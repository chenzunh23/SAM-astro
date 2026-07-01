#!/usr/bin/env python3
"""Batch LSST detection-only background masks for denoised/noisy full patches.

Run this inside an LSST stack environment, for example:

    source ~/lsst_stack/loadLSST.sh
    setup lsst_distrib
    python lsst_pipeline/batch_detect_background.py --patches 4,5 --groups group_01

The generated ``background_mask.npz`` files contain True for pixels outside
LSST detection footprints.  They are intended for AstroCELLECT PU background
supervision and can be cropped cheaply during preprocessing.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from astropy.io import fits


DEFAULT_INPUT_ROOT = Path("/nvme0/zc/scarlet/denoised_fits")
DEFAULT_OUTPUT_ROOT = Path("/nvme0/zc/scarlet/lsst_background_masks")
DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")
DEFAULT_VARIANTS = ("denoised",)


def _split_tokens(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for token in str(value).replace(";", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


def _expand_patches(values: Sequence[str]) -> list[str]:
    patches = _split_tokens(values)
    if not patches:
        return ["4,5"]
    out: list[str] = []
    for patch in patches:
        if patch.lower() == "all":
            out.extend(f"{x},{y}" for x in range(9) for y in range(9))
        else:
            out.append(patch)
    return list(dict.fromkeys(out))


def _patch_dir(root: Path, patch: str) -> Path:
    x_str, y_str = patch.split(",", 1)
    for candidate in (root / f"patch_{x_str}_{y_str}", root / patch, root / patch.replace(",", "_")):
        if candidate.exists():
            return candidate
    return root / f"patch_{x_str}_{y_str}"


def _discover_groups(patch_dir: Path, groups: Sequence[str]) -> list[str]:
    requested = _split_tokens(groups)
    if requested and requested != ["all"]:
        return [g if g.startswith("group_") else f"group_{int(g):02d}" for g in requested]
    found = sorted(path.name for path in patch_dir.iterdir() if path.is_dir() and path.name.startswith("group_"))
    return found


def _origin_from_ltv(header: fits.Header) -> tuple[int, int]:
    if "LTV1" not in header or "LTV2" not in header:
        return 0, 0
    return -int(round(float(header["LTV1"]))), -int(round(float(header["LTV2"])))


def _image_hdu_info(path: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    with fits.open(path, memmap=True) as hdul:
        if "IMAGE" in hdul:
            hdu = hdul["IMAGE"]
        else:
            hdu = next(h for h in hdul if getattr(h, "data", None) is not None and h.data.ndim == 2)
        data = hdu.data
        if data is None or data.ndim != 2:
            raise ValueError(f"{path} has no 2D image plane")
        return (int(data.shape[0]), int(data.shape[1])), _origin_from_ltv(hdu.header)


def _paint_det_background_mask(det_path: Path, shape_yx: tuple[int, int], origin_xy: tuple[int, int]) -> np.ndarray:
    """Return True outside detection footprints."""
    with fits.open(det_path, memmap=True, ignore_missing_end=True) as hdul:
        if len(hdul) <= 4 or hdul[4].data is None:
            return np.ones(shape_yx, dtype=bool)
        rows = [(int(row["y"]), int(row["x0"]), int(row["x1"])) for row in hdul[4].data]

    def paint(subtract_origin: bool) -> tuple[np.ndarray, int]:
        footprint = np.zeros(shape_yx, dtype=bool)
        painted = 0
        ox, oy = origin_xy if subtract_origin else (0, 0)
        for raw_y, raw_x0, raw_x1 in rows:
            y = raw_y - int(oy)
            if y < 0 or y >= shape_yx[0]:
                continue
            x0 = max(0, raw_x0 - int(ox))
            x1 = min(shape_yx[1] - 1, raw_x1 - int(ox))
            if x1 >= x0:
                footprint[y, x0 : x1 + 1] = True
                painted += x1 - x0 + 1
        return footprint, painted

    footprint, painted = paint(subtract_origin=False)
    if painted == 0 and origin_xy != (0, 0):
        footprint, _ = paint(subtract_origin=True)
    return ~footprint


def _atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _task_output_root(output_root: Path, variant: str, tract: int, patch: str, group: str, band: str) -> Path:
    return output_root / variant / str(tract) / patch / group / band


def _run_one(task: dict) -> dict:
    import lsst.afw.image as afwImage
    import lsst.afw.table as afwTable
    import lsst.meas.algorithms as measAlg
    from lsst.pipe.tasks.multiBand import DetectCoaddSourcesTask

    input_path = Path(task["input_path"])
    out_dir = Path(task["out_dir"])
    variant = str(task["variant"])
    tract = int(task["tract"])
    patch = str(task["patch"])
    group = str(task["group"])
    band = str(task["band"])
    overwrite = bool(task["overwrite"])
    write_products = bool(task["write_products"])

    det_path = out_dir / f"det-{variant}-{band}-{tract}-{patch}-{group}.fits"
    mask_path = out_dir / "background_mask.npz"
    summary_path = out_dir / "summary.json"
    if det_path.exists() and mask_path.exists() and not overwrite:
        return {"status": "skipped", "input": str(input_path), "mask": str(mask_path)}

    shape_yx, origin_xy = _image_hdu_info(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not det_path.exists() or overwrite:
        config = DetectCoaddSourcesTask.ConfigClass()
        config.detection.minPixels = int(task["min_pixels"])
        if task["threshold_value"] is not None:
            config.detection.thresholdValue = float(task["threshold_value"])
        if task["n_sigma_to_grow"] is not None:
            config.detection.nSigmaToGrow = float(task["n_sigma_to_grow"])
        if bool(task["disable_bright_prelim"]):
            config.detection.doBrightPrelimDetection = False
        if bool(task["disable_temp_backgrounds"]):
            config.detection.doTempLocalBackground = False
            config.detection.doTempWideBackground = False
        if task["combined_grow"] is not None:
            config.detection.combinedGrow = bool(task["combined_grow"])
        detect_task = DetectCoaddSourcesTask(config=config)
        exposure = afwImage.ExposureF(str(input_path))
        psf_sigma = float(task["psf_sigma"])
        psf_size = int(max(7, 2 * int(np.ceil(psf_sigma * 4.0)) + 1))
        exposure.setPsf(measAlg.SingleGaussianPsf(psf_size, psf_size, psf_sigma))
        original_detection_run = detect_task.detection.run

        def _run_detection_with_sigma(table, exposure, *run_args, **run_kwargs):
            run_kwargs.setdefault("sigma", psf_sigma)
            return original_detection_run(table, exposure, *run_args, **run_kwargs)

        detect_task.detection.run = _run_detection_with_sigma
        result = detect_task.run(
            exposure=exposure,
            idFactory=afwTable.IdFactory.makeSimple(),
            expId=0,
        )
        tmp_det = det_path.with_suffix(det_path.suffix + f".tmp.{os.getpid()}")
        result.outputSources.writeFits(str(tmp_det))
        tmp_det.replace(det_path)
        if write_products:
            result.outputExposure.writeFits(str(out_dir / f"calexp-detect-{variant}-{band}-{tract}-{patch}-{group}.fits"))
            result.outputBackgrounds.writeFits(str(out_dir / f"det_bkgd-{variant}-{band}-{tract}-{patch}-{group}.fits"))

    background_mask = _paint_det_background_mask(det_path, shape_yx, origin_xy)
    _atomic_npz(
        mask_path,
        background_mask=background_mask.astype(np.bool_),
        origin_xy=np.asarray(origin_xy, dtype=np.int32),
        shape_yx=np.asarray(shape_yx, dtype=np.int32),
        input_fits=np.asarray(str(input_path)),
        det_fits=np.asarray(str(det_path)),
    )
    summary = {
        "status": "ok",
        "input": str(input_path),
        "det": str(det_path),
        "mask": str(mask_path),
        "variant": variant,
        "tract": tract,
        "patch": patch,
        "group": group,
        "band": band,
        "shape_yx": list(shape_yx),
        "origin_xy": list(origin_xy),
        "background_pixels": int(np.count_nonzero(background_mask)),
        "footprint_pixels": int(background_mask.size - np.count_nonzero(background_mask)),
        "background_fraction": float(np.count_nonzero(background_mask) / max(background_mask.size, 1)),
        "min_pixels": int(task["min_pixels"]),
        "threshold_value": task["threshold_value"],
        "n_sigma_to_grow": task["n_sigma_to_grow"],
        "psf_sigma": float(task["psf_sigma"]),
        "disable_bright_prelim": bool(task["disable_bright_prelim"]),
        "disable_temp_backgrounds": bool(task["disable_temp_backgrounds"]),
        "combined_grow": task["combined_grow"],
    }
    _atomic_json(summary_path, summary)
    return summary


def _build_tasks(args: argparse.Namespace) -> list[dict]:
    tasks: list[dict] = []
    for patch in _expand_patches(args.patches):
        patch_dir = _patch_dir(args.input_root, patch)
        if not patch_dir.exists():
            raise FileNotFoundError(f"missing denoised patch dir: {patch_dir}")
        for group in _discover_groups(patch_dir, args.groups):
            for band in args.bands:
                for variant in args.variants:
                    input_path = patch_dir / group / band / f"{variant}.fits"
                    if not input_path.exists():
                        if args.ignore_missing:
                            print(f"[missing] {input_path}", flush=True)
                            continue
                        raise FileNotFoundError(f"missing input FITS: {input_path}")
                    tasks.append(
                        {
                            "input_path": str(input_path),
                            "out_dir": str(_task_output_root(args.output_root, variant, args.tract, patch, group, band)),
                            "variant": variant,
                            "tract": args.tract,
                            "patch": patch,
                            "group": group,
                            "band": band,
                            "overwrite": args.overwrite,
                            "write_products": args.write_products,
                            "min_pixels": args.min_pixels,
                            "threshold_value": args.threshold_value,
                            "n_sigma_to_grow": args.n_sigma_to_grow,
                            "psf_sigma": args.psf_sigma,
                            "disable_bright_prelim": args.disable_bright_prelim,
                            "disable_temp_backgrounds": args.disable_temp_backgrounds,
                            "combined_grow": args.combined_grow,
                        }
                    )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tract", type=int, default=9813)
    parser.add_argument("--patches", nargs="+", default=["4,5"])
    parser.add_argument("--groups", nargs="+", default=["all"])
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-pixels", type=int, default=15)
    parser.add_argument("--threshold-value", type=float, default=None)
    parser.add_argument("--n-sigma-to-grow", type=float, default=None)
    parser.add_argument("--disable-bright-prelim", action="store_true")
    parser.add_argument("--disable-temp-backgrounds", action="store_true")
    parser.add_argument(
        "--combined-grow",
        type=int,
        choices=(0, 1),
        default=None,
        help="Override LSST detection.combinedGrow with 0/1. Default leaves LSST config unchanged.",
    )
    parser.add_argument(
        "--psf-sigma",
        type=float,
        default=1.5,
        help="Gaussian PSF sigma in pixels injected when an input FITS has no PSF. Default roughly matches HSC seeing.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-products", action="store_true")
    parser.add_argument("--ignore-missing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    tasks = _build_tasks(args)
    print(f"[batch-detect] tasks={len(tasks)} output_root={args.output_root}", flush=True)
    if not tasks:
        return 0

    workers = max(1, int(args.workers))
    failures: list[dict] = []
    if workers == 1:
        for idx, task in enumerate(tasks, 1):
            try:
                result = _run_one(task)
                print(f"[{idx}/{len(tasks)}] {result.get('status', 'ok')} {result.get('mask')}", flush=True)
            except Exception as exc:
                failures.append({"task": task, "error": repr(exc)})
                print(f"[{idx}/{len(tasks)}] FAILED {task['input_path']}: {exc}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, task): task for task in tasks}
            for idx, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                try:
                    result = future.result()
                    print(f"[{idx}/{len(tasks)}] {result.get('status', 'ok')} {result.get('mask')}", flush=True)
                except Exception as exc:
                    failures.append({"task": task, "error": repr(exc)})
                    print(f"[{idx}/{len(tasks)}] FAILED {task['input_path']}: {exc}", flush=True)

    summary = {
        "tasks": len(tasks),
        "failures": failures,
        "output_root": str(args.output_root),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    _atomic_json(args.output_root / "batch_summary.json", summary)
    if failures:
        raise RuntimeError(f"{len(failures)} LSST detection task(s) failed; see {args.output_root / 'batch_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
