"""Run and evaluate a grid of LSST/SAM deblend experiments.

The script prepares 512x512 coadd cutouts, optionally runs the LSST-native and
SAM-detection pipelines, crops the reference meas catalog to the same cutouts,
and computes centroid-match completeness/purity by magnitude.  It can also run
in ``--plot-only`` mode to recompute CSV summaries and figures from existing
pipeline outputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import pickle
import shlex
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from crop_lsst_catalog import CutoutBox, crop_meas_catalog  # noqa: E402
from evaluate_centroid_matches import (  # noqa: E402
    DEFAULT_PIXEL_SCALE,
    DEFAULT_RADIUS_ARCSEC,
    _load_points,
    match_nearest_unique,
)


BANDS =  ("HSC-I", "HSC-R", "HSC-G") # ("HSC-G", "HSC-R", "HSC-I")
PIXEL_PLANES = ("IMAGE", "MASK", "VARIANCE")
DEFAULT_PREVIOUS_ORIGIN = (18204, 20924)


@dataclass(frozen=True)
class CutoutSpec:
    """One square cutout defined in parent-patch pixel coordinates."""

    name: str
    x0: int
    y0: int
    size: int

    @property
    def box(self) -> CutoutBox:
        return CutoutBox(float(self.x0), float(self.y0), float(self.size), float(self.size))


def _origin_from_ltv(header: fits.Header) -> tuple[int, int]:
    if "LTV1" not in header or "LTV2" not in header:
        raise KeyError("IMAGE header needs LTV1/LTV2 to infer parent-patch coordinates")
    return -int(round(float(header["LTV1"]))), -int(round(float(header["LTV2"])))


def _cropped_header(header: fits.Header, *, local_x0: int, local_y0: int) -> fits.Header:
    out = header.copy()
    if "LTV1" in out:
        out["LTV1"] = float(out["LTV1"]) - local_x0
    if "LTV2" in out:
        out["LTV2"] = float(out["LTV2"]) - local_y0
    if "CRPIX1" in out:
        out["CRPIX1"] = float(out["CRPIX1"]) - local_x0
    if "CRPIX2" in out:
        out["CRPIX2"] = float(out["CRPIX2"]) - local_y0
    if "CRVAL1A" in out:
        out["CRVAL1A"] = float(out["CRVAL1A"]) + local_x0
    if "CRVAL2A" in out:
        out["CRVAL2A"] = float(out["CRVAL2A"]) + local_y0
    return out


def _finite_replacement(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else 0.0


def _new_image_hdu_like(hdu, *, data: np.ndarray, header: fits.Header):
    if isinstance(hdu, fits.PrimaryHDU):
        return fits.PrimaryHDU(data=data, header=header)
    return fits.ImageHDU(data=data, header=header, name=hdu.name)


def crop_exposure_cutout(
    *,
    source_path: Path,
    output_path: Path,
    parent_x0: int,
    parent_y0: int,
    size: int,
    clean_nonfinite: bool,
) -> None:
    with fits.open(source_path, memmap=False) as hdul:
        if "IMAGE" not in hdul:
            raise KeyError(f"{source_path} has no IMAGE extension")
        source_origin = _origin_from_ltv(hdul["IMAGE"].header)
        local_x0 = int(parent_x0 - source_origin[0])
        local_y0 = int(parent_y0 - source_origin[1])

        for plane in PIXEL_PLANES:
            if plane not in hdul:
                raise KeyError(f"{source_path} has no {plane} extension")
            data = hdul[plane].data
            if data is None or data.ndim != 2:
                raise ValueError(f"{source_path}[{plane}] is not a 2D image")
            if local_x0 < 0 or local_y0 < 0 or local_x0 + size > data.shape[1] or local_y0 + size > data.shape[0]:
                raise ValueError(
                    f"{source_path}[{plane}] cannot cover parent cutout "
                    f"x={parent_x0}:{parent_x0 + size}, y={parent_y0}:{parent_y0 + size}; "
                    f"source origin={source_origin}, shape={data.shape}"
                )

        out_hdus = []
        for hdu in hdul:
            if hdu.name in PIXEL_PLANES:
                data = np.asarray(hdu.data[local_y0 : local_y0 + size, local_x0 : local_x0 + size]).copy()
                if clean_nonfinite and np.issubdtype(data.dtype, np.floating) and not np.all(np.isfinite(data)):
                    fill = _finite_replacement(data)
                    data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill).astype(data.dtype, copy=False)
                header = _cropped_header(hdu.header, local_x0=local_x0, local_y0=local_y0)
                out_hdus.append(_new_image_hdu_like(hdu, data=data, header=header))
            else:
                # LSST archive tables may contain heap-backed variable-length
                # arrays; avoid Astropy deep-copying those tables.
                out_hdus.append(hdu)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(out_hdus).writeto(output_path, overwrite=True)


def _coadd_path(root: Path, band: str, tract: int, patch: str) -> Path:
    return root / band / f"deepCoadd-{band}-{tract}-{patch}.fits"


def make_cutouts(args: argparse.Namespace, specs: list[CutoutSpec]) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for spec in specs:
        cutout_paths: dict[str, Path] = {}
        for band in BANDS:
            src = _coadd_path(args.coadd_root, band, args.tract, args.patch)
            dst = args.output_root / "cutouts" / spec.name / band / src.name
            if args.dry_run:
                cutout_paths[band] = dst
                continue
            if not dst.exists() or not args.skip_existing:
                crop_exposure_cutout(
                    source_path=src,
                    output_path=dst,
                    parent_x0=spec.x0,
                    parent_y0=spec.y0,
                    size=spec.size,
                    clean_nonfinite=not args.no_clean_nonfinite,
                )
            cutout_paths[band] = dst
        result[spec.name] = cutout_paths
    return result


def _run_pipeline(
    *,
    args: argparse.Namespace,
    method: str,
    spec: CutoutSpec,
    cutout_paths: dict[str, Path],
    env_update: dict[str, str] | None = None,
) -> Path:
    outdir = args.output_root / "runs" / spec.name / method
    manifest = outdir / "manifest.json"
    if args.skip_existing and manifest.exists():
        return outdir

    cmd = [
        args.python,
        str(args.pipeline_script),
        "--detection-mode",
        method,
        "--repo",
        str(args.repo),
        "--tract",
        str(args.tract),
        "--patch",
        str(args.patch),
        "--output-dir",
        str(outdir),
        "--clip-sky-sources-to-exposure-bbox",
    ]
    for band in BANDS:
        cmd.extend(["--coadd", f"{band}={cutout_paths[band]}"])
    if method == "lsst" and args.lsst_extra_args:
        cmd.extend(shlex.split(args.lsst_extra_args))
    if method == "sam" and args.sam_extra_args:
        cmd.extend(shlex.split(args.sam_extra_args))

    outdir.parent.mkdir(parents=True, exist_ok=True)
    log_path = outdir.with_suffix(f".{method}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_update:
        env.update(env_update)
    label = f"{method} {spec.name}"
    if env_update and "CUDA_VISIBLE_DEVICES" in env_update:
        label += f" gpu={env_update['CUDA_VISIBLE_DEVICES']}"
    print(f"running {label}:", " ".join(shlex.quote(str(part)) for part in cmd))
    if args.dry_run:
        return outdir
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=args.pipeline_root, stdout=log, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        if args.continue_on_error:
            print(f"WARNING: {method} {spec.name} failed; see {log_path}")
            return outdir
        raise RuntimeError(f"{method} {spec.name} failed with exit code {proc.returncode}; see {log_path}")
    return outdir


def _submit_runs(
    *,
    args: argparse.Namespace,
    method: str,
    specs: list[CutoutSpec],
    cutouts: dict[str, dict[str, Path]],
) -> None:
    """Run one method over all cutouts with method-appropriate parallelism."""
    if method == "lsst":
        max_workers = max(1, int(args.lsst_workers))
        env_update = {
            "OMP_NUM_THREADS": str(args.lsst_threads_per_worker),
            "OPENBLAS_NUM_THREADS": str(args.lsst_threads_per_worker),
            "MKL_NUM_THREADS": str(args.lsst_threads_per_worker),
            "NUMEXPR_NUM_THREADS": str(args.lsst_threads_per_worker),
        }

        def task(spec: CutoutSpec) -> Path:
            return _run_pipeline(args=args, method=method, spec=spec, cutout_paths=cutouts[spec.name], env_update=env_update)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task, spec) for spec in specs]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        return

    if method == "sam":
        gpus = [gpu.strip() for gpu in args.sam_gpus.split(",") if gpu.strip()]
        if not gpus:
            gpus = ["0"]
        slots = [gpu for gpu in gpus for _ in range(max(1, int(args.sam_workers_per_gpu)))]
        max_workers = len(slots)

        def task(item: tuple[int, CutoutSpec]) -> Path:
            index, spec = item
            gpu = slots[index % len(slots)]
            env_update = {"CUDA_VISIBLE_DEVICES": gpu}
            return _run_pipeline(args=args, method=method, spec=spec, cutout_paths=cutouts[spec.name], env_update=env_update)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task, item) for item in enumerate(specs)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        return

    raise ValueError(f"unknown method: {method}")


def crop_reference_catalog(args: argparse.Namespace, spec: CutoutSpec) -> Path:
    out = args.output_root / "reference_catalogs" / f"{spec.name}_meas.fits"
    if out.exists() and args.skip_existing:
        return out
    table = crop_meas_catalog(
        args.reference_catalog,
        box=spec.box,
        x_col=args.ref_x,
        y_col=args.ref_y,
        margin=float(args.reference_margin),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write(out, format="fits", overwrite=True)
    return out


def _flux_to_mag(flux: np.ndarray, zeropoint: float) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    mag = np.full(flux.shape, np.nan, dtype=float)
    good = np.isfinite(flux) & (flux > 0)
    mag[good] = float(zeropoint) - 2.5 * np.log10(flux[good])
    return mag


def _choose_flux_col(table: Table, preferred: str, fallbacks: tuple[str, ...]) -> str:
    for name in (preferred, *fallbacks):
        if name in {"", "auto", "none", "None"}:
            continue
        if name in table.colnames:
            return name
    raise KeyError(f"none of the flux columns exists: {(preferred, *fallbacks)}")


def _band_to_spectrum_label(band: str) -> str:
    label = band.strip()
    if label.startswith("HSC-"):
        label = label.split("-", 1)[1]
    return label.lower()


def _source_spectrum_from_csv(run_dir: Path, pred_points, *, band: str) -> np.ndarray | None:
    csv_path = run_dir / "vis" / "source_panels" / "source_spectra.csv"
    if not csv_path.exists():
        return None
    spectrum_col = f"spectrum_{_band_to_spectrum_label(band)}"
    by_id: dict[int, float] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or spectrum_col not in reader.fieldnames:
            return None
        for row in reader:
            try:
                by_id[int(row["source_id"])] = float(row[spectrum_col])
            except (KeyError, TypeError, ValueError):
                continue
    flux = np.full(pred_points.n, np.nan, dtype=float)
    for i, source_id in enumerate(pred_points.ids):
        if int(source_id) in by_id:
            flux[i] = by_id[int(source_id)]
    return flux


def _source_spectrum_from_model(run_dir: Path, pred_points, *, band: str) -> np.ndarray | None:
    pickle_path = run_dir / "deblend" / "deepCoadd_scarletModelData.pickle"
    if not pickle_path.exists():
        return None
    with pickle_path.open("rb") as handle:
        try:
            model_data = pickle.load(handle)
        except ModuleNotFoundError as exc:
            print(f"WARNING: cannot read scarlet model pickle without module {exc.name!r}: {pickle_path}")
            return None
    if not getattr(model_data, "blends", None):
        return None

    first_blend = next(iter(model_data.blends.values()))
    bands = [str(value).lower() for value in getattr(first_blend, "bands", [])]
    band_label = _band_to_spectrum_label(band)
    if band_label not in bands:
        return None
    band_index = bands.index(band_label)

    by_id: dict[int, float] = {}
    for blend in model_data.blends.values():
        for source_id, source in blend.sources.items():
            total = 0.0
            for component in getattr(source, "factorized_components", []):
                spectrum = np.asarray(component.spectrum, dtype=float)
                if band_index < spectrum.size and np.isfinite(spectrum[band_index]):
                    total += float(spectrum[band_index])
            by_id[int(source_id)] = total

    flux = np.full(pred_points.n, np.nan, dtype=float)
    for i, source_id in enumerate(pred_points.ids):
        if int(source_id) in by_id:
            flux[i] = by_id[int(source_id)]
    return flux


def _source_spectrum_flux_for_binning(run_dir: Path, pred_points, *, band: str) -> np.ndarray | None:
    """Load scarlet source-spectrum fluxes aligned to prediction source ids."""
    flux = _source_spectrum_from_csv(run_dir, pred_points, band=band)
    if flux is not None and np.count_nonzero(np.isfinite(flux) & (flux > 0)) > 0:
        return flux
    flux = _source_spectrum_from_model(run_dir, pred_points, band=band)
    if flux is not None and np.count_nonzero(np.isfinite(flux) & (flux > 0)) > 0:
        return flux
    return None


def _prediction_flux_for_binning(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    pred_points,
) -> tuple[np.ndarray, str]:
    if args.pred_flux_col in {"", "auto", "source_spectrum"}:
        flux = _source_spectrum_flux_for_binning(run_dir, pred_points, band=args.pred_spectrum_band)
        if flux is not None:
            return flux, f"scarlet_spectrum_{_band_to_spectrum_label(args.pred_spectrum_band)}"

    candidate_cols = []
    if args.pred_flux_col not in {"", "auto", "none", "None"}:
        candidate_cols.append(args.pred_flux_col)
    candidate_cols.extend(["deblend_scarletFlux", "deblend_peak_instFlux"])
    for col in candidate_cols:
        if col not in pred_points.table.colnames:
            continue
        flux = np.asarray(pred_points.table[col], dtype=float)[pred_points.table_indices]
        if np.count_nonzero(np.isfinite(flux) & (flux > 0)) > 0:
            return flux, col

    return np.full(pred_points.n, np.nan, dtype=float), "unavailable"


def _mag_bin_mask(mag: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = np.isfinite(mag)
    if np.isfinite(lo):
        mask &= mag >= lo
    if np.isfinite(hi):
        mask &= mag < hi
    return mask


def _magnitude_bins(mag_min: float, mag_max: float, bin_size: float) -> list[tuple[float, float, float]]:
    """Return underflow, regular, and overflow magnitude bins.

    The displayed range is fixed by mag_min/mag_max.  Values outside that range
    are kept in the first/last bins instead of stretching plots to extreme
    outliers such as very faint SAM false positives.
    """
    bins: list[tuple[float, float, float]] = []
    bins.append((-np.inf, mag_min, mag_min - 0.5 * bin_size))
    edges = np.arange(mag_min, mag_max + bin_size * 0.5, bin_size)
    bins.extend((float(lo), float(hi), float(0.5 * (lo + hi))) for lo, hi in zip(edges[:-1], edges[1:]))
    bins.append((mag_max, np.inf, mag_max + 0.5 * bin_size))
    return bins


def _format_count_bins(counts: dict[float, int], *, bin_size: float, top_n: int) -> str:
    if not counts:
        return ""
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    return "; ".join(f"{lo:g}-{lo + bin_size:g}:{count}" for lo, count in items)


def _parse_count_bins(value: str) -> dict[float, int]:
    counts: dict[float, int] = {}
    if not value:
        return counts
    for part in str(value).split(";"):
        part = part.strip()
        if not part or ":" not in part or "-" not in part:
            continue
        label, count_text = part.rsplit(":", 1)
        lo_text = label.split("-", 1)[0]
        try:
            counts[float(lo_text)] = counts.get(float(lo_text), 0) + int(count_text)
        except ValueError:
            continue
    return counts


def _fp_detail_bins(pred_mag: np.ndarray, pred_used: np.ndarray, row_mask: np.ndarray, *, bin_size: float, top_n: int) -> str:
    fp_mag = pred_mag[row_mask & ~pred_used & np.isfinite(pred_mag)]
    if fp_mag.size == 0:
        return ""
    bucket_left = np.floor(fp_mag / bin_size) * bin_size
    counts: dict[float, int] = {}
    for lo in bucket_left:
        counts[float(lo)] = counts.get(float(lo), 0) + 1
    return _format_count_bins(counts, bin_size=bin_size, top_n=top_n)


def _rows_for_bins(
    *,
    cutout: str,
    run_dir: Path,
    method: str,
    ref_points,
    pred_points,
    ref_used: np.ndarray,
    pred_used: np.ndarray,
    args: argparse.Namespace,
) -> list[dict]:
    """Compute per-cutout completeness and purity rows for magnitude bins.

    Reference magnitudes come from the cropped catalog.  Prediction magnitudes
    come from scarlet source spectra when available, so purity is binned by the
    deblend model flux rather than by catalog flux.
    """
    ref_flux_col = _choose_flux_col(
        ref_points.table,
        args.ref_flux_col,
        ("modelfit_CModel_instFlux", "base_PsfFlux_instFlux", "base_SdssShape_instFlux"),
    )
    ref_flux = np.asarray(ref_points.table[ref_flux_col], dtype=float)[ref_points.table_indices]
    pred_flux, pred_flux_col = _prediction_flux_for_binning(args=args, run_dir=run_dir, pred_points=pred_points)
    ref_mag = _flux_to_mag(ref_flux, args.mag_zero_point)
    pred_mag = _flux_to_mag(pred_flux, args.mag_zero_point)

    finite = np.concatenate([ref_mag[np.isfinite(ref_mag)], pred_mag[np.isfinite(pred_mag)]])
    if finite.size == 0:
        return []
    mag_min = float(args.mag_min) if args.mag_min is not None else np.floor(np.nanmin(finite) / args.bin_size) * args.bin_size
    mag_max = float(args.mag_max) if args.mag_max is not None else np.ceil(np.nanmax(finite) / args.bin_size) * args.bin_size
    bins = _magnitude_bins(mag_min, mag_max, float(args.bin_size))

    rows = []
    for lo, hi, center in bins:
        ref_in = _mag_bin_mask(ref_mag, lo, hi)
        pred_in = _mag_bin_mask(pred_mag, lo, hi)
        ref_total = int(np.count_nonzero(ref_in))
        pred_total = int(np.count_nonzero(pred_in))
        ref_matched = int(np.count_nonzero(ref_in & ref_used))
        pred_matched = int(np.count_nonzero(pred_in & pred_used))
        fp_detail = ""
        if not np.isfinite(lo) or not np.isfinite(hi):
            fp_detail = _fp_detail_bins(
                pred_mag,
                pred_used,
                pred_in,
                bin_size=float(args.fp_detail_bin_size),
                top_n=int(args.fp_detail_top_n),
            )
        rows.append(
            {
                "cutout": cutout,
                "method": method,
                "mag_left": float(lo),
                "mag_right": float(hi),
                "mag_center": float(center),
                "reference_flux_col": ref_flux_col,
                "prediction_flux_col": pred_flux_col,
                "reference_total": ref_total,
                "reference_matched": ref_matched,
                "completeness": ref_matched / ref_total if ref_total else np.nan,
                "prediction_total": pred_total,
                "prediction_matched": pred_matched,
                "purity": pred_matched / pred_total if pred_total else np.nan,
                "prediction_fp_detail_bins": fp_detail,
            }
        )
    return rows


def evaluate_run(args: argparse.Namespace, *, spec: CutoutSpec, method: str, run_dir: Path, ref_catalog: Path) -> list[dict]:
    pred_catalog = run_dir / "deblend" / "deepCoadd_deblendedFlux.fits"
    if not pred_catalog.exists():
        print(f"WARNING: missing prediction catalog: {pred_catalog}")
        return []
    radius_pix = float(args.match_radius_arcsec) / float(args.pixel_scale)
    ref_points = _load_points(
        ref_catalog,
        x_col=args.ref_x,
        y_col=args.ref_y,
        role="ref",
        hdu=1,
        leaf_only=True,
        drop_flagged_centroids=True,
        require_science_model=False,
    )
    pred_points = _load_points(
        pred_catalog,
        x_col=args.pred_x,
        y_col=args.pred_y,
        role="pred",
        hdu=1,
        leaf_only=True,
        drop_flagged_centroids=True,
        require_science_model=True,
    )
    matches, ref_used, pred_used = match_nearest_unique(ref_points, pred_points, radius_pix)
    summary = {
        "cutout": spec.name,
        "method": method,
        "reference_count": ref_points.n,
        "prediction_count": pred_points.n,
        "matched_count": len(matches),
        "recall": len(matches) / ref_points.n if ref_points.n else np.nan,
        "precision": len(matches) / pred_points.n if pred_points.n else np.nan,
        "match_radius_pix": radius_pix,
        "match_radius_arcsec": float(args.match_radius_arcsec),
    }
    summary_path = run_dir / "magnitude_match_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return _rows_for_bins(
        cutout=spec.name,
        run_dir=run_dir,
        method=method,
        ref_points=ref_points,
        pred_points=pred_points,
        ref_used=ref_used,
        pred_used=pred_used,
        args=args,
    )


def _prediction_catalog_path(args: argparse.Namespace, spec: CutoutSpec, method: str) -> Path:
    return args.output_root / "runs" / spec.name / method / "deblend" / "deepCoadd_deblendedFlux.fits"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cutout",
        "method",
        "mag_left",
        "mag_right",
        "mag_center",
        "reference_flux_col",
        "prediction_flux_col",
        "reference_total",
        "reference_matched",
        "completeness",
        "prediction_total",
        "prediction_matched",
        "purity",
        "prediction_fp_detail_bins",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Sum per-cutout rows into method-level magnitude-bin totals."""
    grouped: dict[tuple[str, float, float], dict] = {}
    for row in rows:
        key = (row["method"], row["mag_left"], row["mag_right"])
        item = grouped.setdefault(
            key,
            {
                "method": row["method"],
                "mag_left": row["mag_left"],
                "mag_right": row["mag_right"],
                "mag_center": row["mag_center"],
                "reference_total": 0,
                "reference_matched": 0,
                "prediction_total": 0,
                "prediction_matched": 0,
                "_prediction_fp_detail_counts": {},
            },
        )
        item["reference_total"] += int(row["reference_total"])
        item["reference_matched"] += int(row["reference_matched"])
        item["prediction_total"] += int(row["prediction_total"])
        item["prediction_matched"] += int(row["prediction_matched"])
        for lo, count in _parse_count_bins(row.get("prediction_fp_detail_bins", "")).items():
            detail_counts = item["_prediction_fp_detail_counts"]
            detail_counts[lo] = detail_counts.get(lo, 0) + count
    out = []
    for item in grouped.values():
        ref_total = item["reference_total"]
        pred_total = item["prediction_total"]
        item["completeness"] = item["reference_matched"] / ref_total if ref_total else np.nan
        item["purity"] = item["prediction_matched"] / pred_total if pred_total else np.nan
        item["prediction_fp_detail_bins"] = _format_count_bins(
            item.pop("_prediction_fp_detail_counts"),
            bin_size=1.0,
            top_n=5,
        )
        out.append(item)
    return sorted(out, key=lambda r: (r["method"], r["mag_left"]))

def plot_curves(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted({row["method"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        x = np.array([row["mag_center"] for row in method_rows], dtype=float)
        # Only consider bins in [23, 30] for plotting, to avoid noisy extremes with very few sources.
        mask = (x >= 23) & (x <= 30)
        x = x[mask]
        completeness = np.array([row["completeness"] for row in method_rows], dtype=float)
        completeness = completeness[mask]
        purity = np.array([row["purity"] for row in method_rows], dtype=float)
        purity = purity[mask]
        # Keep the line order and displayed x-axis in increasing magnitude.
        order = np.argsort(x)
        x = x[order]
        completeness = completeness[order]
        purity = purity[order]
        axes[0].plot(x, completeness, marker="o", linewidth=1.8, label=method)
        axes[1].plot(x, purity, marker="o", linewidth=1.8, label=method)
    for ax, title in zip(axes, ("Completeness by catalog magnitude", "Purity by deblend magnitude")):
        ax.set_title(title)
        ax.set_xlabel("instrumental magnitude")
        ax.set_ylabel("score")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    if methods:
        axes[0].legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _ordered_methods(rows: list[dict]) -> list[str]:
    methods = sorted({str(row["method"]) for row in rows})
    preferred = [name for name in ("lsst", "sam") if name in methods]
    return preferred + [name for name in methods if name not in preferred]


def _mag_bin_label(lo: float, hi: float) -> str:
    if not np.isfinite(lo):
        return f"<{hi:g}"
    if not np.isfinite(hi):
        return f">={lo:g}"
    return f"{lo:g}-{hi:g}"


def plot_count_histograms(completeness_path: Path, purity_path: Path, rows: list[dict]) -> None:
    """Write count histograms used to audit the curve metrics.

    The completeness histogram compares catalog counts with TP counts by catalog
    magnitude.  The purity histogram is a stacked TP+FP bar chart by prediction
    magnitude and deliberately omits catalog counts because the x-axis uses a
    different flux definition.
    """
    completeness_path.parent.mkdir(parents=True, exist_ok=True)
    purity_path.parent.mkdir(parents=True, exist_ok=True)
    methods = _ordered_methods(rows)
    bins = sorted({(float(row["mag_left"]), float(row["mag_right"]), float(row["mag_center"])) for row in rows})
    if not bins or not methods:
        return

    by_method_bin = {
        (str(row["method"]), float(row["mag_left"]), float(row["mag_right"])): row
        for row in rows
    }
    reference_by_bin = []
    for lo, hi, _ in bins:
        candidates = [
            int(row["reference_total"])
            for row in rows
            if float(row["mag_left"]) == lo and float(row["mag_right"]) == hi
        ]
        reference_by_bin.append(max(candidates) if candidates else 0)

    x = np.arange(len(bins), dtype=float)
    labels = [_mag_bin_label(lo, hi) for lo, hi, _ in bins]
    width = min(0.28, 0.75 / max(1, len(methods)))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    figsize = (max(11.0, 0.48 * len(bins)), 5.2)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.bar(x, reference_by_bin, width=0.86, color="0.82", edgecolor="0.55", linewidth=0.6, label="catalog sources")
    for offset, method in zip(offsets, methods):
        tp = [
            int(by_method_bin.get((method, lo, hi), {}).get("reference_matched", 0))
            for lo, hi, _ in bins
        ]
        ax.bar(x + offset, tp, width=width, label=f"{method.upper()} TP")
    ax.set_title("Catalog sources and matched true positives by catalog magnitude")
    ax.set_xlabel("catalog instrumental magnitude bin")
    ax.set_ylabel("source count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=min(3, len(methods) + 1))
    fig.savefig(completeness_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    overflow_notes = []
    for offset, method in zip(offsets, methods):
        tp = [
            int(by_method_bin.get((method, lo, hi), {}).get("prediction_matched", 0))
            for lo, hi, _ in bins
        ]
        fp = [
            max(
                0,
                int(by_method_bin.get((method, lo, hi), {}).get("prediction_total", 0))
                - int(by_method_bin.get((method, lo, hi), {}).get("prediction_matched", 0)),
            )
            for lo, hi, _ in bins
        ]
        ax.bar(x + offset, tp, width=width, label=f"{method.upper()} TP")
        ax.bar(x + offset, fp, width=width, bottom=tp, label=f"{method.upper()} FP")
        for lo, hi, _ in bins:
            if np.isfinite(hi):
                continue
            detail = by_method_bin.get((method, lo, hi), {}).get("prediction_fp_detail_bins", "")
            if detail:
                overflow_notes.append(f"{method.upper()} >= {lo:g} FP peaks: {detail}")
    ax.set_title("Predicted sources by deblend magnitude")
    ax.set_xlabel("predicted instrumental magnitude bin")
    ax.set_ylabel("source count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=min(3, len(methods) + 1))
    if overflow_notes:
        ax.text(
            0.99,
            0.98,
            "\n".join(overflow_notes),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.75", "linewidth": 0.6, "pad": 3},
        )
    fig.savefig(purity_path, dpi=180)
    plt.close(fig)


def _image_bounds(coadd_root: Path, *, tract: int, patch: str, band: str = "HSC-I") -> tuple[int, int, int, int]:
    path = _coadd_path(coadd_root, band, tract, patch)
    with fits.open(path, memmap=False) as hdul:
        data = hdul["IMAGE"].data
        if data is None or data.ndim != 2:
            raise ValueError(f"{path}[IMAGE] is not a 2D image")
        x0, y0 = _origin_from_ltv(hdul["IMAGE"].header)
        height, width = data.shape
    return int(x0), int(y0), int(width), int(height)


def _anchored_axis_origins(
    *,
    start: int,
    length: int,
    tile: int,
    count: int,
    anchor: int,
) -> tuple[list[int], str]:
    max_origin = start + length - tile
    if count <= 0:
        return [], "empty"
    if length < tile:
        raise ValueError(f"axis length {length} is smaller than tile size {tile}")
    if count * tile > length:
        raise ValueError(f"axis cannot fit {count} non-edge tiles of size {tile} into length {length}")
    if not (start <= anchor <= max_origin):
        raise ValueError(f"anchor {anchor} is outside valid origin range {start}:{max_origin}")

    candidate_regular = []
    for anchor_index in range(count):
        first = anchor - anchor_index * tile
        last = first + (count - 1) * tile
        if start <= first and last <= max_origin:
            candidate_regular.append((abs(anchor_index - (count - 1) / 2), first))
    if candidate_regular:
        _, first = min(candidate_regular)
        return [first + i * tile for i in range(count)], "regular_anchor_aligned"

    origins = [start + i * tile for i in range(count)]
    replace_index = min(range(count), key=lambda i: abs(origins[i] - anchor))
    replaced = origins[replace_index]
    origins[replace_index] = anchor
    origins = sorted(set(origins))
    if len(origins) != count:
        for value in [start + i * tile for i in range(length // tile)]:
            if value not in origins and start <= value <= max_origin:
                origins.append(value)
            if len(origins) == count:
                break
    origins = sorted(origins)
    if len(origins) != count:
        raise RuntimeError(f"could not generate {count} valid origins including anchor {anchor}")
    return origins, f"nonuniform_anchor_inserted_replaced_{replaced}"


def generate_grid_specs(args: argparse.Namespace) -> tuple[list[CutoutSpec], dict]:
    image_x0, image_y0, width, height = _image_bounds(args.coadd_root, tract=args.tract, patch=args.patch)
    anchor_x, anchor_y = [int(float(v)) for v in args.grid_anchor.split(",", 1)]
    x_origins, x_mode = _anchored_axis_origins(
        start=image_x0,
        length=width,
        tile=args.size,
        count=args.grid_cols,
        anchor=anchor_x,
    )
    y_origins, y_mode = _anchored_axis_origins(
        start=image_y0,
        length=height,
        tile=args.size,
        count=args.grid_rows,
        anchor=anchor_y,
    )
    specs = [
        CutoutSpec(name=f"grid_r{row:02d}_c{col:02d}_x{x0}_y{y0}", x0=x0, y0=y0, size=args.size)
        for row, y0 in enumerate(y_origins)
        for col, x0 in enumerate(x_origins)
    ]
    metadata = {
        "image_origin": [image_x0, image_y0],
        "image_shape": [width, height],
        "tile_size": args.size,
        "grid_cols": args.grid_cols,
        "grid_rows": args.grid_rows,
        "grid_count": len(specs),
        "grid_anchor": [anchor_x, anchor_y],
        "x_origins": x_origins,
        "y_origins": y_origins,
        "x_generation_mode": x_mode,
        "y_generation_mode": y_mode,
        "contains_anchor_origin": any(spec.x0 == anchor_x and spec.y0 == anchor_y for spec in specs),
    }
    if not metadata["contains_anchor_origin"]:
        raise RuntimeError(f"generated grid does not contain anchor origin {anchor_x},{anchor_y}")
    return specs, metadata


def parse_origins(values: list[str], *, size: int) -> list[CutoutSpec]:
    specs = []
    seen = set()
    for index, value in enumerate(values, start=1):
        x_text, y_text = value.split(",", 1)
        x0 = int(float(x_text))
        y0 = int(float(y_text))
        key = (x0, y0)
        if key in seen:
            continue
        seen.add(key)
        name = "previous" if key == DEFAULT_PREVIOUS_ORIGIN else f"cutout_{index:03d}_x{x0}_y{y0}"
        specs.append(CutoutSpec(name=name, x0=x0, y0=y0, size=size))
    return specs


def build_specs(args: argparse.Namespace) -> tuple[list[CutoutSpec], dict]:
    if args.origin:
        specs = parse_origins(args.origin, size=args.size)
        try:
            grid_specs, _ = generate_grid_specs(args)
            grid_names = {(spec.x0, spec.y0): spec.name for spec in grid_specs}
            specs = [
                CutoutSpec(name=grid_names.get((spec.x0, spec.y0), spec.name), x0=spec.x0, y0=spec.y0, size=spec.size)
                for spec in specs
            ]
        except Exception:
            pass
        metadata = {
            "mode": "manual",
            "grid_count": len(specs),
            "origins": [[spec.x0, spec.y0] for spec in specs],
        }
        return specs, metadata
    specs, metadata = generate_grid_specs(args)
    metadata["mode"] = "grid"
    return specs, metadata


def main() -> None:
    warnings.filterwarnings("ignore", category=AstropyWarning)
    parser = argparse.ArgumentParser(
        description=(
            "Run LSST/SAM deblend experiments on 512x512 cutouts from a large coadd, "
            "then plot completeness and purity as magnitude-binned curves."
        )
    )
    parser.add_argument("--coadd-root", type=Path, default=PIPELINE_ROOT / "fits/projection_cutout")
    parser.add_argument("--reference-catalog", type=Path, default=Path("/home/chenzunhao/2026-05-01_scarlet_deblend_demo/catalog/meas-HSC-I-9813-4,5.fits"))
    parser.add_argument("--repo", type=Path, default=PIPELINE_ROOT / "fits/repo")
    parser.add_argument("--output-root", type=Path, default=PIPELINE_ROOT / "output/cutout_magnitude_experiment")
    parser.add_argument("--pipeline-root", type=Path, default=PIPELINE_ROOT)
    parser.add_argument("--pipeline-script", type=Path, default=PIPELINE_ROOT / "scarlet_deblend_from_fits.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tract", type=int, default=9813)
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--origin", action="append", default=[], help="Manual parent-patch origin x,y. If omitted, run the automatic grid.")
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--grid-rows", type=int, default=7)
    parser.add_argument("--grid-anchor", default=f"{DEFAULT_PREVIOUS_ORIGIN[0]},{DEFAULT_PREVIOUS_ORIGIN[1]}")
    parser.add_argument("--methods", nargs="+", default=["lsst", "sam"], choices=["lsst", "sam"])
    parser.add_argument("--lsst-workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--lsst-threads-per-worker", type=int, default=1)
    parser.add_argument("--sam-gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--sam-workers-per-gpu", type=int, default=2)
    parser.add_argument("--lsst-extra-args", default="", help="Extra args appended only to LSST runs, shell-style string.")
    parser.add_argument("--sam-extra-args", default="", help="Extra args appended only to SAM runs, shell-style string.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Do not crop or run pipelines; evaluate existing outputs.")
    parser.add_argument(
        "--allow-incomplete-cutouts",
        action="store_true",
        help="Evaluate each method independently even when another requested method is missing for the same cutout.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-clean-nonfinite", action="store_true")
    parser.add_argument("--reference-margin", type=float, default=0.0)
    parser.add_argument("--ref-x", default="base_SdssCentroid_x")
    parser.add_argument("--ref-y", default="base_SdssCentroid_y")
    parser.add_argument("--pred-x", default=None)
    parser.add_argument("--pred-y", default=None)
    parser.add_argument("--ref-flux-col", default="base_PsfFlux_instFlux")
    parser.add_argument(
        "--pred-flux-col",
        default="auto",
        help=(
            "Prediction flux column for purity binning. Default auto uses the "
            "scarlet source spectrum from source_spectra.csv or model_data pickle, "
            "then falls back to positive finite deblend flux columns."
        ),
    )
    parser.add_argument("--pred-spectrum-band", default="HSC-I", choices=list(BANDS))
    parser.add_argument("--mag-zero-point", type=float, default=27.0)
    parser.add_argument("--bin-size", type=float, default=1.0)
    parser.add_argument("--mag-min", type=float, default=18.0, help="Lower displayed magnitude edge; values below this go into a single underflow bin.")
    parser.add_argument("--mag-max", type=float, default=30.0, help="Upper displayed magnitude edge; values at or above this go into a single overflow bin.")
    parser.add_argument("--fp-detail-bin-size", type=float, default=1.0)
    parser.add_argument("--fp-detail-top-n", type=int, default=5)
    parser.add_argument("--match-radius-arcsec", type=float, default=DEFAULT_RADIUS_ARCSEC)
    parser.add_argument("--pixel-scale", type=float, default=DEFAULT_PIXEL_SCALE)
    args = parser.parse_args()

    args.coadd_root = args.coadd_root.expanduser().resolve()
    args.reference_catalog = args.reference_catalog.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.pipeline_root = args.pipeline_root.expanduser().resolve()
    args.pipeline_script = args.pipeline_script.expanduser().resolve()

    specs, grid_metadata = build_specs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    grid_metadata_path = args.output_root / "grid_metadata.json"
    grid_metadata_path.write_text(json.dumps(grid_metadata, indent=2, sort_keys=True) + "\n")
    print(f"prepared {len(specs)} cutouts; wrote {grid_metadata_path}")

    if not args.plot_only:
        cutouts = make_cutouts(args, specs)
        for method in args.methods:
            _submit_runs(args=args, method=method, specs=specs, cutouts=cutouts)
        if args.dry_run:
            print("dry-run complete; no cutouts, pipeline outputs, or metrics were written beyond grid metadata.")
            return

    all_rows = []
    skipped_cutouts = []
    for spec in specs:
        missing_methods = [
            method
            for method in args.methods
            if not _prediction_catalog_path(args, spec, method).exists()
        ]
        if missing_methods and not args.allow_incomplete_cutouts:
            skipped_cutouts.append(
                {
                    "cutout": spec.name,
                    "origin": [spec.x0, spec.y0],
                    "missing_methods": missing_methods,
                }
            )
            print(
                "WARNING: skipping incomplete cutout "
                f"{spec.name}; missing prediction catalog for {','.join(missing_methods)}"
            )
            continue

        ref_catalog = crop_reference_catalog(args, spec)
        for method in args.methods:
            run_dir = args.output_root / "runs" / spec.name / method
            all_rows.extend(evaluate_run(args, spec=spec, method=method, run_dir=run_dir, ref_catalog=ref_catalog))

    per_cutout_csv = args.output_root / "magnitude_metrics_per_cutout.csv"
    aggregate_csv = args.output_root / "magnitude_metrics_aggregate.csv"
    plot_path = args.output_root / "magnitude_curves.png"
    completeness_counts_path = args.output_root / "magnitude_completeness_counts.png"
    purity_fp_counts_path = args.output_root / "magnitude_purity_fp_counts.png"
    evaluation_metadata_path = args.output_root / "magnitude_evaluation_metadata.json"
    write_csv(per_cutout_csv, all_rows)
    aggregate = aggregate_rows(all_rows)
    write_csv(aggregate_csv, aggregate)
    plot_curves(plot_path, aggregate)
    plot_count_histograms(completeness_counts_path, purity_fp_counts_path, aggregate)
    evaluation_metadata_path.write_text(
        json.dumps(
            {
                "candidate_cutout_count": len(specs),
                "evaluated_cutout_count": len(specs) - len(skipped_cutouts),
                "skipped_cutout_count": len(skipped_cutouts),
                "allow_incomplete_cutouts": bool(args.allow_incomplete_cutouts),
                "skipped_cutouts": skipped_cutouts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"wrote {per_cutout_csv}")
    print(f"wrote {aggregate_csv}")
    print(f"wrote {plot_path}")
    print(f"wrote {completeness_counts_path}")
    print(f"wrote {purity_fp_counts_path}")
    print(f"wrote {evaluation_metadata_path}")


if __name__ == "__main__":
    main()
