#!/usr/bin/env python3
"""Run Zangetsu six-ROI mask experiments for denoised/noisy HSC data."""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import run_roi_experiments as roi


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (REPO_ROOT / "../Zangetsu_4.30arcmin").resolve()
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "zangetsu_small_roi_masks"
DEFAULT_DETECT_TEMPLATE = "chi2_hsc_irg_{case}.fits"
DEFAULT_SWARP_BIN = Path(os.environ.get("SWARP_BIN", "/home/chenzunhao/swarp/src/swarp"))
DEFAULT_MODES = ("none", "astro_rgb")
DEFAULT_CASES = ("denoised", "noisy")
DEFAULT_SAM_BANDS = ("HSC-I", "HSC-G", "HSC-R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the six predefined 256x256 ROIs on Zangetsu_4.30arcmin. "
            "SExtractor detection is the case-specific chi2_hsc_irg_<case>.fits, measurement is HSC-G "
            "denoised/noisy, and SAM uses HSC-I/G/R denoised/noisy triplets."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--detect-fits",
        type=Path,
        default=None,
        help="Optional override detection FITS used for every case. By default each case uses --detect-template.",
    )
    parser.add_argument(
        "--detect-template",
        default=DEFAULT_DETECT_TEMPLATE,
        help="Case-specific detection FITS name under --data-root. Must contain '{case}'.",
    )
    parser.add_argument(
        "--swarp-bin",
        type=Path,
        default=DEFAULT_SWARP_BIN,
        help="SWarp binary used to auto-build missing case-specific chi^2 images.",
    )
    parser.add_argument("--cases", nargs="+", choices=DEFAULT_CASES, default=list(DEFAULT_CASES))
    parser.add_argument(
        "--astro-rgb-modes",
        nargs="+",
        choices=["none", "astro_rgb", "astro_rgb1", "astro_rgb2"],
        default=list(DEFAULT_MODES),
    )
    parser.add_argument(
        "--sam-bands",
        nargs=3,
        default=list(DEFAULT_SAM_BANDS),
        help="Three Zangetsu band directories to pass to SAM in order.",
    )
    parser.add_argument("--model-type", default="vit_h", choices=["default", "vit_h", "vit_l", "vit_b"])
    parser.add_argument("--checkpoint", default="/home/chenzunhao/sam_vit_h_4b8939.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hdu", type=int, default=0)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=128)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-overlap-ratio", type=float, default=512 / 1500)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=0)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.5)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--astro-hardcode-mean", type=float, nargs=3, default=None)
    parser.add_argument("--astro-hardcode-std", type=float, nargs=3, default=None)
    parser.add_argument("--astro-hardcode-clip-hi", type=float, nargs=3, default=None)
    parser.add_argument("--astro-hardcode-z-clip", type=float, nargs=2, default=None)
    parser.add_argument("--astro-preprocess-in-model", action="store_true")
    parser.add_argument("--astro-preprocess-clip-sigma", type=float, default=3.0)
    parser.add_argument("--astro-preprocess-sigma-iters", type=int, default=-1)
    parser.add_argument("--astro-preprocess-z-clip", type=float, nargs=2, default=None)
    parser.add_argument("--skip-sextractor", action="store_true")
    parser.add_argument("--skip-sam", action="store_true")
    parser.add_argument(
        "--keep-sam-fits",
        action="store_true",
        help="Keep SAM label-map/bundle FITS files. By default they are removed.",
    )
    return parser.parse_args()


def build_experiments(args: argparse.Namespace):
    return [
        roi.SamExperiment(
            group="astro_rgb_mode",
            name=f"mode_{mode}",
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            crop_n_layers=args.crop_n_layers,
            astro_rgb_mode=mode,
        )
        for mode in args.astro_rgb_modes
    ]


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_case_chi2(data_root: Path, case: str, out_path: Path, swarp_bin: Path) -> None:
    swarp_bin = swarp_bin.expanduser().resolve()
    if not swarp_bin.exists():
        raise FileNotFoundError(f"SWarp binary not found: {swarp_bin}")

    inputs = [
        require_file(data_root / "HSC-I" / f"{case}.fits"),
        require_file(data_root / "HSC-R" / f"{case}.fits"),
        require_file(data_root / "HSC-G" / f"{case}.fits"),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(swarp_bin),
        *(str(p) for p in inputs),
        "-COMBINE",
        "Y",
        "-COMBINE_TYPE",
        "CHI-MEAN",
        "-WEIGHT_TYPE",
        "NONE",
        "-RESAMPLE",
        "N",
        "-SUBTRACT_BACK",
        "N",
        "-IMAGEOUT_NAME",
        str(out_path),
        "-WEIGHTOUT_NAME",
        str(out_path.with_suffix(".weight.fits")),
        "-XML_NAME",
        str(out_path.with_suffix(".swarp.xml")),
        "-VERBOSE_TYPE",
        "NORMAL",
    ]
    print("Building case-specific chi^2 image:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    require_file(out_path)


def case_detect_fits(args: argparse.Namespace, data_root: Path, case: str) -> Path:
    if args.detect_fits is not None:
        return require_file(args.detect_fits)
    if "{case}" not in args.detect_template:
        raise ValueError("--detect-template must contain '{case}' when --detect-fits is not set.")

    detect_fits = (data_root / args.detect_template.format(case=case)).expanduser().resolve()
    if not detect_fits.exists():
        build_case_chi2(data_root, case, detect_fits, args.swarp_bin)
    return require_file(detect_fits)


def case_paths(data_root: Path, case: str, sam_bands: List[str]) -> Dict[str, object]:
    measure_fits = require_file(data_root / "HSC-G" / f"{case}.fits")
    sam_fits = [require_file(data_root / band / f"{case}.fits") for band in sam_bands]
    return {"measure_fits": measure_fits, "sam_fits": sam_fits}


def run_case(args: argparse.Namespace, case: str) -> None:
    data_root = args.data_root.expanduser().resolve()
    detect_fits = case_detect_fits(args, data_root, case)
    paths = case_paths(data_root, case, args.sam_bands)
    out_root = (args.out_root.expanduser().resolve() / case)
    out_root.mkdir(parents=True, exist_ok=True)

    run_args = {
        "measure_fits": str(paths["measure_fits"]),
        "detect_fits": str(detect_fits),
        "sam_fits": [str(p) for p in paths["sam_fits"]],
        "out_root": str(out_root),
        "hdu": args.hdu,
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "points_per_batch": args.points_per_batch,
        "stability_score_thresh": args.stability_score_thresh,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
        "max_mask_area_ratio": args.max_mask_area_ratio,
        "overlay_alpha": args.overlay_alpha,
        "astro_hardcode_mean": args.astro_hardcode_mean,
        "astro_hardcode_std": args.astro_hardcode_std,
        "astro_hardcode_clip_hi": args.astro_hardcode_clip_hi,
        "astro_hardcode_z_clip": args.astro_hardcode_z_clip,
        "astro_preprocess_in_model": args.astro_preprocess_in_model,
        "astro_preprocess_clip_sigma": args.astro_preprocess_clip_sigma,
        "astro_preprocess_sigma_iters": args.astro_preprocess_sigma_iters,
        "astro_preprocess_z_clip": args.astro_preprocess_z_clip,
        "parallel_rois": 1,
        "skip_sextractor": args.skip_sextractor,
        "skip_sam": args.skip_sam,
        "keep_sam_fits": args.keep_sam_fits,
    }

    print(f"[CASE] {case}", flush=True)
    print(f"  SExtractor detection: {detect_fits}", flush=True)
    print(f"  SExtractor measurement: {paths['measure_fits']}", flush=True)
    print("  SAM inputs:", *paths["sam_fits"], sep="\n    ", flush=True)
    print(f"  Output: {out_root}", flush=True)

    for i, _xy in enumerate(roi.ROI_LOWER_LEFT):
        roi.run_roi(i, run_args)


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()
    if args.detect_fits is not None:
        args.detect_fits = args.detect_fits.expanduser().resolve()
    args.swarp_bin = args.swarp_bin.expanduser().resolve()

    experiments = build_experiments(args)
    roi.experiments = lambda: experiments

    print("Experiments:", ", ".join(exp.name for exp in experiments), flush=True)
    for case in args.cases:
        run_case(args, case)


if __name__ == "__main__":
    main()
