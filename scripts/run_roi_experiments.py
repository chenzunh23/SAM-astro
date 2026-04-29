#!/usr/bin/env python3
"""Run ROI SExtractor/SAM comparison experiments on FITS images."""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from amg_fits_core import (
    SamAutomaticMaskGenerator,
    run_one_single,
    run_one_triplet,
    sam_model_registry,
)
from run_sextractor_crop import crop_fits, run_sextractor
from visualize_sextractor import main as visualize_main


ROI_SIZE = 256
ROI_LOWER_LEFT = [
    (1032, 46),
    (608, 325),
    (578, 622),
    (1257, 661),
    (910, 1211),
    (1185, 1174),
]


@dataclass(frozen=True)
class SamExperiment:
    group: str
    name: str
    points_per_side: int = 32
    pred_iou_thresh: float = 0.8
    crop_n_layers: int = 1
    astro_rgb_mode: str = "astro_rgb2"
    astro_rgb_low_sigma: Optional[float] = None
    astro_rgb_none_std: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop six 256x256 FITS ROIs, run SExtractor once per ROI, then run the "
            "requested SAM parameter sweeps and draw per-ROI mask-count summaries."
        )
    )
    parser.add_argument("measure_fits", help="Measurement FITS for ROI cropping and SExtractor.")
    parser.add_argument("--detect-fits", help="Optional detection FITS for SExtractor dual-image mode.")
    parser.add_argument(
        "--sam-fits",
        nargs="+",
        help=(
            "FITS input(s) for SAM. Provide one single-band FITS, or three FITS files "
            "in R G B order for astro_rgb/astro_rgb2 experiments. Defaults to measure_fits."
        ),
    )
    parser.add_argument("--out-root", default="ROI", help="Output root directory (default: ROI).")
    parser.add_argument("--hdu", type=int, default=0, help="FITS HDU index (default: 0).")
    parser.add_argument("--model-type", default="vit_h", choices=["default", "vit_h", "vit_l", "vit_b"])
    parser.add_argument("--checkpoint", default="/home/chenzunhao/sam_vit_h_4b8939.pth")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--points-per-batch", type=int, default=128)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-overlap-ratio", type=float, default=512 / 1500)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=0)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.15)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--astro-hardcode-mean", type=float, nargs=3, default=None)
    parser.add_argument("--astro-hardcode-std", type=float, nargs=3, default=None)
    parser.add_argument("--astro-hardcode-clip-hi", type=float, nargs=3, default=None)
    parser.add_argument("--astro-hardcode-z-clip", type=float, nargs=2, default=None)
    parser.add_argument("--astro-preprocess-in-model", action="store_true")
    parser.add_argument("--astro-preprocess-clip-sigma", type=float, default=3.0)
    parser.add_argument("--astro-preprocess-sigma-iters", type=int, default=-1)
    parser.add_argument("--astro-preprocess-z-clip", type=float, nargs=2, default=None)
    parser.add_argument("--parallel-rois", type=int, default=1, help="Number of ROI workers. Increase only if GPU memory allows it.")
    parser.add_argument("--skip-sextractor", action="store_true")
    parser.add_argument("--skip-sam", action="store_true")
    parser.add_argument("--keep-sam-fits", action="store_true", help="Keep temporary SAM label-map/bundle FITS files.")
    return parser.parse_args()


def read_shape(path: Path, hdu: int) -> Tuple[int, int]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[hdu].data
        if data is None:
            raise ValueError(f"No image data in HDU {hdu}: {path}")
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D FITS image in {path}, got shape {data.shape}")
        return int(data.shape[0]), int(data.shape[1])


def roi_name(index: int, x_left: int, y_bottom: int) -> str:
    return f"roi_{index:02d}_x{x_left}_yb{y_bottom}"


def crop_roi_inputs(
    *,
    roi_dir: Path,
    measure_fits: Path,
    detect_fits: Path,
    sam_fits: Sequence[Path],
    hdu: int,
    x_left: int,
    y_bottom: int,
    image_height: int,
) -> Tuple[Path, Path, List[Path]]:
    y_top = image_height - y_bottom - ROI_SIZE
    if y_top < 0:
        raise ValueError(
            f"ROI upper-left y={y_bottom} with size {ROI_SIZE} exceeds image height {image_height}"
        )

    inputs_dir = roi_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    measure_crop = inputs_dir / "measure_roi.fits"
    crop_fits(measure_fits, measure_crop, x0=x_left, y0=y_top, width=ROI_SIZE, height=ROI_SIZE, hdu_index=hdu)

    if detect_fits == measure_fits:
        detect_crop = measure_crop
    else:
        detect_crop = inputs_dir / "detect_roi.fits"
        crop_fits(detect_fits, detect_crop, x0=x_left, y0=y_top, width=ROI_SIZE, height=ROI_SIZE, hdu_index=hdu)

    sam_crops: List[Path] = []
    for i, sam_path in enumerate(sam_fits):
        out_path = inputs_dir / f"sam_{i + 1}_{sam_path.stem}.fits"
        crop_fits(sam_path, out_path, x0=x_left, y0=y_top, width=ROI_SIZE, height=ROI_SIZE, hdu_index=hdu)
        sam_crops.append(out_path)

    meta = {
        "x_left": x_left,
        "y_bottom": y_bottom,
        "x0_numpy": x_left,
        "y0_numpy_top_origin": y_top,
        "size": ROI_SIZE,
        "measure_crop": str(measure_crop),
        "detect_crop": str(detect_crop),
        "sam_crops": [str(p) for p in sam_crops],
    }
    (roi_dir / "roi_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return measure_crop, detect_crop, sam_crops


def visualize(
    image: Path,
    segmentation: Optional[Path],
    catalog: Optional[Path],
    out: Path,
    *,
    label_ellipses: bool = False,
    no_mask_overlay: bool = False,
) -> None:
    import sys

    argv = [
        "visualize_sextractor.py",
        "--image",
        str(image),
        "--out",
        str(out),
    ]
    if segmentation is not None:
        argv.extend(["--segmentation", str(segmentation)])
    if catalog is not None:
        argv.extend(["--catalog", str(catalog)])
    if label_ellipses:
        argv.append("--label-ellipses")
    if no_mask_overlay:
        argv.append("--no-mask-overlay")

    old_argv = sys.argv
    try:
        sys.argv = argv
        visualize_main()
    finally:
        sys.argv = old_argv


def run_sextractor_for_roi(measure_crop: Path, detect_crop: Path, roi_dir: Path) -> int:
    out_dir = roi_dir / "sextractor_results"
    run_sextractor(detect_crop, measure_crop, out_dir)

    seg = out_dir / "check_segmentation.fits"
    cat = out_dir / "result.cat"
    for suffix in ("png", "tif"):
        visualize(measure_crop, seg, None, roi_dir / f"sextractor_segmentation_overlay.{suffix}")
        visualize(measure_crop, None, cat, roi_dir / f"sextractor_ellipses.{suffix}")

    return count_sextractor_catalog(cat)


def count_sextractor_catalog(path: Path) -> int:
    if not path.exists():
        return 0
    for fmt in ("ascii.sextractor", "ascii.commented_header", "ascii"):
        try:
            return len(Table.read(path, format=fmt))
        except Exception:
            continue
    return 0


def count_metadata_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def experiments() -> List[SamExperiment]:
    exps: List[SamExperiment] = []
    for mode in ("astro_rgb2", "astro_rgb"):
        for pps in (16, 32, 64):
            for iou in (0.8, 0.84, 0.88):
                exps.append(
                    SamExperiment(
                        group="points_iou_rgbmode",
                        name=f"pps{pps}_iou{iou:g}_{mode}",
                        points_per_side=pps,
                        pred_iou_thresh=iou,
                        astro_rgb_mode=mode,
                    )
                )

    for mode in ("astro_rgb2", "astro_rgb"):
        for crop_layers in (0, 1, 2):
            exps.append(
                SamExperiment(
                    group="crop_layers_rgbmode",
                    name=f"crop{crop_layers}_{mode}",
                    crop_n_layers=crop_layers,
                    astro_rgb_mode=mode,
                )
            )

    for low_sigma in (0.5, 1.0, 2.0, 4.0, 5.0):
        exps.append(
            SamExperiment(
                group="astro_rgb2_low_sigma",
                name=f"astro_rgb2_low{low_sigma:g}_high3",
                astro_rgb_mode="astro_rgb2",
                astro_rgb_low_sigma=low_sigma,
            )
        )
    return exps


def load_sam(
    model_type: str,
    checkpoint: str,
    device: str,
    scaling_mode: str,
    astro_rgb_mode: str,
    astro_rgb_low_sigma: Optional[float],
    astro_rgb_none_std: Optional[float],
    astro_preprocess_in_model: bool,
    astro_preprocess_clip_sigma: float,
    astro_preprocess_sigma_iters: int,
    astro_preprocess_z_clip: Optional[Sequence[float]],
):
    sam = sam_model_registry[model_type](
        checkpoint=checkpoint,
        scaling_mode=scaling_mode,
        astro_rgb_mode=astro_rgb_mode,
        astro_rgb_low_sigma=astro_rgb_low_sigma,
        astro_rgb_none_std=astro_rgb_none_std,
        astro_preprocess_in_model=astro_preprocess_in_model,
        astro_preprocess_clip_sigma=astro_preprocess_clip_sigma,
        astro_preprocess_sigma_iters=astro_preprocess_sigma_iters,
        astro_preprocess_z_clip=astro_preprocess_z_clip,
    )
    return sam.to(device=device)


def run_sam_experiment(
    *,
    exp: SamExperiment,
    sam_crops: Sequence[Path],
    measure_crop: Path,
    roi_dir: Path,
    args: argparse.Namespace,
    model_cache: Dict[Tuple[str, str, Optional[float], Optional[float]], object],
) -> int:
    exp_dir = roi_dir / "sam_results" / exp.group / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    scaling_mode = "astro_rgb" if len(sam_crops) == 3 else "robust"
    model_cache_key = (
        scaling_mode,
        exp.astro_rgb_mode,
        exp.astro_rgb_low_sigma,
        exp.astro_rgb_none_std,
        args.astro_preprocess_in_model,
        args.astro_preprocess_clip_sigma,
        args.astro_preprocess_sigma_iters,
        tuple(args.astro_preprocess_z_clip) if args.astro_preprocess_z_clip is not None else None,
    )
    if model_cache_key not in model_cache:
        model_cache[model_cache_key] = load_sam(
            args.model_type,
            args.checkpoint,
            args.device,
            scaling_mode,
            exp.astro_rgb_mode,
            exp.astro_rgb_low_sigma,
            exp.astro_rgb_none_std,
            args.astro_preprocess_in_model,
            args.astro_preprocess_clip_sigma,
            args.astro_preprocess_sigma_iters,
            args.astro_preprocess_z_clip,
        )
    sam = model_cache[model_cache_key]
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=exp.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=exp.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=exp.crop_n_layers,
        crop_nms_thresh=args.crop_nms_thresh,
        crop_overlap_ratio=args.crop_overlap_ratio,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
        output_mode="binary_mask",
    )

    common = dict(
        output=str(exp_dir),
        hdu=args.hdu,
        low_percentile=0.1,
        high_percentile=99.5,
        overlay_alpha=args.overlay_alpha,
        boundary_color=[255, 255, 255],
        overlay_style="fill",
        scaling_mode=scaling_mode,
        astro_rgb_mode=exp.astro_rgb_mode,
        astro_rgb_low_sigma=exp.astro_rgb_low_sigma,
        astro_stats_mode="sigmaclip",
        astro_crop_size=ROI_SIZE,
        astro_stats_input=[str(p) for p in args.sam_fits] if len(sam_crops) == 3 else None,
        astro_hardcode_mean=args.astro_hardcode_mean,
        astro_hardcode_std=args.astro_hardcode_std,
        astro_hardcode_clip_hi=args.astro_hardcode_clip_hi,
        astro_hardcode_z_clip=args.astro_hardcode_z_clip,
        astro_preprocess_in_model=args.astro_preprocess_in_model,
        astro_preprocess_clip_sigma=args.astro_preprocess_clip_sigma,
        astro_preprocess_sigma_iters=args.astro_preprocess_sigma_iters,
        astro_preprocess_z_clip=args.astro_preprocess_z_clip,
        min_mask_region_area=args.min_mask_region_area,
        max_mask_area_ratio=args.max_mask_area_ratio,
        no_save_fits=False,
        save_json=False,
    )
    core_args = argparse.Namespace(**common)
    if len(sam_crops) == 3:
        run_one_triplet(core_args, generator, list(sam_crops), exp_dir, astro_mode=exp.astro_rgb_mode)
    elif len(sam_crops) == 1:
        run_one_single(core_args, generator, sam_crops[0], exp_dir)
    else:
        raise ValueError("SAM needs either one FITS file or exactly three FITS files.")

    label_maps = sorted(exp_dir.glob("*_sam_labelmap.fits"))
    if not label_maps:
        raise FileNotFoundError(f"SAM label map was not produced in {exp_dir}")
    label_map = label_maps[0]
    for suffix in ("png", "tif"):
        visualize(measure_crop, label_map, None, exp_dir / f"{exp.name}_sam_segmentation_overlay.{suffix}")
        visualize(
            measure_crop,
            label_map,
            None,
            exp_dir / f"{exp.name}_sam_ellipses.{suffix}",
            label_ellipses=True,
            no_mask_overlay=True,
        )

    metadata = sorted(exp_dir.glob("*_metadata.csv"))
    count = count_metadata_rows(metadata[0]) if metadata else 0
    run_meta = {
        "mask_count": count,
        "experiment": exp.__dict__,
        "scaling_mode": scaling_mode,
        "sam_crops": [str(p) for p in sam_crops],
        "astro_stats_input": [str(p) for p in args.sam_fits] if len(sam_crops) == 3 else None,
        "astro_hardcode_mean": args.astro_hardcode_mean,
        "astro_hardcode_std": args.astro_hardcode_std,
        "astro_hardcode_clip_hi": args.astro_hardcode_clip_hi,
        "astro_hardcode_z_clip": args.astro_hardcode_z_clip,
        "astro_preprocess_in_model": args.astro_preprocess_in_model,
        "astro_preprocess_clip_sigma": args.astro_preprocess_clip_sigma,
        "astro_preprocess_sigma_iters": args.astro_preprocess_sigma_iters,
        "astro_preprocess_z_clip": args.astro_preprocess_z_clip,
        "astro_stats_flow": "full image raw sigma bright-source 3sigma removal, astropy sigma_clipped_stats, ROI clipped by full-image raw sigma then normalized by full-image mean/std",
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
    }
    (exp_dir / "experiment_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    if not args.keep_sam_fits:
        for path in exp_dir.glob("*_sam_bundle.fits"):
            path.unlink()
    return count


def plot_counts(rows: Sequence[Dict], out_base: Path) -> None:
    names = [row["name"] for row in rows]
    counts = [row["mask_count"] for row in rows]
    colors = ["#4a4a4a" if row["name"] == "SExtractor" else "#2f80ed" for row in rows]

    fig, ax = plt.subplots(figsize=(max(8, min(18, len(rows) * 0.42)), 4.5))
    x = np.arange(len(rows))
    ax.bar(x, counts, color=colors)
    ax.set_ylabel("Mask / source count")
    ax.set_title(out_base.name)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=180)
    fig.savefig(out_base.with_suffix(".tif"))
    plt.close(fig)


def run_roi(index: int, args_dict: Dict) -> None:
    args = argparse.Namespace(**args_dict)
    measure_fits = Path(args.measure_fits).expanduser().resolve()
    detect_fits = Path(args.detect_fits).expanduser().resolve() if args.detect_fits else measure_fits
    sam_fits = [Path(p).expanduser().resolve() for p in args.sam_fits]
    out_root = Path(args.out_root).expanduser().resolve()
    image_height, _ = read_shape(measure_fits, args.hdu)
    x_left, y_bottom = ROI_LOWER_LEFT[index]
    roi_dir = out_root / roi_name(index + 1, x_left, y_bottom)
    roi_dir.mkdir(parents=True, exist_ok=True)

    measure_crop, detect_crop, sam_crops = crop_roi_inputs(
        roi_dir=roi_dir,
        measure_fits=measure_fits,
        detect_fits=detect_fits,
        sam_fits=sam_fits,
        hdu=args.hdu,
        x_left=x_left,
        y_bottom=y_bottom,
        image_height=image_height,
    )

    rows: List[Dict] = []
    if not args.skip_sextractor:
        rows.append({"name": "SExtractor", "mask_count": run_sextractor_for_roi(measure_crop, detect_crop, roi_dir)})

    if not args.skip_sam:
        model_cache: Dict[Tuple[str, str, Optional[float], Optional[float]], object] = {}
        for exp in experiments():
            rows.append(
                {
                    "name": exp.name,
                    "group": exp.group,
                    "mask_count": run_sam_experiment(
                        exp=exp,
                        sam_crops=sam_crops,
                        measure_crop=measure_crop,
                        roi_dir=roi_dir,
                        args=args,
                        model_cache=model_cache,
                    ),
                }
            )

    if rows:
        by_group: Dict[str, List[Dict]] = {}
        sextractor_rows = [row for row in rows if row["name"] == "SExtractor"]
        for row in rows:
            if row["name"] == "SExtractor":
                continue
            group = str(row.get("group", row["name"].split("_")[0]))
            by_group.setdefault(group, [*sextractor_rows]).append(row)
        plot_counts(rows, roi_dir / "all_mask_counts")
        for group, group_rows in by_group.items():
            plot_counts(group_rows, roi_dir / f"{group}_mask_counts")

    print(f"[ROI OK] {roi_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.sam_fits is None:
        args.sam_fits = [args.measure_fits]
    if len(args.sam_fits) not in (1, 3):
        raise ValueError("--sam-fits must contain either one FITS file or exactly three FITS files.")
    if args.parallel_rois < 1:
        raise ValueError("--parallel-rois must be >= 1")

    args_dict = vars(args)
    if args.parallel_rois == 1:
        for index in range(len(ROI_LOWER_LEFT)):
            run_roi(index, args_dict)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel_rois) as pool:
            futures = [pool.submit(run_roi, index, args_dict) for index in range(len(ROI_LOWER_LEFT))]
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
