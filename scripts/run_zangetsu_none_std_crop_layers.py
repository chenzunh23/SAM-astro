#!/usr/bin/env python3
"""Run Zangetsu six-ROI crop-layer sweeps for astro_rgb_mode=none std settings."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_roi_experiments as roi
import run_zangetsu_small_roi_masks as zangetsu


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "zangetsu_none_std_crop_layers"
STD_VARIANTS: Tuple[Tuple[str, float, str], ...] = (
    ("std_1over48", 1.0 / 48.0, "1/48 sigma"),
    ("std_1x", 1.0, "1x sigma"),
)
CROP_LAYERS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the six predefined Zangetsu 256x256 ROIs with astro_rgb_mode=none, "
            "comparing SAM preprocess pixel_std=1/48 and pixel_std=1 across "
            "crop_n_layers=0,1,2. A line plot is written after the runs."
        )
    )
    parser.add_argument("--data-root", type=Path, default=zangetsu.DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--detect-fits", type=Path, default=None)
    parser.add_argument("--detect-template", default=zangetsu.DEFAULT_DETECT_TEMPLATE)
    parser.add_argument(
        "--swarp-bin",
        type=Path,
        default=Path(os.environ.get("SWARP_BIN", str(zangetsu.DEFAULT_SWARP_BIN))),
    )
    parser.add_argument("--cases", nargs="+", choices=zangetsu.DEFAULT_CASES, default=list(zangetsu.DEFAULT_CASES))
    parser.add_argument("--sam-bands", nargs=3, default=list(zangetsu.DEFAULT_SAM_BANDS))
    parser.add_argument("--model-type", default="vit_h", choices=["default", "vit_h", "vit_l", "vit_b"])
    parser.add_argument("--checkpoint", default="/home/chenzunhao/sam_vit_h_4b8939.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hdu", type=int, default=0)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=128)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
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
    parser.add_argument("--keep-sam-fits", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def build_experiments(args: argparse.Namespace) -> List[roi.SamExperiment]:
    exps: List[roi.SamExperiment] = []
    for std_name, std_value, _ in STD_VARIANTS:
        for crop_layers in CROP_LAYERS:
            exps.append(
                roi.SamExperiment(
                    group="none_std_crop_layers",
                    name=f"{std_name}_crop{crop_layers}",
                    points_per_side=args.points_per_side,
                    pred_iou_thresh=args.pred_iou_thresh,
                    crop_n_layers=crop_layers,
                    astro_rgb_mode="none",
                    astro_rgb_none_std=std_value,
                )
            )
    return exps


def roi_sort_key(path: Path) -> Tuple[int, str]:
    parts = path.name.split("_")
    if len(parts) > 1 and parts[1].isdigit():
        return int(parts[1]), path.name
    return 999, path.name


def roi_label(roi_dir: Path) -> str:
    meta_path = roi_dir / "roi_metadata.json"
    roi_id = roi_dir.name.split("_")[1] if "_" in roi_dir.name else roi_dir.name
    if not meta_path.exists():
        return f"ROI {roi_id}"
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    return f"ROI {roi_id} ({meta['x_left']}, {meta['y_bottom']})"


def read_count(root: Path, case: str, roi_dir: Path, std_name: str, crop_layers: int) -> int:
    meta_path = (
        root
        / case
        / roi_dir.name
        / "sam_results"
        / "none_std_crop_layers"
        / f"{std_name}_crop{crop_layers}"
        / "experiment_metadata.json"
    )
    if not meta_path.exists():
        return 0
    with meta_path.open(encoding="utf-8") as f:
        return int(json.load(f).get("mask_count", 0))


def collect_case(root: Path, case: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    case_dir = root / case
    roi_dirs = sorted((p for p in case_dir.glob("roi_*") if p.is_dir()), key=roi_sort_key)
    if not roi_dirs:
        raise FileNotFoundError(f"No ROI result directories found under {case_dir}")

    labels = [roi_label(p) for p in roi_dirs]
    values: Dict[str, np.ndarray] = {}
    for std_name, _, _ in STD_VARIANTS:
        rows = [
            [read_count(root, case, roi_dir, std_name, crop_layers) for crop_layers in CROP_LAYERS]
            for roi_dir in roi_dirs
        ]
        values[std_name] = np.asarray(rows, dtype=int)
    return labels, values


def plot_counts(root: Path, cases: List[str], dpi: int) -> Path:
    collected = {case: collect_case(root, case) for case in cases}
    ymax = max(int(v.max()) for _, values in collected.values() for v in values.values())

    fig, axes = plt.subplots(
        len(cases),
        len(STD_VARIANTS),
        figsize=(6.2 * len(STD_VARIANTS), 4.2 * len(cases)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(cases), len(STD_VARIANTS))
    x = np.asarray(CROP_LAYERS)

    for row, case in enumerate(cases):
        labels, values = collected[case]
        for col, (std_name, _, std_label) in enumerate(STD_VARIANTS):
            ax = axes[row, col]
            counts = values[std_name]
            for i, label in enumerate(labels):
                ax.plot(x, counts[i], marker="o", linewidth=1.2, alpha=0.55, label=label)
            ax.plot(x, counts.mean(axis=0), marker="o", color="#111111", linewidth=2.6, label="Mean")
            ax.set_title(f"{case} - {std_label}")
            ax.set_xticks(x)
            ax.set_xlabel("crop_n_layers")
            ax.grid(alpha=0.25)
            ax.set_ylim(0, ymax * 1.12 if ymax else 1)
            if col == 0:
                ax.set_ylabel("SAM mask count")
            if row == 0 and col == len(STD_VARIANTS) - 1:
                ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.suptitle("astro_rgb_mode=none: preprocess std and crop layer sweep", fontsize=14)
    out = root / "none_std_crop_layers_lineplot.png"
    fig.savefig(out, dpi=dpi)
    fig.savefig(out.with_suffix(".tif"), dpi=dpi)
    plt.close(fig)
    return out


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
        zangetsu.run_case(args, case)

    if not args.skip_sam:
        out = plot_counts(args.out_root, args.cases, args.dpi)
        print(f"[PLOT OK] {out}", flush=True)


if __name__ == "__main__":
    main()
