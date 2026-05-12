#!/usr/bin/env python
"""Plot IRG64 SAM iou-threshold and LSST nSigmaToGrow comparisons."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_aggregate(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "mag_left",
            "mag_right",
            "mag_center",
            "completeness",
            "purity",
            "reference_total",
            "prediction_total",
            "reference_matched",
            "prediction_matched",
        ):
            try:
                row[key] = float(row[key])
            except ValueError:
                row[key] = np.nan
    return rows


def series(rows: list[dict], method: str, metric: str, mag_min: float, mag_max: float) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["method"] == method]
    x = np.array([row["mag_center"] for row in selected], dtype=float)
    y = np.array([row[metric] for row in selected], dtype=float)
    mask = (x >= mag_min) & (x <= mag_max)
    order = np.argsort(x[mask])
    return x[mask][order], y[mask][order]


def style_axes(ax, title: str, mag_min: float, mag_max: float) -> None:
    ax.set_title(title)
    ax.set_xlabel("instrumental magnitude")
    ax.set_ylabel("score")
    ax.set_xlim(mag_min - 0.1, mag_max + 0.1)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot_iou_comparison(
    *,
    base_root: Path,
    output: Path,
    mag_min: float,
    mag_max: float,
) -> None:
    iou80 = read_aggregate(base_root / "irg64_denoised_noisy_lsst_sam" / "magnitude_metrics_aggregate.csv")
    iou85 = read_aggregate(base_root / "irg64_denoised_noisy_lsst_sam" / "iou85" / "magnitude_metrics_aggregate.csv")

    curves = [
        ("noisy_LSST", iou85, "noisy_LSST", "#4e79a7", "-"),
        ("denoised_LSST", iou85, "denoised_LSST", "#f28e2b", "-"),
        ("denoised_SAM_80", iou80, "denoised_SAM", "#59a14f", "--"),
        ("denoised_SAM_85", iou85, "denoised_SAM", "#e15759", "--"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for label, rows, method, color, linestyle in curves:
        for ax, metric in zip(axes, ("completeness", "purity")):
            x, y = series(rows, method, metric, mag_min, mag_max)
            ax.plot(x, y, marker="o", linewidth=1.9, color=color, linestyle=linestyle, label=label)
    style_axes(axes[0], "Completeness by catalog magnitude", mag_min, mag_max)
    style_axes(axes[1], "Purity by measured prediction magnitude", mag_min, mag_max)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("IRG64 SAM pred_iou_thresh comparison: default 0.80 vs 0.85", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_grow_comparison(
    *,
    base_root: Path,
    output: Path,
    mag_min: float,
    mag_max: float,
) -> None:
    zero = read_aggregate(base_root / "irg64_denoised_noisy_lsst_sam" / "iou85" / "magnitude_metrics_aggregate.csv")
    default = read_aggregate(
        base_root / "irg64_lsst_default_denoised_noisy_lsst_sam" / "iou85" / "magnitude_metrics_aggregate.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True, sharex=True, sharey=True)
    curves = [
        ("noisy_zero_sigma", zero, "noisy_LSST", "#4e79a7", "-"),
        ("noisy_default", default, "noisy_LSST", "#4e79a7", "--"),
        ("denoised_zero_sigma", zero, "denoised_LSST", "#f28e2b", "-"),
        ("denoised_default", default, "denoised_LSST", "#f28e2b", "--"),
    ]

    for ax, metric, title in zip(
        axes,
        ("completeness", "purity"),
        ("Completeness by catalog magnitude", "Purity by measured prediction magnitude"),
    ):
        for label, data, method, color, linestyle in curves:
            x, y = series(data, method, metric, mag_min, mag_max)
            ax.plot(x, y, marker="o", linewidth=1.9, color=color, linestyle=linestyle, label=label)
        style_axes(ax, title, mag_min, mag_max)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("LSST nSigmaToGrow comparison on IRG64 iou85 runs", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=Path("output/cutout_magnitude_experiment_grid"))
    parser.add_argument("--output-root", type=Path, default=Path("output/cutout_magnitude_experiment_grid/comparison_plots"))
    parser.add_argument("--copy-root", type=Path, default=Path("~/transfer"))
    parser.add_argument("--mag-min", type=float, default=23.0)
    parser.add_argument("--mag-max", type=float, default=30.0)
    args = parser.parse_args()

    output_root = args.output_root.expanduser()
    copy_root = args.copy_root.expanduser()
    iou_path = output_root / "irg64_pred_iou80_vs_iou85_curves.png"
    grow_path = output_root / "lsst_nSigmaToGrow_zero_vs_default_irg64_iou85_curves.png"
    plot_iou_comparison(base_root=args.base_root, output=iou_path, mag_min=args.mag_min, mag_max=args.mag_max)
    plot_grow_comparison(base_root=args.base_root, output=grow_path, mag_min=args.mag_min, mag_max=args.mag_max)

    copy_root.mkdir(parents=True, exist_ok=True)
    for path in (iou_path, grow_path):
        copied = copy_root / path.name
        shutil.copy2(path, copied)
        print(f"wrote {path}")
        print(f"copied {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
