#!/usr/bin/env python
"""Plot GRI-64 LSST/SAM curves on common noisy and denoised cutouts.

The default comparison uses the intersection of cutouts present in both
``gri_64_noisy`` and ``gri_64_denoised`` for both LSST and SAM.  This avoids
mixing the 45-cutout noisy run with the 41-cutout denoised run.

Example
-------
python utils/plot_gri64_denoised_noisy_lsst_sam_curves.py \
    --copy-png ~/transfer/gri64_denoised_noisy_lsst_sam_curves.png
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_float(value: str) -> float:
    if value == "-inf":
        return -math.inf
    if value == "inf":
        return math.inf
    return float(value)


def _format_float(value: float) -> str:
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return f"{value:g}"


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def cutouts_for(path: Path, method: str) -> set[str]:
    return {row["cutout"] for row in read_rows(path) if row["method"] == method}


def select_rows(path: Path, method: str, cutouts: set[str], label: str) -> list[dict]:
    rows = [
        row
        for row in read_rows(path)
        if row["method"] == method and row["cutout"] in cutouts
    ]
    present = {row["cutout"] for row in rows}
    missing = sorted(cutouts - present)
    if missing:
        raise RuntimeError(f"{label} is missing {len(missing)} common cutouts")
    return rows


def aggregate(rows: list[dict], label: str) -> list[dict]:
    grouped: dict[tuple[float, float], dict] = {}
    for row in rows:
        lo = _parse_float(row["mag_left"])
        hi = _parse_float(row["mag_right"])
        item = grouped.setdefault(
            (lo, hi),
            {
                "cutout": "",
                "method": label,
                "mag_left": lo,
                "mag_right": hi,
                "mag_center": _parse_float(row["mag_center"]),
                "reference_flux_col": "",
                "prediction_flux_col": "",
                "reference_total": 0,
                "reference_matched": 0,
                "prediction_total": 0,
                "prediction_matched": 0,
                "completeness": math.nan,
                "purity": math.nan,
                "prediction_fp_detail_bins": "",
            },
        )
        item["reference_total"] += int(row["reference_total"])
        item["reference_matched"] += int(row["reference_matched"])
        item["prediction_total"] += int(row["prediction_total"])
        item["prediction_matched"] += int(row["prediction_matched"])

    out = []
    for item in grouped.values():
        ref_total = item["reference_total"]
        pred_total = item["prediction_total"]
        item["completeness"] = item["reference_matched"] / ref_total if ref_total else math.nan
        item["purity"] = item["prediction_matched"] / pred_total if pred_total else math.nan
        out.append(item)
    return sorted(out, key=lambda row: row["mag_left"])


def write_csv(path: Path, rows: list[dict]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            for key in ("mag_left", "mag_right", "mag_center"):
                item[key] = _format_float(float(item[key]))
            writer.writerow(item)


def plot(path: Path, rows: list[dict], mag_min: float, mag_max: float, n_cutouts: int) -> None:
    labels = ["denoised_LSST", "denoised_SAM", "noisy_LSST", "noisy_SAM"]
    colors = {
        "denoised_LSST": "#4e79a7",
        "denoised_SAM": "#f28e2b",
        "noisy_LSST": "#59a14f",
        "noisy_SAM": "#e15759",
    }
    styles = {
        "denoised_LSST": "-",
        "denoised_SAM": "-",
        "noisy_LSST": "--",
        "noisy_SAM": "--",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), constrained_layout=True)
    for label in labels:
        label_rows = [row for row in rows if row["method"] == label]
        x = np.array([float(row["mag_center"]) for row in label_rows], dtype=float)
        completeness = np.array([float(row["completeness"]) for row in label_rows], dtype=float)
        purity = np.array([float(row["purity"]) for row in label_rows], dtype=float)
        mask = (x >= mag_min) & (x <= mag_max)
        order = np.argsort(x[mask])
        x = x[mask][order]
        axes[0].plot(
            x,
            completeness[mask][order],
            marker="o",
            linewidth=1.9,
            linestyle=styles[label],
            color=colors[label],
            label=label,
        )
        axes[1].plot(
            x,
            purity[mask][order],
            marker="o",
            linewidth=1.9,
            linestyle=styles[label],
            color=colors[label],
            label=label,
        )

    axes[0].set_title(f"Completeness by catalog magnitude ({n_cutouts} common cutouts)")
    axes[1].set_title(f"Purity by measured prediction magnitude ({n_cutouts} common cutouts)")
    for ax in axes:
        ax.set_xlabel("instrumental magnitude")
        ax.set_ylabel("score")
        ax.set_xlim(mag_min - 0.1, mag_max + 0.1)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def copy_file(src: Path, dst: Path) -> None:
    dst = dst.expanduser()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=Path("output/cutout_magnitude_experiment_grid"))
    parser.add_argument("--noisy-root", type=Path, default=Path("irg_64_iou85_noisy"))
    parser.add_argument("--denoised-root", type=Path, default=Path("irg_64_iou85_denoised"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/cutout_magnitude_experiment_grid/irg64_lsst_default_denoised_noisy_lsst_sam/iou85"),
    )
    parser.add_argument("--copy-png", type=Path, default=None)
    parser.add_argument("--mag-min", type=float, default=23.0)
    parser.add_argument("--mag-max", type=float, default=30.0)
    args = parser.parse_args()

    noisy_csv = args.base_root / args.noisy_root / "magnitude_metrics_per_cutout.csv"
    denoised_csv = args.base_root / args.denoised_root / "magnitude_metrics_per_cutout.csv"

    common_cutouts = (
        cutouts_for(noisy_csv, "lsst")
        & cutouts_for(noisy_csv, "sam")
        & cutouts_for(denoised_csv, "lsst")
        & cutouts_for(denoised_csv, "sam")
    )
    if not common_cutouts:
        raise RuntimeError("no common cutouts found")

    series = [
        ("denoised_LSST", denoised_csv, "lsst"),
        ("denoised_SAM", denoised_csv, "sam"),
        ("noisy_LSST", noisy_csv, "lsst"),
        ("noisy_SAM", noisy_csv, "sam"),
    ]
    aggregate_rows: list[dict] = []
    filtered_rows: list[dict] = []
    for label, path, method in series:
        selected = select_rows(path, method, common_cutouts, label)
        aggregate_rows.extend(aggregate(selected, label))
        for row in selected:
            item = dict(row)
            item["method"] = label
            filtered_rows.append(item)

    out_root = args.output_root.expanduser()
    aggregate_csv = out_root / "magnitude_metrics_aggregate.csv"
    filtered_csv = out_root / "magnitude_metrics_per_cutout_common.csv"
    cutout_list = out_root / "common_cutouts.txt"
    plot_path = out_root / "magnitude_curves.png"

    write_csv(aggregate_csv, aggregate_rows)
    write_csv(filtered_csv, filtered_rows)
    cutout_list.parent.mkdir(parents=True, exist_ok=True)
    cutout_list.write_text("\n".join(sorted(common_cutouts)) + "\n")
    plot(plot_path, aggregate_rows, args.mag_min, args.mag_max, len(common_cutouts))
    if args.copy_png is not None:
        copy_file(plot_path, args.copy_png)

    print(f"common cutouts: {len(common_cutouts)}")
    print(f"wrote {aggregate_csv}")
    print(f"wrote {filtered_csv}")
    print(f"wrote {cutout_list}")
    print(f"wrote {plot_path}")
    if args.copy_png is not None:
        print(f"copied plot to {args.copy_png.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
