#!/usr/bin/env python
"""Plot magnitude curves for several methods on a common cutout subset.

This is intended for fair comparisons between experiments that finished on
different grid subsets.  By default it compares noisy LSST, noisy IRG-64 SAM,
and noisy GRI-64 SAM, using the GRI SAM cutouts as the common 45-cutout subset.

Example
-------
python utils/plot_common_cutout_method_curves.py \
    --output-root output/cutout_magnitude_experiment_grid/irg_gri_64_common45_noisy \
    --copy-png ~/transfer/irg_gri_64_common45_noisy_curves.png
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


DEFAULT_BASE = Path("output/cutout_magnitude_experiment_grid")


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


def _parse_count_bins(text: str) -> dict[float, int]:
    out: dict[float, int] = {}
    if not text:
        return out
    for item in text.split(";"):
        item = item.strip()
        if not item or ":" not in item:
            continue
        bin_text, count_text = item.split(":", 1)
        lo_text = bin_text.split("-", 1)[0].strip()
        try:
            lo = float(lo_text)
            count = int(count_text)
        except ValueError:
            continue
        out[lo] = out.get(lo, 0) + count
    return out


def _format_count_bins(counts: dict[float, int], top_n: int = 5) -> str:
    if not counts:
        return ""
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    return "; ".join(f"{lo:g}-{lo + 1:g}:{count}" for lo, count in items)


def read_per_cutout_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def cutouts_for(path: Path, method: str) -> set[str]:
    return {row["cutout"] for row in read_per_cutout_rows(path) if row["method"] == method}


def aggregate_rows(rows: list[dict], label: str) -> list[dict]:
    grouped: dict[tuple[float, float], dict] = {}
    for row in rows:
        lo = _parse_float(row["mag_left"])
        hi = _parse_float(row["mag_right"])
        key = (lo, hi)
        item = grouped.setdefault(
            key,
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
                "_prediction_fp_detail_counts": {},
            },
        )
        item["reference_total"] += int(row["reference_total"])
        item["reference_matched"] += int(row["reference_matched"])
        item["prediction_total"] += int(row["prediction_total"])
        item["prediction_matched"] += int(row["prediction_matched"])
        for fp_lo, count in _parse_count_bins(row.get("prediction_fp_detail_bins", "")).items():
            detail_counts = item["_prediction_fp_detail_counts"]
            detail_counts[fp_lo] = detail_counts.get(fp_lo, 0) + count

    out: list[dict] = []
    for item in grouped.values():
        ref_total = item["reference_total"]
        pred_total = item["prediction_total"]
        item["completeness"] = item["reference_matched"] / ref_total if ref_total else math.nan
        item["purity"] = item["prediction_matched"] / pred_total if pred_total else math.nan
        item["prediction_fp_detail_bins"] = _format_count_bins(item.pop("_prediction_fp_detail_counts"))
        out.append(item)
    return sorted(out, key=lambda row: row["mag_left"])


def select_rows(path: Path, method: str, cutouts: set[str]) -> list[dict]:
    rows = [
        row
        for row in read_per_cutout_rows(path)
        if row["method"] == method and row["cutout"] in cutouts
    ]
    present_cutouts = {row["cutout"] for row in rows}
    missing = sorted(cutouts - present_cutouts)
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise RuntimeError(f"{path} lacks {len(missing)} requested cutouts: {preview}{more}")
    return rows


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
        for row in rows:
            formatted = dict(row)
            for key in ("mag_left", "mag_right", "mag_center"):
                formatted[key] = _format_float(float(formatted[key]))
            writer.writerow(formatted)


def plot_curves(path: Path, rows: list[dict], mag_min: float, mag_max: float, n_cutouts: int) -> None:
    labels = ["LSST", "IRG", "GRI"]
    colors = {"LSST": "#4e79a7", "IRG": "#f28e2b", "GRI": "#59a14f"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)

    for label in labels:
        method_rows = [row for row in rows if row["method"] == label]
        if not method_rows:
            continue
        x = np.array([float(row["mag_center"]) for row in method_rows], dtype=float)
        completeness = np.array([float(row["completeness"]) for row in method_rows], dtype=float)
        purity = np.array([float(row["purity"]) for row in method_rows], dtype=float)
        mask = (x >= mag_min) & (x <= mag_max)
        order = np.argsort(x[mask])
        x = x[mask][order]
        completeness = completeness[mask][order]
        purity = purity[mask][order]
        axes[0].plot(x, completeness, marker="o", linewidth=1.9, label=label, color=colors[label])
        axes[1].plot(x, purity, marker="o", linewidth=1.9, label=label, color=colors[label])

    axes[0].set_title(f"Completeness by catalog magnitude ({n_cutouts} common cutouts)")
    axes[1].set_title(f"Purity by measured prediction magnitude ({n_cutouts} common cutouts)")
    for ax in axes:
        ax.set_xlabel("instrumental magnitude")
        ax.set_ylabel("score")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(mag_min - 0.1, mag_max + 0.1)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def copy_file(src: Path, dst: Path) -> None:
    dst = dst.expanduser()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-aggregate LSST/IRG/GRI magnitude curves on the same cutout subset. "
            "Defaults target the noisy IRG-64-vs-GRI-64 comparison."
        )
    )
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--lsst-root", type=Path, default=Path("gri_64_denoised"))
    parser.add_argument("--irg-root", type=Path, default=Path("irg_64_denoised"))
    parser.add_argument("--gri-root", type=Path, default=Path("gri_64_denoised"))
    parser.add_argument(
        "--cutout-source-root",
        type=Path,
        default=Path("gri_64_denoised"),
        help="Experiment root whose cutouts define the common subset.",
    )
    parser.add_argument("--cutout-source-method", default="sam")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BASE / "irg_gri_64_common45_denoised")
    parser.add_argument("--copy-png", type=Path, default=None)
    parser.add_argument("--mag-min", type=float, default=23.0)
    parser.add_argument("--mag-max", type=float, default=30.0)
    args = parser.parse_args()

    base = args.base_root
    lsst_csv = base / args.lsst_root / "magnitude_metrics_per_cutout.csv"
    irg_csv = base / args.irg_root / "magnitude_metrics_per_cutout.csv"
    gri_csv = base / args.gri_root / "magnitude_metrics_per_cutout.csv"
    cutout_csv = base / args.cutout_source_root / "magnitude_metrics_per_cutout.csv"

    common_cutouts = cutouts_for(cutout_csv, args.cutout_source_method)
    if not common_cutouts:
        raise RuntimeError(f"no cutouts found in {cutout_csv} for method={args.cutout_source_method}")

    lsst_rows = select_rows(lsst_csv, "lsst", common_cutouts)
    irg_rows = select_rows(irg_csv, "sam", common_cutouts)
    gri_rows = select_rows(gri_csv, "sam", common_cutouts)

    aggregate: list[dict] = []
    aggregate.extend(aggregate_rows(lsst_rows, "LSST"))
    aggregate.extend(aggregate_rows(irg_rows, "IRG"))
    aggregate.extend(aggregate_rows(gri_rows, "GRI"))

    out_root = args.output_root.expanduser()
    aggregate_csv = out_root / "magnitude_metrics_aggregate.csv"
    filtered_csv = out_root / "magnitude_metrics_per_cutout_common.csv"
    cutout_list = out_root / "common_cutouts.txt"
    plot_path = out_root / "magnitude_curves.png"

    write_csv(aggregate_csv, aggregate)
    filtered_rows = []
    for label, rows in (("LSST", lsst_rows), ("IRG", irg_rows), ("GRI", gri_rows)):
        for row in rows:
            item = dict(row)
            item["method"] = label
            filtered_rows.append(item)
    write_csv(filtered_csv, filtered_rows)
    cutout_list.parent.mkdir(parents=True, exist_ok=True)
    cutout_list.write_text("\n".join(sorted(common_cutouts)) + "\n")
    plot_curves(plot_path, aggregate, args.mag_min, args.mag_max, len(common_cutouts))
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
