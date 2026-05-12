#!/usr/bin/env python
"""Summarize packed LSST SourceCatalog flags for false positives by magnitude."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from evaluate_centroid_matches import DEFAULT_PIXEL_SCALE, DEFAULT_RADIUS_ARCSEC, _load_points, match_nearest_unique


def _flux_to_mag(flux: np.ndarray, zeropoint: float) -> np.ndarray:
    mag = np.full(flux.shape, np.nan, dtype=float)
    good = np.isfinite(flux) & (flux > 0)
    mag[good] = float(zeropoint) - 2.5 * np.log10(flux[good])
    return mag


def _flag_names(path: Path, hdu: int) -> list[str]:
    with fits.open(path) as hdul:
        hdr = hdul[hdu].header
        names = []
        for idx in range(1, 10000):
            key = f"TFLAG{idx}"
            if key not in hdr:
                break
            names.append(str(hdr[key]))
    return names


def _resolve_prediction_path(path: Path, band: str) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        meas = path / "measure" / band / "deepCoadd_meas.fits"
        if meas.exists():
            return meas.resolve()
        deblend = path / "deblend" / "deepCoadd_deblendedFlux.fits"
        if deblend.exists():
            return deblend.resolve()
        raise FileNotFoundError(f"{path} has neither {meas.relative_to(path)} nor {deblend.relative_to(path)}")
    raise FileNotFoundError(path)


def _diagnostic_flag(name: str) -> bool:
    boring_prefixes = (
        "merge_footprint_",
        "merge_peak_",
        "detect_is",
        "detect_fromBlend",
    )
    return not name.startswith(boring_prefixes)


def _bin_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}"


def _load_fp_data(args: argparse.Namespace):
    pred_path = _resolve_prediction_path(args.prediction, args.band)
    ref_path = args.reference.expanduser().resolve()
    ref = _load_points(
        ref_path,
        x_col=None,
        y_col=None,
        role="ref",
        hdu=args.ref_hdu,
        leaf_only=True,
        drop_flagged_centroids=True,
        require_science_model=False,
    )
    pred = _load_points(
        pred_path,
        x_col=None,
        y_col=None,
        role="pred",
        hdu=args.pred_hdu,
        leaf_only=True,
        drop_flagged_centroids=True,
        require_science_model=True,
    )
    _, _, pred_used = match_nearest_unique(ref, pred, args.match_radius_arcsec / args.pixel_scale)
    if args.flux_col not in pred.table.colnames:
        raise KeyError(f"{pred_path} missing flux column {args.flux_col!r}")
    flux = np.asarray(pred.table[args.flux_col], dtype=float)[pred.table_indices]
    mag = _flux_to_mag(flux, args.pred_mag_zero_point)

    full_table = Table.read(pred_path, hdu=args.pred_hdu)
    flags = np.asarray(full_table["flags"], dtype=bool)
    flag_names = _flag_names(pred_path, args.pred_hdu)
    if flags.ndim != 2:
        raise RuntimeError(f"expected packed flags array with shape (n, nflags), got {flags.shape}")
    if flags.shape[1] != len(flag_names):
        raise RuntimeError(f"flags bit count {flags.shape[1]} does not match TFLAG count {len(flag_names)}")
    selected_flags = flags[pred.table_indices]
    fp_mask = ~pred_used
    return pred_path, pred, fp_mask, mag, selected_flags, flag_names


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mag_bin",
        "mag_left",
        "mag_right",
        "fp_count",
        "flag",
        "flag_count",
        "flag_fraction",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_heatmap(path: Path, rows: list[dict], *, title: str, top_n: int, diagnostic_only: bool) -> None:
    rows_for_flags = [r for r in rows if (not diagnostic_only or _diagnostic_flag(str(r["flag"])))]
    totals: dict[str, int] = {}
    for row in rows_for_flags:
        totals[row["flag"]] = totals.get(row["flag"], 0) + int(row["flag_count"])
    flags = [name for name, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:top_n]]
    bins = []
    for row in rows:
        label = row["mag_bin"]
        if label not in bins:
            bins.append(label)
    matrix = np.full((len(flags), len(bins)), np.nan, dtype=float)
    by_key = {(row["flag"], row["mag_bin"]): float(row["flag_fraction"]) for row in rows}
    for i, flag in enumerate(flags):
        for j, label in enumerate(bins):
            matrix[i, j] = by_key.get((flag, label), np.nan)

    fig_w = max(9.0, 0.58 * len(bins))
    fig_h = max(6.0, 0.28 * len(flags))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels(bins, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(flags)))
    ax.set_yticklabels(flags, fontsize=8)
    ax.set_xlabel("prediction magnitude bin")
    ax.set_ylabel("flag")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("FP fraction with flag")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="fp_flags")
    parser.add_argument("--band", default="HSC-I", help="Band used when --prediction is a run directory.")
    parser.add_argument("--mag-min", type=float, default=23.0)
    parser.add_argument("--mag-max", type=float, default=30.0)
    parser.add_argument("--bin-size", type=float, default=0.5)
    parser.add_argument("--pred-mag-zero-point", type=float, default=31.4)
    parser.add_argument("--flux-col", default="base_PsfFlux_instFlux")
    parser.add_argument("--pixel-scale", type=float, default=DEFAULT_PIXEL_SCALE)
    parser.add_argument("--match-radius-arcsec", type=float, default=DEFAULT_RADIUS_ARCSEC)
    parser.add_argument("--ref-hdu", type=int, default=1)
    parser.add_argument("--pred-hdu", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=35)
    args = parser.parse_args()

    pred_path, pred, fp_mask, mag, flags, flag_names = _load_fp_data(args)
    edges = np.arange(args.mag_min, args.mag_max + args.bin_size * 0.5, args.bin_size)
    rows: list[dict] = []
    bin_summary = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = fp_mask & np.isfinite(mag) & (mag >= lo) & (mag < hi)
        fp_count = int(np.count_nonzero(in_bin))
        if fp_count == 0:
            continue
        flag_counts = np.count_nonzero(flags[in_bin], axis=0)
        bin_summary.append({"mag_bin": _bin_label(lo, hi), "fp_count": fp_count})
        for idx, count in enumerate(flag_counts):
            if count == 0:
                continue
            rows.append(
                {
                    "mag_bin": _bin_label(lo, hi),
                    "mag_left": float(lo),
                    "mag_right": float(hi),
                    "fp_count": fp_count,
                    "flag": flag_names[idx],
                    "flag_count": int(count),
                    "flag_fraction": float(count / fp_count),
                }
            )

    outdir = args.output_dir.expanduser().resolve()
    stem = args.label
    csv_path = outdir / f"{stem}_fp_flag_distribution_by_mag.csv"
    png_path = outdir / f"{stem}_fp_flag_distribution_by_mag.png"
    summary_path = outdir / f"{stem}_fp_flag_distribution_summary.json"
    _write_csv(csv_path, rows)
    _plot_heatmap(
        png_path,
        rows,
        title=f"{args.label}: diagnostic flag fraction among FPs",
        top_n=int(args.top_n),
        diagnostic_only=True,
    )
    summary = {
        "reference": str(args.reference.expanduser().resolve()),
        "prediction": str(pred_path),
        "prediction_count_after_filters": pred.n,
        "fp_count_all_magnitudes": int(np.count_nonzero(fp_mask)),
        "mag_min": float(args.mag_min),
        "mag_max": float(args.mag_max),
        "bin_size": float(args.bin_size),
        "bin_summary": bin_summary,
        "csv": str(csv_path),
        "png": str(png_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
