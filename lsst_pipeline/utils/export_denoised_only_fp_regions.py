#!/usr/bin/env python
"""Export denoised-only SAM false-positive regions for one cutout.

A denoised-only FP is defined as a denoised prediction that is unmatched to the
reference catalog and has no noisy FP within the comparison radius.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from evaluate_centroid_matches import (  # noqa: E402
    DEFAULT_PIXEL_SCALE,
    DEFAULT_RADIUS_ARCSEC,
    _load_points,
    match_nearest_unique,
)


def _flux_to_mag(flux: np.ndarray, zeropoint: float) -> np.ndarray:
    mag = np.full(flux.shape, np.nan, dtype=float)
    good = np.isfinite(flux) & (flux > 0)
    mag[good] = float(zeropoint) - 2.5 * np.log10(flux[good])
    return mag


def _origin_from_background(path: Path) -> tuple[float, float]:
    with fits.open(path) as hdul:
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None and "LTV1" in hdu.header and "LTV2" in hdu.header:
                return -float(hdu.header["LTV1"]), -float(hdu.header["LTV2"])
    raise KeyError(f"{path} has no image HDU with LTV1/LTV2")


def _shape_prefix(table) -> str | None:
    for prefix in (
        "base_SdssShape",
        "ext_shapeHSM_HsmSourceMoments",
        "ext_shapeHSM_HsmSourceMomentsRound",
        "modelfit_CModel_ellipse",
    ):
        if all(f"{prefix}_{suffix}" in table.colnames for suffix in ("xx", "yy", "xy")):
            return prefix
    return None


def _ellipse_params(table, row_index: int, *, shape_scale: float) -> tuple[float, float, float, str] | None:
    prefix = _shape_prefix(table)
    if prefix is None:
        return None
    try:
        values = [table[f"{prefix}_{suffix}"][row_index] for suffix in ("xx", "yy", "xy")]
        if any(np.ma.is_masked(value) for value in values):
            return None
        xx, yy, xy = [float(value) for value in values]
    except Exception:
        return None
    cov = np.array([[xx, xy], [xy, yy]], dtype=float)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
        return None
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    semi_major = float(shape_scale * np.sqrt(vals[0]))
    semi_minor = float(shape_scale * np.sqrt(vals[1]))
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    if not np.all(np.isfinite([semi_major, semi_minor, angle])):
        return None
    return semi_major, semi_minor, angle, prefix


def _catalog_path(run_dir: Path, band: str) -> Path:
    meas = run_dir / "measure" / band / "deepCoadd_meas.fits"
    if meas.exists():
        return meas
    deblend = run_dir / "deblend" / "deepCoadd_deblendedFlux.fits"
    if deblend.exists():
        return deblend
    raise FileNotFoundError(f"no measurement/deblend catalog under {run_dir}")


def _load_prediction(run_dir: Path, reference: Path, args: argparse.Namespace):
    pred_path = _catalog_path(run_dir, args.band)
    ref = _load_points(
        reference,
        x_col=None,
        y_col=None,
        role="ref",
        hdu=1,
        leaf_only=True,
        drop_flagged_centroids=True,
        require_science_model=False,
    )
    pred = _load_points(
        pred_path,
        x_col=None,
        y_col=None,
        role="pred",
        hdu=1,
        leaf_only=True,
        drop_flagged_centroids=True,
        require_science_model=True,
    )
    _, _, pred_used = match_nearest_unique(ref, pred, args.radius_pix)
    if args.flux_col not in pred.table.colnames:
        raise KeyError(f"{pred_path} missing {args.flux_col}")
    flux = np.asarray(pred.table[args.flux_col], dtype=float)[pred.table_indices]
    mag = _flux_to_mag(flux, args.pred_mag_zero_point)
    return pred_path, pred, pred_used, flux, mag


def _nearest_distances(x: np.ndarray, y: np.ndarray, other_x: np.ndarray, other_y: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros(0, dtype=float)
    if other_x.size == 0:
        return np.full(x.shape, np.inf, dtype=float)
    dx = x[:, None] - other_x[None, :]
    dy = y[:, None] - other_y[None, :]
    return np.min(np.hypot(dx, dy), axis=1)


def _write_reg(path: Path, rows: list[dict], *, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write(
            f'global color={color} dashlist=8 3 width=2 font="helvetica 14 bold roman" '
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
        )
        handle.write("image\n")
        for row in rows:
            label = f"id={row['den_id']} mag={float(row['den_mag']):.2f}"
            if row.get("ellipse_a_pix") not in ("", None):
                handle.write(
                    f"ellipse({float(row['image_x']):.3f},{float(row['image_y']):.3f},"
                    f"{float(row['ellipse_a_pix']):.3f},{float(row['ellipse_b_pix']):.3f},"
                    f"{float(row['ellipse_angle_deg']):.2f}) # text={{{label}}}\n"
                )
            handle.write(
                f"point({float(row['image_x']):.3f},{float(row['image_y']):.3f}) "
                f"# point=cross 8 color=yellow text={{{label}}}\n"
            )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cutout",
        "den_table_index",
        "den_id",
        "den_parent",
        "den_mag",
        "den_flux",
        "global_x",
        "global_y",
        "image_x",
        "image_y",
        "nearest_noisy_fp_dist_pix",
        "nearest_noisy_fp_dist_arcsec",
        "ellipse_a_pix",
        "ellipse_b_pix",
        "ellipse_angle_deg",
        "ellipse_source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--denoised-run", type=Path, required=True)
    parser.add_argument("--noisy-run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutout", default="grid_r02_c04_x18204_y20924")
    parser.add_argument("--band", default="HSC-I")
    parser.add_argument("--mag-min", type=float, default=24.5)
    parser.add_argument("--mag-max", type=float, default=26.5)
    parser.add_argument("--pred-mag-zero-point", type=float, default=31.4)
    parser.add_argument("--flux-col", default="base_PsfFlux_instFlux")
    parser.add_argument("--pixel-scale", type=float, default=DEFAULT_PIXEL_SCALE)
    parser.add_argument("--match-radius-arcsec", type=float, default=DEFAULT_RADIUS_ARCSEC)
    parser.add_argument("--shape-scale", type=float, default=2.0)
    parser.add_argument("--color", default="red")
    args = parser.parse_args()
    args.radius_pix = float(args.match_radius_arcsec) / float(args.pixel_scale)

    den_run = args.denoised_run.expanduser().resolve()
    noisy_run = args.noisy_run.expanduser().resolve()
    reference = args.reference.expanduser().resolve()
    background = args.background.expanduser().resolve()
    origin_x, origin_y = _origin_from_background(background)

    den_path, den, den_used, den_flux, den_mag = _load_prediction(den_run, reference, args)
    noisy_path, noisy, noisy_used, _, noisy_mag = _load_prediction(noisy_run, reference, args)

    den_fp = ~den_used
    noisy_fp = ~noisy_used
    den_selected = den_fp & np.isfinite(den_mag) & (den_mag >= args.mag_min) & (den_mag < args.mag_max)
    noisy_fp_x = noisy.x[noisy_fp]
    noisy_fp_y = noisy.y[noisy_fp]
    selected_indices = np.flatnonzero(den_selected)
    nearest = _nearest_distances(den.x[selected_indices], den.y[selected_indices], noisy_fp_x, noisy_fp_y)
    den_only_mask = nearest > args.radius_pix

    rows: list[dict] = []
    for filtered_index, nearest_dist in zip(selected_indices[den_only_mask], nearest[den_only_mask]):
        table_index = int(den.table_indices[filtered_index])
        source_id = int(den.ids[filtered_index])
        parent = int(den.table["parent"][table_index]) if "parent" in den.table.colnames else -1
        gx = float(den.x[filtered_index])
        gy = float(den.y[filtered_index])
        ellipse = _ellipse_params(den.table, table_index, shape_scale=float(args.shape_scale))
        rows.append(
            {
                "cutout": args.cutout,
                "den_table_index": table_index,
                "den_id": source_id,
                "den_parent": parent,
                "den_mag": float(den_mag[filtered_index]),
                "den_flux": float(den_flux[filtered_index]),
                "global_x": gx,
                "global_y": gy,
                "image_x": gx - origin_x + 1.0,
                "image_y": gy - origin_y + 1.0,
                "nearest_noisy_fp_dist_pix": float(nearest_dist),
                "nearest_noisy_fp_dist_arcsec": float(nearest_dist * args.pixel_scale),
                "ellipse_a_pix": "" if ellipse is None else float(ellipse[0]),
                "ellipse_b_pix": "" if ellipse is None else float(ellipse[1]),
                "ellipse_angle_deg": "" if ellipse is None else float(ellipse[2]),
                "ellipse_source": "" if ellipse is None else ellipse[3],
            }
        )

    outdir = args.output_dir.expanduser().resolve()
    stem = f"{args.cutout}_denoised_sam_fp_mag{args.mag_min:g}_{args.mag_max:g}_not_noisy_fp"
    reg_path = outdir / f"{stem}.reg"
    csv_path = outdir / f"{stem}.csv"
    summary_path = outdir / f"{stem}_summary.json"
    _write_reg(reg_path, rows, color=args.color)
    _write_csv(csv_path, rows)
    summary = {
        "cutout": args.cutout,
        "denoised_prediction": str(den_path),
        "noisy_prediction": str(noisy_path),
        "reference": str(reference),
        "background": str(background),
        "denoised_prediction_count": den.n,
        "noisy_prediction_count": noisy.n,
        "denoised_fp_count": int(np.count_nonzero(den_fp)),
        "noisy_fp_count": int(np.count_nonzero(noisy_fp)),
        "denoised_fp_in_mag_bin_count": int(np.count_nonzero(den_selected)),
        "denoised_fp_not_noisy_fp_count": len(rows),
        "compare_radius_pix": float(args.radius_pix),
        "compare_radius_arcsec": float(args.match_radius_arcsec),
        "mag_min": float(args.mag_min),
        "mag_max": float(args.mag_max),
        "reg": str(reg_path),
        "csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
