#!/usr/bin/env python
"""Export magnitude-selected false-positive centroids to DS9 region files.

False positives are defined with the same one-to-one centroid matching used by
``run_cutout_magnitude_experiment.py``: leaf science-model predictions are
matched to leaf reference catalog sources within 0.5 arcsec by default.  The
prediction magnitude is computed from a measurement flux column, defaulting to
``base_PsfFlux_instFlux`` with a 31.4 zero point.  Region output defaults to
ellipses from measured second moments.  If a Kron radius column is available,
the ellipse semi-axes are ``Kron radius * sqrt(moment eigenvalues)``; otherwise
they are ``shape-scale * sqrt(moment eigenvalues)``.

Example
-------
python utils/export_fp_centroid_regions.py \
    --reference output/cutout_magnitude_experiment_grid/gri_64_denoised/reference_catalogs/grid_r02_c04_x18204_y20924_meas.fits \
    --run-dir output/cutout_magnitude_experiment_grid/gri_64_denoised/runs/grid_r02_c04_x18204_y20924/sam \
    --background output/cutout_magnitude_experiment_grid/gri_64_denoised/cutouts/grid_r02_c04_x18204_y20924/HSC-I/deepCoadd-HSC-I-9813-4,5.fits \
    --label gri64_denoised_sam \
    --mag-min 27.5 --mag-max 29 \
    --output-dir ~/transfer/gri64_denoised_fp_centroids_27p5_29
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


def _prediction_path(run_dir: Path, band: str) -> Path:
    meas = run_dir / "measure" / band / "deepCoadd_meas.fits"
    if meas.exists():
        return meas
    deblend = run_dir / "deblend" / "deepCoadd_deblendedFlux.fits"
    if deblend.exists():
        return deblend
    raise FileNotFoundError(f"no measurement/deblend catalog found under {run_dir}")


def _origin_from_background(path: Path | None) -> tuple[float, float] | None:
    if path is None:
        return None
    with fits.open(path) as hdul:
        for hdu in hdul:
            header = hdu.header
            if "LTV1" in header and "LTV2" in header:
                return -float(header["LTV1"]), -float(header["LTV2"])
    raise KeyError(f"{path} does not contain LTV1/LTV2")


def _fallback_origin_from_points(pred_x: np.ndarray, pred_y: np.ndarray) -> tuple[float, float]:
    # Cutout measurements in this workflow retain parent-patch pixel
    # coordinates.  The lower integer edge recovers the 512x512 cutout origin.
    return math.floor(float(np.nanmin(pred_x))), math.floor(float(np.nanmin(pred_y)))


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


def _kron_radius_col(table) -> str | None:
    for name in (
        "ext_photometryKron_KronFlux_radius",
        "ext_photometryKron_KronFlux_radius_for_radius",
    ):
        if name in table.colnames:
            return name
    return None


def _ellipse_params(table, row_index: int, *, shape_scale: float) -> tuple[float, float, float, str] | None:
    prefix = _shape_prefix(table)
    if prefix is None:
        return None
    try:
        xx, yy, xy = [float(table[f"{prefix}_{suffix}"][row_index]) for suffix in ("xx", "yy", "xy")]
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

    scale = float(shape_scale)
    source = f"{prefix}*shape_scale"
    kron_col = _kron_radius_col(table)
    if kron_col is not None:
        try:
            kron_radius = float(table[kron_col][row_index])
            if np.isfinite(kron_radius) and kron_radius > 0:
                scale = kron_radius
                source = f"{prefix}*{kron_col}"
        except Exception:
            pass

    semi_major = float(scale * np.sqrt(vals[0]))
    semi_minor = float(scale * np.sqrt(vals[1]))
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    if not np.all(np.isfinite([semi_major, semi_minor, angle])) or semi_major <= 0 or semi_minor <= 0:
        return None
    return semi_major, semi_minor, angle, source


def _write_reg(
    path: Path,
    rows: list[dict],
    *,
    color: str,
    show_text: bool,
    point_size: int,
    region_shape: str,
    fallback_point: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write(
            f'global color={color} dashlist=8 3 width=2 font="helvetica 14 bold roman" '
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
        )
        handle.write("image\n")
        for row in rows:
            text = ""
            if show_text:
                text = f" # text={{{row['label']} id={row['id']} mag={float(row['pred_mag']):.2f}}}"
            if region_shape == "ellipse" and row.get("ellipse_a_pix") not in ("", None):
                handle.write(
                    f"ellipse({float(row['image_x']):.3f},{float(row['image_y']):.3f},"
                    f"{float(row['ellipse_a_pix']):.3f},{float(row['ellipse_b_pix']):.3f},"
                    f"{float(row['ellipse_angle_deg']):.2f}){text}\n"
                )
            elif region_shape == "point" or fallback_point:
                handle.write(
                    f"point({float(row['image_x']):.3f},{float(row['image_y']):.3f}) "
                    f"# point=circle {point_size}{text}\n"
                )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "table_index",
        "id",
        "parent",
        "pred_mag",
        "pred_flux",
        "global_x",
        "global_y",
        "image_x",
        "image_y",
        "centroid_x_col",
        "centroid_y_col",
        "flux_col",
        "ellipse_a_pix",
        "ellipse_b_pix",
        "ellipse_angle_deg",
        "ellipse_source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_false_positive_centroids(args: argparse.Namespace) -> dict:
    reference = args.reference.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    prediction = (args.prediction.expanduser().resolve() if args.prediction else _prediction_path(run_dir, args.band))
    radius_pix = float(args.match_radius_arcsec) / float(args.pixel_scale)

    ref_points = _load_points(
        reference,
        x_col=args.ref_x,
        y_col=args.ref_y,
        role="ref",
        hdu=args.ref_hdu,
        leaf_only=True,
        drop_flagged_centroids=True,
        require_science_model=False,
    )
    pred_points = _load_points(
        prediction,
        x_col=args.pred_x,
        y_col=args.pred_y,
        role="pred",
        hdu=args.pred_hdu,
        leaf_only=True,
        drop_flagged_centroids=True,
        require_science_model=True,
    )
    matches, ref_used, pred_used = match_nearest_unique(ref_points, pred_points, radius_pix)

    if args.flux_col not in pred_points.table.colnames:
        raise KeyError(f"{prediction} missing flux column {args.flux_col!r}")
    pred_flux = np.asarray(pred_points.table[args.flux_col], dtype=float)[pred_points.table_indices]
    pred_mag = _flux_to_mag(pred_flux, args.pred_mag_zero_point)

    origin = _origin_from_background(args.background.expanduser().resolve() if args.background else None)
    if origin is None:
        origin = _fallback_origin_from_points(pred_points.x, pred_points.y)
    origin_x, origin_y = origin

    selected = (~pred_used) & np.isfinite(pred_mag) & (pred_mag >= args.mag_min) & (pred_mag < args.mag_max)
    rows: list[dict] = []
    table = pred_points.table
    for filtered_index in np.flatnonzero(selected):
        table_index = int(pred_points.table_indices[filtered_index])
        source_id = int(pred_points.ids[filtered_index])
        parent = int(table["parent"][table_index]) if "parent" in table.colnames else -1
        gx = float(pred_points.x[filtered_index])
        gy = float(pred_points.y[filtered_index])
        ellipse = _ellipse_params(table, table_index, shape_scale=float(args.shape_scale))
        rows.append(
            {
                "label": args.label,
                "table_index": table_index,
                "id": source_id,
                "parent": parent,
                "pred_mag": float(pred_mag[filtered_index]),
                "pred_flux": float(pred_flux[filtered_index]),
                "global_x": gx,
                "global_y": gy,
                "image_x": gx - origin_x + 1.0,
                "image_y": gy - origin_y + 1.0,
                "centroid_x_col": pred_points.x_col,
                "centroid_y_col": pred_points.y_col,
                "flux_col": args.flux_col,
                "ellipse_a_pix": "" if ellipse is None else float(ellipse[0]),
                "ellipse_b_pix": "" if ellipse is None else float(ellipse[1]),
                "ellipse_angle_deg": "" if ellipse is None else float(ellipse[2]),
                "ellipse_source": "" if ellipse is None else ellipse[3],
            }
        )

    outdir = args.output_dir.expanduser().resolve()
    safe_label = args.label.replace("/", "_")
    csv_path = outdir / f"{safe_label}_fp_centroids_mag{args.mag_min:g}_{args.mag_max:g}.csv"
    reg_path = outdir / f"{safe_label}_fp_{args.region_shape}_mag{args.mag_min:g}_{args.mag_max:g}.reg"
    _write_csv(csv_path, rows)
    _write_reg(
        reg_path,
        rows,
        color=args.color,
        show_text=args.show_text,
        point_size=args.point_size,
        region_shape=args.region_shape,
        fallback_point=bool(args.fallback_point),
    )

    summary = {
        "label": args.label,
        "reference": str(reference),
        "prediction": str(prediction),
        "run_dir": str(run_dir),
        "reference_count": ref_points.n,
        "prediction_count": pred_points.n,
        "matched_count": len(matches),
        "false_positive_count_all_magnitudes": int(np.count_nonzero(~pred_used)),
        "selected_false_positive_count": len(rows),
        "selected_false_positive_with_ellipse_count": int(sum(row.get("ellipse_a_pix") not in ("", None) for row in rows)),
        "mag_min": float(args.mag_min),
        "mag_max": float(args.mag_max),
        "pred_mag_zero_point": float(args.pred_mag_zero_point),
        "flux_col": args.flux_col,
        "match_radius_arcsec": float(args.match_radius_arcsec),
        "match_radius_pix": radius_pix,
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "csv": str(csv_path),
        "reg": str(reg_path),
        "region_shape": args.region_shape,
        "shape_scale": float(args.shape_scale),
    }
    summary_path = outdir / f"{safe_label}_fp_centroids_mag{args.mag_min:g}_{args.mag_max:g}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, default=None)
    parser.add_argument("--background", type=Path, default=None, help="Cutout FITS used to convert global pixel coords to DS9 image coords.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--band", default="HSC-I")
    parser.add_argument("--mag-min", type=float, required=True)
    parser.add_argument("--mag-max", type=float, required=True)
    parser.add_argument("--pred-mag-zero-point", type=float, default=31.4)
    parser.add_argument("--flux-col", default="base_PsfFlux_instFlux")
    parser.add_argument("--pixel-scale", type=float, default=DEFAULT_PIXEL_SCALE)
    parser.add_argument("--match-radius-arcsec", type=float, default=DEFAULT_RADIUS_ARCSEC)
    parser.add_argument("--ref-hdu", type=int, default=1)
    parser.add_argument("--pred-hdu", type=int, default=1)
    parser.add_argument("--ref-x", default=None)
    parser.add_argument("--ref-y", default=None)
    parser.add_argument("--pred-x", default=None)
    parser.add_argument("--pred-y", default=None)
    parser.add_argument("--color", default="red")
    parser.add_argument("--region-shape", choices=("ellipse", "point"), default="ellipse")
    parser.add_argument("--shape-scale", type=float, default=2.0, help="Fallback semi-axis scale for moment ellipses when no Kron radius exists.")
    parser.add_argument("--fallback-point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--point-size", type=int, default=7)
    parser.add_argument("--show-text", action="store_true")
    args = parser.parse_args()
    export_false_positive_centroids(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
