"""Match TP sources, compare magnitudes, and export largest-difference FITS stamps.

Example
-------
python utils/export_tp_mag_diff_stamps.py \
    --run output/sam_denoised_32_meas \
    --reference fits/catalog/meas-HSC-I-9813-4,5.fits \
    --output-dir ~/transfer/sam_denoised_32_meas_tp_mag_diff \
    --ref-zero-point 27 \
    --pred-zero-point 31.4 \
    --top-n 6
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.units import UnitsWarning
import warnings

try:
    import lsst.afw.detection  # noqa: F401  # registers Footprint bindings
    import lsst.afw.table as afwTable
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Run this script inside an initialized LSST stack environment.") from exc


warnings.simplefilter("ignore", UnitsWarning)


def _mag(flux: float, zero_point: float) -> float:
    try:
        flux = float(flux)
    except Exception:
        return np.nan
    return float(zero_point - 2.5 * np.log10(flux)) if np.isfinite(flux) and flux > 0 else np.nan


def _record_get(record, name: str, default=np.nan):
    try:
        return record.get(name)
    except Exception:
        try:
            return record[name]
        except Exception:
            return default


def _has_schema(catalog, name: str) -> bool:
    try:
        catalog.schema.find(name)
        return True
    except Exception:
        return False


def _is_sky_source(record) -> bool:
    for name in ("merge_footprint_sky", "merge_peak_sky"):
        try:
            if bool(record.get(name)):
                return True
        except Exception:
            pass
    return False


def _stamp_bounds(center_x: float, center_y: float, shape: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    height, width = shape
    ix = int(round(center_x))
    iy = int(round(center_y))
    x0 = max(0, min(width - size, ix - size // 2))
    y0 = max(0, min(height - size, iy - size // 2))
    return x0, y0, x0 + size, y0 + size


def _footprint_mask(footprint, global_x0: int, global_y0: int, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if footprint is None:
        return mask
    indices = np.asarray(footprint.getSpans().indices(), dtype=np.int64)
    if indices.shape[0] != 2 or indices.shape[1] == 0:
        return mask
    y = indices[0]
    x = indices[1]
    inside = (x >= global_x0) & (x < global_x0 + width) & (y >= global_y0) & (y < global_y0 + height)
    if np.any(inside):
        mask[y[inside] - global_y0, x[inside] - global_x0] = True
    return mask


def _load_reference_points(table: Table, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(table[args.ref_x], dtype=float)
    y = np.asarray(table[args.ref_y], dtype=float)
    flux = np.asarray(table[args.flux_col], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(flux) & (flux > 0)
    if args.leaf_only:
        if "deblend_nChild" in table.colnames:
            mask &= np.asarray(table["deblend_nChild"], dtype=int) == 0
        elif "parent" in table.colnames:
            mask &= np.asarray(table["parent"], dtype=int) != 0
    return np.flatnonzero(mask), x, y


def _load_prediction_points(catalog, args: argparse.Namespace) -> list[tuple[int, object, float, float, float]]:
    rows = []
    for index, record in enumerate(catalog):
        if args.leaf_only:
            parent = int(_record_get(record, "parent", 0))
            n_child = int(_record_get(record, "deblend_nChild", 0))
            if parent == 0 or n_child != 0:
                continue
        if _is_sky_source(record):
            continue
        flux = _record_get(record, args.flux_col, np.nan)
        try:
            flux = float(flux)
        except Exception:
            flux = np.nan
        if not (np.isfinite(flux) and flux > 0):
            continue
        x = _record_get(record, args.pred_x, np.nan)
        y = _record_get(record, args.pred_y, np.nan)
        if not (np.isfinite(x) and np.isfinite(y)):
            x = _record_get(record, "deblend_peak_center_x", np.nan)
            y = _record_get(record, "deblend_peak_center_y", np.nan)
        if np.isfinite(x) and np.isfinite(y):
            rows.append((index, record, float(x), float(y), flux))
    return rows


def _greedy_matches(
    *,
    ref_indices: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    pred_rows: list[tuple[int, object, float, float, float]],
    radius_pix: float,
) -> list[tuple[float, int, int]]:
    pairs = []
    for ref_index in ref_indices:
        for pred_index, (_, _, pred_x, pred_y, _) in enumerate(pred_rows):
            distance = float(np.hypot(pred_x - ref_x[ref_index], pred_y - ref_y[ref_index]))
            if np.isfinite(distance) and distance <= radius_pix:
                pairs.append((distance, int(ref_index), int(pred_index)))
    pairs.sort(key=lambda item: item[0])

    used_ref = set()
    used_pred = set()
    matches = []
    for distance, ref_index, pred_index in pairs:
        if ref_index in used_ref or pred_index in used_pred:
            continue
        used_ref.add(ref_index)
        used_pred.add(pred_index)
        matches.append((distance, ref_index, pred_index))
    return matches


def _image_hdu(path: Path):
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul["IMAGE"] if "IMAGE" in hdul else next(
            item for item in hdul if item.data is not None and getattr(item.data, "ndim", 0) == 2
        )
        return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()


def _write_largest_fits(
    *,
    rows: list[dict],
    pred_catalog,
    ref_table: Table,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    image_path: Path,
    fits_dir: Path,
    top_n: int,
    stamp_size: int,
) -> None:
    image, image_header = _image_hdu(image_path)
    origin_x = int(round(-float(image_header.get("LTV1", 0.0))))
    origin_y = int(round(-float(image_header.get("LTV2", 0.0))))
    largest = sorted(rows, key=lambda row: row["abs_delta_mag"], reverse=True)[:top_n]

    fits_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(largest, start=1):
        record = pred_catalog[row["pred_table_index"]]
        local_x = ref_x[row["ref_table_index"]] - origin_x
        local_y = ref_y[row["ref_table_index"]] - origin_y
        x0, y0, x1, y1 = _stamp_bounds(local_x, local_y, image.shape, stamp_size)
        global_x0 = x0 + origin_x
        global_y0 = y0 + origin_y
        stamp = image[y0:y1, x0:x1].astype(np.float32, copy=True)
        fp_mask = _footprint_mask(record.getFootprint(), global_x0, global_y0, x1 - x0, y1 - y0)
        fp_image = np.full(stamp.shape, np.nan, dtype=np.float32)
        fp_image[fp_mask] = stamp[fp_mask]

        summary = Table()
        for key, value in row.items():
            summary[key] = [value]

        header = fits.Header()
        header["RANK"] = rank
        header["REFROW"] = int(row["ref_table_index"])
        header["REFID"] = int(row["ref_id"])
        header["PREDID"] = int(row["pred_id"])
        header["DMAG"] = float(row["delta_mag_pred_minus_gt"])
        header["ABSDMAG"] = float(row["abs_delta_mag"])
        header["GTMAG"] = float(row["gt_mag"])
        header["PREDMAG"] = float(row["pred_mag"])
        header["DISTASEC"] = float(row["distance_arcsec"])
        header["FPAREA"] = int(row["footprint_area"])
        header["LTV1"] = -float(global_x0)
        header["LTV2"] = -float(global_y0)

        summary_hdu = fits.table_to_hdu(summary)
        summary_hdu.name = "SUMMARY"
        ref_hdu = fits.table_to_hdu(ref_table[row["ref_table_index"] : row["ref_table_index"] + 1])
        ref_hdu.name = "REF_ROW"
        out_path = fits_dir / (
            f"rank{rank:02d}_refrow{row['ref_table_index']:04d}_ref{row['ref_id']}"
            f"_pred{row['pred_id']}_dmag{row['delta_mag_pred_minus_gt']:+.3f}.fits"
        )
        fits.HDUList(
            [
                fits.PrimaryHDU(data=stamp, header=header),
                fits.ImageHDU(data=fp_mask.astype(np.uint8), name="FOOTPRINT_MASK"),
                fits.ImageHDU(data=fp_image, name="FOOTPRINT_IMAGE"),
                summary_hdu,
                ref_hdu,
            ]
        ).writeto(out_path, overwrite=True, output_verify="silentfix")
        row["stamp_fits"] = str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Pipeline run directory.")
    parser.add_argument("--reference", type=Path, default=Path("fits/catalog/meas-HSC-I-9813-4,5.fits"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--band", default="HSC-I")
    parser.add_argument("--flux-col", default="base_PsfFlux_instFlux")
    parser.add_argument("--ref-x", default="base_SdssCentroid_x")
    parser.add_argument("--ref-y", default="base_SdssCentroid_y")
    parser.add_argument("--pred-x", default="base_SdssCentroid_x")
    parser.add_argument("--pred-y", default="base_SdssCentroid_y")
    parser.add_argument("--ref-zero-point", type=float, default=27.0)
    parser.add_argument("--pred-zero-point", type=float, default=31.4)
    parser.add_argument("--match-radius-arcsec", type=float, default=0.5)
    parser.add_argument("--pixel-scale", type=float, default=0.168)
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--stamp-size", type=int, default=64)
    parser.add_argument("--leaf-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    run_dir = args.run.expanduser()
    output_dir = args.output_dir.expanduser()
    fits_dir = output_dir / f"largest_{args.top_n}_fits"
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_table = Table.read(args.reference.expanduser(), hdu=1)
    pred_path = run_dir / "measure" / args.band / "deepCoadd_meas.fits"
    image_path = run_dir / "detect" / args.band / "deepCoadd_calexp.fits"
    pred_catalog = afwTable.SourceCatalog.readFits(str(pred_path))

    ref_indices, ref_x, ref_y = _load_reference_points(ref_table, args)
    pred_rows = _load_prediction_points(pred_catalog, args)
    radius_pix = args.match_radius_arcsec / args.pixel_scale
    matched = _greedy_matches(ref_indices=ref_indices, ref_x=ref_x, ref_y=ref_y, pred_rows=pred_rows, radius_pix=radius_pix)

    rows = []
    ref_flux = np.asarray(ref_table[args.flux_col], dtype=float)
    for distance_pix, ref_index, pred_point_index in matched:
        pred_table_index, record, pred_x, pred_y, pred_flux = pred_rows[pred_point_index]
        gt_flux = float(ref_flux[ref_index])
        gt_mag = _mag(gt_flux, args.ref_zero_point)
        pred_mag = _mag(pred_flux, args.pred_zero_point)
        delta_mag = pred_mag - gt_mag if np.isfinite(gt_mag) and np.isfinite(pred_mag) else np.nan
        footprint = record.getFootprint()
        row = {
            "ref_table_index": int(ref_index),
            "ref_id": int(ref_table["id"][ref_index]) if "id" in ref_table.colnames else int(ref_index),
            "pred_table_index": int(pred_table_index),
            "pred_id": int(record.getId()),
            "pred_parent": int(_record_get(record, "parent", -1)),
            "distance_pix": float(distance_pix),
            "distance_arcsec": float(distance_pix * args.pixel_scale),
            "gt_flux": gt_flux,
            "pred_flux": float(pred_flux),
            "gt_mag": gt_mag,
            "pred_mag": pred_mag,
            "delta_mag_pred_minus_gt": delta_mag,
            "abs_delta_mag": abs(delta_mag) if np.isfinite(delta_mag) else np.nan,
            "flux_ratio_pred_over_gt": float(pred_flux / gt_flux) if gt_flux else np.nan,
            "footprint_area": int(footprint.getArea()) if footprint is not None else 0,
        }
        for flag in (
            "base_PsfFlux_flag",
            "base_PsfFlux_flag_edge",
            "base_PixelFlags_flag_inexact_psfCenter",
            "base_PixelFlags_flag_clippedCenter",
        ):
            if _has_schema(pred_catalog, flag):
                row[flag] = bool(_record_get(record, flag, False))
        rows.append(row)

    _write_largest_fits(
        rows=rows,
        pred_catalog=pred_catalog,
        ref_table=ref_table,
        ref_x=ref_x,
        ref_y=ref_y,
        image_path=image_path,
        fits_dir=fits_dir,
        top_n=args.top_n,
        stamp_size=args.stamp_size,
    )

    csv_path = output_dir / "tp_mag_differences.csv"
    if rows:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("")

    finite_diff = np.array([row["delta_mag_pred_minus_gt"] for row in rows], dtype=float)
    finite_diff = finite_diff[np.isfinite(finite_diff)]
    summary = [
        f"reference={args.reference}",
        f"prediction={pred_path}",
        f"ref_zero_point={args.ref_zero_point}",
        f"pred_zero_point={args.pred_zero_point}",
        f"match_radius_arcsec={args.match_radius_arcsec}",
        f"reference_leaf_positive_flux_count={len(ref_indices)}",
        f"prediction_leaf_positive_flux_count={len(pred_rows)}",
        f"tp_count={len(rows)}",
    ]
    if finite_diff.size:
        summary.extend(
            [
                f"delta_mag_mean={np.nanmean(finite_diff):.6g}",
                f"delta_mag_median={np.nanmedian(finite_diff):.6g}",
                f"delta_mag_std={np.nanstd(finite_diff):.6g}",
                "delta_mag_percentiles_5_16_50_84_95="
                + ",".join(f"{value:.6g}" for value in np.nanpercentile(finite_diff, [5, 16, 50, 84, 95])),
                f"abs_delta_mag_max={np.nanmax(np.abs(finite_diff)):.6g}",
            ]
        )
    summary_path = output_dir / "tp_mag_difference_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote FITS stamps to {fits_dir}")


if __name__ == "__main__":
    main()
