#!/usr/bin/env python
"""Export masked image stamps for selected false-positive sources.

The source mask is taken from the LSST SourceCatalog Footprint of each selected
prediction id.  Each stamp is clipped to the image boundary; pixels outside the
source footprint are written as NaN.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval

try:
    import lsst.afw.detection  # noqa: F401  # registers Footprint Python bindings
    import lsst.afw.table as afwTable
except ImportError as exc:  # pragma: no cover
    afwTable = None
    AFW_IMPORT_ERROR = exc
else:
    AFW_IMPORT_ERROR = None


BANDS = ("HSC-I", "HSC-R", "HSC-G")


def _origin_from_hdu(header: fits.Header) -> tuple[int, int]:
    return -int(round(float(header["LTV1"]))), -int(round(float(header["LTV2"])))


def _load_band_images(run_dir: Path) -> dict[str, tuple[np.ndarray, fits.Header, int, int]]:
    out = {}
    for band in BANDS:
        path = run_dir / "detect" / band / "deepCoadd_calexp.fits"
        if not path.exists():
            continue
        with fits.open(path) as hdul:
            hdu = hdul["IMAGE"] if "IMAGE" in hdul else next(h for h in hdul if getattr(h, "data", None) is not None)
            data = np.asarray(hdu.data, dtype=np.float32)
            header = hdu.header.copy()
            ox, oy = _origin_from_hdu(header)
        out[band] = (data, header, ox, oy)
    if not out:
        raise FileNotFoundError(f"no detect/*/deepCoadd_calexp.fits images found under {run_dir}")
    return out


def _read_selected_rows(path: Path, *, y_min: float, limit: int | None) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["image_y"]) > y_min:
                rows.append(row)
    rows.sort(key=lambda row: float(row.get("ellipse_a_pix") or 0.0), reverse=True)
    return rows if limit is None else rows[:limit]


def _load_footprints(catalog_path: Path) -> dict[int, object]:
    if afwTable is None:
        raise ImportError(f"lsst.afw.table is required to read footprints: {AFW_IMPORT_ERROR}")
    catalog = afwTable.SourceCatalog.readFits(str(catalog_path))
    return {int(record.getId()): record.getFootprint() for record in catalog if record.getFootprint() is not None}


def _catalog_path(run_dir: Path, band: str) -> Path:
    meas = run_dir / "measure" / band / "deepCoadd_meas.fits"
    if meas.exists():
        return meas
    deblend = run_dir / "deblend" / "deepCoadd_deblendedFlux.fits"
    if deblend.exists():
        return deblend
    raise FileNotFoundError(f"no measurement/deblend catalog found under {run_dir}")


def _footprint_arrays(footprint) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    indices = np.asarray(footprint.getSpans().indices(), dtype=np.int64)
    if indices.shape[0] != 2 or indices.shape[1] == 0:
        raise RuntimeError("empty footprint spans")
    bbox = footprint.getBBox()
    return (
        indices[1],
        indices[0],
        (int(bbox.getMinX()), int(bbox.getMinY()), int(bbox.getMaxX()) + 1, int(bbox.getMaxY()) + 1),
    )


def _masked_crop(
    image: np.ndarray,
    *,
    image_origin_x: int,
    image_origin_y: int,
    footprint,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    fp_x, fp_y, (fp_x0, fp_y0, fp_x1, fp_y1) = _footprint_arrays(footprint)
    height, width = image.shape
    image_x0 = image_origin_x
    image_y0 = image_origin_y
    image_x1 = image_x0 + width
    image_y1 = image_y0 + height
    gx0 = max(fp_x0, image_x0)
    gy0 = max(fp_y0, image_y0)
    gx1 = min(fp_x1, image_x1)
    gy1 = min(fp_y1, image_y1)
    if gx1 <= gx0 or gy1 <= gy0:
        raise RuntimeError("footprint bbox does not overlap image")

    crop = image[gy0 - image_y0 : gy1 - image_y0, gx0 - image_x0 : gx1 - image_x0].astype(np.float32, copy=True)
    mask = np.zeros(crop.shape, dtype=bool)
    inside = (fp_x >= gx0) & (fp_x < gx1) & (fp_y >= gy0) & (fp_y < gy1)
    mask[fp_y[inside] - gy0, fp_x[inside] - gx0] = True
    masked = np.full(crop.shape, np.nan, dtype=np.float32)
    masked[mask] = crop[mask]
    return masked, mask, (gx0, gy0, gx1, gy1)


def _write_png(path: Path, arrays: dict[str, np.ndarray], title: str) -> None:
    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (band, data) in zip(axes, arrays.items()):
        finite = data[np.isfinite(data)]
        if finite.size:
            vmin, vmax = ZScaleInterval().get_limits(finite)
        else:
            vmin, vmax = 0.0, 1.0
        ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(band)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    rows = _read_selected_rows(args.fp_csv.expanduser().resolve(), y_min=args.y_min, limit=args.limit)
    band_images = _load_band_images(run_dir)
    catalog_path = _catalog_path(run_dir, args.catalog_band)
    footprints = _load_footprints(catalog_path)
    outdir = args.output_dir.expanduser().resolve()
    fits_dir = outdir / "fits"
    png_dir = outdir / "png"
    fits_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for seq, row in enumerate(rows, start=1):
        source_id = int(row["id"])
        footprint = footprints.get(source_id)
        if footprint is None:
            print(f"WARNING: no footprint for source id={source_id}")
            continue

        hdus = []
        png_arrays = {}
        bounds = None
        mask_area = None
        for band, (image, header, ox, oy) in band_images.items():
            masked, mask, this_bounds = _masked_crop(image, image_origin_x=ox, image_origin_y=oy, footprint=footprint)
            bounds = this_bounds
            mask_area = int(np.count_nonzero(mask))
            hdu_header = header.copy()
            gx0, gy0, gx1, gy1 = this_bounds
            hdu_header["LTV1"] = (-float(gx0), "LSST image origin x offset")
            hdu_header["LTV2"] = (-float(gy0), "LSST image origin y offset")
            hdu_header["SRCID"] = source_id
            hdu_header["PARENT"] = int(row["parent"])
            hdu_header["PREDMAG"] = float(row["pred_mag"])
            hdu_header["GX0"] = gx0
            hdu_header["GY0"] = gy0
            hdu_header["GX1"] = gx1 - 1
            hdu_header["GY1"] = gy1 - 1
            if not hdus:
                hdus.append(fits.PrimaryHDU(data=masked, header=hdu_header))
                hdus.append(fits.ImageHDU(data=mask.astype(np.uint8), name="SOURCE_MASK"))
            else:
                hdus.append(fits.ImageHDU(data=masked, header=hdu_header, name=f"MASKED_{band}"))
            png_arrays[band] = masked

        if bounds is None:
            continue
        filename = f"fp_{seq:02d}_id{source_id}_y{float(row['image_y']):.1f}_mag{float(row['pred_mag']):.2f}"
        fits_path = fits_dir / f"{filename}.fits"
        png_path = png_dir / f"{filename}.png"
        fits.HDUList(hdus).writeto(fits_path, overwrite=True, output_verify="silentfix")
        _write_png(png_path, png_arrays, f"id={source_id} parent={row['parent']} mag={float(row['pred_mag']):.2f}")
        summary.append(
            {
                "id": source_id,
                "parent": int(row["parent"]),
                "pred_mag": float(row["pred_mag"]),
                "image_x": float(row["image_x"]),
                "image_y": float(row["image_y"]),
                "mask_area": mask_area,
                "gx0": bounds[0],
                "gy0": bounds[1],
                "gx1": bounds[2] - 1,
                "gy1": bounds[3] - 1,
                "fits": str(fits_path),
                "png": str(png_path),
            }
        )

    summary_path = outdir / "masked_stamp_summary.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = list(summary[0]) if summary else [
            "id",
            "parent",
            "pred_mag",
            "image_x",
            "image_y",
            "mask_area",
            "gx0",
            "gy0",
            "gx1",
            "gy1",
            "fits",
            "png",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"selected_rows={len(rows)} wrote={len(summary)} outdir={outdir} summary={summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp-csv", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--y-min", type=float, default=400.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--catalog-band", default="HSC-I")
    args = parser.parse_args()
    export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
