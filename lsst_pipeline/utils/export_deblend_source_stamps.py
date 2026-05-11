"""Export deblended source stamps for selected reference sources.

This utility compares selected reference-catalog sources against one or more
pipeline run directories.  For each matched prediction it writes a DS9-friendly
multi-HDU FITS file containing the input coadd stamp, the scarlet deblend model
for each available band, and compact GT/prediction metadata.  Unmatched
reference sources are written to a false-negative CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table
from astropy.units import UnitsWarning

try:
    import lsst.afw.detection  # noqa: F401  # registers Footprint Python bindings
    import lsst.afw.table as afwTable
except ImportError:  # pragma: no cover - optional outside an initialized LSST env
    afwTable = None


warnings.simplefilter("ignore", UnitsWarning)
warnings.simplefilter("ignore", VerifyWarning)


DEFAULT_REFERENCE = Path("fits/catalog/meas-HSC-I-9813-4,5.fits")
DEFAULT_REF_X = "base_SdssCentroid_x"
DEFAULT_REF_Y = "base_SdssCentroid_y"
DEFAULT_REF_FLUX = "base_PsfFlux_instFlux"
DEFAULT_MATCH_RADIUS_ARCSEC = 0.5
DEFAULT_PIXEL_SCALE = 0.168
AB_NJY_ZEROPOINT = 31.4
IRG_BANDS = ("i", "r", "g")
FN_FIELDS = [
    "seq",
    "ref_row",
    "ref_id",
    "ref_x",
    "ref_y",
    "ref_mag",
    "nearest_distance_pix",
    "nearest_distance_arcsec",
    "reason",
]


@dataclass(frozen=True)
class BandImage:
    label: str
    path: Path
    data: np.ndarray
    header: fits.Header
    origin_x: int
    origin_y: int


@dataclass(frozen=True)
class SourceModel:
    cube: np.ndarray
    spectrum: np.ndarray
    origin_y: int
    origin_x: int


@dataclass(frozen=True)
class FootprintData:
    y: np.ndarray
    x: np.ndarray
    area: int
    bbox_min_x: int
    bbox_min_y: int
    bbox_max_x: int
    bbox_max_y: int


def _repo_root_from_run(run_dir: Path) -> Path:
    parts = run_dir.resolve().parts
    if "output" in parts:
        output_index = parts.index("output")
        return Path(*parts[:output_index])
    return run_dir.resolve().parent


def _resolve_path(path_text: str | Path, *, base: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _read_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing manifest: {path}")
    return json.loads(path.read_text())


def _load_band_images(run_dir: Path, manifest: dict) -> dict[str, BandImage]:
    repo_root = _repo_root_from_run(run_dir)
    by_label: dict[str, BandImage] = {}
    for band_name, info in manifest.get("detect", {}).items():
        label = str(info.get("band_label", "")).lower()
        if not label:
            continue
        path_text = info.get("input_fits") or info.get("post_detect_calexp_fits")
        if path_text is None:
            continue
        path = _resolve_path(path_text, base=repo_root)
        with fits.open(path, memmap=True) as hdul:
            hdu = hdul["IMAGE"] if "IMAGE" in hdul else next(
                h for h in hdul if h.data is not None and getattr(h.data, "ndim", 0) == 2
            )
            header = hdu.header.copy()
            data = np.asarray(hdu.data, dtype=np.float32)
        origin_x = int(round(-float(header.get("LTV1", 0.0))))
        origin_y = int(round(-float(header.get("LTV2", 0.0))))
        by_label[label] = BandImage(
            label=label,
            path=path,
            data=data,
            header=header,
            origin_x=origin_x,
            origin_y=origin_y,
        )
    if not by_label:
        raise RuntimeError(f"could not find input coadd images from {run_dir / 'manifest.json'}")
    return by_label


def _load_model_data(run_dir: Path, manifest: dict):
    repo_root = _repo_root_from_run(run_dir)
    path_text = manifest.get("deblend", {}).get("scarlet_model_path")
    if path_text is None:
        path = run_dir / "deblend" / "deepCoadd_scarletModelData.pickle"
    else:
        path = _resolve_path(path_text, base=repo_root)
        if not path.exists():
            path = run_dir / "deblend" / "deepCoadd_scarletModelData.pickle"
    if not path.exists():
        raise FileNotFoundError(f"missing scarlet model pickle: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _model_bands(model_data) -> list[str]:
    first_blend = next(iter(model_data.blends.values()))
    return [str(value).lower() for value in getattr(first_blend, "bands", [])]


def _source_model(source, *, n_bands: int) -> SourceModel:
    pieces = []
    spectra = []
    for component in getattr(source, "factorized_components", []):
        spectrum = np.asarray(component.spectrum, dtype=np.float32)
        morph = np.asarray(component.morph, dtype=np.float32)
        # Normalize to 1 total morph
        morph_sum = np.nansum(morph)        
        if morph_sum > 0:
            morph /= morph_sum
        cube = spectrum[:, None, None] * morph[None, :, :]
        origin_y, origin_x = tuple(component.origin)
        pieces.append((cube, int(origin_y), int(origin_x)))
        spectra.append(spectrum)
    if not pieces:
        return SourceModel(
            cube=np.zeros((n_bands, 1, 1), dtype=np.float32),
            spectrum=np.zeros(n_bands, dtype=np.float32),
            origin_y=0,
            origin_x=0,
        )

    y0 = min(origin_y for _, origin_y, _ in pieces)
    x0 = min(origin_x for _, _, origin_x in pieces)
    y1 = max(origin_y + cube.shape[1] for cube, origin_y, _ in pieces)
    x1 = max(origin_x + cube.shape[2] for cube, _, origin_x in pieces)
    full = np.zeros((n_bands, y1 - y0, x1 - x0), dtype=np.float32)
    for cube, origin_y, origin_x in pieces:
        yy = origin_y - y0
        xx = origin_x - x0
        full[:, yy : yy + cube.shape[1], xx : xx + cube.shape[2]] += cube
    spectrum = np.sum(np.vstack(spectra), axis=0).astype(np.float32)
    return SourceModel(cube=full, spectrum=spectrum, origin_y=y0, origin_x=x0)


def _load_prediction_footprints(pred_path: Path) -> dict[int, FootprintData]:
    """Read LSST Footprints by source id from a SourceCatalog FITS file."""
    if afwTable is None:
        warnings.warn(
            "LSST afwTable is not available; footprint masks and footprint sums will be omitted.",
            RuntimeWarning,
        )
        return {}
    catalog = afwTable.SourceCatalog.readFits(str(pred_path))
    footprints: dict[int, FootprintData] = {}
    for record in catalog:
        footprint = record.getFootprint()
        if footprint is None:
            continue
        indices = np.asarray(footprint.getSpans().indices(), dtype=np.int64)
        if indices.shape[0] != 2 or indices.shape[1] == 0:
            continue
        bbox = footprint.getBBox()
        footprints[int(record.getId())] = FootprintData(
            y=indices[0].copy(),
            x=indices[1].copy(),
            area=int(footprint.getArea()),
            bbox_min_x=int(bbox.getMinX()),
            bbox_min_y=int(bbox.getMinY()),
            bbox_max_x=int(bbox.getMaxX()),
            bbox_max_y=int(bbox.getMaxY()),
        )
    return footprints


def _footprint_mask(
    footprint: FootprintData | None,
    *,
    global_x0: int,
    global_y0: int,
    size: int,
) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    if footprint is None:
        return mask
    inside = (
        (footprint.x >= global_x0)
        & (footprint.x < global_x0 + size)
        & (footprint.y >= global_y0)
        & (footprint.y < global_y0 + size)
    )
    if np.any(inside):
        mask[footprint.y[inside] - global_y0, footprint.x[inside] - global_x0] = True
    return mask


def _footprint_region_mask(
    footprint: FootprintData | None,
    *,
    global_x0: int,
    global_y0: int,
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if footprint is None:
        return mask
    inside = (
        (footprint.x >= global_x0)
        & (footprint.x < global_x0 + width)
        & (footprint.y >= global_y0)
        & (footprint.y < global_y0 + height)
    )
    if np.any(inside):
        mask[footprint.y[inside] - global_y0, footprint.x[inside] - global_x0] = True
    return mask


def _masked_plane(plane: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.full(plane.shape, np.nan, dtype=np.float32)
    masked[mask] = plane[mask]
    return masked


def _mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _footprint_bbox_bounds(
    footprint: FootprintData | None,
    image: BandImage,
) -> tuple[int, int, int, int] | None:
    if footprint is None:
        return None
    image_height, image_width = image.data.shape
    image_global_x0 = image.origin_x
    image_global_y0 = image.origin_y
    image_global_x1 = image_global_x0 + image_width
    image_global_y1 = image_global_y0 + image_height
    global_x0 = max(int(footprint.bbox_min_x), image_global_x0)
    global_y0 = max(int(footprint.bbox_min_y), image_global_y0)
    global_x1 = min(int(footprint.bbox_max_x) + 1, image_global_x1)
    global_y1 = min(int(footprint.bbox_max_y) + 1, image_global_y1)
    if global_x1 <= global_x0 or global_y1 <= global_y0:
        return None
    return global_x0, global_y0, global_x1, global_y1


def _prediction_mask(table: Table) -> np.ndarray:
    x = np.asarray(table["deblend_peak_center_x"], dtype=float)
    y = np.asarray(table["deblend_peak_center_y"], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if "parent" in table.colnames:
        mask &= np.asarray(table["parent"], dtype=np.int64) != 0
    if "deblend_nChild" in table.colnames:
        mask &= np.asarray(table["deblend_nChild"], dtype=np.int64) == 0
    if "merge_footprint_sky" in table.colnames:
        mask &= ~np.asarray(table["merge_footprint_sky"], dtype=bool)
    if "deblend_modelType" in table.colnames:
        values = np.asarray(table["deblend_modelType"])
        mask &= np.array(
            [str(value.decode() if isinstance(value, bytes) else value).strip() != "" for value in values]
        )
    return mask


def _parse_int_list(text: str | None) -> list[int]:
    if text is None or text.strip() == "":
        return []
    values = []
    for item in text.replace(",", " ").split():
        values.append(int(item))
    return values


def _source_rows_from_csv(path: Path) -> tuple[list[int], list[int]]:
    rows: list[int] = []
    ids: list[int] = []
    with path.expanduser().open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key in ("row_index", "gt_row", "cat_row", "row"):
                if row.get(key, "") not in {"", None}:
                    rows.append(int(float(row[key])))
                    break
            for key in ("source_id", "gt_id", "id"):
                if row.get(key, "") not in {"", None}:
                    # Source ids are 64-bit integers; parsing through float can
                    # silently change the last few digits.
                    ids.append(int(str(row[key]).strip()))
                    break
    return rows, ids


def _target_indices(args: argparse.Namespace, ref: Table) -> list[int]:
    rows = _parse_int_list(args.rows)
    ids = _parse_int_list(args.source_ids)
    if args.sources_csv is not None:
        csv_rows, csv_ids = _source_rows_from_csv(args.sources_csv)
        rows.extend(csv_rows)
        ids.extend(csv_ids)

    out: list[int] = []
    for row in rows:
        if row < 0 or row >= len(ref):
            raise IndexError(f"reference row out of range: {row}")
        out.append(int(row))

    if ids:
        if "id" not in ref.colnames:
            raise KeyError("reference catalog has no 'id' column")
        by_id = {int(value): idx for idx, value in enumerate(np.asarray(ref["id"], dtype=np.int64))}
        for source_id in ids:
            if source_id not in by_id:
                raise KeyError(f"source id not found in reference catalog: {source_id}")
            out.append(int(by_id[source_id]))

    seen = set()
    unique = []
    for row in out:
        if row not in seen:
            unique.append(row)
            seen.add(row)
    if not unique:
        raise RuntimeError("no target sources supplied; use --rows, --source-ids, or --sources-csv")
    return unique


def _flux_to_mag(flux: float, zeropoint: float) -> float:
    if not np.isfinite(flux) or flux <= 0:
        return np.nan
    return float(zeropoint - 2.5 * np.log10(flux))


def _finite_table_float(value) -> float:
    if np.ma.is_masked(value):
        return np.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def _stamp_bounds(center_x: float, center_y: float, *, size: int, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape
    ix = int(round(center_x))
    iy = int(round(center_y))
    x0 = max(0, min(width - size, ix - size // 2))
    y0 = max(0, min(height - size, iy - size // 2))
    return x0, y0, x0 + size, y0 + size


def _insert_model_stamp(
    model: SourceModel,
    *,
    band_index: int,
    global_x0: int,
    global_y0: int,
    size: int,
) -> np.ndarray:
    return _insert_model_region(
        model,
        band_index=band_index,
        global_x0=global_x0,
        global_y0=global_y0,
        width=size,
        height=size,
    )


def _insert_model_region(
    model: SourceModel,
    *,
    band_index: int,
    global_x0: int,
    global_y0: int,
    width: int,
    height: int,
) -> np.ndarray:
    stamp = np.zeros((height, width), dtype=np.float32)
    if band_index >= model.cube.shape[0]:
        stamp[:] = np.nan
        return stamp
    plane = model.cube[band_index]
    model_x0 = int(model.origin_x)
    model_y0 = int(model.origin_y)
    model_x1 = model_x0 + plane.shape[1]
    model_y1 = model_y0 + plane.shape[0]
    global_x1 = global_x0 + width
    global_y1 = global_y0 + height
    ox0 = max(global_x0, model_x0)
    ox1 = min(global_x1, model_x1)
    oy0 = max(global_y0, model_y0)
    oy1 = min(global_y1, model_y1)
    if ox1 <= ox0 or oy1 <= oy0:
        return stamp
    sx0 = ox0 - model_x0
    sx1 = ox1 - model_x0
    sy0 = oy0 - model_y0
    sy1 = oy1 - model_y0
    dx0 = ox0 - global_x0
    dx1 = ox1 - global_x0
    dy0 = oy0 - global_y0
    dy1 = oy1 - global_y0
    stamp[dy0:dy1, dx0:dx1] = plane[sy0:sy1, sx0:sx1]
    return stamp


def _matched_prediction(
    *,
    ref_x: float,
    ref_y: float,
    pred: Table,
    pred_mask: np.ndarray,
    radius_pix: float,
) -> tuple[int | None, float]:
    px = np.asarray(pred["deblend_peak_center_x"], dtype=float)
    py = np.asarray(pred["deblend_peak_center_y"], dtype=float)
    d2 = (px - ref_x) ** 2 + (py - ref_y) ** 2
    d2[~pred_mask] = np.inf
    if not np.any(np.isfinite(d2)):
        return None, np.inf
    index = int(np.argmin(d2))
    distance = float(np.sqrt(d2[index]))
    if distance > radius_pix:
        return None, distance
    return index, distance


def _table_hdu(table: Table, name: str) -> fits.BinTableHDU:
    hdu = fits.table_to_hdu(table)
    hdu.name = name
    return hdu


def _write_footprint_bbox_fits(
    *,
    path: Path,
    hdr: fits.Header,
    footprint: FootprintData | None,
    model: SourceModel,
    bands: list[str],
    band_index: dict[str, int],
    band_images: dict[str, BandImage],
    reference_band: BandImage,
    row_table: Table,
) -> bool:
    bounds = _footprint_bbox_bounds(footprint, reference_band)
    if bounds is None:
        return False
    global_x0, global_y0, global_x1, global_y1 = bounds
    width = global_x1 - global_x0
    height = global_y1 - global_y0
    mask = _footprint_region_mask(
        footprint,
        global_x0=global_x0,
        global_y0=global_y0,
        width=width,
        height=height,
    )
    if not np.any(mask):
        return False

    box_hdr = hdr.copy()
    box_hdr["BOXGX0"] = (int(global_x0), "footprint bbox global min x")
    box_hdr["BOXGY0"] = (int(global_y0), "footprint bbox global min y")
    box_hdr["BOXGX1"] = (int(global_x1 - 1), "footprint bbox global max x")
    box_hdr["BOXGY1"] = (int(global_y1 - 1), "footprint bbox global max y")
    box_hdr["BOXSIZE"] = (f"{width}x{height}", "footprint bbox array size")
    box_hdr["LTV1"] = (-float(global_x0), "LSST image origin x offset")
    box_hdr["LTV2"] = (-float(global_y0), "LSST image origin y offset")

    primary_label = "i" if "i" in band_images else reference_band.label
    primary_image = None
    if primary_label in band_images:
        image = band_images[primary_label]
        lx0 = global_x0 - image.origin_x
        ly0 = global_y0 - image.origin_y
        primary_image = _masked_plane(image.data[ly0 : ly0 + height, lx0 : lx0 + width], mask)
    if primary_image is None:
        primary_image = mask.astype(np.float32)

    hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [
        fits.PrimaryHDU(data=primary_image.astype(np.float32, copy=False), header=box_hdr),
        fits.ImageHDU(data=mask.astype(np.uint8), name="FOOTPRINT_MASK"),
    ]

    for label in bands:
        if label in band_images:
            image = band_images[label]
            lx0 = global_x0 - image.origin_x
            ly0 = global_y0 - image.origin_y
            original = image.data[ly0 : ly0 + height, lx0 : lx0 + width].astype(np.float32, copy=True)
            hdus.append(fits.ImageHDU(data=_masked_plane(original, mask), name=f"FPBOX_ORIG_{label.upper()}"))
        model_plane = _insert_model_region(
            model,
            band_index=band_index[label],
            global_x0=global_x0,
            global_y0=global_y0,
            width=width,
            height=height,
        )
        hdus.append(fits.ImageHDU(data=_masked_plane(model_plane, mask), name=f"FPBOX_MODEL_{label.upper()}"))

    hdus.append(_table_hdu(row_table, "SUMMARY"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True, output_verify="silentfix")
    return True


def _write_run_outputs(args: argparse.Namespace, run_dir: Path, target_rows: list[int], ref: Table) -> None:
    run_dir = run_dir.expanduser().resolve()
    manifest = _read_manifest(run_dir)
    band_images = _load_band_images(run_dir, manifest)
    reference_band = band_images.get("i") or next(iter(band_images.values()))
    model_data = _load_model_data(run_dir, manifest)
    bands = _model_bands(model_data)
    band_index = {label: index for index, label in enumerate(bands)}

    source_by_id = {}
    source_parent_by_id = {}
    for parent_id, blend in model_data.blends.items():
        for source_id, source in blend.sources.items():
            source_by_id[int(source_id)] = source
            source_parent_by_id[int(source_id)] = int(parent_id)

    pred_path_text = manifest.get("deblend", {}).get("deblended_catalog_fits")
    if pred_path_text is None:
        pred_path = run_dir / "deblend" / "deepCoadd_deblendedFlux.fits"
    else:
        pred_path = _resolve_path(pred_path_text, base=_repo_root_from_run(run_dir))
        if not pred_path.exists():
            pred_path = run_dir / "deblend" / "deepCoadd_deblendedFlux.fits"
    pred = Table.read(pred_path, hdu=1)
    pred_mask = _prediction_mask(pred)
    pred_ids = np.asarray(pred["id"], dtype=np.int64)
    footprints_by_id = _load_prediction_footprints(pred_path)

    run_output = args.output_dir.expanduser() / run_dir.name
    fits_dir = run_output / "fits"
    footprint_fits_dir = run_output / "footprint_fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    footprint_fits_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    fn_rows = []
    radius_pix = float(args.radius_arcsec) / float(args.pixel_scale)

    for seq, ref_row in enumerate(target_rows, start=1):
        ref_x = float(ref[args.ref_x][ref_row])
        ref_y = float(ref[args.ref_y][ref_row])
        ref_local_x = ref_x - reference_band.origin_x
        ref_local_y = ref_y - reference_band.origin_y
        ref_id = int(ref["id"][ref_row]) if "id" in ref.colnames else int(ref_row)
        ref_flux = float(ref[args.ref_flux_col][ref_row]) if args.ref_flux_col in ref.colnames else np.nan
        ref_mag = _flux_to_mag(ref_flux, args.ref_mag_zero_point)

        pred_index, distance_pix = _matched_prediction(
            ref_x=ref_x,
            ref_y=ref_y,
            pred=pred,
            pred_mask=pred_mask,
            radius_pix=radius_pix,
        )
        if pred_index is None:
            fn_rows.append(
                {
                    "seq": seq,
                    "ref_row": ref_row,
                    "ref_id": ref_id,
                    "ref_x": ref_x,
                    "ref_y": ref_y,
                    "ref_mag": ref_mag,
                    "nearest_distance_pix": distance_pix,
                    "nearest_distance_arcsec": distance_pix * float(args.pixel_scale)
                    if np.isfinite(distance_pix)
                    else np.nan,
                    "reason": "no_prediction_within_radius",
                }
            )
            continue

        pred_id = int(pred_ids[pred_index])
        if pred_id not in source_by_id:
            fn_rows.append(
                {
                    "seq": seq,
                    "ref_row": ref_row,
                    "ref_id": ref_id,
                    "ref_x": ref_x,
                    "ref_y": ref_y,
                    "ref_mag": ref_mag,
                    "nearest_distance_pix": distance_pix,
                    "nearest_distance_arcsec": distance_pix * float(args.pixel_scale),
                    "reason": "prediction_has_no_scarlet_model",
                }
            )
            continue

        x0, y0, x1, y1 = _stamp_bounds(
            ref_local_x,
            ref_local_y,
            size=int(args.stamp_size),
            image_shape=reference_band.data.shape,
        )
        global_x0 = x0 + reference_band.origin_x
        global_y0 = y0 + reference_band.origin_y
        model = _source_model(source_by_id[pred_id], n_bands=len(bands))
        pred_parent = (
            int(pred["parent"][pred_index]) if "parent" in pred.colnames else source_parent_by_id.get(pred_id, -1)
        )
        child_footprint = footprints_by_id.get(pred_id)
        parent_footprint = footprints_by_id.get(pred_parent)
        child_fp_mask = _footprint_mask(
            child_footprint,
            global_x0=global_x0,
            global_y0=global_y0,
            size=int(args.stamp_size),
        )
        parent_fp_mask = _footprint_mask(
            parent_footprint,
            global_x0=global_x0,
            global_y0=global_y0,
            size=int(args.stamp_size),
        )
        if _mask_area(child_fp_mask) > 0:
            footprint = child_footprint
            fp_mask = child_fp_mask
            footprint_source = "child"
        elif _mask_area(parent_fp_mask) > 0:
            footprint = parent_footprint
            fp_mask = parent_fp_mask
            footprint_source = "parent"
        else:
            footprint = child_footprint or parent_footprint
            fp_mask = child_fp_mask
            footprint_source = "none"
        footprint_area_in_stamp = _mask_area(fp_mask)

        hdr = fits.Header()
        hdr["RUN"] = (run_dir.name, "pipeline run directory name")
        hdr["REFROW"] = (int(ref_row), "row in reference catalog")
        hdr["REFID"] = (ref_id, "reference source id")
        hdr["PREDID"] = (pred_id, "matched prediction source id")
        hdr["PREDPAR"] = (pred_parent, "matched prediction parent id")
        hdr["REFMAG"] = (ref_mag if np.isfinite(ref_mag) else np.nan, "reference magnitude")
        hdr["REFFLUX"] = (ref_flux if np.isfinite(ref_flux) else np.nan, f"reference {args.ref_flux_col}")
        hdr["DISTPIX"] = (float(distance_pix), "centroid match distance in pixels")
        hdr["DISTASEC"] = (float(distance_pix) * float(args.pixel_scale), "centroid match distance in arcsec")
        hdr["XGLOBAL"] = (ref_x, "reference x in global coadd coordinates")
        hdr["YGLOBAL"] = (ref_y, "reference y in global coadd coordinates")
        hdr["XLOCAL"] = (ref_local_x, "reference x in input image array coordinates")
        hdr["YLOCAL"] = (ref_local_y, "reference y in input image array coordinates")
        hdr["GX0"] = (int(global_x0), "stamp global min x")
        hdr["GY0"] = (int(global_y0), "stamp global min y")
        hdr["STAMPSZ"] = (int(args.stamp_size), "stamp size in pixels")
        hdr["FPAREA"] = (int(footprint.area) if footprint is not None else 0, "LSST footprint area")
        hdr["FPINSTMP"] = (footprint_area_in_stamp, "footprint pixels inside this stamp")
        hdr["FPSOURCE"] = (footprint_source, "selected footprint source")
        hdr["CFPINST"] = (_mask_area(child_fp_mask), "child footprint pixels inside stamp")
        hdr["PFPINST"] = (_mask_area(parent_fp_mask), "parent footprint pixels inside stamp")
        if footprint is not None:
            hdr["FPX0"] = (int(footprint.bbox_min_x), "footprint bbox min x")
            hdr["FPY0"] = (int(footprint.bbox_min_y), "footprint bbox min y")
            hdr["FPX1"] = (int(footprint.bbox_max_x), "footprint bbox max x")
            hdr["FPY1"] = (int(footprint.bbox_max_y), "footprint bbox max y")
        for label in bands:
            idx = band_index[label]
            value = float(model.spectrum[idx]) if idx < model.spectrum.size else np.nan
            hdr[f"SPEC_{label.upper()}"] = (value if np.isfinite(value) else np.nan, "scarlet spectrum")
            hdr[f"MAG_{label.upper()}"] = (_flux_to_mag(value, args.pred_mag_zero_point), "scarlet spectrum mag")

        hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = []
        primary_label = "i" if "i" in band_images else reference_band.label
        primary_image = band_images[primary_label].data[y0:y1, x0:x1].astype(np.float32, copy=True)
        hdus.append(fits.PrimaryHDU(data=primary_image, header=hdr))
        hdus.append(fits.ImageHDU(data=fp_mask.astype(np.uint8), name="FOOTPRINT_MASK"))
        hdus.append(fits.ImageHDU(data=child_fp_mask.astype(np.uint8), name="CHILD_FP_MASK"))
        hdus.append(fits.ImageHDU(data=parent_fp_mask.astype(np.uint8), name="PARENT_FP_MASK"))

        model_planes = {}
        original_planes = {}
        for label in bands:
            if label in band_images:
                original = band_images[label].data[y0:y1, x0:x1].astype(np.float32, copy=True)
                original_planes[label] = original
                hdus.append(fits.ImageHDU(data=original, name=f"ORIG_{label.upper()}"))
                hdus.append(fits.ImageHDU(data=_masked_plane(original, fp_mask), name=f"FP_ORIG_{label.upper()}"))
            stamp = _insert_model_stamp(
                model,
                band_index=band_index[label],
                global_x0=global_x0,
                global_y0=global_y0,
                size=int(args.stamp_size),
            )
            model_planes[label] = stamp
            hdus.append(fits.ImageHDU(data=stamp, name=f"MODEL_{label.upper()}"))
            hdus.append(fits.ImageHDU(data=_masked_plane(stamp, fp_mask), name=f"FP_MODEL_{label.upper()}"))

        if all(label in band_images for label in IRG_BANDS):
            irg = np.stack(
                [band_images[label].data[y0:y1, x0:x1].astype(np.float32, copy=True) for label in IRG_BANDS],
                axis=0,
            )
            hdus.append(fits.ImageHDU(data=irg, name="ORIG_IRG_CUBE"))
        if all(label in model_planes for label in IRG_BANDS):
            irg_model = np.stack([model_planes[label] for label in IRG_BANDS], axis=0)
            hdus.append(fits.ImageHDU(data=irg_model, name="MODEL_IRG_CUBE"))

        mask_source = next((plane for plane in model_planes.values() if np.any(plane > 0)), None)
        if mask_source is not None:
            hdus.append(fits.ImageHDU(data=(mask_source > 0).astype(np.uint8), name="MODEL_MASK"))

        row_table = Table()
        row_table["run"] = [run_dir.name]
        row_table["ref_row"] = [int(ref_row)]
        row_table["ref_id"] = [ref_id]
        row_table["pred_id"] = [pred_id]
        row_table["pred_parent"] = [pred_parent]
        row_table["distance_pix"] = [float(distance_pix)]
        row_table["distance_arcsec"] = [float(distance_pix) * float(args.pixel_scale)]
        row_table["ref_flux"] = [ref_flux]
        row_table["ref_mag"] = [ref_mag]
        row_table["footprint_area"] = [int(footprint.area) if footprint is not None else 0]
        row_table["footprint_area_in_stamp"] = [footprint_area_in_stamp]
        row_table["footprint_source"] = [footprint_source]
        row_table["child_footprint_area_in_stamp"] = [_mask_area(child_fp_mask)]
        row_table["parent_footprint_area_in_stamp"] = [_mask_area(parent_fp_mask)]
        for label in bands:
            idx = band_index[label]
            flux_value = float(model.spectrum[idx]) if idx < model.spectrum.size else np.nan
            model_fp_sum = float(np.nansum(model_planes[label][fp_mask])) if footprint_area_in_stamp > 0 else np.nan
            row_table[f"spectrum_{label}"] = [flux_value]
            row_table[f"mag_{label}"] = [_flux_to_mag(flux_value, args.pred_mag_zero_point)]
            row_table[f"model_sum_{label}"] = [float(np.nansum(model_planes[label]))]
            row_table[f"footprint_model_sum_{label}"] = [model_fp_sum]
            row_table[f"footprint_model_mag_{label}"] = [_flux_to_mag(model_fp_sum, args.pred_mag_zero_point)]
            if label in original_planes:
                orig_fp_sum = float(np.nansum(original_planes[label][fp_mask])) if footprint_area_in_stamp > 0 else np.nan
                row_table[f"footprint_orig_sum_{label}"] = [orig_fp_sum]
                row_table[f"footprint_orig_mag_{label}"] = [_flux_to_mag(orig_fp_sum, args.pred_mag_zero_point)]
        for name in ("deblend_scarletFlux", "deblend_peak_instFlux"):
            if name in pred.colnames:
                row_table[name] = [_finite_table_float(pred[name][pred_index])]
        hdus.append(_table_hdu(row_table, "SUMMARY"))
        hdus.append(_table_hdu(ref[ref_row : ref_row + 1], "REF_ROW"))
        hdus.append(_table_hdu(pred[pred_index : pred_index + 1], "PRED_ROW"))

        out_path = fits_dir / f"source_{seq:03d}_refrow{ref_row:04d}_ref{ref_id}_pred{pred_id}.fits"
        fits.HDUList(hdus).writeto(out_path, overwrite=True, output_verify="silentfix")

        summary_row = {name: row_table[name][0] for name in row_table.colnames}
        summary_row["fits"] = str(out_path)
        footprint_out_path = (
            footprint_fits_dir / f"source_{seq:03d}_refrow{ref_row:04d}_ref{ref_id}_pred{pred_id}_footprint.fits"
        )
        wrote_footprint = _write_footprint_bbox_fits(
            path=footprint_out_path,
            hdr=hdr,
            footprint=footprint,
            model=model,
            bands=bands,
            band_index=band_index,
            band_images=band_images,
            reference_band=reference_band,
            row_table=row_table,
        )
        summary_row["footprint_fits"] = str(footprint_out_path) if wrote_footprint else ""
        summary_rows.append(summary_row)

    summary_path = run_output / "deblend_source_summary.csv"
    fn_path = run_output / "false_negatives.csv"
    if summary_rows:
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        summary_path.write_text("")
    if fn_rows:
        with fn_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FN_FIELDS)
            writer.writeheader()
            writer.writerows(fn_rows)
    else:
        with fn_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FN_FIELDS)
            writer.writeheader()

    print(
        f"{run_dir.name}: matched={len(summary_rows)} false_negative={len(fn_rows)} "
        f"fits_dir={fits_dir} summary={summary_path} fn={fn_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=Path, help="Run output directory; may be repeated.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-ids", default=None, help="Comma/space-separated reference source ids.")
    parser.add_argument("--rows", default=None, help="Comma/space-separated reference table row indices.")
    parser.add_argument("--sources-csv", type=Path, default=None, help="CSV with row_index/gt_row and/or source_id/gt_id.")
    parser.add_argument("--ref-x", default=DEFAULT_REF_X)
    parser.add_argument("--ref-y", default=DEFAULT_REF_Y)
    parser.add_argument("--ref-flux-col", default=DEFAULT_REF_FLUX)
    parser.add_argument("--ref-mag-zero-point", type=float, default=AB_NJY_ZEROPOINT)
    parser.add_argument("--pred-mag-zero-point", type=float, default=27.0)
    parser.add_argument("--radius-arcsec", type=float, default=DEFAULT_MATCH_RADIUS_ARCSEC)
    parser.add_argument("--pixel-scale", type=float, default=DEFAULT_PIXEL_SCALE)
    parser.add_argument("--stamp-size", type=int, default=64)
    args = parser.parse_args()

    ref_path = args.reference.expanduser()
    ref = Table.read(ref_path, hdu=1)
    target_rows = _target_indices(args, ref)

    for run_dir in args.run:
        _write_run_outputs(args, run_dir, target_rows, ref)


if __name__ == "__main__":
    main()
