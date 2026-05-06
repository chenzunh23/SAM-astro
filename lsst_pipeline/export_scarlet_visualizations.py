from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval


def _load_catalog(path: Path):
    import lsst.afw.detection  # noqa: F401
    import lsst.afw.table as afwTable

    return afwTable.SourceCatalog.readFits(str(path))


def _safe_name(value: str) -> str:
    value = value.strip() or "root"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower() or "item"


def _read_exposure_image(path: Path):
    import lsst.afw.image as afwImage

    exposure = afwImage.ExposureF(str(path))
    bbox = exposure.getBBox()
    return exposure, exposure.image.array.astype(np.float32), int(bbox.getMinX()), int(bbox.getMinY())


def _is_sky_source(record) -> bool:
    try:
        return bool(record["merge_footprint_sky"])
    except Exception:
        return False


def _asinh_scale(image: np.ndarray, q: float = 8.0) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.5])
    if not np.isfinite(hi - lo) or hi <= lo:
        hi = lo + 1.0
    x = np.clip((image - lo) / (hi - lo), 0, None)
    y = np.arcsinh(q * x) / np.arcsinh(q)
    return np.clip(np.nan_to_num(y), 0, 1).astype(np.float32)


def _zscale_image(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    try:
        vmin, vmax = ZScaleInterval().get_limits(finite)
    except Exception:
        vmin, vmax = np.percentile(finite, [0.5, 99.5])
    if not np.isfinite(vmax - vmin) or vmax <= vmin:
        vmax = vmin + 1.0
    scaled = (image - vmin) / (vmax - vmin)
    return np.clip(np.nan_to_num(scaled), 0, 1).astype(np.float32)


def _make_rgb(band_images: dict[str, np.ndarray]) -> np.ndarray:
    # Display order is RGB = i,r,g for astronomy-style color.
    labels = list(band_images)
    g = band_images.get("g", band_images[labels[0]])
    r = band_images.get("r", band_images[labels[min(1, len(labels) - 1)]])
    i = band_images.get("i", band_images[labels[min(2, len(labels) - 1)]])
    return np.dstack([_asinh_scale(i), _asinh_scale(r), _asinh_scale(g)])


def _make_reference_display(band_images: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    for band in ("i", "r", "g"):
        if band in band_images:
            return _asinh_scale(band_images[band]), band
    band = next(iter(band_images))
    return _asinh_scale(band_images[band]), band


def _paint_catalog(catalog, shape: tuple[int, int], min_x: int, min_y: int):
    all_mask = np.zeros(shape, dtype=np.int16)
    parent_mask = np.zeros(shape, dtype=np.int16)
    child_mask = np.zeros(shape, dtype=np.int16)
    peaks = np.zeros(shape, dtype=np.int16)
    for record in catalog:
        if _is_sky_source(record):
            continue
        footprint = record.getFootprint()
        if footprint is None:
            continue
        try:
            is_child = int(record["parent"]) != 0
        except Exception:
            is_child = False
        spans = footprint.spans
        for span in spans:
            y = int(span.getY()) - min_y
            x0 = int(span.getX0()) - min_x
            x1 = int(span.getX1()) - min_x
            if 0 <= y < shape[0]:
                xs0 = max(0, x0)
                xs1 = min(shape[1] - 1, x1)
                if xs0 <= xs1:
                    all_mask[y, xs0 : xs1 + 1] = 1
                    (child_mask if is_child else parent_mask)[y, xs0 : xs1 + 1] = 1
        for peak in footprint.getPeaks():
            x = int(round(peak.getFx())) - min_x
            y = int(round(peak.getFy())) - min_y
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                peaks[y, x] = 1
    return all_mask, parent_mask, child_mask, peaks


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(1.0, 0.1, 0.1), alpha: float = 0.45):
    out = rgb.copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.asarray(color)
    return np.clip(out, 0, 1)


def _paint_label_catalog(catalog, shape: tuple[int, int], min_x: int, min_y: int) -> np.ndarray:
    label_map = np.zeros(shape, dtype=np.int32)
    label = 0
    for record in catalog:
        if _is_sky_source(record):
            continue
        footprint = record.getFootprint()
        if footprint is None:
            continue
        label += 1
        for span in footprint.spans:
            y = int(span.getY()) - min_y
            x0 = int(span.getX0()) - min_x
            x1 = int(span.getX1()) - min_x
            if 0 <= y < shape[0]:
                xs0 = max(0, x0)
                xs1 = min(shape[1] - 1, x1)
                if xs0 <= xs1:
                    label_map[y, xs0 : xs1 + 1] = label
    return label_map


def _catalog_mask_and_bboxes(catalog, shape: tuple[int, int], min_x: int, min_y: int):
    mask = np.zeros(shape, dtype=np.int16)
    bboxes: list[tuple[int, int, int, int]] = []
    peaks: list[tuple[float, float]] = []
    for record in catalog:
        if _is_sky_source(record):
            continue
        footprint = record.getFootprint()
        if footprint is None:
            continue
        bbox = footprint.getBBox()
        bx0 = int(bbox.getMinX()) - min_x
        by0 = int(bbox.getMinY()) - min_y
        bx1 = int(bbox.getMaxX()) - min_x
        by1 = int(bbox.getMaxY()) - min_y
        if bx1 >= 0 and by1 >= 0 and bx0 < shape[1] and by0 < shape[0]:
            bboxes.append((max(0, bx0), max(0, by0), min(shape[1] - 1, bx1), min(shape[0] - 1, by1)))
        for span in footprint.spans:
            y = int(span.getY()) - min_y
            x0 = int(span.getX0()) - min_x
            x1 = int(span.getX1()) - min_x
            if 0 <= y < shape[0]:
                xs0 = max(0, x0)
                xs1 = min(shape[1] - 1, x1)
                if xs0 <= xs1:
                    mask[y, xs0 : xs1 + 1] = 1
        for peak in footprint.getPeaks():
            x = float(peak.getFx()) - min_x
            y = float(peak.getFy()) - min_y
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                peaks.append((x, y))
    return mask, bboxes, peaks


def _overlay_labels(rgb: np.ndarray, label_map: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    out = rgb.copy()
    labels = np.unique(label_map)
    labels = labels[labels > 0]
    if labels.size == 0:
        return out
    cmap = plt.get_cmap("tab20", 20)
    for label in labels:
        color = np.asarray(cmap((int(label) - 1) % 20)[:3])
        mask = label_map == label
        out[mask] = (1 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0, 1)


def _record_by_id(catalog) -> dict[int, Any]:
    return {int(record.getId()): record for record in catalog}


def _leaf_records(catalog):
    child_counts: dict[int, int] = {}
    for record in catalog:
        if _is_sky_source(record):
            continue
        parent = int(record["parent"])
        if parent != 0:
            child_counts[parent] = child_counts.get(parent, 0) + 1
    leaves = []
    for record in catalog:
        if _is_sky_source(record):
            continue
        rec_id = int(record.getId())
        parent = int(record["parent"])
        if parent != 0 or child_counts.get(rec_id, 0) == 0:
            leaves.append(record)
    return leaves, child_counts


def _record_n_child(record, child_counts: dict[int, int], schema) -> int:
    if "deblend_nChild" in schema:
        try:
            return int(record["deblend_nChild"])
        except Exception:
            pass
    return int(child_counts.get(int(record.getId()), 0))


def _footprint_xy(record) -> tuple[float, float]:
    footprint = record.getFootprint()
    if footprint is None:
        return np.nan, np.nan
    peaks = footprint.getPeaks()
    if len(peaks):
        peak = peaks[0]
        return float(peak.getFx()), float(peak.getFy())
    bbox = footprint.getBBox()
    return 0.5 * (float(bbox.getMinX()) + float(bbox.getMaxX())), 0.5 * (float(bbox.getMinY()) + float(bbox.getMaxY()))


def _peak_xy(record, schema, *, prefer_deblend: bool = True) -> tuple[float, float]:
    if "deblend_peak_center_x" in schema and "deblend_peak_center_y" in schema:
        x = float(record["deblend_peak_center_x"])
        y = float(record["deblend_peak_center_y"])
        if prefer_deblend and np.isfinite(x) and np.isfinite(y):
            return x, y
    return _footprint_xy(record)


def _bounded_peak_xy(record, schema, min_x: int, min_y: int, shape: tuple[int, int]) -> tuple[float, float]:
    for prefer_deblend in (True, False):
        x, y = _peak_xy(record, schema, prefer_deblend=prefer_deblend)
        if np.isfinite(x) and np.isfinite(y) and min_x <= x < min_x + shape[1] and min_y <= y < min_y + shape[0]:
            return x, y
    return np.nan, np.nan


def _science_leaf_records(catalog, min_x: int, min_y: int, shape: tuple[int, int]):
    _, child_counts = _leaf_records(catalog)
    science = []
    for record in catalog:
        if _is_sky_source(record):
            continue
        if _record_n_child(record, child_counts, catalog.schema) != 0:
            continue
        x, y = _bounded_peak_xy(record, catalog.schema, min_x, min_y, shape)
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if "deblend_modelType" in catalog.schema and str(record["deblend_modelType"]).strip() == "":
            continue
        science.append(record)
    return science, child_counts


def _component_model(component) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    spectrum = np.asarray(component.spectrum, dtype=np.float32)
    morph = np.asarray(component.morph, dtype=np.float32)
    cube = spectrum[:, None, None] * morph[None, :, :]
    return cube, spectrum, tuple(component.origin)


def _source_model(source, n_bands: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    pieces = []
    spectra = []
    for component in source.factorized_components:
        cube, spectrum, origin = _component_model(component)
        pieces.append((cube, origin))
        spectra.append(spectrum)
    if not pieces:
        return np.zeros((n_bands, 1, 1), dtype=np.float32), np.zeros(n_bands, dtype=np.float32), (0, 0)

    y0 = min(origin[0] for _, origin in pieces)
    x0 = min(origin[1] for _, origin in pieces)
    y1 = max(origin[0] + cube.shape[1] for cube, origin in pieces)
    x1 = max(origin[1] + cube.shape[2] for cube, origin in pieces)
    full = np.zeros((n_bands, y1 - y0, x1 - x0), dtype=np.float32)
    for cube, origin in pieces:
        yy = origin[0] - y0
        xx = origin[1] - x0
        full[:, yy : yy + cube.shape[1], xx : xx + cube.shape[2]] += cube
    return full, np.sum(np.vstack(spectra), axis=0), (y0, x0)


def _model_rgb(cube: np.ndarray) -> np.ndarray:
    if cube.shape[0] >= 3:
        return np.dstack([_asinh_scale(cube[2]), _asinh_scale(cube[1]), _asinh_scale(cube[0])])
    scaled = _asinh_scale(np.sum(cube, axis=0))
    return np.dstack([scaled, scaled, scaled])


def export_source_panels(
    model_data,
    deblend_catalog,
    outdir: Path,
    bands: list[str],
    max_sources: int | None,
    min_x: int,
    min_y: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    records = _record_by_id(deblend_catalog)
    rows = []
    count = 0
    for parent_id, blend in model_data.blends.items():
        for source_id, source in blend.sources.items():
            if max_sources is not None and count >= max_sources:
                break
            cube, spectrum, origin = _source_model(source, len(bands))
            rec = records.get(int(source_id))
            parent_rec = records.get(int(parent_id))
            if (rec is not None and _is_sky_source(rec)) or (parent_rec is not None and _is_sky_source(parent_rec)):
                continue
            parent = int(rec["parent"]) if rec is not None else int(parent_id)
            origin_y, origin_x = int(origin[0]), int(origin[1])
            width, height = int(cube.shape[2]), int(cube.shape[1])
            if rec is not None and "deblend_peak_center_x" in deblend_catalog.schema:
                peak_x = int(rec["deblend_peak_center_x"])
                peak_y = int(rec["deblend_peak_center_y"])
            else:
                peak_y, peak_x = tuple(source.factorized_components[0].peak) if source.factorized_components else (origin_y, origin_x)
            peak_local_x = peak_x - min_x
            peak_local_y = peak_y - min_y
            peak_model_x = peak_x - origin_x
            peak_model_y = peak_y - origin_y
            extent = [origin_x - min_x, origin_x - min_x + width, origin_y - min_y, origin_y - min_y + height]

            fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
            axes[0].imshow(_model_rgb(cube), origin="lower", extent=extent, aspect="equal")
            if 0 <= peak_model_x < width and 0 <= peak_model_y < height:
                axes[0].scatter([peak_local_x], [peak_local_y], s=48, marker="+", c="cyan", linewidths=1.4)
            axes[0].set_title(
                f"source {source_id}\n"
                f"peak local=({peak_local_x},{peak_local_y}) global=({peak_x},{peak_y})",
                fontsize=9,
            )
            axes[0].set_xlabel("local x pixel")
            axes[0].set_ylabel("local y pixel")
            axes[0].tick_params(labelsize=8)
            axes[0].grid(color="white", alpha=0.25, linewidth=0.5)
            axes[0].text(
                0.02,
                0.02,
                f"bbox {width}x{height}\norigin local=({origin_x - min_x},{origin_y - min_y})",
                transform=axes[0].transAxes,
                fontsize=7,
                color="white",
                va="bottom",
                bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2},
            )
            axes[1].plot(bands, spectrum, marker="o", linewidth=1.8)
            axes[1].set_title("spectrum")
            axes[1].set_xlabel("band")
            axes[1].set_ylabel("model flux")
            axes[1].grid(True, alpha=0.25)
            fig.savefig(outdir / f"source_{int(source_id):08d}_panel.png", dpi=160)
            plt.close(fig)

            rows.append([
                int(source_id),
                parent,
                *[float(v) for v in spectrum],
                origin_x,
                origin_y,
                width,
                height,
                peak_x,
                peak_y,
                peak_local_x,
                peak_local_y,
            ])
            count += 1
        if max_sources is not None and count >= max_sources:
            break

    with (outdir / "source_spectra.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source_id",
            "parent_id",
            *[f"spectrum_{b}" for b in bands],
            "origin_x",
            "origin_y",
            "width",
            "height",
            "peak_x",
            "peak_y",
            "peak_local_x",
            "peak_local_y",
        ])
        writer.writerows(rows)


def _plot_source_markers(
    ax,
    records,
    schema,
    min_x: int,
    min_y: int,
    shape: tuple[int, int],
    *,
    annotate: bool = False,
) -> None:
    isolated_x = []
    isolated_y = []
    child_x = []
    child_y = []
    for record in records:
        if _is_sky_source(record):
            continue
        x, y = _bounded_peak_xy(record, schema, min_x, min_y, shape)
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        local_x = x - min_x
        local_y = y - min_y
        if int(record["parent"]) == 0:
            isolated_x.append(local_x)
            isolated_y.append(local_y)
        else:
            child_x.append(local_x)
            child_y.append(local_y)
        if annotate:
            ax.text(local_x + 2, local_y + 2, str(int(record.getId())), color="#f5b642", fontsize=6)
    if isolated_x:
        ax.scatter(isolated_x, isolated_y, s=20, facecolors="none", edgecolors="#55d8ff", linewidths=0.8)
    if child_x:
        ax.scatter(child_x, child_y, s=24, c="#f5b642", marker="+", linewidths=0.9)


def export_visual_checks(
    deblend_catalog,
    rgb: np.ndarray,
    reference_image: np.ndarray,
    reference_band: str,
    min_x: int,
    min_y: int,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    leaves, child_counts = _science_leaf_records(deblend_catalog, min_x, min_y, rgb.shape[:2])
    label_map = _paint_label_catalog(leaves, rgb.shape[:2], min_x, min_y)

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(reference_image, origin="lower", cmap="gray", vmin=0, vmax=1)
    _plot_source_markers(ax, leaves, deblend_catalog.schema, min_x, min_y, rgb.shape[:2])
    parent_anchor_x = []
    parent_anchor_y = []
    for record in deblend_catalog:
        if _is_sky_source(record):
            continue
        if int(record["parent"]) != 0 or _record_n_child(record, child_counts, deblend_catalog.schema) <= 1:
            continue
        x, y = _peak_xy(record, deblend_catalog.schema, prefer_deblend=False)
        if np.isfinite(x) and np.isfinite(y) and min_x <= x < min_x + rgb.shape[1] and min_y <= y < min_y + rgb.shape[0]:
            parent_anchor_x.append(x - min_x)
            parent_anchor_y.append(y - min_y)
    if parent_anchor_x:
        ax.scatter(parent_anchor_x, parent_anchor_y, s=32, c="#ff4f5e", marker="x", linewidths=0.9)
    ax.set_title(f"HSC-{reference_band.upper()} cutout: science-ready leaf sources (deblend_nChild == 0)")
    ax.set_xlabel("local x")
    ax.set_ylabel("local y")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#55d8ff", label="leaf isolated"),
        Line2D([0], [0], marker="+", color="#f5b642", linestyle="none", label="leaf child"),
        Line2D([0], [0], marker="x", color="#ff4f5e", linestyle="none", label="blend parent anchor"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.8)
    fig.savefig(outdir / "science_leaf_map.png", dpi=144)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(_overlay_labels(rgb, label_map), origin="lower")
    _plot_source_markers(ax, leaves, deblend_catalog.schema, min_x, min_y, rgb.shape[:2])
    ax.set_title("science leaf footprint labels")
    ax.set_xlabel("local x")
    ax.set_ylabel("local y")
    fig.savefig(outdir / "science_leaf_footprint_map.png", dpi=144)
    plt.close(fig)

    complex_parents = [
        record
        for record in deblend_catalog
        if not _is_sky_source(record)
        and int(record["parent"]) == 0
        and _record_n_child(record, child_counts, deblend_catalog.schema) > 1
    ]
    complex_parents = sorted(
        complex_parents,
        key=lambda record: _record_n_child(record, child_counts, deblend_catalog.schema),
        reverse=True,
    )[:6]
    children_by_parent: dict[int, list[Any]] = {}
    for record in leaves:
        parent = int(record["parent"])
        if parent != 0:
            children_by_parent.setdefault(parent, []).append(record)
    ncols = 3
    nrows = max(1, math.ceil(len(complex_parents) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 4.2), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.set_axis_off()
    for ax, parent in zip(axes, complex_parents):
        footprint = parent.getFootprint()
        bbox = footprint.getBBox()
        x0 = max(0, int(bbox.getMinX()) - min_x - 8)
        y0 = max(0, int(bbox.getMinY()) - min_y - 8)
        x1 = min(reference_image.shape[1], int(bbox.getMaxX()) - min_x + 9)
        y1 = min(reference_image.shape[0], int(bbox.getMaxY()) - min_y + 9)
        crop = reference_image[y0:y1, x0:x1]
        ax.imshow(crop, origin="lower", cmap="gray", vmin=0, vmax=1, extent=[x0, x1, y0, y1])
        children = children_by_parent.get(int(parent.getId()), [])
        _plot_source_markers(ax, children, deblend_catalog.schema, min_x, min_y, rgb.shape[:2], annotate=True)
        n_child = _record_n_child(parent, child_counts, deblend_catalog.schema)
        ax.set_title(f"parent {int(parent.getId())}: leaf children={len(children)} nChild={n_child}", fontsize=9)
    fig.savefig(outdir / "complex_family_leaf_gallery.png", dpi=144)
    plt.close(fig)

    with (outdir / "science_leaf_sources.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_id", "parent_id", "local_x", "local_y", "is_child_leaf", "deblend_scarletFlux", "deblend_blendedness"])
        for record in leaves:
            peak_x, peak_y = _bounded_peak_xy(record, deblend_catalog.schema, min_x, min_y, rgb.shape[:2])
            x = peak_x - min_x if np.isfinite(peak_x) else np.nan
            y = peak_y - min_y if np.isfinite(peak_y) else np.nan
            flux = record["deblend_scarletFlux"] if "deblend_scarletFlux" in deblend_catalog.schema else ""
            blendedness = record["deblend_blendedness"] if "deblend_blendedness" in deblend_catalog.schema else ""
            writer.writerow([int(record.getId()), int(record["parent"]), x, y, int(record["parent"]) != 0, flux, blendedness])


def export_merge_regions(merge_catalog, rgb: np.ndarray, min_x: int, min_y: int, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for record in merge_catalog:
        if _is_sky_source(record):
            continue
        parent_id = int(record.getId())
        single_mask, _, _, peaks = _paint_catalog([record], rgb.shape[:2], min_x, min_y)
        fits.writeto(outdir / f"merge_parent_{parent_id:08d}_mask.fits", single_mask.astype(np.int16), overwrite=True)
        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        ax.imshow(np.flipud(_overlay_mask(rgb, single_mask, color=(1.0, 0.2, 0.1), alpha=0.45)))
        py, px = np.nonzero(peaks)
        if px.size:
            ax.scatter(px, rgb.shape[0] - 1 - py, s=18, c="cyan", marker="+", linewidths=1.0)
        ax.set_title(f"merge parent {parent_id}")
        ax.set_axis_off()
        fig.savefig(outdir / f"merge_parent_{parent_id:08d}.png", dpi=120)
        plt.close(fig)
        summary_rows.append([parent_id, int(np.count_nonzero(single_mask)), int(np.count_nonzero(peaks))])
    with (outdir / "merge_regions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parent_id", "mask_area", "n_peaks"])
        writer.writerows(summary_rows)


def _overlay_mask_on_gray(gray: np.ndarray, mask: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    out = np.dstack([gray, gray, gray]).astype(np.float32)
    if np.count_nonzero(mask) == 0:
        return out
    values = np.unique(mask)
    values = values[values != 0]
    if values.size <= 1:
        m = mask != 0
        color = np.asarray([1.0, 0.12, 0.05], dtype=np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * color
        return np.clip(out, 0, 1)

    cmap = plt.get_cmap("tab20", 20)
    for value in values:
        m = mask == value
        color = np.asarray(cmap((int(value) - 1) % 20)[:3], dtype=np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 1)


def _candidate_image_mask_hdus(path: Path, shape: tuple[int, int]):
    products = []
    with fits.open(path, memmap=False) as hdul:
        for index, hdu in enumerate(hdul):
            data = hdu.data
            if data is None:
                continue
            array = np.asarray(data)
            if array.ndim < 2:
                continue
            if array.ndim > 2:
                array = array.reshape((-1, array.shape[-2], array.shape[-1]))[0]
            if array.shape != shape:
                continue

            hdu_name = str(hdu.name or f"HDU{index}").upper()
            path_name = path.name.upper()
            looks_like_mask = (
                "MASK" in hdu_name
                or "MASK" in path_name
                or "LABELMAP" in path_name
                or "LABEL" in hdu_name
            )
            if not looks_like_mask:
                continue

            mask = np.nan_to_num(array)
            if np.issubdtype(mask.dtype, np.floating):
                mask = mask != 0
            products.append((hdu.name or f"HDU{index}", mask.astype(np.int32)))
    return products


def _save_zscale_overlay(
    outpath: Path,
    gray: np.ndarray,
    title: str,
    *,
    mask: np.ndarray | None = None,
    bboxes: list[tuple[int, int, int, int]] | None = None,
    peaks: list[tuple[float, float]] | None = None,
    note: str | None = None,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    if mask is not None and np.count_nonzero(mask):
        ax.imshow(_overlay_mask_on_gray(gray, mask), origin="lower")
    else:
        ax.imshow(gray, origin="lower", cmap="gray", vmin=0, vmax=1)

    if bboxes:
        for x0, y0, x1, y1 in bboxes:
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    max(1, x1 - x0 + 1),
                    max(1, y1 - y0 + 1),
                    fill=False,
                    edgecolor="#39d5ff",
                    linewidth=0.8,
                    alpha=0.85,
                )
            )
    if peaks:
        x = [p[0] for p in peaks]
        y = [p[1] for p in peaks]
        ax.scatter(x, y, s=14, c="#ffd34d", marker="+", linewidths=0.8)
    if note:
        ax.text(
            0.02,
            0.02,
            note,
            transform=ax.transAxes,
            color="white",
            fontsize=9,
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3},
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("local x")
    ax.set_ylabel("local y")
    fig.savefig(outpath, dpi=144)
    plt.close(fig)


def export_zscale_mask_overlays(root: Path, hsc_i_image: np.ndarray, min_x: int, min_y: int, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    gray = _zscale_image(hsc_i_image)
    shape = gray.shape
    fits_files = sorted(root.rglob("*.fits"))
    # Filter out files produced by the visualization (in "merge_regions/" directory) to avoid confusion.
    fits_files = [f for f in fits_files if not f.parts[-2] == "merge_regions"]
    summary_rows = []

    catalog_names = {"deepCoadd_det.fits", "deepCoadd_mergeDet.fits", "deepCoadd_deblendedFlux.fits"}
    for path in fits_files:
        rel = path.relative_to(root)
        base = _safe_name("__".join(rel.with_suffix("").parts))
        wrote = 0

        if path.name in catalog_names:
            try:
                catalog = _load_catalog(path)
                mask, bboxes, peaks = _catalog_mask_and_bboxes(catalog, shape, min_x, min_y)
                use_bboxes = bboxes if np.count_nonzero(mask) == 0 else []
                note = f"{rel}\nrecords={len(catalog)} mask_pixels={int(np.count_nonzero(mask))}"
                if np.count_nonzero(mask) == 0 and bboxes:
                    note += "\nno footprint spans; showing bbox"
                _save_zscale_overlay(
                    outdir / f"{base}.png",
                    gray,
                    str(rel),
                    mask=mask,
                    bboxes=use_bboxes,
                    peaks=peaks,
                    note=note,
                )
                summary_rows.append([str(rel), "catalog", int(np.count_nonzero(mask)), len(bboxes), len(peaks), f"{base}.png"])
                wrote += 1
            except Exception as exc:
                _save_zscale_overlay(
                    outdir / f"{base}.png",
                    gray,
                    str(rel),
                    note=f"{rel}\ncatalog read failed: {type(exc).__name__}: {exc}",
                )
                summary_rows.append([str(rel), "catalog_error", 0, 0, 0, f"{base}.png"])
                wrote += 1

        if wrote == 0:
            image_masks = []
            try:
                image_masks = _candidate_image_mask_hdus(path, shape)
            except Exception as exc:
                _save_zscale_overlay(
                    outdir / f"{base}.png",
                    gray,
                    str(rel),
                    note=f"{rel}\nFITS read failed: {type(exc).__name__}: {exc}",
                )
                summary_rows.append([str(rel), "fits_error", 0, 0, 0, f"{base}.png"])
                wrote += 1

            for hdu_name, mask in image_masks:
                suffix = _safe_name(str(hdu_name))
                outname = f"{base}__{suffix}.png" if len(image_masks) > 1 else f"{base}.png"
                _save_zscale_overlay(
                    outdir / outname,
                    gray,
                    f"{rel} [{hdu_name}]",
                    mask=mask,
                    note=f"{rel}\nHDU={hdu_name} mask_pixels={int(np.count_nonzero(mask))}",
                )
                summary_rows.append([str(rel), f"image_hdu:{hdu_name}", int(np.count_nonzero(mask)), 0, 0, outname])
                wrote += 1

        if wrote == 0:
            _save_zscale_overlay(
                outdir / f"{base}.png",
                gray,
                str(rel),
                note=f"{rel}\nno mask-like HDU or LSST footprint catalog found",
            )
            summary_rows.append([str(rel), "none", 0, 0, 0, f"{base}.png"])

    with (outdir / "zscale_mask_overlays.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fits_path", "overlay_type", "mask_pixels", "bbox_count", "peak_count", "png"])
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export scarlet/SAM/LSST visualizations from a pipeline output directory.")
    parser.add_argument("--input", type=Path, required=True, help="Pipeline output root, e.g. output/scarlet_sam_multipeak_test.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-source-panels", type=int, default=None, help="Limit source panel PNG count for quick tests.")
    args = parser.parse_args()

    root = args.input.resolve()
    outdir = args.output.resolve()
    deblend_catalog = _load_catalog(root / "deblend" / "deepCoadd_deblendedFlux.fits")
    merge_catalog = _load_catalog(root / "merge" / "deepCoadd_mergeDet.fits")
    with (root / "deblend" / "deepCoadd_scarletModelData.pickle").open("rb") as handle:
        model_data = pickle.load(handle)

    calexp_paths = sorted((root / "detect").glob("HSC-*/deepCoadd_calexp.fits"))
    band_images: dict[str, np.ndarray] = {}
    min_x = min_y = None
    for path in calexp_paths:
        exposure, image, x0, y0 = _read_exposure_image(path)
        band = exposure.getInfo().getFilter().bandLabel
        band_images[str(band)] = image
        min_x = x0 if min_x is None else min_x
        min_y = y0 if min_y is None else min_y
    if min_x is None or min_y is None:
        raise RuntimeError(f"No detect/*/deepCoadd_calexp.fits files found under {root}")

    bands = list(next(iter(model_data.blends.values())).bands) if model_data.blends else list(band_images)
    rgb = _make_rgb(band_images)
    reference_image, reference_band = _make_reference_display(band_images)
    export_source_panels(
        model_data,
        deblend_catalog,
        outdir / "source_panels",
        bands,
        args.max_source_panels,
        int(min_x),
        int(min_y),
    )
    export_visual_checks(
        deblend_catalog,
        rgb,
        reference_image,
        reference_band,
        int(min_x),
        int(min_y),
        outdir / "visual_check",
    )
    export_merge_regions(merge_catalog, rgb, int(min_x), int(min_y), outdir / "merge_regions")
    hsc_i_image = band_images.get("i", next(iter(band_images.values())))
    export_zscale_mask_overlays(root, hsc_i_image, int(min_x), int(min_y), outdir / "zscale_mask_overlays")
    print(f"Wrote visualizations under: {outdir}")


if __name__ == "__main__":
    main()
