from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Patch


@dataclass(frozen=True)
class MaskLayer:
    name: str
    path: Path
    data: np.ndarray


@dataclass(frozen=True)
class Roi:
    name: str
    center_x: float
    center_y: float
    anchor_parent: int | None
    x0: int
    y0: int
    x1: int
    y1: int


def _safe_name(value: str) -> str:
    value = value.strip() or "roi"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower() or "roi"


def _read_first_image(path: Path, *, preferred_names: tuple[str, ...] = ("IMAGE", "PRIMARY")) -> np.ndarray:
    data, _ = _read_first_image_with_header(path, preferred_names=preferred_names)
    return data


def _read_first_image_with_header(path: Path, *, preferred_names: tuple[str, ...] = ("IMAGE", "PRIMARY")) -> tuple[np.ndarray, dict]:
    with fits.open(path, memmap=False) as hdul:
        for name in preferred_names:
            for hdu in hdul:
                if str(hdu.name).upper() == name and hdu.data is not None:
                    data = np.asarray(hdu.data)
                    if data.ndim >= 2:
                        if data.ndim > 2:
                            data = data.reshape((-1, data.shape[-2], data.shape[-1]))[0]
                        return np.array(data, copy=True), dict(hdu.header)
        for hdu in hdul:
            if hdu.data is None:
                continue
            data = np.asarray(hdu.data)
            if data.ndim >= 2 and np.issubdtype(data.dtype, np.number):
                if data.ndim > 2:
                    data = data.reshape((-1, data.shape[-2], data.shape[-1]))[0]
                return np.array(data, copy=True), dict(hdu.header)
    raise RuntimeError(f"No image-like HDU found in {path}")


def _asinh_scale(image: np.ndarray, q: float = 8.0) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.5])
    if not np.isfinite(hi - lo) or hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((image - lo) / (hi - lo), 0, None)
    scaled = np.arcsinh(q * scaled) / np.arcsinh(q)
    return np.clip(np.nan_to_num(scaled), 0, 1).astype(np.float32)


def _zscale_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
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


def _header_origin(header: dict) -> tuple[int, int]:
    if "LTV1" in header and "LTV2" in header:
        return int(round(-float(header["LTV1"]))), int(round(-float(header["LTV2"])))
    return 0, 0


def _load_background(run_root: Path, background: str) -> tuple[np.ndarray, str, int, int]:
    calexp_paths = sorted((run_root / "detect").glob("HSC-*/deepCoadd_calexp.fits"))
    if not calexp_paths:
        raise RuntimeError(f"No detect/HSC-*/deepCoadd_calexp.fits files found under {run_root}")

    images: dict[str, np.ndarray] = {}
    origins: dict[str, tuple[int, int]] = {}
    for path in calexp_paths:
        band = path.parent.name.split("-")[-1].lower()
        image, header = _read_first_image_with_header(path, preferred_names=("IMAGE",))
        images[band] = image
        origins[band] = _header_origin(header)

    if background == "rgb" and len(images) >= 3:
        labels = list(images)
        g = images.get("g", images[labels[0]])
        r = images.get("r", images[labels[min(1, len(labels) - 1)]])
        i = images.get("i", images[labels[min(2, len(labels) - 1)]])
        origin_band = "i" if "i" in origins else labels[0]
        min_x, min_y = origins[origin_band]
        return np.dstack([_asinh_scale(i), _asinh_scale(r), _asinh_scale(g)]), "RGB=i,r,g", min_x, min_y

    band = background.lower()
    if band == "rgb":
        band = "i" if "i" in images else next(iter(images))
    if band not in images:
        available = ", ".join(sorted(images))
        raise RuntimeError(f"Background band {background!r} not found. Available bands: {available}")
    gray = _zscale_image(images[band])
    min_x, min_y = origins[band]
    return np.dstack([gray, gray, gray]), f"HSC-{band.upper()}", min_x, min_y


def _load_sam_layers(run_root: Path, sam_labelmaps: list[Path]) -> list[MaskLayer]:
    paths = sam_labelmaps or sorted((run_root / "sam").glob("*_sam_labelmap.fits"))
    if not paths:
        raise RuntimeError(f"No SAM labelmap FITS found under {run_root / 'sam'}")
    layers = []
    for path in paths:
        data = _read_first_image(path).astype(np.int32)
        name = path.stem.replace("_sam_labelmap", "")
        layers.append(MaskLayer(name=name, path=path, data=data))
    return layers


def _discover_merge_regions(run_root: Path, merge_regions: Path | None) -> Path | None:
    if merge_regions is not None:
        return merge_regions
    direct = run_root / "merge_regions"
    if direct.is_dir() and any(direct.glob("merge_parent_*_mask.fits")):
        return direct

    candidates = []
    candidates.extend(run_root.parent.glob(f"{run_root.name}*_vis*/merge_regions"))
    candidates.extend(run_root.parent.glob(f"{run_root.name}*/merge_regions"))
    valid = [
        candidate
        for candidate in candidates
        if candidate.is_dir() and any(candidate.glob("merge_parent_*_mask.fits"))
    ]
    if valid:
        return max(valid, key=lambda path: path.stat().st_mtime)
    return None


def _parent_id_from_path(path: Path) -> int:
    match = re.search(r"merge_parent_(\d+)_mask\.fits$", path.name)
    if not match:
        raise ValueError(f"Cannot parse parent id from {path.name}")
    return int(match.group(1))


def _load_lsst_layers(merge_regions: Path) -> list[MaskLayer]:
    paths = sorted(merge_regions.glob("merge_parent_*_mask.fits"))
    if not paths:
        raise RuntimeError(f"No merge_parent_*_mask.fits files found under {merge_regions}")
    layers = []
    for path in paths:
        parent_id = _parent_id_from_path(path)
        data = (_read_first_image(path) != 0).astype(np.int16)
        layers.append(MaskLayer(name=f"parent {parent_id}", path=path, data=data))
    return layers


def _layer_parent_id(layer: MaskLayer) -> int:
    try:
        return _parent_id_from_path(layer.path)
    except ValueError:
        match = re.search(r"parent\s+(\d+)", layer.name)
        if match:
            return int(match.group(1))
        raise


def _archive_rows_by_id(archive_index: np.ndarray, *, cat_archive: int, persistable: int | None = None) -> dict[int, tuple[int, int]]:
    rows = {}
    for row in archive_index:
        if int(row["cat.archive"]) != cat_archive:
            continue
        if persistable is not None and int(row["cat.persistable"]) != persistable:
            continue
        rows[int(row["id"])] = (int(row["row0"]), int(row["nrows"]))
    return rows


def _is_sky_source_row(row) -> bool:
    names = getattr(getattr(row, "dtype", None), "names", None)
    if names is None or "merge_footprint_sky" not in names:
        return False
    try:
        return bool(row["merge_footprint_sky"])
    except Exception:
        return False


def _load_layers_from_merge_catalog(
    path: Path,
    shape: tuple[int, int],
    min_x: int,
    min_y: int,
) -> list[MaskLayer]:
    layers = []
    with fits.open(path, memmap=False) as hdul:
        if len(hdul) < 5:
            raise RuntimeError(f"{path} does not look like an LSST SourceCatalog FITS")
        sources = hdul[1].data
        archive_index = hdul[2].data
        footprints = hdul[3].data
        spans = hdul[4].data

        footprint_rows = _archive_rows_by_id(archive_index, cat_archive=1, persistable=0)
        spanset_rows = _archive_rows_by_id(archive_index, cat_archive=2, persistable=0)
        for source in sources:
            if _is_sky_source_row(source):
                continue
            if "parent" in sources.dtype.names and int(source["parent"]) != 0:
                continue
            parent_id = int(source["id"])
            footprint_archive_id = int(source["footprint"])
            footprint_row = footprint_rows.get(footprint_archive_id)
            if footprint_row is None:
                continue
            footprint_row0, _ = footprint_row
            spanset_archive_id = int(footprints[footprint_row0]["id"])
            spans_row = spanset_rows.get(spanset_archive_id)
            if spans_row is None:
                continue

            row0, nrows = spans_row
            mask = np.zeros(shape, dtype=np.int16)
            for span in spans[row0 : row0 + nrows]:
                y = int(span["y"]) - min_y
                x0 = int(span["x0"]) - min_x
                x1 = int(span["x1"]) - min_x
                if 0 <= y < shape[0]:
                    xs0 = max(0, x0)
                    xs1 = min(shape[1] - 1, x1)
                    if xs0 <= xs1:
                        mask[y, xs0 : xs1 + 1] = 1
            if np.count_nonzero(mask):
                layers.append(MaskLayer(name=f"parent {parent_id}", path=path, data=mask))
    if not layers:
        raise RuntimeError(f"No parent footprints could be read from {path}")
    return layers


def _load_merged_layers(
    run_root: Path,
    merge_regions: Path | None,
    shape: tuple[int, int],
    min_x: int,
    min_y: int,
) -> tuple[list[MaskLayer], str]:
    if merge_regions is not None:
        return _load_lsst_layers(merge_regions), str(merge_regions)

    merge_catalog = run_root / "merge" / "deepCoadd_mergeDet.fits"
    if merge_catalog.exists():
        return _load_layers_from_merge_catalog(merge_catalog, shape, min_x, min_y), str(merge_catalog)

    regions = _discover_merge_regions(run_root, None)
    if regions is not None:
        return _load_lsst_layers(regions), str(regions)
    raise RuntimeError(f"No merge/deepCoadd_mergeDet.fits or merge_regions directory found under {run_root}")


def _parse_int_list(values: list[str]) -> list[int]:
    result: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                result.append(int(part))
    return result


def _parse_centers(values: list[str]) -> list[tuple[float, float]]:
    centers = []
    for value in values:
        for item in value.split(";"):
            item = item.strip()
            if not item:
                continue
            x_text, y_text = item.split(",", 1)
            centers.append((float(x_text), float(y_text)))
    return centers


def _mask_center(mask: np.ndarray) -> tuple[float, float]:
    y, x = np.nonzero(mask)
    if x.size == 0:
        return np.nan, np.nan
    return float(np.mean(x)), float(np.mean(y))


def _roi_bounds(center_x: float, center_y: float, size: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    roi_w = min(size, width)
    roi_h = min(size, height)
    x0 = int(round(center_x - roi_w / 2))
    y0 = int(round(center_y - roi_h / 2))
    x0 = min(max(0, x0), width - roi_w)
    y0 = min(max(0, y0), height - roi_h)
    return x0, y0, x0 + roi_w, y0 + roi_h


def _select_rois(
    lsst_layers: list[MaskLayer],
    parents: list[int],
    centers: list[tuple[float, float]],
    max_rois: int,
    roi_size: int,
    shape: tuple[int, int],
) -> list[Roi]:
    by_parent = {_layer_parent_id(layer): layer for layer in lsst_layers}
    roi_specs: list[tuple[str, float, float, int | None]] = []

    for parent_id in parents:
        layer = by_parent.get(parent_id)
        if layer is None:
            raise RuntimeError(f"Requested parent {parent_id} is not present in merge_regions")
        center_x, center_y = _mask_center(layer.data)
        if np.isfinite(center_x) and np.isfinite(center_y):
            roi_specs.append((f"parent_{parent_id:08d}", center_x, center_y, parent_id))

    for index, (center_x, center_y) in enumerate(centers, start=1):
        roi_specs.append((f"center_{index:02d}_{center_x:.1f}_{center_y:.1f}", center_x, center_y, None))

    if not roi_specs:
        ranked = sorted(
            lsst_layers,
            key=lambda layer: int(np.count_nonzero(layer.data)),
            reverse=True,
        )
        for layer in ranked[:max_rois]:
            parent_id = _layer_parent_id(layer)
            center_x, center_y = _mask_center(layer.data)
            if np.isfinite(center_x) and np.isfinite(center_y):
                roi_specs.append((f"parent_{parent_id:08d}", center_x, center_y, parent_id))

    rois = []
    for name, center_x, center_y, parent_id in roi_specs[:max_rois]:
        x0, y0, x1, y1 = _roi_bounds(center_x, center_y, roi_size, shape)
        rois.append(Roi(name=_safe_name(name), center_x=center_x, center_y=center_y, anchor_parent=parent_id, x0=x0, y0=y0, x1=x1, y1=y1))
    return rois


def _color_for_value(value: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20", 20)
    return np.asarray(cmap((int(value) - 1) % 20)[:3], dtype=np.float32)


def _overlay_labelmap(base: np.ndarray, labelmap: np.ndarray, alpha: float) -> tuple[np.ndarray, list[int]]:
    out = np.array(base, copy=True)
    labels = np.unique(labelmap)
    labels = [int(label) for label in labels if int(label) != 0]
    for label in labels:
        mask = labelmap == label
        out[mask] = (1 - alpha) * out[mask] + alpha * _color_for_value(label)
    return np.clip(out, 0, 1), labels


def _merged_labelmap_for_roi(layers: list[MaskLayer], yslice: slice, xslice: slice) -> tuple[np.ndarray, list[int]]:
    shape = layers[0].data[yslice, xslice].shape
    labelmap = np.zeros(shape, dtype=np.int32)
    parent_ids = []
    for layer in layers:
        parent_id = _layer_parent_id(layer)
        crop = layer.data[yslice, xslice] != 0
        if np.count_nonzero(crop) == 0:
            continue
        labelmap[crop] = parent_id
        parent_ids.append(parent_id)
    return labelmap, parent_ids


def _apply_axes_style(ax, title: str, roi: Roi) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("local x")
    ax.set_ylabel("local y")
    ax.set_xlim(roi.x0, roi.x1)
    ax.set_ylim(roi.y0, roi.y1)
    ax.tick_params(labelsize=7)


def _add_compact_legend(ax, labels: list[int], prefix: str, max_labels: int = 10) -> None:
    if not labels:
        return
    shown = labels[:max_labels]
    handles = [Patch(facecolor=_color_for_value(label), edgecolor="none", label=f"{prefix}{label}") for label in shown]
    if len(labels) > max_labels:
        handles.append(Patch(facecolor="none", edgecolor="none", label=f"+{len(labels) - max_labels} more"))
    ax.legend(handles=handles, loc="upper right", fontsize=6, framealpha=0.72)


def _save_roi_figure(
    outpath: Path,
    roi: Roi,
    background: np.ndarray,
    background_name: str,
    sam_layers: list[MaskLayer],
    lsst_layers: list[MaskLayer],
    alpha: float,
) -> tuple[list[int], list[int]]:
    yslice = slice(roi.y0, roi.y1)
    xslice = slice(roi.x0, roi.x1)
    base = background[yslice, xslice]
    extent = [roi.x0, roi.x1, roi.y0, roi.y1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    axes[0].imshow(base, origin="lower", extent=extent)
    axes[0].scatter([roi.center_x], [roi.center_y], marker="+", c="#ffd34d", s=42, linewidths=1.0)
    _apply_axes_style(axes[0], f"{background_name}\n{roi.name}", roi)

    sam_labelmap, sam_parent_ids = _merged_labelmap_for_roi(sam_layers, yslice, xslice)
    sam_overlay, sam_labels = _overlay_labelmap(base, sam_labelmap, alpha)
    axes[1].imshow(sam_overlay, origin="lower", extent=extent)
    _apply_axes_style(axes[1], f"SAM after merge\nparents={len(sam_parent_ids)}", roi)
    _add_compact_legend(axes[1], sam_labels, "s")

    lsst_labelmap, lsst_parent_ids = _merged_labelmap_for_roi(lsst_layers, yslice, xslice)
    lsst_overlay, lsst_labels = _overlay_labelmap(base, lsst_labelmap, alpha)
    axes[2].imshow(lsst_overlay, origin="lower", extent=extent)
    _apply_axes_style(axes[2], f"LSST after merge\nparents={len(lsst_parent_ids)}", roi)
    _add_compact_legend(axes[2], lsst_labels, "l")

    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    return sam_parent_ids, lsst_parent_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize fixed-size ROIs comparing SAM-after-merge and LSST-after-merge parent footprints."
    )
    parser.add_argument("--input", type=Path, default=None, help="Alias for --sam-input.")
    parser.add_argument("--sam-input", type=Path, default=None, help="SAM pipeline output root containing detect/ and merge/.")
    parser.add_argument("--lsst-input", type=Path, default=Path("output/no_docker_test"), help="LSST/no_docker output root.")
    parser.add_argument("--sam-merge-regions", type=Path, default=None, help="SAM merge_regions directory, if already exported.")
    parser.add_argument("--lsst-merge-regions", type=Path, default=None, help="LSST merge_regions directory, if already exported.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for ROI comparison PNGs and CSV.")
    parser.add_argument("--roi-size", type=int, default=128, help="Square ROI size in pixels.")
    parser.add_argument("--max-rois", type=int, default=6, help="Maximum number of ROIs when selecting automatically.")
    parser.add_argument("--parents", action="append", default=[], help="Comma-separated anchor parent ids.")
    parser.add_argument("--centers", action="append", default=[], help="Manual ROI centers as x,y or x,y;x,y in local pixels.")
    parser.add_argument("--anchor", choices=("sam", "lsst"), default="sam", help="Run used for --parents and automatic ROI selection.")
    parser.add_argument("--background", default="i", help="Background band: i/r/g or rgb. Default: i.")
    parser.add_argument("--alpha", type=float, default=0.48, help="Mask overlay alpha.")
    args = parser.parse_args()

    if args.roi_size <= 0:
        raise RuntimeError("--roi-size must be positive")
    if args.max_rois <= 0:
        raise RuntimeError("--max-rois must be positive")

    sam_root = args.sam_input or args.input
    if sam_root is None:
        raise RuntimeError("Pass --sam-input, or use --input as an alias.")
    sam_root = sam_root.resolve()
    lsst_root = args.lsst_input.resolve()
    outdir = args.output.resolve()
    background, background_name, min_x, min_y = _load_background(sam_root, args.background)

    shape = background.shape[:2]
    sam_layers, sam_source = _load_merged_layers(
        sam_root,
        args.sam_merge_regions.resolve() if args.sam_merge_regions else None,
        shape,
        min_x,
        min_y,
    )
    lsst_layers, lsst_source = _load_merged_layers(
        lsst_root,
        args.lsst_merge_regions.resolve() if args.lsst_merge_regions else None,
        shape,
        min_x,
        min_y,
    )
    for layer in [*sam_layers, *lsst_layers]:
        if layer.data.shape != shape:
            raise RuntimeError(f"Shape mismatch for {layer.path}: {layer.data.shape} != {shape}")

    parents = _parse_int_list(args.parents)
    centers = _parse_centers(args.centers)
    anchor_layers = sam_layers if args.anchor == "sam" else lsst_layers
    rois = _select_rois(anchor_layers, parents, centers, args.max_rois, args.roi_size, shape)
    if not rois:
        raise RuntimeError("No ROI could be selected")

    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for index, roi in enumerate(rois, start=1):
        outpath = outdir / f"roi_{index:03d}_{roi.name}.png"
        sam_parent_ids, lsst_parent_ids = _save_roi_figure(
            outpath,
            roi,
            background,
            background_name,
            sam_layers,
            lsst_layers,
            args.alpha,
        )
        summary_rows.append(
            [
                index,
                roi.name,
                args.anchor,
                roi.anchor_parent if roi.anchor_parent is not None else "",
                f"{roi.center_x:.3f}",
                f"{roi.center_y:.3f}",
                roi.x0,
                roi.y0,
                roi.x1,
                roi.y1,
                ";".join(str(parent_id) for parent_id in sam_parent_ids),
                ";".join(str(parent_id) for parent_id in lsst_parent_ids),
                outpath.name,
            ]
        )

    with (outdir / "roi_visualization.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "roi_index",
                "roi_name",
                "anchor",
                "anchor_parent",
                "center_x",
                "center_y",
                "x0",
                "y0",
                "x1",
                "y1",
                "sam_parent_ids",
                "lsst_parent_ids",
                "png",
            ]
        )
        writer.writerows(summary_rows)

    print(f"Wrote {len(rois)} ROI comparison figures under: {outdir}")
    print(f"SAM merge source: {sam_source}")
    print(f"LSST merge source: {lsst_source}")


if __name__ == "__main__":
    main()
