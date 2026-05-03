from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from astropy.io import fits


def _safe_name(value: str) -> str:
    value = value.strip() or "primary"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower() or "hdu"


def _relative_stem(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return _safe_name("__".join(rel.parts))


def _write_image(path: Path, data, header=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(data)
    hdu = fits.PrimaryHDU(data=array, header=header)
    hdu.writeto(path, overwrite=True)


def export_image_hdus(path: Path, root: Path, outdir: Path) -> list[Path]:
    written: list[Path] = []
    with fits.open(path, memmap=False) as hdul:
        for index, hdu in enumerate(hdul):
            data = hdu.data
            if data is None:
                continue
            shape = getattr(data, "shape", ())
            if len(shape) < 2:
                continue
            name = _safe_name(hdu.name if hdu.name else f"hdu{index}")
            dst = outdir / f"{_relative_stem(path, root)}__{index:02d}_{name}.fits"
            _write_image(dst, data, hdu.header)
            written.append(dst)
    return written


def _find_reference_calexp(path: Path, root: Path) -> Path | None:
    if path.name == "deepCoadd_calexp.fits":
        return path
    if path.name == "deepCoadd_det.fits":
        candidate = path.with_name("deepCoadd_calexp.fits")
        if candidate.exists():
            return candidate
    candidates = sorted((root / "detect").glob("HSC-*/deepCoadd_calexp.fits"))
    return candidates[0] if candidates else None


def _load_lsst_reference_bbox(reference_calexp: Path):
    import lsst.afw.image as afwImage

    exposure = afwImage.ExposureF(str(reference_calexp))
    bbox = exposure.getBBox()
    return bbox, int(bbox.getWidth()), int(bbox.getHeight()), int(bbox.getMinX()), int(bbox.getMinY())


def _paint_footprints_with_lsst(path: Path, root: Path, outdir: Path) -> list[Path]:
    import lsst.afw.image as afwImage
    import lsst.afw.table as afwTable

    reference = _find_reference_calexp(path, root)
    if reference is None:
        return []

    catalog = afwTable.SourceCatalog.readFits(str(path))
    if len(catalog) == 0:
        return []

    bbox, width, height, min_x, min_y = _load_lsst_reference_bbox(reference)
    all_mask = afwImage.Mask(bbox)
    parent_mask = afwImage.Mask(bbox)
    child_mask = afwImage.Mask(bbox)
    peaks = np.zeros((height, width), dtype=np.int16)

    has_parent = "parent" in catalog.schema
    for record in catalog:
        footprint = record.getFootprint()
        if footprint is None:
            continue

        footprint.spans.setMask(all_mask, 1)
        is_child = bool(has_parent and int(record["parent"]) != 0)
        footprint.spans.setMask(child_mask if is_child else parent_mask, 1)

        for peak in footprint.getPeaks():
            x = int(round(peak.getFx())) - min_x
            y = int(round(peak.getFy())) - min_y
            if 0 <= x < width and 0 <= y < height:
                peaks[y, x] = 1

    base = outdir / _relative_stem(path, root)
    outputs = [
        base.with_name(base.name + "__footprints.fits"),
        base.with_name(base.name + "__parent_footprints.fits"),
        base.with_name(base.name + "__child_footprints.fits"),
        base.with_name(base.name + "__peaks.fits"),
    ]
    _write_image(outputs[0], all_mask.array.astype(np.int16))
    _write_image(outputs[1], parent_mask.array.astype(np.int16))
    _write_image(outputs[2], child_mask.array.astype(np.int16))
    _write_image(outputs[3], peaks)
    return outputs


def convert_tree(root: Path, outdir: Path) -> None:
    fits_files = sorted(root.rglob("*.fits"))
    if not fits_files:
        raise FileNotFoundError(f"no FITS files found under {root}")

    for path in fits_files:
        print(f"\n{path}")
        image_outputs = export_image_hdus(path, root, outdir)
        for dst in image_outputs:
            print(f"  image -> {dst}")

        table_outputs: list[Path] = []
        looks_like_catalog = path.name in {
            "deepCoadd_det.fits",
            "deepCoadd_mergeDet.fits",
            "deepCoadd_deblendedFlux.fits",
        }
        if looks_like_catalog:
            try:
                table_outputs = _paint_footprints_with_lsst(path, root, outdir)
            except Exception as exc:
                print(f"  catalog map skipped: {type(exc).__name__}: {exc}")
        elif not image_outputs:
            table_outputs = []
        for dst in table_outputs:
            print(f"  map   -> {dst}")

        if not image_outputs and not table_outputs:
            print("  no directly viewable image product generated")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LSST/afw FITS outputs into simple image FITS files for DS9/FIJI."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/scarlet_demo"),
        help="Root directory containing LSST FITS outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/viewable_fits"),
        help="Directory for converted image FITS files.",
    )
    args = parser.parse_args()
    convert_tree(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
