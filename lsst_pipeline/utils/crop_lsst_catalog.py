from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits
from astropy.table import Table


DEFAULT_MEAS_X = "base_SdssCentroid_x"
DEFAULT_MEAS_Y = "base_SdssCentroid_y"
DEFAULT_DET_PEAK_X = "f_x"
DEFAULT_DET_PEAK_Y = "f_y"


@dataclass(frozen=True)
class CutoutBox:
    """A rectangular cutout in parent-patch pixel coordinates."""

    x0: float
    y0: float
    width: float
    height: float

    @property
    def x1(self) -> float:
        return self.x0 + self.width

    @property
    def y1(self) -> float:
        return self.y0 + self.height

    def contains(self, x: np.ndarray, y: np.ndarray, *, margin: float = 0.0) -> np.ndarray:
        finite = np.isfinite(x) & np.isfinite(y)
        return (
            finite
            & (x >= self.x0 - margin)
            & (x < self.x1 + margin)
            & (y >= self.y0 - margin)
            & (y < self.y1 + margin)
        )


def infer_parent_origin_from_exposure(path: Path, *, hdu: str | int = "IMAGE") -> tuple[int, int]:
    """Return the parent-patch pixel origin encoded by an LSST Exposure FITS."""
    with fits.open(path, memmap=False) as hdul:
        header = hdul[hdu].header
        if "LTV1" not in header or "LTV2" not in header:
            raise KeyError(f"{path}[{hdu}] does not contain LTV1/LTV2")
        return -int(round(float(header["LTV1"]))), -int(round(float(header["LTV2"])))


def infer_shape_from_exposure(path: Path, *, hdu: str | int = "IMAGE") -> tuple[int, int]:
    """Return image shape as width, height for an LSST Exposure FITS image HDU."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul[hdu].data
        if data is None or data.ndim != 2:
            raise ValueError(f"{path}[{hdu}] is not a 2D image")
        height, width = data.shape
        return int(width), int(height)


def cutout_box_from_exposure(path: Path, *, hdu: str | int = "IMAGE") -> CutoutBox:
    x0, y0 = infer_parent_origin_from_exposure(path, hdu=hdu)
    width, height = infer_shape_from_exposure(path, hdu=hdu)
    return CutoutBox(float(x0), float(y0), float(width), float(height))


def _require_columns(table: Table, names: Iterable[str]) -> None:
    missing = [name for name in names if name not in table.colnames]
    if missing:
        raise KeyError(f"missing required column(s): {', '.join(missing)}")


def _with_local_columns(table: Table, *, x_col: str, y_col: str, box: CutoutBox, prefix: str) -> Table:
    out = table.copy(copy_data=True)
    out[f"{prefix}_local_x"] = np.asarray(out[x_col], dtype=float) - box.x0
    out[f"{prefix}_local_y"] = np.asarray(out[y_col], dtype=float) - box.y0
    return out


def crop_table_by_position(
    table: Table,
    *,
    box: CutoutBox,
    x_col: str,
    y_col: str,
    margin: float = 0.0,
    add_local: bool = True,
    local_prefix: str = "centroid",
) -> Table:
    """Crop a source table by parent-patch pixel coordinates."""
    _require_columns(table, (x_col, y_col))
    x = np.asarray(table[x_col], dtype=float)
    y = np.asarray(table[y_col], dtype=float)
    cropped = table[box.contains(x, y, margin=margin)]
    if add_local:
        cropped = _with_local_columns(cropped, x_col=x_col, y_col=y_col, box=box, prefix=local_prefix)
    return cropped


def read_meas_sources(path: Path, *, source_hdu: int = 1) -> Table:
    return Table.read(path, hdu=source_hdu)


def read_det_sources(path: Path, *, source_hdu: int = 1) -> Table:
    return Table.read(path, hdu=source_hdu)


def read_det_peaks(path: Path, *, peaks_hdu: int = 5) -> Table:
    return Table.read(path, hdu=peaks_hdu)


def crop_meas_catalog(
    path: Path,
    *,
    box: CutoutBox,
    x_col: str = DEFAULT_MEAS_X,
    y_col: str = DEFAULT_MEAS_Y,
    margin: float = 0.0,
) -> Table:
    table = read_meas_sources(path)
    return crop_table_by_position(
        table,
        box=box,
        x_col=x_col,
        y_col=y_col,
        margin=margin,
        local_prefix="centroid",
    )


def crop_det_peak_catalog(
    path: Path,
    *,
    box: CutoutBox,
    x_col: str = DEFAULT_DET_PEAK_X,
    y_col: str = DEFAULT_DET_PEAK_Y,
    margin: float = 0.0,
) -> Table:
    table = read_det_peaks(path)
    return crop_table_by_position(
        table,
        box=box,
        x_col=x_col,
        y_col=y_col,
        margin=margin,
        local_prefix="peak",
    )


def _write_table(table: Table, output: Path, *, overwrite: bool = True) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".csv":
        _table_for_csv(table).write(output, format="ascii.csv", overwrite=overwrite)
    elif suffix in {".fits", ".fit", ".fz"}:
        table.write(output, format="fits", overwrite=overwrite)
    elif suffix in {".ecsv", ".txt"}:
        table.write(output, format="ascii.ecsv", overwrite=overwrite)
    else:
        raise ValueError(f"unsupported output suffix {output.suffix!r}; use .fits, .csv, or .ecsv")


def _table_for_csv(table: Table) -> Table:
    """Return a CSV-safe table by serializing multidimensional columns."""
    out = Table()
    for name in table.colnames:
        values = table[name]
        arr = np.asarray(values)
        if arr.ndim <= 1:
            out[name] = values
        else:
            out[name] = [json.dumps(np.asarray(row).tolist(), separators=(",", ":")) for row in arr]
    return out


def _make_box(args: argparse.Namespace) -> CutoutBox:
    if args.exposure is not None:
        box = cutout_box_from_exposure(args.exposure.expanduser(), hdu=args.exposure_hdu)
        if args.x0 is None and args.y0 is None and args.width is None and args.height is None:
            return box
        return CutoutBox(
            float(box.x0 if args.x0 is None else args.x0),
            float(box.y0 if args.y0 is None else args.y0),
            float(box.width if args.width is None else args.width),
            float(box.height if args.height is None else args.height),
        )

    if args.x0 is None or args.y0 is None or args.width is None or args.height is None:
        raise ValueError("pass either --exposure or all of --x0 --y0 --width --height")
    return CutoutBox(float(args.x0), float(args.y0), float(args.width), float(args.height))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crop LSST det/meas catalog tables to a cutout in parent-patch pixel coordinates. "
            "For a 512x512 cutout Exposure, --exposure can infer x0/y0/width/height from LTV and IMAGE shape."
        )
    )
    parser.add_argument("catalog", type=Path, help="Input LSST catalog FITS, e.g. meas-*.fits or det-*.fits.")
    parser.add_argument("--output", required=True, type=Path, help="Output table path: .fits, .csv, or .ecsv.")
    parser.add_argument("--kind", choices=["meas", "det-peaks"], default="meas")
    parser.add_argument("--x-col", default=None, help="Position x column. Defaults depend on --kind.")
    parser.add_argument("--y-col", default=None, help="Position y column. Defaults depend on --kind.")
    parser.add_argument("--margin", type=float, default=0.0, help="Extra margin around the cutout in pixels.")
    parser.add_argument("--x0", type=float, default=None, help="Cutout parent-patch x origin.")
    parser.add_argument("--y0", type=float, default=None, help="Cutout parent-patch y origin.")
    parser.add_argument("--width", type=float, default=None, help="Cutout width in pixels.")
    parser.add_argument("--height", type=float, default=None, help="Cutout height in pixels.")
    parser.add_argument("--exposure", type=Path, default=None, help="Cutout Exposure FITS to infer x0/y0/width/height.")
    parser.add_argument("--exposure-hdu", default="IMAGE", help="Image HDU in --exposure, default IMAGE.")
    args = parser.parse_args()

    box = _make_box(args)
    catalog_path = args.catalog.expanduser()
    if args.kind == "meas":
        table = crop_meas_catalog(
            catalog_path,
            box=box,
            x_col=args.x_col or DEFAULT_MEAS_X,
            y_col=args.y_col or DEFAULT_MEAS_Y,
            margin=float(args.margin),
        )
    else:
        table = crop_det_peak_catalog(
            catalog_path,
            box=box,
            x_col=args.x_col or DEFAULT_DET_PEAK_X,
            y_col=args.y_col or DEFAULT_DET_PEAK_Y,
            margin=float(args.margin),
        )

    _write_table(table, args.output.expanduser())
    print(
        f"wrote {len(table)} rows to {args.output} "
        f"for box x={box.x0:g}:{box.x1:g}, y={box.y0:g}:{box.y1:g}"
    )


if __name__ == "__main__":
    main()
