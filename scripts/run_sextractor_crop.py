#!/usr/bin/env python3
"""Run SExtractor on a cropped FITS region and save check-image FITS files."""

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


REPO_ROOT = Path(__file__).resolve().parents[1]
SEXTRACTOR_DIR = (REPO_ROOT / "../sextractor").resolve()
SEXTRACTOR_BIN = Path(os.environ.get("SEXTRACTOR_BIN", SEXTRACTOR_DIR / "src/sex"))
FILTER_NAME = os.environ.get("FILTER_NAME", "gauss_5.0_9x9")
DETECT_THRESH = 0.5 # Default 1.5
ANALYSIS_THRESH = 0.5 # Default 1.5
DETECT_MINAREA = 15 # Default 3

def read_2d_fits(path: Path, hdu_index: int):
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[hdu_index]
        if hdu.data is None:
            raise ValueError(f"No image data in HDU {hdu_index}: {path}")
        data = np.asarray(hdu.data)
        header = hdu.header.copy()

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D FITS image in {path}, got shape {data.shape}")
    return data, header


def crop_fits(
    in_path: Path,
    out_path: Path,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    hdu_index: int,
):
    data, header = read_2d_fits(in_path, hdu_index)
    full_h, full_w = data.shape

    if x0 < 0 or y0 < 0:
        raise ValueError(f"Crop origin must be non-negative, got x0={x0}, y0={y0}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Crop size must be positive, got width={width}, height={height}")
    if x0 + width > full_w or y0 + height > full_h:
        raise ValueError(
            f"Crop [{x0}:{x0 + width}, {y0}:{y0 + height}] exceeds image shape "
            f"(height={full_h}, width={full_w})"
        )

    cropped = np.asarray(data[y0 : y0 + height, x0 : x0 + width])

    # Keep celestial WCS usable when present. FITS CRPIX is 1-based, while x0/y0 are
    # zero-based NumPy coordinates.
    try:
        wcs = WCS(header)
        if wcs.has_celestial:
            header.update(wcs.slice((slice(y0, y0 + height), slice(x0, x0 + width))).to_header())
        else:
            if "CRPIX1" in header:
                header["CRPIX1"] = header["CRPIX1"] - x0
            if "CRPIX2" in header:
                header["CRPIX2"] = header["CRPIX2"] - y0
    except Exception:
        if "CRPIX1" in header:
            header["CRPIX1"] = header["CRPIX1"] - x0
        if "CRPIX2" in header:
            header["CRPIX2"] = header["CRPIX2"] - y0

    header["CROPX0"] = (x0, "0-based source x origin")
    header["CROPY0"] = (y0, "0-based source y origin")
    header["CROPW"] = (width, "crop width")
    header["CROPH"] = (height, "crop height")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=cropped, header=header).writeto(out_path, overwrite=True)


def run_sextractor(detect_fits: Path, measure_fits: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "conda",
        "run",
        "-n",
        "base",
        str(SEXTRACTOR_BIN),
        f"{detect_fits},{measure_fits}",
        "-c",
        "config/default.sex",
        "-PARAMETERS_NAME",
        "config/rich.param",
        "-FILTER_NAME",
        f"config/{FILTER_NAME}.conv",
        "-STARNNW_NAME",
        "config/default.nnw",
        "-DETECT_THRESH",
        str(DETECT_THRESH),
        "-ANALYSIS_THRESH",
        str(ANALYSIS_THRESH),
        "-DETECT_MINAREA",
        str(DETECT_MINAREA),
        "-MEMORY_OBJSTACK",
        "10000",
        "-MEMORY_PIXSTACK",
        "3000000",
        "-CATALOG_TYPE",
        "ASCII_HEAD",
        "-CATALOG_NAME",
        str(out_dir / "result.cat"),
        "-WRITE_XML",
        "Y",
        "-XML_NAME",
        str(out_dir / "run_meta.xml"),
        "-CHECKIMAGE_TYPE",
        "FILTERED,SEGMENTATION,BACKGROUND",
        "-CHECKIMAGE_NAME",
        ",".join(
            [
                str(out_dir / "check_filtered.fits"),
                str(out_dir / "check_segmentation.fits"),
                str(out_dir / "check_background.fits"),
            ]
        ),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=SEXTRACTOR_DIR, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Crop a FITS image to a selected region, then run SExtractor with the "
            "same check-image outputs as ../sextractor/run.sh."
        )
    )
    parser.add_argument("measure_fits", help="Measurement FITS. Also used for detection unless --detect-fits is set.")
    parser.add_argument("--detect-fits", help="Optional detection FITS for dual-image mode.")
    parser.add_argument("--x0", type=int, required=True, help="Zero-based left x coordinate of the crop.")
    parser.add_argument("--y0", type=int, required=True, help="Zero-based top y coordinate of the crop.")
    parser.add_argument("--size", type=int, default=1024, help="Square crop size in pixels (default: 1024).")
    parser.add_argument("--width", type=int, help="Crop width. Overrides --size.")
    parser.add_argument("--height", type=int, help="Crop height. Overrides --size.")
    parser.add_argument("--hdu", type=int, default=0, help="FITS HDU index to read (default: 0).")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: results/sextractor_crop/<input>_x<X>_y<Y>_<W>x<H>",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    measure_fits = Path(args.measure_fits).expanduser().resolve()
    detect_fits = Path(args.detect_fits).expanduser().resolve() if args.detect_fits else measure_fits
    width = args.width if args.width is not None else args.size
    height = args.height if args.height is not None else args.size

    if not measure_fits.exists():
        raise FileNotFoundError(f"Measurement FITS not found: {measure_fits}")
    if not detect_fits.exists():
        raise FileNotFoundError(f"Detection FITS not found: {detect_fits}")
    if not SEXTRACTOR_BIN.exists():
        raise FileNotFoundError(f"SExtractor binary not found: {SEXTRACTOR_BIN}")

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        stem = detect_fits.stem
        out_dir = REPO_ROOT / "results" / "sextractor_crop" / f"{stem}_x{args.x0}_y{args.y0}_{width}x{height}"

    crop_dir = out_dir / "inputs"
    detect_crop = crop_dir / "detect_crop.fits"
    measure_crop = crop_dir / "measure_crop.fits"

    crop_fits(
        detect_fits,
        detect_crop,
        x0=args.x0,
        y0=args.y0,
        width=width,
        height=height,
        hdu_index=args.hdu,
    )
    if detect_fits == measure_fits:
        measure_crop = detect_crop
    else:
        crop_fits(
            measure_fits,
            measure_crop,
            x0=args.x0,
            y0=args.y0,
            width=width,
            height=height,
            hdu_index=args.hdu,
        )

    run_sextractor(detect_crop, measure_crop, out_dir)
    print(f"Saved outputs in {out_dir}")
    print(f"  {out_dir / 'check_background.fits'}")
    print(f"  {out_dir / 'check_filtered.fits'}")
    print(f"  {out_dir / 'check_segmentation.fits'}")


if __name__ == "__main__":
    main()
