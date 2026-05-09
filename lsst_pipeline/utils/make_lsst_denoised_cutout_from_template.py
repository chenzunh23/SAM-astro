"""Build a self-consistent denoised LSST Exposure cutout.

The template cutout supplies the desired sky footprint and output size.  The
denoised full-patch FITS supplies the pixel planes and, by default, the full HDU
archive structure.  IMAGE, MASK, and VARIANCE are cropped together and WCS/LTV
metadata are shifted so the resulting FITS can be consumed by the LSST pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


PIXEL_PLANES = ("IMAGE", "MASK", "VARIANCE")


def _origin_from_ltv(header: fits.Header) -> tuple[int, int]:
    if "LTV1" not in header or "LTV2" not in header:
        raise KeyError("header does not contain LTV1/LTV2; pass --x0 and --y0 explicitly")
    return -int(round(float(header["LTV1"]))), -int(round(float(header["LTV2"])))


def _finite_replacement(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.median(finite))


def _cropped_header(header: fits.Header, *, x0: int, y0: int) -> fits.Header:
    """Shift pixel-coordinate header keywords after a local crop."""
    out = header.copy()
    if "LTV1" in out:
        out["LTV1"] = float(out["LTV1"]) - x0
    if "LTV2" in out:
        out["LTV2"] = float(out["LTV2"]) - y0
    if "CRPIX1" in out:
        out["CRPIX1"] = float(out["CRPIX1"]) - x0
    if "CRPIX2" in out:
        out["CRPIX2"] = float(out["CRPIX2"]) - y0
    if "CRVAL1A" in out:
        out["CRVAL1A"] = float(out["CRVAL1A"]) + x0
    if "CRVAL2A" in out:
        out["CRVAL2A"] = float(out["CRVAL2A"]) + y0
    return out


def _new_hdu_like(hdu, *, data: np.ndarray | None = None, header: fits.Header | None = None):
    header = hdu.header.copy() if header is None else header
    if isinstance(hdu, fits.PrimaryHDU):
        return fits.PrimaryHDU(data=data, header=header)
    if isinstance(hdu, fits.ImageHDU):
        return fits.ImageHDU(data=data, header=header, name=hdu.name)
    return hdu.copy()


def make_cutout(
    *,
    template_path: Path,
    denoised_path: Path,
    output_path: Path,
    x0: int | None,
    y0: int | None,
    clean_nonfinite: bool,
    structure_source: str,
) -> None:
    """Write a denoised cutout whose pixel planes and headers agree.

    ``x0``/``y0`` are local pixel origins in the denoised full image.  If omitted,
    they are inferred from the difference between the template and denoised LTV
    origins, which is the common case for matching a known noisy cutout.
    """
    with fits.open(template_path, memmap=False) as template_hdul, fits.open(denoised_path, memmap=False) as den_hdul:
        if "IMAGE" not in template_hdul:
            raise KeyError(f"{template_path} has no IMAGE extension")
        if "IMAGE" not in den_hdul:
            raise KeyError(f"{denoised_path} has no IMAGE extension")

        template_image = template_hdul["IMAGE"].data
        if template_image is None or template_image.ndim != 2:
            raise ValueError(f"{template_path} IMAGE extension is not a 2D image")

        height, width = template_image.shape
        if x0 is None or y0 is None:
            template_origin = _origin_from_ltv(template_hdul["IMAGE"].header)
            denoised_origin = _origin_from_ltv(den_hdul["IMAGE"].header)
            auto_x0 = template_origin[0] - denoised_origin[0]
            auto_y0 = template_origin[1] - denoised_origin[1]
            x0 = auto_x0 if x0 is None else x0
            y0 = auto_y0 if y0 is None else y0

        for plane in PIXEL_PLANES:
            if plane not in den_hdul:
                raise KeyError(f"{denoised_path} has no {plane} extension")
            data = den_hdul[plane].data
            if data is None or data.ndim != 2:
                raise ValueError(f"{denoised_path} {plane} extension is not a 2D image")
            if x0 < 0 or y0 < 0 or x0 + width > data.shape[1] or y0 + height > data.shape[0]:
                raise ValueError(
                    f"cutout x={x0}:{x0 + width}, y={y0}:{y0 + height} exceeds "
                    f"denoised {plane} shape width={data.shape[1]}, height={data.shape[0]}"
                )

        base_hdul = den_hdul if structure_source == "denoised" else template_hdul
        out_hdus = []
        for hdu in base_hdul:
            if hdu.name in PIXEL_PLANES:
                if hdu.name not in den_hdul:
                    raise KeyError(f"{denoised_path} has no {hdu.name} extension")
                den_data = den_hdul[hdu.name].data
                cutout = np.asarray(den_data[y0 : y0 + height, x0 : x0 + width]).copy()
                if hdu.data is not None:
                    cutout = cutout.astype(np.asarray(hdu.data).dtype, copy=False)
                if clean_nonfinite and np.issubdtype(cutout.dtype, np.floating) and not np.all(np.isfinite(cutout)):
                    fill = _finite_replacement(cutout)
                    cutout = np.nan_to_num(cutout, nan=fill, posinf=fill, neginf=fill).astype(cutout.dtype, copy=False)
                header = _cropped_header(den_hdul[hdu.name].header, x0=x0, y0=y0)
                out_hdus.append(_new_hdu_like(hdu, data=cutout, header=header))
            else:
                # Some LSST archive tables use variable-length heap columns
                # that Astropy cannot reliably deep-copy.  Reuse the HDU while
                # the input file is open; HDUList.writeto serializes it before
                # the source file is closed.
                out_hdus.append(hdu)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(out_hdus).writeto(output_path, overwrite=True)

    print(
        f"wrote {output_path} from {denoised_path} "
        f"x={x0}:{x0 + width}, y={y0}:{y0 + height}; "
        f"structure={structure_source}, pixel_planes={','.join(PIXEL_PLANES)}, template_bbox={template_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an LSST-readable denoised cutout.  The template supplies the "
            "target cutout bbox, while IMAGE/MASK/VARIANCE are cropped from the "
            "denoised full-patch Exposure FITS.  By default the output preserves "
            "the denoised FITS HDU/archive structure."
        )
    )
    parser.add_argument("--template", required=True, type=Path, help="Known-good noisy cutout LSST Exposure FITS.")
    parser.add_argument("--denoised", required=True, type=Path, help="Denoised full-patch LSST Exposure FITS.")
    parser.add_argument("--output", required=True, type=Path, help="Output denoised cutout FITS.")
    parser.add_argument("--x0", type=int, default=None, help="Optional x origin in the denoised IMAGE.")
    parser.add_argument("--y0", type=int, default=None, help="Optional y origin in the denoised IMAGE.")
    parser.add_argument(
        "--no-clean-nonfinite",
        action="store_true",
        help="Do not replace NaN/Inf in floating IMAGE/VARIANCE cutouts.",
    )
    parser.add_argument(
        "--structure-source",
        choices=["denoised", "template"],
        default="denoised",
        help=(
            "Which FITS HDU/archive structure to preserve.  Default denoised keeps "
            "the full denoised product structure while cropping its pixel planes; "
            "template keeps the old known-good cutout structure."
        ),
    )
    args = parser.parse_args()

    make_cutout(
        template_path=args.template.expanduser(),
        denoised_path=args.denoised.expanduser(),
        output_path=args.output.expanduser(),
        x0=args.x0,
        y0=args.y0,
        clean_nonfinite=not args.no_clean_nonfinite,
        structure_source=args.structure_source,
    )


if __name__ == "__main__":
    main()
