from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


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


def make_cutout(
    *,
    template_path: Path,
    denoised_path: Path,
    output_path: Path,
    x0: int | None,
    y0: int | None,
    clean_nonfinite: bool,
) -> None:
    with fits.open(template_path, memmap=False) as template_hdul, fits.open(denoised_path, memmap=False) as den_hdul:
        if "IMAGE" not in template_hdul:
            raise KeyError(f"{template_path} has no IMAGE extension")
        if "IMAGE" not in den_hdul:
            raise KeyError(f"{denoised_path} has no IMAGE extension")

        template_image = template_hdul["IMAGE"].data
        den_image = den_hdul["IMAGE"].data
        if template_image is None or template_image.ndim != 2:
            raise ValueError(f"{template_path} IMAGE extension is not a 2D image")
        if den_image is None or den_image.ndim != 2:
            raise ValueError(f"{denoised_path} IMAGE extension is not a 2D image")

        height, width = template_image.shape
        if x0 is None or y0 is None:
            template_origin = _origin_from_ltv(template_hdul["IMAGE"].header)
            denoised_origin = _origin_from_ltv(den_hdul["IMAGE"].header)
            auto_x0 = template_origin[0] - denoised_origin[0]
            auto_y0 = template_origin[1] - denoised_origin[1]
            x0 = auto_x0 if x0 is None else x0
            y0 = auto_y0 if y0 is None else y0

        if x0 < 0 or y0 < 0 or x0 + width > den_image.shape[1] or y0 + height > den_image.shape[0]:
            raise ValueError(
                f"cutout x={x0}:{x0 + width}, y={y0}:{y0 + height} exceeds "
                f"denoised IMAGE shape width={den_image.shape[1]}, height={den_image.shape[0]}"
            )

        cutout = np.asarray(den_image[y0 : y0 + height, x0 : x0 + width], dtype=template_image.dtype).copy()
        if clean_nonfinite and not np.all(np.isfinite(cutout)):
            fill = _finite_replacement(cutout)
            cutout = np.nan_to_num(cutout, nan=fill, posinf=fill, neginf=fill).astype(template_image.dtype, copy=False)

        out_hdus = []
        for hdu in template_hdul:
            if hdu.name == "IMAGE":
                if isinstance(hdu, fits.PrimaryHDU):
                    out_hdus.append(fits.PrimaryHDU(data=cutout, header=hdu.header.copy()))
                else:
                    out_hdus.append(type(hdu)(data=cutout, header=hdu.header.copy(), name=hdu.name))
            else:
                out_hdus.append(hdu)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(out_hdus).writeto(output_path, overwrite=True)

    print(
        f"wrote {output_path} from {denoised_path} "
        f"x={x0}:{x0 + width}, y={y0}:{y0 + height} using template {template_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an LSST-readable denoised cutout by preserving a known-good "
            "cutout Exposure FITS template and replacing only its IMAGE plane."
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
        help="Do not replace NaN/Inf in the output IMAGE cutout.",
    )
    args = parser.parse_args()

    make_cutout(
        template_path=args.template.expanduser(),
        denoised_path=args.denoised.expanduser(),
        output_path=args.output.expanduser(),
        x0=args.x0,
        y0=args.y0,
        clean_nonfinite=not args.no_clean_nonfinite,
    )


if __name__ == "__main__":
    main()
