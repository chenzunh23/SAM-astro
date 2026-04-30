#!/usr/bin/env python3
"""Visualize pixels clipped by Sam.astro_preprocess().

The script reproduces the per-channel preprocessing in
segment_anything/modeling/sam.py without loading a SAM checkpoint:

1. replace non-finite values by the sigma-clipped mean
2. clip bright pixels above median + clip_sigma * raw_std
3. convert to z-score using astropy sigma_clipped_stats on bright-clipped data
4. optionally clip z-scores to [LOW, HIGH]

The FITS outputs are intended for DS9/zscale inspection.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import ZScaleInterval
except Exception as exc:  # pragma: no cover
    raise RuntimeError("astropy is required. Install with: pip install astropy") from exc

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required. Install with: pip install pillow") from exc


CHANNEL_NAMES = ("R", "G", "B")


@dataclass(frozen=True)
class ChannelClipStats:
    channel: str
    finite_count: int
    raw_median: float
    raw_sigma: float
    clip_hi: float
    raw_clipped_count: int
    raw_clipped_fraction: float
    raw_excess_sum: float
    mean: float
    std: float
    z_low: float | None
    z_high: float | None
    z_low_clipped_count: int
    z_high_clipped_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create FITS/PNG visualizations of the pixels clipped in Sam.astro_preprocess()."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Either one 2D/3-band FITS file or exactly three 2D FITS files in R G B order.",
    )
    parser.add_argument("--output", required=True, help="Output prefix, without suffix.")
    parser.add_argument("--hdu", type=int, default=0)
    parser.add_argument(
        "--clip-sigma",
        type=float,
        default=3.0,
        help="Matches Sam.astro_preprocess_clip_sigma.",
    )
    parser.add_argument(
        "--sigma-iters",
        type=int,
        default=-1,
        help="Matches Sam.astro_preprocess_sigma_iters; -1 means iterate to convergence.",
    )
    parser.add_argument(
        "--z-clip",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Optional final z-score clip, matching Sam.astro_preprocess_z_clip.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=None,
        help="Optional square crop size before computing stats and visualizations.",
    )
    parser.add_argument(
        "--crop-origin",
        choices=["top-left", "top-right", "bottom-left", "bottom-right", "center"],
        default="top-right",
        help="Crop origin used when --crop-size is set. Existing astro AMG crop uses top-right.",
    )
    parser.add_argument(
        "--stats-input",
        nargs=3,
        default=None,
        help=(
            "Optional full-frame 2D FITS triplet used only for estimating clip/statistics. "
            "Useful when visualizing an ROI with full-image normalization."
        ),
    )
    parser.add_argument("--stats-hdu", type=int, default=None)
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write one multi-extension FITS containing all output products.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG quicklook outputs.",
    )
    return parser.parse_args()


def read_fits(path: Path, hdu: int) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        if hdu >= len(hdul):
            raise IndexError(f"HDU index {hdu} out of range for file: {path}")
        data = hdul[hdu].data
        if data is None:
            raise ValueError(f"No image data in HDU {hdu}: {path}")
        return np.asarray(data, dtype=np.float32), hdul[hdu].header.copy()


def read_input(paths: Sequence[str], hdu: int) -> tuple[np.ndarray, fits.Header]:
    fit_paths = [Path(p).expanduser().resolve() for p in paths]
    if len(fit_paths) == 1:
        arr, header = read_fits(fit_paths[0], hdu)
        if arr.ndim == 2:
            arr = arr[..., None]
        elif arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
            pass
        else:
            raise ValueError(f"Expected 2D or 1/3-band FITS in {fit_paths[0]}, got {arr.shape}")
        return arr.astype(np.float32, copy=False), header

    if len(fit_paths) != 3:
        raise ValueError("--input must receive either one FITS file or exactly three 2D FITS files.")

    arrays = []
    header = None
    for path in fit_paths:
        arr, h = read_fits(path, hdu)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D FITS for triplet input, got {arr.shape} in {path}")
        arrays.append(arr)
        if header is None:
            header = h
    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Input FITS shapes differ: {sorted(shapes)}")
    return np.stack(arrays, axis=-1).astype(np.float32, copy=False), header or fits.Header()


def crop_image(arr: np.ndarray, crop_size: int | None, origin: str) -> tuple[np.ndarray, int, int]:
    if crop_size is None:
        return arr, 0, 0
    h, w = arr.shape[:2]
    ch = min(h, int(crop_size))
    cw = min(w, int(crop_size))
    if origin.startswith("top"):
        y0 = 0
    elif origin.startswith("bottom"):
        y0 = h - ch
    else:
        y0 = (h - ch) // 2
    if origin.endswith("left"):
        x0 = 0
    elif origin.endswith("right"):
        x0 = w - cw
    else:
        x0 = (w - cw) // 2
    return arr[y0 : y0 + ch, x0 : x0 + cw], y0, x0


def finite_values(arr: np.ndarray) -> np.ndarray:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        raise ValueError("A channel has no finite values.")
    return vals.astype(np.float64, copy=False)


def sigma_clipped_mean_std(vals: np.ndarray, sigma: float, maxiters: int | None) -> tuple[float, float]:
    mean, _median, std = sigma_clipped_stats(vals, sigma=sigma, maxiters=maxiters)
    if not np.isfinite(mean):
        mean = float(np.nanmean(vals))
    if not np.isfinite(std) or std <= 0:
        std = float(np.nanstd(vals))
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return float(mean), float(std)


def preprocess_channel(
    image: np.ndarray,
    stats_image: np.ndarray,
    name: str,
    clip_sigma: float,
    sigma_iters: int,
    z_clip: Sequence[float] | None,
) -> tuple[dict[str, np.ndarray], ChannelClipStats]:
    vals = finite_values(stats_image)
    raw_median = float(np.median(vals))
    raw_sigma = float(np.std(vals))
    if not np.isfinite(raw_sigma) or raw_sigma <= 0:
        raw_sigma = 1.0
    clip_hi = raw_median + float(clip_sigma) * raw_sigma
    clipped_vals = np.minimum(vals, clip_hi)
    maxiters = None if sigma_iters < 0 else int(sigma_iters)
    mean, std = sigma_clipped_mean_std(clipped_vals, float(clip_sigma), maxiters)

    finite = np.isfinite(image)
    safe = np.where(finite, image, mean).astype(np.float32, copy=False)
    raw_clip_mask = (safe > clip_hi) & finite
    raw_clip_excess = np.where(raw_clip_mask, safe - clip_hi, 0.0).astype(np.float32)
    clipped = np.minimum(safe, clip_hi)
    z_before = ((clipped - mean) / std).astype(np.float32)

    z_low = None
    z_high = None
    z_low_mask = np.zeros(image.shape, dtype=bool)
    z_high_mask = np.zeros(image.shape, dtype=bool)
    z_after = z_before
    if z_clip is not None:
        z_low = float(z_clip[0])
        z_high = float(z_clip[1])
        if not (np.isfinite(z_low) and np.isfinite(z_high) and z_low < z_high):
            raise ValueError("--z-clip requires finite LOW < HIGH.")
        z_low_mask = z_before < z_low
        z_high_mask = z_before > z_high
        z_after = np.clip(z_before, z_low, z_high).astype(np.float32)

    products = {
        "raw_clip_excess": raw_clip_excess,
        "raw_clip_mask": raw_clip_mask.astype(np.float32),
        "z_before_clip": z_before,
        "z_after_clip": z_after.astype(np.float32, copy=False),
        "z_low_clip_mask": z_low_mask.astype(np.float32),
        "z_high_clip_mask": z_high_mask.astype(np.float32),
    }
    stats = ChannelClipStats(
        channel=name,
        finite_count=int(np.count_nonzero(finite)),
        raw_median=raw_median,
        raw_sigma=raw_sigma,
        clip_hi=clip_hi,
        raw_clipped_count=int(np.count_nonzero(raw_clip_mask)),
        raw_clipped_fraction=float(np.count_nonzero(raw_clip_mask) / max(np.count_nonzero(finite), 1)),
        raw_excess_sum=float(np.sum(raw_clip_excess, dtype=np.float64)),
        mean=mean,
        std=std,
        z_low=z_low,
        z_high=z_high,
        z_low_clipped_count=int(np.count_nonzero(z_low_mask)),
        z_high_clipped_count=int(np.count_nonzero(z_high_mask)),
    )
    return products, stats


def stack_channels(channel_products: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    names = channel_products[0].keys()
    return {
        name: np.stack([products[name] for products in channel_products], axis=0).astype(np.float32)
        for name in names
    }


def update_header(header: fits.Header, args: argparse.Namespace, y0: int, x0: int) -> fits.Header:
    out = header.copy()
    out["SAMCLIP"] = (True, "Visualization of Sam.astro_preprocess clipping")
    out["CLIPSIG"] = (float(args.clip_sigma), "Bright clip sigma")
    out["SIGITER"] = (int(args.sigma_iters), "Sigma clipped stats maxiters; -1 means None")
    out["CROPY0"] = (int(y0), "Crop origin y in source image")
    out["CROPX0"] = (int(x0), "Crop origin x in source image")
    if args.z_clip is not None:
        out["ZCLIPLO"] = (float(args.z_clip[0]), "Final z clip low")
        out["ZCLIPHI"] = (float(args.z_clip[1]), "Final z clip high")
    return out


def write_fits_products(products: dict[str, np.ndarray], prefix: Path, header: fits.Header) -> list[Path]:
    paths = []
    for name, data in products.items():
        path = prefix.with_name(f"{prefix.name}_{name}").with_suffix(".fits")
        fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)
        paths.append(path)
    return paths


def write_combined(products: dict[str, np.ndarray], prefix: Path, header: fits.Header) -> Path:
    hdus = [fits.PrimaryHDU(header=header)]
    for name, data in products.items():
        hdus.append(fits.ImageHDU(data=data, name=name.upper()[:68]))
    path = prefix.with_name(f"{prefix.name}_sam_clip_products").with_suffix(".fits")
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def write_stats(stats: Sequence[ChannelClipStats], prefix: Path) -> Path:
    path = prefix.with_name(f"{prefix.name}_clip_stats").with_suffix(".csv")
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(stats[0].__dict__.keys()))
        writer.writeheader()
        for row in stats:
            writer.writerow(row.__dict__)
    return path


def astropy_zscale_to_uint8(image: np.ndarray) -> np.ndarray:
    vals = finite_values(image)
    lo, hi = ZScaleInterval().get_limits(vals)
    if not np.isfinite(hi - lo) or hi <= lo:
        lo, hi = float(np.min(vals)), float(np.max(vals))
    if not np.isfinite(hi - lo) or hi <= lo:
        hi = lo + 1e-6
    out = (np.clip(image, lo, hi) - lo) / (hi - lo)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.round(out * 255.0).astype(np.uint8)


def positive_excess_to_uint8(image: np.ndarray) -> np.ndarray:
    positive = image[np.isfinite(image) & (image > 0)]
    if positive.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    lo, hi = ZScaleInterval().get_limits(positive.astype(np.float64, copy=False))
    if not np.isfinite(hi - lo) or hi <= lo:
        lo = float(np.min(positive))
        hi = float(np.percentile(positive, 99.0))
    if not np.isfinite(hi - lo) or hi <= lo:
        hi = float(np.max(positive))
    if not np.isfinite(hi - lo) or hi <= lo:
        hi = lo + 1e-6
    clipped = np.clip(image, lo, hi)
    scaled = (clipped - lo) / (hi - lo)
    scaled = np.where(image > 0, scaled, 0.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    return np.round(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def raw_excess_highlight_png(data: np.ndarray) -> np.ndarray:
    if data.shape[0] == 1:
        gray = positive_excess_to_uint8(data[0])
        rgb = np.stack([gray // 3, gray // 6, np.zeros_like(gray)], axis=-1)
        mask = dilate_mask(data[0] > 0)
        rgb[mask] = np.maximum(rgb[mask], np.array([255, 80, 0], dtype=np.uint8))
        return rgb

    colors = np.array(
        [
            [255, 40, 40],
            [40, 230, 80],
            [60, 120, 255],
        ],
        dtype=np.uint8,
    )
    base_channels = [positive_excess_to_uint8(data[c]) for c in range(min(3, data.shape[0]))]
    rgb = np.stack(base_channels, axis=-1)
    rgb = (rgb.astype(np.float32) * 0.55).astype(np.uint8)
    for c, color in enumerate(colors[: data.shape[0]]):
        mask = dilate_mask(data[c] > 0)
        rgb[mask] = np.maximum(rgb[mask], color)
    return rgb


def write_png_quicklooks(products: dict[str, np.ndarray], prefix: Path) -> list[Path]:
    paths = []
    for name in ("raw_clip_excess", "raw_clip_mask", "z_before_clip", "z_after_clip"):
        data = products[name]
        if name == "raw_clip_excess":
            png = raw_excess_highlight_png(data)
            mode = "RGB"
        elif data.shape[0] == 1:
            png = astropy_zscale_to_uint8(data[0])
            mode = "L"
        else:
            channels = [astropy_zscale_to_uint8(data[c]) for c in range(min(3, data.shape[0]))]
            png = np.stack(channels, axis=-1)
            mode = "RGB"
        png = np.flipud(png)
        path = prefix.with_name(f"{prefix.name}_{name}").with_suffix(".png")
        Image.fromarray(png, mode=mode).save(path)
        paths.append(path)
    return paths


def channel_names(n_channels: int) -> tuple[str, ...]:
    if n_channels == 1:
        return ("CH0",)
    if n_channels == 3:
        return CHANNEL_NAMES
    return tuple(f"CH{i}" for i in range(n_channels))


def main() -> None:
    args = parse_args()
    prefix = Path(args.output).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    image, header = read_input(args.input, args.hdu)
    image, y0, x0 = crop_image(image, args.crop_size, args.crop_origin)

    stats_image = image
    if args.stats_input is not None:
        stats_hdu = args.hdu if args.stats_hdu is None else args.stats_hdu
        stats_image, _ = read_input(args.stats_input, stats_hdu)

    if image.shape[-1] != stats_image.shape[-1]:
        raise ValueError(
            f"Input has {image.shape[-1]} channels but stats input has {stats_image.shape[-1]} channels."
        )

    names = channel_names(image.shape[-1])
    channel_products = []
    stats = []
    for idx, name in enumerate(names):
        products, channel_stats = preprocess_channel(
            image[..., idx],
            stats_image[..., idx],
            name,
            args.clip_sigma,
            args.sigma_iters,
            args.z_clip,
        )
        channel_products.append(products)
        stats.append(channel_stats)

    products = stack_channels(channel_products)
    out_header = update_header(header, args, y0, x0)
    fits_paths = write_fits_products(products, prefix, out_header)
    stats_path = write_stats(stats, prefix)
    png_paths = [] if args.no_png else write_png_quicklooks(products, prefix)
    combined_path = write_combined(products, prefix, out_header) if args.combined else None

    for row in stats:
        print(
            f"[{row.channel}] clip_hi={row.clip_hi:.6g} "
            f"raw_clipped={row.raw_clipped_count}/{row.finite_count} "
            f"({row.raw_clipped_fraction:.6%}) mean={row.mean:.6g} std={row.std:.6g} "
            f"z_low_clipped={row.z_low_clipped_count} z_high_clipped={row.z_high_clipped_count}"
        )
    print(f"Saved stats: {stats_path}")
    print("Saved FITS:")
    for path in fits_paths:
        print(f"  {path}")
    if combined_path is not None:
        print(f"Saved combined FITS: {combined_path}")
    if png_paths:
        print("Saved PNG quicklooks:")
        for path in png_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
