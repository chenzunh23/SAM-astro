#!/usr/bin/env python3
"""Preprocess a 3-band FITS using astro_rgb-style clipping and mapping.

Usage:
  python scripts/astro_preprocess.py --input file.fits --out out_prefix --method robust --size 1024

This script:
- reads a FITS containing 3 bands (either shape 3,H,W or H,W,3)
- for each channel clips values to mean +/- 3*sigma (computed per-channel)
- applies the selected mapping, including the same robust percentile mapping used
  by scripts/amg_fits_overlay.py for single-band SAM inputs
- optionally resizes via the repo's ResizeLongestSide transform
- converts the result to a PIL image and saves PNG + saves processed FITS
"""
import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits
from torchvision.transforms.functional import to_pil_image
from PIL import Image

try:
    from segment_anything.utils.transforms import ResizeLongestSide
except Exception:
    # fallback: minimal resize function
    ResizeLongestSide = None


def read_fits_rgb(path: Path):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
    arr = np.asarray(data)
    # Accept shapes: (3,H,W), (H,W,3), (H,W) (single-band)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.stack([arr[0], arr[1], arr[2]], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        # single-band -> H,W,1
        return arr.astype(np.float32)[..., None]
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr.astype(np.float32)
    raise ValueError(f"Input FITS must contain 1 or 3 bands. Got shape {arr.shape}")


def compute_clip_bounds(arr: np.ndarray):
    # arr: HxWxC where C==1 or 3
    means = []
    stds = []
    if arr.ndim != 3 or arr.shape[2]==1:
        means = np.array([float(np.nanmean(arr))], dtype=np.float32)
        stds = np.array([float(np.nanstd(arr))], dtype=np.float32)
    else:
        for c in range(3):
            ch = arr[..., c]
            finite = np.isfinite(ch)
            if not np.any(finite):
                m = 0.0
                s = 1.0
            else:
                m = float(np.nanmean(ch[finite]))
                s = float(np.nanstd(ch[finite]))
            means.append(m)
            stds.append(s)
        means = np.array(means, dtype=np.float32)
        stds = np.array(stds, dtype=np.float32)
    lo = means - 3.0 * stds
    hi = means + 3.0 * stds
    return means, stds, lo, hi


def robust_to_uint8(arr: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    channels = arr.shape[-1]
    out = np.empty(arr.shape, dtype=np.uint8)
    for c in range(channels):
        ch = arr[..., c]
        finite = np.isfinite(ch)
        if not np.any(finite):
            raise ValueError("Image has no finite values.")
        vals = ch[finite]
        lo = np.percentile(vals, low_pct)
        hi = np.percentile(vals, high_pct)
        if hi <= lo:
            hi = lo + 1e-6

        clipped = np.clip(ch, lo, hi)
        norm = (clipped - lo) / (hi - lo)
        norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
        out[..., c] = (norm * 255.0).astype(np.uint8)
        print(
            f"robust channel {c}: low_pct={low_pct}, high_pct={high_pct}, "
            f"lo={lo:.6g}, hi={hi:.6g}, mean_u8={out[..., c].mean():.3f}, "
            f"nonzero={np.count_nonzero(out[..., c])}/{out[..., c].size}"
        )
    return out


def clip_and_map(arr: np.ndarray, lo: np.ndarray, hi: np.ndarray, method: str, low_pct: float, high_pct: float):
    if method == 'robust':
        return robust_to_uint8(arr, low_pct, high_pct)

    channels = arr.shape[-1]
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(channels):
        ch = arr[..., c]
        l = lo[c]
        h = hi[c]
        if not np.isfinite(l) or not np.isfinite(h) or h <= l:
            clipped = np.nan_to_num(ch, nan=0.0)
        else:
            clipped = np.clip(ch, l, h)
        out[..., c] = clipped

    if method == 'norm01':
        # map per-channel linear to [0,1]
        for c in range(channels):
            l = lo[c]
            h = hi[c]
            mean = (l + h) / 2.0
            if h <= l:
                out[..., c] = 0.0
            else:
                out[..., c] = np.clip((out[..., c] - l) / (h - l), 0.0, 1.0)
        return out.astype(np.float32)
    elif method == 'norm0255':
        res = np.empty_like(out, dtype=np.float32)
        for c in range(channels):
            l = lo[c]
            h = hi[c]
            if h <= l:
                mapped = np.zeros_like(out[..., c], dtype=np.float32)
            else:
                mapped = np.clip((out[..., c] - l) / (h - l), 0.0, 1.0)
                mapped = mapped * 255.0
            res[..., c] = mapped
        return res
    elif method == 'no_map':
        clipped = np.clip(out, -3, 3)
        return clipped.astype(np.float32)
    elif method == 'astro_norm':
        norm = 6 * (out - (lo + hi) / 2.0) / (hi - lo)
        return norm.astype(np.float32)
    elif method == 'astro_rgb' or method == 'astro':
        # derive mean/std from lo/hi (lo = mean - 3*sigma)
        mean = (lo + hi) / 2.0
        std = (hi - lo) / 6.0
        # avoid zero std
        std = np.where(std <= 0, 1.0, std)
        # standardize per-channel and clip to [-3,3]
        # reshape for broadcasting: mean/std shape (C,) -> (1,1,C)
        mean_r = mean.reshape((1, 1, -1))
        std_r = std.reshape((1, 1, -1))
        standardized = (arr - mean_r) / std_r
        clipped = np.clip(standardized, -3.0, 3.0)
        u8 = ((clipped + 3.0) / 6.0) * 255.0
        return u8
    else:
        raise ValueError(f"Unknown method {method}")


def save_outputs(arr, out_prefix: Path, method: str, header=None):
    # arr may be float32 HxWxC or uint8 HxWxC where C==1 or 3
    png_path = out_prefix.with_suffix('.png')
    fits_path = out_prefix.with_suffix('.fits')

    # Create PIL image
    if arr.dtype == np.uint8:
        if arr.shape[-1] == 3:
            pil = Image.fromarray(arr, mode='RGB')
        else:
            pil = Image.fromarray(arr[..., 0], mode='L')
            if method == 'robust':
                # Match amg_fits_overlay.py: SAM receives a repeated grayscale
                # channel as RGB, and the saved input preview is RGB.
                pil = pil.convert('RGB')
    else:
        # to_pil_image will scale floats in [0,1] to 0-255; for general floats
        # it multiplies by 255 as well, which is acceptable for PNG preview.
        if arr.shape[-1] == 3:
            pil = to_pil_image(arr)
        else:
            # single-channel float -> make PIL grayscale
            single = arr[..., 0]
            pil = to_pil_image(single)

    pil.save(png_path)
    if method == 'robust':
        display_path = out_prefix.with_name(out_prefix.name + '_display').with_suffix('.png')
        Image.fromarray(np.flipud(np.array(pil))).save(display_path)
        print(f"Saved display-oriented PNG: {display_path}")

    # Save FITS: write the exact pixel values used for the PNG so FITS matches PIL
    pil_arr = np.array(pil)
    if pil_arr.ndim == 3:
        data_for_fits = pil_arr.astype(np.float32).transpose(2, 0, 1)
    else:
        data_for_fits = pil_arr.astype(np.float32)

    hdu = fits.PrimaryHDU(data=data_for_fits, header=header)
    hdu.writeto(fits_path, overwrite=True)

    print(f"Saved PNG: {png_path}, FITS: {fits_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', required=True, help='output prefix (no suffix)')
    p.add_argument('--method', choices=['robust', 'norm01', 'norm0255', 'no_map', 'astro_rgb', 'astro_norm'], default='norm01')
    p.add_argument('--low-percentile', type=float, default=0.1)
    p.add_argument('--high-percentile', type=float, default=99.5)
    p.add_argument('--size', type=int, default=None, help='optional target long side for ResizeLongestSide')
    args = p.parse_args()

    inp = Path(args.input)
    outp = Path(args.out)

    arr = read_fits_rgb(inp)
    means, stds, lo, hi = compute_clip_bounds(arr)
    print(f"means={means}, stds={stds}")

    mapped_np = clip_and_map(arr, lo, hi, args.method, args.low_percentile, args.high_percentile)
    header = None
    # optional resize via repo transform
    # if args.size is not None and ResizeLongestSide is not None:
    #     tr = ResizeLongestSide(args.size)
    #     mapped_np = tr.apply_image(mapped if isinstance(mapped, np.ndarray) else np.array(mapped))
    # else:
    #     mapped_np = mapped

    # ensure single-channel arrays have explicit channel axis (H,W,1)
    if isinstance(mapped_np, np.ndarray) and mapped_np.ndim == 2:
        mapped_np = mapped_np[..., None]
    save_outputs(mapped_np, outp, args.method, header=header)


if __name__ == '__main__':
    main()
