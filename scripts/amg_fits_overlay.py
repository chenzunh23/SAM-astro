#!/usr/bin/env python3
"""Compatibility wrapper for the cleaned FITS AMG implementation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from amg_fits_core import main


if __name__ == "__main__":
    main()
    raise SystemExit

'''
"""
Run SAM automatic mask generation on FITS images and save:
1) label-map FITS (all masks, same HxW)
2) overlay image(s) (PNG/TIFF)
3) multi-extension FITS that keeps original float32 image unchanged

This script is designed for astronomical 2D FITS images (e.g. 1536x1536 float32).
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import csv
import json

try:
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "astropy is required. Install with: pip install astropy"
    ) from exc

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Pillow is required. Install with: pip install pillow"
    ) from exc

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SAM segmentation for FITS images. Converts float FITS to 8-bit RGB for SAM, "
            "while preserving original float32 FITS in outputs."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        nargs='+',  # Allow multiple input files
        help="Input FITS file(s), or directory containing FITS files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_h",
        choices=["default", "vit_h", "vit_l", "vit_b"],
        help="SAM model type.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/chenzunhao/sam_vit_h_4b8939.pth",
        help="Path to SAM checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    parser.add_argument(
        "--hdu",
        type=int,
        default=0,
        help="FITS HDU index to read image from (default: 0).",
    )
    parser.add_argument(
        "--low-percentile",
        type=float,
        default=0.1,
        help="Lower percentile for robust intensity scaling to SAM input.",
    )
    parser.add_argument(
        "--high-percentile",
        type=float,
        default=99.5,
        help="Upper percentile for robust intensity scaling to SAM input.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Alpha for filled mask overlay in [0,1].",
    )
    parser.add_argument(
        "--boundary-color",
        type=int,
        nargs=3,
        default=[255, 255, 255],
        help="Boundary color RGB for overlay images.",
    )
    parser.add_argument(
        "--scaling-mode",
        type=str,
        default="robust",
        choices=["robust", "lupton_rgb", "linear_rgb", "lupton", "astro_rgb"],
        help="Scaling mode for input images (default: robust).",
    )
    parser.add_argument(
        "--astro-rgb-mode",
        type=str,
        default="none",
        choices=["none", "astro_rgb", "astro_rgb1", "astro_rgb2"],
        help=(
            "Shared astro normalization mode used both for SAM pixel_mean/std and "
            "for input mapping. none uses original RGB normalization; astro_rgb means [-3σ,3σ]->[0,1]; astro_rgb1 means [-σ,3σ]->[0,1]."
        ),
    )
    parser.add_argument(
        "--astro-rgb-low-sigma",
        type=float,
        default=None,
        help=(
            "Override the lower clipping bound for astro_rgb2-like normalization. "
            "For example 6 means [mean-6*sigma, mean+3*sigma]."
        ),
    )
    parser.add_argument(
        "--no-save-fits",
        action="store_true",
        help="Do not keep SAM label-map/bundle FITS outputs; image and metadata outputs are still saved.",
    )

    # AMG options
    parser.add_argument("--points-per-side", type=int, default=64)
    parser.add_argument("--points-per-batch", type=int, default=128)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-overlap-ratio", type=float, default=512 / 1500)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=0)
    parser.add_argument(
        "--max-mask-area-ratio",
        type=float,
        default=0.15, # 0.5 for 256*256
        help=(
            "Drop masks whose area exceeds this fraction of full image area. "
            "Set to 1.0 to disable."
        ),
    )

    return parser.parse_args()


def collect_fits_paths(path_str: str) -> List[Path]:
    p = Path(path_str)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Input path not found: {p}")

    exts = {".fits", ".fit", ".fts", ".fits.gz"}
    files: List[Path] = []
    for child in sorted(p.iterdir()):
        name_lower = child.name.lower()
        if child.is_file() and any(name_lower.endswith(ext) for ext in exts):
            files.append(child)
    if not files:
        raise FileNotFoundError(f"No FITS files found in directory: {p}")
    return files


def robust_to_uint8(image_f32: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    finite = np.isfinite(image_f32)
    if not np.any(finite):
        raise ValueError("Image has no finite values.")

    vals = image_f32[finite]
    lo = np.percentile(vals, low_pct)
    hi = np.percentile(vals, high_pct)
    if hi <= lo:
        hi = lo + 1e-6

    clipped = np.clip(image_f32, lo, hi)
    norm = (clipped - lo) / (hi - lo)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    return (norm * 255.0).astype(np.uint8)

def lupton_to_uint8(image_f32: np.ndarray, low_pct: float, high_pct: float, Q: float = 10.0) -> np.ndarray:
    # Lupton et al. 2004 "Preparing Red-Green-Blue Images from CCD Data"
    # https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L
    # For grayscale input we use the same arcsinh transfer curve as the RGB recipe,
    # with a robust black point from the lower percentile and a stretch based on the
    # percentile span.
    if image_f32.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image_f32.shape}")

    finite = np.isfinite(image_f32)
    if not np.any(finite):
        raise ValueError("Image has no finite values.")

    vals = image_f32[finite].astype(np.float64)

    if not (0.0 <= low_pct < high_pct <= 100.0):
        raise ValueError("Require 0 <= low_pct < high_pct <= 100.")

    minimum = float(np.percentile(vals, low_pct))
    upper = float(np.percentile(vals, high_pct))
    stretch = upper - minimum

    if not np.isfinite(stretch) or stretch <= 0:
        stretch = float(np.nanmax(vals) - minimum)

    if not np.isfinite(stretch) or stretch <= 0:
        stretch = 1e-6

    # Shift to black point and normalize by stretch.
    x = (image_f32.astype(np.float64) - minimum) / stretch

    # Clip negative values to 0; keep bright values > 1 so arcsinh can compress them.
    x = np.clip(x, 0.0, None)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if Q is None or Q <= 0:
            y = np.clip(x, 0.0, 1.0)
        else:
            # Lupton-style nonlinearity, normalized so x=1 maps to y=1.
            y = np.arcsinh(Q * x) / np.arcsinh(Q)

    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    y = np.clip(y, 0.0, 1.0)

    return np.round(y * 255.0).astype(np.uint8)

def estimate_background(image):
    """
    简单稳健背景估计：median + MAD
    """
    finite = np.isfinite(image)
    vals = image[finite]

    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    sigma = 1.4826 * mad

    return med, sigma


def subtract_background(image):
    bg, _ = estimate_background(image)
    return image - bg

def make_lupton_rgb(r, g, b, low_pct=1.0, high_pct=99.5, Q=10.0):
    """
    三波段 Lupton（更接近 DeepDISC）
    """

    # 背景扣除
    r = subtract_background(r)
    g = subtract_background(g)
    b = subtract_background(b)

    # 总强度（关键！）
    I = (r + g + b) / 3.0

    finite = np.isfinite(I)
    vals = I[finite]

    minimum = np.percentile(vals, low_pct)
    upper = np.percentile(vals, high_pct)
    stretch = upper - minimum

    if stretch <= 0 or not np.isfinite(stretch):
        stretch = np.max(vals) - minimum
    if stretch <= 0 or not np.isfinite(stretch):
        stretch = 1e-6

    # Lupton核心
    scaled_I = np.clip((I - minimum) / stretch, 0.0, None)

    if Q > 0:
        f = np.arcsinh(Q * scaled_I) / np.arcsinh(Q)
    else:
        f = np.clip(scaled_I, 0.0, 1.0)

    # 避免除零
    I_safe = np.where(I <= 0, 1e-6, I)

    R = r * f / I_safe
    G = g * f / I_safe
    B = b * f / I_safe

    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb.astype(np.float32)

def get_mean_std(r, g, b, mode="naive"):
    # Placeholder for a more sophisticated method to estimate mean and std for each channel, e.g. using robust statistics or fitting a model to the histogram.
    # Currently only consider one set of FITS images
    r_no_inf = r[np.isfinite(r)]
    g_no_inf = g[np.isfinite(g)]
    b_no_inf = b[np.isfinite(b)]
    if mode == "naive":
        r_mean = np.mean(r_no_inf)
        g_mean = np.mean(g_no_inf)
        b_mean = np.mean(b_no_inf)
        r_std = np.std(r_no_inf)
        g_std = np.std(g_no_inf)
        b_std = np.std(b_no_inf)
        mean_v = np.expand_dims(np.array([r_mean, g_mean, b_mean]), axis=(0,1))
        std_v = np.expand_dims(np.array([r_std, g_std, b_std]), axis=(0,1))
    elif mode == "bgd":
        r_med, r_std = estimate_background(r)
        g_med, g_std = estimate_background(g)
        b_med, b_std = estimate_background(b)
        mean_v = np.expand_dims(np.array([r_med, g_med, b_med]), axis=(0,1))
        std_v = np.expand_dims(np.array([r_std, g_std, b_std]), axis=(0,1))
    elif mode == "sigmaclip":
        r_mean, r_med, r_std = sigma_clipped_stats(r_no_inf, sigma=3.0)
        g_mean, g_med, g_std = sigma_clipped_stats(g_no_inf, sigma=3.0)
        b_mean, b_med, b_std = sigma_clipped_stats(b_no_inf, sigma=3.0)
        print(f'[DEBUG] Sigma-clipped stats: R mean={r_mean:.2f}, std={r_std:.2f}, med={r_med:.2f}; G mean={g_mean:.2f}, std={g_std:.2f}, med={g_med:.2f}; B mean={b_mean:.2f}, std={b_std:.2f}, med={b_med:.2f}')
        mean_v = np.expand_dims(np.array([r_mean, g_mean, b_mean]), axis=(0,1))
        std_v = np.expand_dims(np.array([r_std, g_std, b_std]), axis=(0,1))
    else:
        raise NotImplementedError(f"Unknown mean/std estimation mode: {mode}")
    return mean_v, std_v

def make_astro_rgb(r, g, b):
    # DEBUG
    r_mean = np.mean(r[np.isfinite(r)])
    g_mean = np.mean(g[np.isfinite(g)])
    b_mean = np.mean(b[np.isfinite(b)])
    r_std = np.std(r[np.isfinite(r)])
    g_std = np.std(g[np.isfinite(g)])
    b_std = np.std(b[np.isfinite(b)])
    print(f"[ASTRO] Basic RGB means: R={r_mean:.2f}, G={g_mean:.2f}, B={b_mean:.2f}")
    print(f"[ASTRO] Basic RGB stds: R={r_std:.2f}, G={g_std:.2f}, B={b_std:.2f}")
    xsize = min(r.shape[1], g.shape[1], b.shape[1], 1024)
    ysize = min(r.shape[0], g.shape[0], b.shape[0], 1024)
    # crop bottom-left region (preserve consistent origin)
    # y0 = r.shape[0] - ysize
    y0 = 0
    # x0 = 0
    x0 = r.shape[1] - xsize
    r1 = r[y0 : y0 + ysize, x0 : x0 + xsize]
    g1 = g[y0 : y0 + ysize, x0 : x0 + xsize]
    b1 = b[y0 : y0 + ysize, x0 : x0 + xsize]
    rgb = np.stack([r1, g1, b1], axis=-1)
    return rgb.astype(np.float32), (y0, x0)


def astro_rgb_clip_params(
    mean_rgb: np.ndarray,
    std_rgb: np.ndarray,
    astro_rgb_mode: str,
    low_sigma_override: float = None,
):
    if astro_rgb_mode == "none":
        clip_lo = None
        clip_hi = None
        scale = None
        offset = None
    elif astro_rgb_mode == "astro_rgb":
        clip_lo = mean_rgb - 3.0 * std_rgb
        clip_hi = mean_rgb + 3.0 * std_rgb
        scale = 6.0
        offset = 0.5
    elif astro_rgb_mode == "astro_rgb1":
        clip_lo = mean_rgb - 1.0 * std_rgb
        clip_hi = mean_rgb + 3.0 * std_rgb
        scale = 4.0
        offset = 0.25
    elif astro_rgb_mode == "astro_rgb2":
        low_sigma = 5.0 if low_sigma_override is None else float(low_sigma_override)
        clip_lo = mean_rgb - low_sigma * std_rgb
        clip_hi = mean_rgb + 3.0 * std_rgb
        scale = low_sigma + 3.0
        offset = low_sigma / scale
    else:
        raise ValueError(f"Unknown astro_rgb_mode: {astro_rgb_mode}")
    return clip_lo, clip_hi, scale, offset

def make_linear_rgb(r, g, b, low_pct=0.5, high_pct=99.5):
    """
    简单线性三波段 RGB 组合
    """
    r_u8 = robust_to_uint8(r, low_pct, high_pct)
    g_u8 = robust_to_uint8(g, low_pct, high_pct)
    b_u8 = robust_to_uint8(b, low_pct, high_pct)

    rgb_u8 = np.stack([r_u8, g_u8, b_u8], axis=-1)
    return rgb_u8.astype(np.float32) / 255.0

def make_label_map(masks: List[Dict], h: int, w: int) -> np.ndarray:
    # Sort by area desc so larger masks get stable IDs first.
    masks_sorted = sorted(masks, key=lambda m: float(m.get("area", 0)), reverse=True)
    label_map = np.zeros((h, w), dtype=np.int32)

    obj_id = 1
    for m in masks_sorted:
        seg = m["segmentation"]
        if seg.shape != (h, w):
            continue
        seg_bool = seg.astype(bool)
        # Keep first assignment to avoid unstable overwrites.
        assign = seg_bool & (label_map == 0)
        if np.any(assign):
            label_map[assign] = obj_id
            obj_id += 1
    return label_map


def _bbox_xywh_from_seg(seg: np.ndarray) -> List[int]:
    ys, xs = np.where(seg)
    if ys.size == 0 or xs.size == 0:
        return [0, 0, 0, 0]
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def filter_large_masks(
    masks: List[Dict], h: int, w: int, max_mask_area_ratio: float
) -> Tuple[List[Dict], int]:
    if max_mask_area_ratio >= 1.0:
        return masks, 0

    if max_mask_area_ratio <= 0.0:
        raise ValueError("--max-mask-area-ratio must be in (0, 1].")

    area_limit = float(h * w) * max_mask_area_ratio
    print(f'[INFO] Filtering masks larger than {max_mask_area_ratio:.2%} of image area ({area_limit:.0f} pixels)')
    kept: List[Dict] = []
    removed = 0

    for m in masks:
        seg = m.get("segmentation", None)
        if isinstance(seg, np.ndarray) and seg.shape == (h, w):
            area = int(np.count_nonzero(seg))
            m["area"] = area
            m["bbox"] = _bbox_xywh_from_seg(seg.astype(bool))
        else:
            area = int(m.get("area", 0))

        if area > area_limit:
            removed += 1
            continue

        kept.append(m)

    return kept, removed


def label_boundaries(label_map: np.ndarray) -> np.ndarray:
    # Boundary where neighboring labels differ (excluding background-only region).
    center = label_map
    up = np.zeros_like(center)
    up[1:, :] = center[:-1, :]
    down = np.zeros_like(center)
    down[:-1, :] = center[1:, :]
    left = np.zeros_like(center)
    left[:, 1:] = center[:, :-1]
    right = np.zeros_like(center)
    right[:, :-1] = center[:, 1:]

    b = (
        ((center != up) | (center != down) | (center != left) | (center != right))
        & (center > 0)
    )
    return b


def colorize_labels(label_map: np.ndarray, seed: int = 1234) -> np.ndarray:
    n = int(label_map.max())
    lut = np.zeros((n + 1, 3), dtype=np.uint8)
    if n > 0:
        rng = np.random.default_rng(seed)
        lut[1:] = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    return lut[label_map]


def blend_overlay(base_gray_u8: np.ndarray, label_map: np.ndarray, alpha: float, boundary_rgb: Tuple[int, int, int]) -> np.ndarray:
    base_rgb = np.repeat(base_gray_u8[..., None], 3, axis=2)
    seg_rgb = colorize_labels(label_map)

    overlay = base_rgb.astype(np.float32)
    mask = label_map > 0
    if np.any(mask):
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * seg_rgb[mask].astype(np.float32)

    boundary = label_boundaries(label_map)
    if np.any(boundary):
        overlay[boundary] = np.array(boundary_rgb, dtype=np.float32)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_outputs(
    in_path: Path,
    out_dir: Path,
    original_f32: np.ndarray,
    header,
    sam_rgb: np.ndarray,
    label_map: np.ndarray,
    overlay_u8: np.ndarray,
    n_masks: int,
    masks: List[Dict],
    output_mode: str,
    scaling_mode: str,
    save_fits: bool = True,
) -> None:
    stem = in_path.name
    for suf in [".fits", ".fit", ".fts", ".fits.gz"]:
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
            break

    out_dir.mkdir(parents=True, exist_ok=True)

    if save_fits:
        label_fits = out_dir / f"{stem}_sam_labelmap.fits"
        fits.writeto(label_fits, label_map.astype(np.int32), header=header, overwrite=True)

        # Multi-extension FITS keeps the original float image untouched.
        bundle_fits = out_dir / f"{stem}_sam_bundle.fits"
        hdu0 = fits.PrimaryHDU(data=original_f32.astype(np.float32), header=header)
        hdu0.header["HIERARCH SAM.NMASK"] = int(n_masks)
        hdu0.header["HIERARCH SAM.DESC"] = "Original image unchanged"

        hdu1 = fits.ImageHDU(data=label_map.astype(np.int32), name="LABELMAP")
        hdu1.header["HIERARCH SAM.DESC"] = "Label ID map, 0=background"

        hdu2 = fits.ImageHDU(data=np.transpose(overlay_u8, (2, 0, 1)), name="OVERLAYRGB")
        hdu2.header["HIERARCH SAM.DESC"] = "RGB overlay for quicklook"

        hdu3 = fits.ImageHDU(data=sam_rgb.astype(np.uint8), name="SAMINPUT8")
        hdu3.header["HIERARCH SAM.DESC"] = "8-bit grayscale used to build SAM RGB input"

        fits.HDUList([hdu0, hdu1, hdu2, hdu3]).writeto(bundle_fits, overwrite=True)

    lupton_path = out_dir / f"{stem}_{scaling_mode}_input.png"
    # Invert sam_rgb back to original orientation
    sam_rgb_flipped = np.flipud(sam_rgb)
    # if scaling_mode == "astro_rgb":
    #     # For astro_rgb, sam_rgb is already in float32 format. We need to convert it to uint8 for saving as PNG.
    #     min_val = np.nanmin(sam_rgb_flipped)
    #     sam_rgb_u8 = np.round(np.clip((sam_rgb_flipped - min_val) / (np.nanmax(sam_rgb_flipped) - min_val) * 255.0, 0, 255)).astype(np.uint8)
    #     sam_rgb_flipped = sam_rgb_u8
    # `sam_rgb` may be a 2D grayscale image or a 3-channel image. Handle both.
    if sam_rgb.ndim == 2:
        Image.fromarray(sam_rgb_flipped, mode="L").convert("RGB").save(lupton_path)
    elif sam_rgb.ndim == 3 and sam_rgb.shape[2] == 3:
        Image.fromarray(sam_rgb_flipped, mode="RGB").save(lupton_path)
    else:
        # Fallback: try to save as grayscale
        Image.fromarray(sam_rgb_flipped).save(lupton_path)

    overlay_u8_flipped = np.flipud(overlay_u8)
    png_path = out_dir / f"{stem}_{scaling_mode}_overlay.png"
    Image.fromarray(overlay_u8_flipped, mode="RGB").save(png_path)

    # Save TIFF quicklook in a broadly compatible RGB format.
    tiff_path = out_dir / f"{stem}_{scaling_mode}_overlay.tif"
    Image.fromarray(overlay_u8_flipped, mode="RGB").save(tiff_path, compression="tiff_lzw")

    if output_mode == "binary_mask":
        # Save masks metadata to CSV in SAM's original format
        csv_path = out_dir / f"{stem}_{scaling_mode}_metadata.csv"
        header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header.split(","))
            for i, mask in enumerate(masks):
                row = [
                    i + 1,
                    mask.get("area", 0),
                    *mask.get("bbox", [0, 0, 0, 0]),
                    *mask.get("point_coords", [[0, 0]])[0],
                    mask.get("predicted_iou", 0),
                    mask.get("stability_score", 0),
                    *mask.get("crop_box", [0, 0, 0, 0]),
                ]
                writer.writerow(row)
    else:
        # Convert masks to JSON serializable format
        serializable_masks = convert_masks_to_serializable(masks)

        # Save masks metadata to JSON
        json_path = out_dir / f"{stem}_{scaling_mode}_masks.json"
        with open(json_path, "w") as f:
            json.dump(serializable_masks, f, indent=4)

        # Save masks metadata to CSV in SAM's original format
        csv_path = out_dir / f"{stem}_{scaling_mode}_metadata.csv"
        header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header.split(","))
            for i, mask in enumerate(serializable_masks):
                row = [
                    i + 1,
                    mask.get("area", 0),
                    *mask.get("bbox", [0, 0, 0, 0]),
                    *mask.get("point_coords", [[0, 0]])[0],
                    mask.get("predicted_iou", 0),
                    mask.get("stability_score", 0),
                    *mask.get("crop_box", [0, 0, 0, 0]),
                ]
                writer.writerow(row)


def convert_masks_to_serializable(masks: List[Dict]) -> List[Dict]:
    """Convert masks to JSON serializable format."""
    serializable_masks = []
    for mask in masks:
        serializable_mask = {}
        for key, value in mask.items():
            if isinstance(value, np.ndarray):
                serializable_mask[key] = value.tolist()
            else:
                serializable_mask[key] = value
        serializable_masks.append(serializable_mask)
    return serializable_masks


def run_one(
    generator: SamAutomaticMaskGenerator,
    fits_path: Path,
    out_dir: Path,
    hdu: int,
    low_pct: float,
    high_pct: float,
    alpha: float,
    boundary_rgb: Tuple[int, int, int],
    scaling_mode: str,
    astro_rgb_mode: str,
    astro_rgb_low_sigma: float,
    max_mask_area_ratio: float,
    save_fits: bool,
) -> None:
    with fits.open(fits_path, memmap=True) as hdul:
        if hdu >= len(hdul):
            raise IndexError(f"HDU index {hdu} out of range for file: {fits_path}")
        data = hdul[hdu].data
        if data is None:
            raise ValueError(f"No image data in HDU {hdu}: {fits_path}")
        if data.ndim != 2:
            raise ValueError(f"Only 2D FITS supported. Got shape {data.shape} in {fits_path}")

        original_f32 = np.asarray(data, dtype=np.float32)
        header = hdul[hdu].header.copy()

    sam_rgb = None
    sam_u8 = None

    if scaling_mode == "lupton_rgb":
        # If the FITS contains three channels (3,H,W) or (H,W,3), use make_lupton_rgb
        if original_f32.ndim == 3 and original_f32.shape[0] == 3:
            r, g, b = original_f32[0].astype(np.float32), original_f32[1].astype(np.float32), original_f32[2].astype(np.float32)
            rgb_float = make_lupton_rgb(r, g, b, low_pct, high_pct)
            sam_rgb = np.round(np.clip(rgb_float * 255.0, 0, 255)).astype(np.uint8)
            # derive a grayscale 8-bit version for overlay/saving
            sam_u8 = np.round(np.dot(rgb_float, [0.2989, 0.5870, 0.1140]) * 255.0).astype(np.uint8)
        elif original_f32.ndim == 3 and original_f32.shape[2] == 3:
            r, g, b = original_f32[..., 0].astype(np.float32), original_f32[..., 1].astype(np.float32), original_f32[..., 2].astype(np.float32)
            rgb_float = make_lupton_rgb(r, g, b, low_pct, high_pct)
            sam_rgb = np.round(np.clip(rgb_float * 255.0, 0, 255)).astype(np.uint8)
            sam_u8 = np.round(np.dot(rgb_float, [0.2989, 0.5870, 0.1140]) * 255.0).astype(np.uint8)
        else:
            # No explicit three-channel data in this file: fall back to single-band Lupton curve
            sam_u8 = lupton_to_uint8(original_f32, low_pct, high_pct)
            sam_rgb = np.repeat(sam_u8[..., None], 3, axis=2)
    else:
        sam_u8 = robust_to_uint8(original_f32, low_pct, high_pct)
        sam_rgb = np.repeat(sam_u8[..., None], 3, axis=2)

    # Ensure sam_rgb is HxWx3 uint8
    if sam_rgb is None:
        raise RuntimeError("Failed to construct SAM RGB input")

    masks = generator.generate(sam_rgb)
    masks, removed_large = filter_large_masks(
        masks, sam_rgb.shape[0], sam_rgb.shape[1], max_mask_area_ratio
    )
    label_map = make_label_map(masks, original_f32.shape[0], original_f32.shape[1])
    overlay_u8 = blend_overlay(sam_u8, label_map, alpha=alpha, boundary_rgb=boundary_rgb)

    save_outputs(
        fits_path,
        out_dir,
        original_f32,
        header,
        sam_rgb,
        label_map,
        overlay_u8,
        n_masks=len(masks),
        masks=masks,
        output_mode="binary_mask",
        scaling_mode=scaling_mode,
        save_fits=save_fits,
    )

    print(
        f"[OK] {fits_path.name}: {len(masks)} masks "
        f"(removed_large={removed_large}, max_area_ratio={max_mask_area_ratio})"
    )


def run_one_triplet(
    generator: SamAutomaticMaskGenerator,
    fits_paths: List[Path],
    out_dir: Path,
    hdu: int,
    low_pct: float,
    high_pct: float,
    alpha: float,
    boundary_rgb: Tuple[int, int, int],
    scaling_mode: str,
    astro_rgb_mode: str,
    astro_rgb_low_sigma: float,
    max_mask_area_ratio: float,
    save_fits: bool,
) -> None:
    if len(fits_paths) != 3:
        raise ValueError("run_one_triplet requires exactly 3 FITS paths")

    if scaling_mode not in ["lupton_rgb", "linear_rgb", "astro_rgb"]:
        raise ValueError(f"Unsupported scaling mode for triplet: {scaling_mode}")

    # Read the three band images and ensure consistent shapes
    imgs = []
    headers = []
    for p in fits_paths:
        with fits.open(p, memmap=True) as hdul:
            if hdu >= len(hdul):
                raise IndexError(f"HDU index {hdu} out of range for file: {p}")
            data = hdul[hdu].data
            if data is None:
                raise ValueError(f"No image data in HDU {hdu}: {p}")
            if data.ndim != 2:
                raise ValueError(f"Only 2D FITS supported. Got shape {data.shape} in {p}")
            imgs.append(np.asarray(data, dtype=np.float32))
            headers.append(hdul[hdu].header.copy())

    r, g, b = imgs[0], imgs[1], imgs[2]
    if r.shape != g.shape or r.shape != b.shape:
        raise ValueError("Input band images must have identical shapes")

    if scaling_mode == "lupton_rgb":
        rgb_float = make_lupton_rgb(r, g, b, low_pct, high_pct)
    elif scaling_mode == "astro_rgb":
        rgb_float, (crop_y0, crop_x0) = make_astro_rgb(r, g, b)
    else:
        rgb_float = make_linear_rgb(r, g, b, low_pct, high_pct)

    if scaling_mode == "astro_rgb":
        mean_rgb, std_rgb = get_mean_std(r, g, b)
        clip_lo, clip_hi, scale, offset = astro_rgb_clip_params(
            mean_rgb,
            std_rgb,
            astro_rgb_mode,
            low_sigma_override=astro_rgb_low_sigma,
        )
        if astro_rgb_mode == "none":
            print(f'[DEBUG] Astro RGB mode=none, raw crop mean={mean_rgb}, std={std_rgb}')
            sam_rgb = (rgb_float - mean_rgb) / std_rgb # For debug purpose only
            print(f"[DEBUG] Astro RGB mode=none, raw crop mean={np.mean(sam_rgb)}")
        else:
            sam_rgb_clipped = np.clip(rgb_float, clip_lo, clip_hi)
            sam_rgb = (sam_rgb_clipped - mean_rgb) / (scale * std_rgb) + offset
            sam_rgb = np.clip(sam_rgb, 0, 1)
            sam_rgb = np.round(sam_rgb * 255.0).astype(np.uint8)
            print(
                f"[DEBUG] Astro RGB mode={astro_rgb_mode}, after transform: "
                f"mean={np.mean(sam_rgb)}, max={np.max(sam_rgb)}, min={np.min(sam_rgb)}"
            )
    else:
        sam_rgb = np.round(np.clip(rgb_float * 255.0, 0, 255)).astype(np.uint8)
    
    masks = generator.generate(sam_rgb)

    # If using astro_rgb we generated masks on a cropped float image; map each
    # mask's segmentation back into the full-resolution frame using crop offsets.
    if scaling_mode == "astro_rgb":
        # crop dims
        ch, cw = sam_rgb.shape[0], sam_rgb.shape[1]
        full_h, full_w = r.shape[0], r.shape[1]
        for m in masks:
            seg = m.get("segmentation")
            if seg is None:
                continue
            # only expand if shapes mismatch
            if seg.shape != (full_h, full_w):
                full_seg = np.zeros((full_h, full_w), dtype=seg.dtype)
                full_seg[crop_y0 : crop_y0 + ch, crop_x0 : crop_x0 + cw] = seg
                m["segmentation"] = full_seg

        # For overlay saving, compute a Lupton-based uint8 quicklook for the full image
        lupton_rgb = make_lupton_rgb(r, g, b, low_pct, high_pct)
        sam_u8 = np.round(np.dot(lupton_rgb, [0.2989, 0.5870, 0.1140]) * 255.0).astype(np.uint8)
        sam_rgb = np.round(np.clip(lupton_rgb * 255.0, 0, 255)).astype(np.uint8)
    else:
        sam_u8 = np.round(np.dot(rgb_float, [0.2989, 0.5870, 0.1140]) * 255.0).astype(np.uint8)

    masks, removed_large = filter_large_masks(
        masks, sam_rgb.shape[0], sam_rgb.shape[1], max_mask_area_ratio
    )
    label_map = make_label_map(masks, r.shape[0], r.shape[1])
    overlay_u8 = blend_overlay(sam_u8, label_map, alpha=alpha, boundary_rgb=boundary_rgb)

    # Build a synthetic in_path for naming using the three stems
    stem = f"{fits_paths[0].stem}_{fits_paths[1].stem}_{fits_paths[2].stem}_{scaling_mode}"
    synthetic_path = Path(stem + ".fits")

    # Stack original float bands into (3,H,W) to preserve all channels in bundle
    original_stack = np.stack([r, g, b], axis=0)

    # Use the first header as basis, but don't overwrite shape-sensitive keywords
    header = headers[0]

    save_outputs(
        synthetic_path,
        out_dir,
        original_stack,
        header,
        sam_rgb,
        label_map,
        overlay_u8,
        n_masks=len(masks),
        masks=masks,
        output_mode="binary_mask",
        scaling_mode=scaling_mode,
        save_fits=save_fits,
    )

    print(
        f"[OK] {stem}: {len(masks)} masks "
        f"(removed_large={removed_large}, max_area_ratio={max_mask_area_ratio})"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    input_paths = []
    for path_str in args.input:
        input_paths.extend(collect_fits_paths(path_str))

    astro_rgb_mode = args.astro_rgb_mode

    print("Loading SAM model...")
    sam = sam_model_registry[args.model_type](
        checkpoint=args.checkpoint,
        astro_rgb_mode=astro_rgb_mode,
        astro_rgb_low_sigma=args.astro_rgb_low_sigma,
    )
    _ = sam.to(device=args.device)

    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
        "output_mode": "binary_mask",
    }
    generator = SamAutomaticMaskGenerator(sam, **amg_kwargs)

    boundary_rgb = tuple(int(x) for x in args.boundary_color)

    if args.scaling_mode == "lupton_rgb" or args.scaling_mode == "linear_rgb" or args.scaling_mode == "astro_rgb":
        # Expect input FITS to be provided in groups of three (R,G,B).
        if len(input_paths) % 3 != 0:
            raise ValueError("When using --scaling-mode lupton_rgb or linear_rgb or astro_rgb, provide input FITS in groups of three (R G B).")
        for i in range(0, len(input_paths), 3):
            triplet = input_paths[i : i + 3]
            run_one_triplet(
                generator=generator,
                fits_paths=triplet,
                out_dir=output_dir,
                hdu=args.hdu,
                low_pct=args.low_percentile,
                high_pct=args.high_percentile,
                alpha=float(args.overlay_alpha),
                boundary_rgb=boundary_rgb,
                scaling_mode=args.scaling_mode,
                astro_rgb_mode=astro_rgb_mode,
                astro_rgb_low_sigma=args.astro_rgb_low_sigma,
                max_mask_area_ratio=args.max_mask_area_ratio,
                save_fits=not args.no_save_fits,
            )
    else:
        for p in input_paths:
            run_one(
                generator=generator,
                fits_path=p,
                out_dir=output_dir,
                hdu=args.hdu,
                low_pct=args.low_percentile,
                high_pct=args.high_percentile,
                alpha=float(args.overlay_alpha),
                boundary_rgb=boundary_rgb,
                scaling_mode=args.scaling_mode,
                astro_rgb_mode=astro_rgb_mode,
                astro_rgb_low_sigma=args.astro_rgb_low_sigma,
                max_mask_area_ratio=args.max_mask_area_ratio,
                save_fits=not args.no_save_fits,
            )

    print("Done.")


if __name__ == "__main__":
    main()
'''
