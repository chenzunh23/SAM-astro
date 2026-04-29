#!/usr/bin/env python3
"""Analyze pixel distributions after AMG image preprocessing.

This script reproduces the image path used by AMG before image encoder input:
1) resize longest side to model image size (same as ResizeLongestSide.apply_image)
2) normalize with SAM pixel mean/std
3) optional square padding to img_size (same as Sam.preprocess)

It processes images in batches and supports GPU execution for fast histogram stats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segment_anything.build_sam import convert_astro
from segment_anything.utils.transforms import ResizeLongestSide


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ASTRO_RGB2_LOW_SIGMAS = (0.5, 1.0, 3.0, 4.0, 5.0, 6.0)


@dataclass
class ChannelStats:
    count: int = 0
    sum: float = 0.0
    sq_sum: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        self.count += int(values.numel())
        self.sum += float(values.sum().item())
        self.sq_sum += float((values * values).sum().item())

    def mean(self) -> float:
        return self.sum / max(self.count, 1)

    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        mean = self.mean()
        var = self.sq_sum / self.count - mean * mean
        return math.sqrt(max(var, 0.0))


def normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF computed via erf."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def gaussian_expected_counts_with_tail_folding(edges: np.ndarray, total_count: int) -> np.ndarray:
    """Expected per-bin counts under N(0,1).

    Interior bins use exact interval probability. Left/right tail mass outside
    [edges[0], edges[-1]] is folded into first/last bins for direct comparison.
    """
    cdf_vals = normal_cdf(edges)
    probs = cdf_vals[1:] - cdf_vals[:-1]
    probs[0] += cdf_vals[0]
    probs[-1] += 1.0 - cdf_vals[-1]
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / max(probs.sum(), 1e-12)
    return probs * float(total_count)


def safe_label(value: str) -> str:
    return value.replace(".", "p").replace("-", "m").replace("+", "p")

def format_bin_width_display(bin_width: float | Sequence[float]) -> str:
    if isinstance(bin_width, (int, float)):
        return f"{float(bin_width):.2f}"
    return "[" + ", ".join(f"{float(v):.2f}" for v in bin_width) + "]"


def mode_label(mode: str, low_sigma: float | None) -> str:
    if mode == "none":
        return "astro_rgb_none"
    if mode == "astro_rgb2":
        low = 5.0 if low_sigma is None else float(low_sigma)
        return f"astro_rgb2_low{safe_label(f'{low:g}')}"
    return mode


def load_fits_2d(path: Path, hdu: int) -> np.ndarray:
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError("Reading FITS inputs requires astropy. Install astropy or use image inputs.") from exc

    with fits.open(path, memmap=True) as hdul:
        if hdu >= len(hdul):
            raise IndexError(f"HDU index {hdu} out of range for file: {path}")
        data = hdul[hdu].data
        if data is None:
            raise ValueError(f"No image data in HDU {hdu}: {path}")
        arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D FITS image in {path}, got shape {arr.shape}")
    return arr


def crop_bottom_right(arr: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = arr.shape
    ysize = min(h, crop_size)
    xsize = min(w, crop_size)
    return arr[h - ysize : h, w - xsize : w]


def finite_values(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("Image channel has no finite values.")
    return values[finite].astype(np.float64, copy=False)


def robust_location_scale(values: np.ndarray, mode: str) -> Tuple[float, float, float, float, float, float]:
    vals = finite_values(values)
    raw_mean = float(np.mean(vals))
    raw_median = float(np.median(vals))
    raw_sigma = float(np.std(vals))
    if not np.isfinite(raw_sigma) or raw_sigma <= 0:
        raw_sigma = 1.0
    bright_limit = raw_median + 3.0 * raw_sigma
    bright_clipped = np.minimum(vals, bright_limit)

    if mode == "bgd":
        center = float(np.median(bright_clipped))
        mad = float(np.median(np.abs(bright_clipped - center)))
        sigma = 1.4826 * mad
        mean = float(np.mean(bright_clipped))
    elif mode == "sigmaclip":
        try:
            from astropy.stats import sigma_clipped_stats
        except ImportError as exc:
            raise RuntimeError("Using --astro-stats-mode sigmaclip requires astropy.") from exc
        mean, median, std = sigma_clipped_stats(bright_clipped, sigma=3.0, maxiters=None)
        center = float(median)
        sigma = float(std)
        mean = float(mean)
    elif mode == "legacy":
        mean = float(np.mean(vals))
        center = mean
        sigma = float(np.std(bright_clipped))
    else:
        raise ValueError(f"Unknown astro stats mode: {mode}")

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(vals))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return mean, center, sigma, raw_mean, raw_median, raw_sigma


def estimate_rgb_location_scale(rgb: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.zeros(3, dtype=np.float32)
    sigmas = np.ones(3, dtype=np.float32)
    raw_medians = np.zeros(3, dtype=np.float32)
    raw_sigmas = np.ones(3, dtype=np.float32)
    for c, name in enumerate(("R", "G", "B")):
        mean, center, sigma, raw_mean, raw_median, raw_sigma = robust_location_scale(rgb[..., c], mode)
        centers[c] = mean
        sigmas[c] = sigma
        raw_medians[c] = raw_median
        raw_sigmas[c] = raw_sigma
        n_bright = int(np.count_nonzero(finite_values(rgb[..., c]) > raw_median + 3.0 * raw_sigma))
        print(
            f"[ASTRO {mode} {name}] raw_mean={raw_mean:.6g} raw_median={raw_median:.6g} "
            f"raw_sigma={raw_sigma:.6g} "
            f"clip_hi={raw_median + 3.0 * raw_sigma:.6g} clipped_pixels={n_bright} "
            f"mean={mean:.6g} center={center:.6g} sigma={sigma:.6g}"
        )
    return centers, sigmas, raw_medians, raw_sigmas


def map_astro_rgb_for_amg(
    rgb: np.ndarray,
    stats_rgb: np.ndarray,
    mode: str,
    low_sigma: float | None,
    stats_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    center_rgb, sigma_rgb, raw_median_rgb, raw_sigma_rgb = estimate_rgb_location_scale(stats_rgb, stats_mode)
    safe_rgb = rgb.copy()
    hist_mask = np.ones(rgb.shape, dtype=bool)
    for c in range(3):
        safe_rgb[..., c] = np.where(np.isfinite(safe_rgb[..., c]), safe_rgb[..., c], center_rgb[c])
        hist_mask[..., c] = np.isfinite(rgb[..., c])
        safe_rgb[..., c] = np.minimum(safe_rgb[..., c], raw_median_rgb[c] + 3.0 * raw_sigma_rgb[c])

    z = (safe_rgb - center_rgb.reshape(1, 1, 3)) / sigma_rgb.reshape(1, 1, 3)

    if mode == "none":
        print(f"[DEBUG] Using continuous z-score astro input with {stats_mode} center/sigma")
        return z.astype(np.float32), hist_mask
    if mode == "astro_rgb":
        low = 3.0
    elif mode == "astro_rgb1":
        low = 1.0
    elif mode == "astro_rgb2":
        low = 5.0 if low_sigma is None else float(low_sigma)
    else:
        raise ValueError(f"Unknown astro_rgb mode: {mode}")

    high = 3.0
    mapped = (np.clip(z, -low, high) + low) / (low + high)
    mapped = np.clip(mapped, 0.0, 1.0)
    return np.round(mapped * 255.0).astype(np.uint8), hist_mask


class SAMImageDataset(Dataset):
    def __init__(
        self,
        data_dir: Path | None,
        img_size: int,
        max_images: int | None = None,
        astro_rgb_fits: Sequence[Path] | None = None,
        astro_fits_hdu: int = 0,
        astro_crop_size: int = 1024,
        astro_rgb_mode: str = "astro_rgb",
        astro_rgb_low_sigma: float | None = None,
        astro_stats_mode: str = "sigmaclip",
    ) -> None:
        self.transform = ResizeLongestSide(img_size)
        self.astro_rgb: np.ndarray | None = None
        self.astro_hist_mask: np.ndarray | None = None
        self.astro_name = ""

        if astro_rgb_fits is not None:
            if len(astro_rgb_fits) != 3:
                raise ValueError("--astro-rgb-fits must receive exactly three FITS paths: HSC-I HSC-R HSC-G")
            full_bands = [load_fits_2d(Path(p), astro_fits_hdu) for p in astro_rgb_fits]
            bands = [crop_bottom_right(band, astro_crop_size) for band in full_bands]
            if len({band.shape for band in bands}) != 1:
                raise ValueError(f"Astro FITS crops must have identical shapes, got {[b.shape for b in bands]}")
            if len({band.shape for band in full_bands}) != 1:
                raise ValueError(f"Astro FITS full-frame inputs must have identical shapes, got {[b.shape for b in full_bands]}")
            raw_rgb = np.stack(bands, axis=-1)
            stats_rgb = np.stack(full_bands, axis=-1)
            self.astro_rgb, self.astro_hist_mask = map_astro_rgb_for_amg(
                raw_rgb,
                stats_rgb,
                astro_rgb_mode,
                astro_rgb_low_sigma,
                astro_stats_mode,
            )
            self.astro_name = (
                f"{mode_label(astro_rgb_mode, astro_rgb_low_sigma)}_{astro_stats_mode}:"
                f"{','.join(str(p) for p in astro_rgb_fits)}"
            )
            self.paths: List[Path] = []
            return

        if data_dir is None:
            raise ValueError("Either --data-dir or --astro-rgb-fits must be provided")
        self.data_dir = data_dir

        paths = [p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        paths = sorted(paths)
        if max_images is not None:
            paths = paths[:max_images]
        self.paths = paths

        if not self.paths:
            raise RuntimeError(f"No images found in {data_dir} with extensions: {sorted(IMG_EXTS)}")

    def __len__(self) -> int:
        if self.astro_rgb is not None:
            return 1
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor | None, Tuple[int, int], str]:
        if self.astro_rgb is not None:
            resized = self.transform.apply_image(self.astro_rgb)
            h, w = resized.shape[:2]
            tensor = torch.from_numpy(resized).permute(2, 0, 1).to(dtype=torch.float32)
            mask_tensor = None
            if self.astro_hist_mask is not None:
                resized_mask = self.transform.apply_image(self.astro_hist_mask.astype(np.uint8))
                mask_tensor = torch.from_numpy(resized_mask > 0).permute(2, 0, 1)
            return tensor, mask_tensor, (h, w), self.astro_name

        path = self.paths[index]
        with Image.open(path) as img:
            rgb = np.array(img.convert("RGB"), dtype=np.uint8)

        resized = self.transform.apply_image(rgb)  # HxWx3 uint8
        h, w = resized.shape[:2]
        tensor = torch.from_numpy(resized).permute(2, 0, 1).to(dtype=torch.float32)  # 3xHxW
        return tensor, None, (h, w), str(path)


def collate_keep_lists(batch: Sequence[Tuple[torch.Tensor, torch.Tensor | None, Tuple[int, int], str]]):
    images = [x[0] for x in batch]
    masks = [x[1] for x in batch]
    sizes = [x[2] for x in batch]
    paths = [x[3] for x in batch]
    return images, masks, sizes, paths


def get_pixel_stats(
    mode: str,
    low_sigma: float | None,
    none_stats: str,
    use_astro_input: bool,
) -> Tuple[List[float], List[float]]:
    if mode == "none":
        if none_stats == "auto":
            # For normal SAM RGB images, keep original SAM/ImageNet normalization.
            # For astro-mapped input, identity stats are often expected.
            resolved = "identity" if use_astro_input else "imagenet"
        else:
            resolved = none_stats

        if resolved == "identity":
            return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
        return [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
    mean, std = convert_astro([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], mode=mode, low_sigma=low_sigma)
    return mean, std


def iter_preprocessed(
    images: List[torch.Tensor],
    masks: List[torch.Tensor | None],
    device: torch.device,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    img_size: int,
    include_padding: bool,
) -> List[Tuple[torch.Tensor, torch.Tensor | None]]:
    processed: List[Tuple[torch.Tensor, torch.Tensor | None]] = []
    for img, mask in zip(images, masks):
        x = img.to(device, non_blocking=True)
        x = (x - pixel_mean) / pixel_std
        mask_device = mask.to(device, non_blocking=True) if mask is not None else None
        if include_padding:
            h, w = x.shape[-2:]
            padh = img_size - h
            padw = img_size - w
            x = F.pad(x, (0, padw, 0, padh))
            if mask_device is not None:
                mask_device = F.pad(mask_device, (0, padw, 0, padh), value=False)
        processed.append((x, mask_device))
    return processed


def channel_values(x: torch.Tensor, mask: torch.Tensor | None, c: int) -> torch.Tensor:
    vals = x[c]
    if mask is not None:
        vals = vals[mask[c]]
    return vals.reshape(-1)


def progress_iter(iterable: Iterable, desc: str, total: int):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)


def sample_values(vals: torch.Tensor, max_samples: int) -> np.ndarray:
    flat = vals.reshape(-1)
    if flat.numel() <= max_samples:
        return flat.detach().cpu().numpy().astype(np.float64, copy=False)
    step = int(math.ceil(flat.numel() / max_samples))
    return flat[::step][:max_samples].detach().cpu().numpy().astype(np.float64, copy=False)


def compute_channel_range(
    loader: DataLoader,
    device: torch.device,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    img_size: int,
    include_padding: bool,
    hist_percentiles: Tuple[float, float] | None,
    percentile_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    min_vals = torch.full((3,), float("inf"), dtype=torch.float64, device=device)
    max_vals = torch.full((3,), float("-inf"), dtype=torch.float64, device=device)
    samples: List[List[np.ndarray]] = [[], [], []]

    for images, masks, _, _ in progress_iter(loader, desc="Pass1 range", total=len(loader)):
        processed = iter_preprocessed(images, masks, device, pixel_mean, pixel_std, img_size, include_padding)
        for x, mask in processed:
            if hist_percentiles is not None:
                per_tensor_budget = max(1024, percentile_samples // max(len(loader.dataset), 1))
            for c in range(3):
                vals = channel_values(x, mask, c)
                if vals.numel() == 0:
                    continue
                min_vals[c] = torch.minimum(min_vals[c], vals.amin().to(torch.float64))
                max_vals[c] = torch.maximum(max_vals[c], vals.amax().to(torch.float64))
                if hist_percentiles is not None:
                    samples[c].append(sample_values(vals, per_tensor_budget))

    pct_min = None
    pct_max = None
    if hist_percentiles is not None:
        pct_min = np.zeros(3, dtype=np.float64)
        pct_max = np.zeros(3, dtype=np.float64)
        for c in range(3):
            merged = np.concatenate(samples[c]) if samples[c] else np.array([], dtype=np.float64)
            if merged.size == 0:
                pct_min[c] = float(min_vals[c].item())
                pct_max[c] = float(max_vals[c].item())
            else:
                pct_min[c], pct_max[c] = np.percentile(merged, hist_percentiles)

    return min_vals.cpu().numpy(), max_vals.cpu().numpy(), pct_min, pct_max


def build_edges(min_v: float, max_v: float, bin_width: float, max_bins: int) -> np.ndarray:
    span = max(max_v - min_v, bin_width)
    bins = int(math.ceil(span / bin_width))
    bins = max(32, min(bins, max_bins))
    return np.linspace(min_v, max_v, bins + 1, dtype=np.float64)


def apply_astro_none_hist_window(
    edge_min: np.ndarray,
    edge_max: np.ndarray,
    full_min: np.ndarray,
    full_max: np.ndarray,
    window: Tuple[float, float] | None,
    enabled: bool,
) -> Tuple[np.ndarray, np.ndarray, str | None]:
    if not enabled or window is None:
        return edge_min, edge_max, None

    low, high = window
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        raise ValueError("--astro-none-hist-range requires finite LOW < HIGH.")

    clipped_min = np.maximum(edge_min, low)
    clipped_max = np.minimum(edge_max, high)
    clipped_min = np.minimum(clipped_min, full_max)
    clipped_max = np.maximum(clipped_max, full_min)

    # If all sampled values are outside the requested window for a channel,
    # fall back to the explicit window so the plot still communicates overflow.
    bad = clipped_max <= clipped_min
    if np.any(bad):
        clipped_min[bad] = low
        clipped_max[bad] = high

    return clipped_min, clipped_max, f"[{low:g}, {high:g}]"

def infer_bin_widths(
    pixel_std: Sequence[float],
    mode: str,
    use_astro_input: bool,
    requested_bin_width: float | None,
    fallback_bin_width: float = 0.10,
) -> Tuple[List[float], str]:
    if requested_bin_width is not None:
        return [float(requested_bin_width)] * 3, "explicit"

    # Regular RGB images and clipped astro RGB modes are uint8 before SAM
    # normalization, so adjacent quantized levels are separated by 1 / std.
    if not (use_astro_input and mode == "none"):
        widths = []
        for std in pixel_std:
            if std <= 0 or not math.isfinite(std):
                widths.append(fallback_bin_width)
            else:
                widths.append(1.0 / float(std))
        return widths, "quantization_step"

    # astro_rgb_mode=none for FITS keeps continuous float values through resize
    # and normalization, so there is no uint8 quantization step to align to.
    return [fallback_bin_width] * 3, "fallback_continuous"


def compute_histograms(
    loader: DataLoader,
    device: torch.device,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    img_size: int,
    include_padding: bool,
    edges_per_channel: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[ChannelStats], List[int], List[int]]:
    counts = [np.zeros(len(e) - 1, dtype=np.float64) for e in edges_per_channel]
    stats = [ChannelStats(), ChannelStats(), ChannelStats()]
    underflow = [0, 0, 0]
    overflow = [0, 0, 0]

    for images, masks, _, _ in progress_iter(loader, desc="Pass2 hist", total=len(loader)):
        processed = iter_preprocessed(images, masks, device, pixel_mean, pixel_std, img_size, include_padding)

        for x, mask in processed:
            for c in range(3):
                vals = channel_values(x, mask, c)
                if vals.numel() == 0:
                    continue
                stats[c].update(vals)
                e = edges_per_channel[c]
                underflow[c] += int((vals < float(e[0])).sum().item())
                overflow[c] += int((vals > float(e[-1])).sum().item())
                h = torch.histc(vals, bins=len(e) - 1, min=float(e[0]), max=float(e[-1]))
                counts[c] += h.detach().cpu().numpy()

    return counts, stats, underflow, overflow


def compute_sam_range_counts(
    loader: DataLoader,
    device: torch.device,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    img_size: int,
    include_padding: bool,
) -> Tuple[List[int], List[int]]:
    """Count values outside default SAM/ImageNet-normalized [0, 255] bounds.

    This intentionally uses fixed SAM RGB normalization statistics instead of
    mode-specific stats (e.g., identity in astro none), so the counts are
    directly comparable to standard SAM image preprocessing ranges.
    """

    _ = (pixel_mean, pixel_std)
    sam_mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32, device=device).view(3, 1, 1)
    sam_std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32, device=device).view(3, 1, 1)
    low = (0.0 - sam_mean) / sam_std
    high = (255.0 - sam_mean) / sam_std
    below = [0, 0, 0]
    above = [0, 0, 0]

    for images, masks, _, _ in progress_iter(loader, desc="Pass2 sam-range", total=len(loader)):
        processed = iter_preprocessed(images, masks, device, pixel_mean, pixel_std, img_size, include_padding)
        for x, mask in processed:
            for c in range(3):
                vals = channel_values(x, mask, c)
                if vals.numel() == 0:
                    continue
                below[c] += int((vals < float(low[c])).sum().item())
                above[c] += int((vals > float(high[c])).sum().item())

    return below, above


def save_plots(
    output_dir: Path,
    label: str,
    edges_per_channel: Sequence[np.ndarray],
    counts: Sequence[np.ndarray],
    stats: Sequence[ChannelStats],
    full_min: Sequence[float],
    full_max: Sequence[float],
    underflow: Sequence[int],
    overflow: Sequence[int],
    sam_below: Sequence[int] | None,
    sam_above: Sequence[int] | None,
    include_padding: bool,
    bin_width: float | Sequence[float],
    bin_width_source: str,
    log_y: bool,
    hist_percentiles: Tuple[float, float] | None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    ch_names = ["R", "G", "B"]
    colors = ["#E74C3C", "#2ECC71", "#3498DB"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    for i, ax in enumerate(axes):
        edges = edges_per_channel[i]
        ctrs = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        gauss_counts = gaussian_expected_counts_with_tail_folding(edges, stats[i].count)
        sam_below_text = "N/A" if sam_below is None else str(int(sam_below[i]))
        sam_above_text = "N/A" if sam_above is None else str(int(sam_above[i]))

        ax.bar(ctrs, counts[i], width=widths * 0.95, color=colors[i], alpha=0.85, edgecolor="none")
        ax.plot(
            ctrs,
            gauss_counts,
            color="#111111",
            linewidth=1.8,
            label="Std Gaussian (tails folded to edge bins)",
        )
        ax.set_title(
            f"{ch_names[i]} channel\n"
            f"underflow={underflow[i]}  overflow={overflow[i]}  sam_below={sam_below_text}  sam_above={sam_above_text}"
        )
        ax.set_xlabel("Value after preprocessing")
        ax.set_ylabel("Pixel count")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="upper right")

    fig.suptitle(
        f"AMG preprocessed input distribution: {label} "
        f"(include_padding={include_padding}, bin_width={format_bin_width_display(bin_width)})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig_path = output_dir / f"amg_input_hist_{label}.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    summary = {
        "include_padding": include_padding,
        "histogram_source": (
            "For FITS astro inputs, histogram/range/stat values are computed from the actual "
            "bright-clipped preprocessed image: finite pixels above raw_median+3*raw_sigma are "
            "included after being set to raw_median+3*raw_sigma."
        ),
        "bin_width_requested": bin_width,
        "bin_width_source": bin_width_source,
        "bin_width_display": format_bin_width_display(bin_width),
        "hist_percentiles": list(hist_percentiles) if hist_percentiles is not None else None,
        "channels": {
            ch_names[i]: {
                "count": stats[i].count,
                "mean": stats[i].mean(),
                "std": stats[i].std(),
                "full_min": float(full_min[i]),
                "full_max": float(full_max[i]),
                "hist_min": float(edges_per_channel[i][0]),
                "hist_max": float(edges_per_channel[i][-1]),
                "bins": int(len(edges_per_channel[i]) - 1),
                "underflow": int(underflow[i]),
                "overflow": int(overflow[i]),
                "sam_below": None if sam_below is None else int(sam_below[i]),
                "sam_above": None if sam_above is None else int(sam_above[i]),
            }
            for i in range(3)
        },
        "hist_png": str(fig_path),
        "hist_npz": str(output_dir / f"amg_input_hist_{label}.npz"),
    }

    with (output_dir / f"amg_input_hist_summary_{label}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        output_dir / f"amg_input_hist_{label}.npz",
        r_edges=edges_per_channel[0],
        g_edges=edges_per_channel[1],
        b_edges=edges_per_channel[2],
        r_counts=counts[0],
        g_counts=counts[1],
        b_counts=counts[2],
    )

    return summary


def requested_modes(args: argparse.Namespace) -> List[Tuple[str, float | None, str]]:
    if args.astro_rgb_fits is not None:
        modes: List[Tuple[str, float | None, str]] = []
        mode_names = args.astro_rgb_modes or ["astro_rgb", "none", "astro_rgb2"]
        for mode in mode_names:
            if mode == "astro_rgb2":
                lows = args.astro_rgb_low_sigmas or list(ASTRO_RGB2_LOW_SIGMAS)
                for low in lows:
                    modes.append((mode, low, f"{mode_label(mode, low)}_{args.astro_stats_mode}"))
            else:
                modes.append((mode, None, f"{mode_label(mode, None)}_{args.astro_stats_mode}"))
        return modes

    label = "rgb" if args.astro_rgb_mode == "none" else mode_label(args.astro_rgb_mode, args.astro_rgb_low_sigma)
    return [(args.astro_rgb_mode, args.astro_rgb_low_sigma, label)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SAM AMG preprocessed input distributions.")
    parser.add_argument("--data-dir", type=Path, default=Path("../sa_data"), help="Directory with SA images.")
    parser.add_argument(
        "--astro-rgb-fits",
        type=Path,
        nargs=3,
        default=None,
        metavar=("HSC-I", "HSC-R", "HSC-G"),
        help="Analyze one astro RGB crop from three FITS files. Paths are mapped to SAM R/G/B channels in this order.",
    )
    parser.add_argument("--astro-fits-hdu", type=int, default=0, help="FITS HDU used for --astro-rgb-fits.")
    parser.add_argument(
        "--astro-crop-size",
        type=int,
        default=1024,
        help="Right-bottom square crop size for astro FITS inputs.",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for image loading.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu.")
    parser.add_argument("--img-size", type=int, default=1024, help="SAM image encoder input size.")
    parser.add_argument(
        "--astro-rgb-mode",
        type=str,
        default="none",
        choices=["none", "astro_rgb", "astro_rgb1", "astro_rgb2"],
        help="Normalization mode aligned with build_sam.py",
    )
    parser.add_argument(
        "--astro-rgb-low-sigma",
        type=float,
        default=None,
        help="Low sigma for astro_rgb2 mode.",
    )
    parser.add_argument(
        "--none-stats",
        type=str,
        default="auto",
        choices=["auto", "imagenet", "identity"],
        help=(
            "Normalization stats when astro_rgb_mode=none. "
            "auto: imagenet for regular RGB inputs and identity for --astro-rgb-fits inputs."
        ),
    )
    parser.add_argument(
        "--astro-rgb-modes",
        nargs="+",
        choices=["none", "astro_rgb", "astro_rgb1", "astro_rgb2"],
        default=None,
        help="Modes to emit for --astro-rgb-fits. Defaults to astro_rgb, none, and astro_rgb2.",
    )
    parser.add_argument(
        "--astro-rgb-low-sigmas",
        type=float,
        nargs="+",
        default=None,
        help="Low-sigma values emitted for astro_rgb2 when using --astro-rgb-fits.",
    )
    parser.add_argument(
        "--astro-stats-mode",
        type=str,
        default="sigmaclip",
        choices=["bgd", "sigmaclip", "legacy"],
        help=(
            "Mean/sigma estimation for --astro-rgb-fits. "
            "bgd uses median/MAD; sigmaclip uses sigma-clipped median/std; "
            "legacy uses original global mean/std."
        ),
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=None,
        help=(
            "Target histogram bin width. Defaults to the normalized uint8 quantization step "
            "1/pixel_std for discrete inputs, or 0.10 for continuous astro_rgb_mode=none FITS input."
        ),
    )
    parser.add_argument("--max-bins", type=int, default=1024, help="Upper bound for bins per channel.")
    parser.add_argument(
        "--hist-percentiles",
        type=float,
        nargs=2,
        default=(0.1, 99.9),
        metavar=("LOW", "HIGH"),
        help="Use percentile-clipped x range for histogram bins. Use --full-range to disable.",
    )
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="Use exact min/max as histogram range instead of --hist-percentiles.",
    )
    parser.add_argument(
        "--astro-none-hist-range",
        type=float,
        nargs=2,
        default=(-10.0, 50.0),
        metavar=("LOW", "HIGH"),
        help=(
            "Histogram x-range for continuous FITS astro_rgb_mode=none z-score inputs. "
            "Set both values with --full-range only if you want to include extreme bright tails."
        ),
    )
    parser.add_argument(
        "--percentile-samples",
        type=int,
        default=1_000_000,
        help="Maximum sampled values per channel for estimating --hist-percentiles.",
    )
    parser.add_argument(
        "--exclude-padding",
        action="store_true",
        help="If set, histogram excludes zero padding introduced by Sam.preprocess.",
    )
    parser.add_argument("--log-y", action="store_true", help="Use log scale on y axis.")
    parser.add_argument("--output-dir", type=Path, default=Path("./amg_dist_outputs"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    include_padding = not args.exclude_padding
    hist_percentiles = None if args.full_range else tuple(args.hist_percentiles)

    all_summaries = {}
    for mode, low_sigma, label in requested_modes(args):
        pixel_mean, pixel_std = get_pixel_stats(
            mode,
            low_sigma,
            none_stats=args.none_stats,
            use_astro_input=(args.astro_rgb_fits is not None),
        )
        bin_widths, bin_width_source = infer_bin_widths(
            pixel_std,
            mode,
            use_astro_input=(args.astro_rgb_fits is not None),
            requested_bin_width=args.bin_width,
        )
        print(f"\n[{label}] Using pixel_mean={pixel_mean}, pixel_std={pixel_std}")
        print(f"[{label}] Using bin_widths={format_bin_width_display(bin_widths)} ({bin_width_source})")

        dataset = SAMImageDataset(
            args.data_dir if args.astro_rgb_fits is None else None,
            img_size=args.img_size,
            max_images=args.max_images,
            astro_rgb_fits=args.astro_rgb_fits,
            astro_fits_hdu=args.astro_fits_hdu,
            astro_crop_size=args.astro_crop_size,
            astro_rgb_mode=mode,
            astro_rgb_low_sigma=low_sigma,
            astro_stats_mode=args.astro_stats_mode,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_keep_lists,
            persistent_workers=(args.num_workers > 0),
        )
        source = "astro FITS crop" if args.astro_rgb_fits is not None else str(args.data_dir)
        print(f"[{label}] Found {len(dataset)} image(s) from {source}")

        pm = torch.tensor(pixel_mean, dtype=torch.float32, device=device).view(3, 1, 1)
        ps = torch.tensor(pixel_std, dtype=torch.float32, device=device).view(3, 1, 1)

        min_vals, max_vals, pct_min, pct_max = compute_channel_range(
            loader,
            device,
            pm,
            ps,
            args.img_size,
            include_padding,
            hist_percentiles,
            args.percentile_samples,
        )
        edge_min = pct_min if pct_min is not None else min_vals
        edge_max = pct_max if pct_max is not None else max_vals
        edge_min, edge_max, hist_window_label = apply_astro_none_hist_window(
            edge_min=np.array(edge_min, dtype=np.float64, copy=True),
            edge_max=np.array(edge_max, dtype=np.float64, copy=True),
            full_min=np.array(min_vals, dtype=np.float64, copy=False),
            full_max=np.array(max_vals, dtype=np.float64, copy=False),
            window=tuple(args.astro_none_hist_range),
            enabled=(args.astro_rgb_fits is not None and mode == "none"),
        )
        if hist_window_label is not None:
            print(f"[{label}] Applying astro none histogram window {hist_window_label}")
        edges_per_channel = [
            build_edges(float(edge_min[c]), float(edge_max[c]), bin_widths[c], args.max_bins) for c in range(3)
        ]

        counts, stats, underflow, overflow = compute_histograms(
            loader,
            device,
            pm,
            ps,
            args.img_size,
            include_padding,
            edges_per_channel,
        )

        sam_below = sam_above = None
        if args.astro_rgb_fits is not None and mode == "none":
            sam_below, sam_above = compute_sam_range_counts(
                loader,
                device,
                pm,
                ps,
                args.img_size,
                include_padding,
            )

        summary = save_plots(
            output_dir=args.output_dir,
            label=label,
            edges_per_channel=edges_per_channel,
            counts=counts,
            stats=stats,
            full_min=min_vals,
            full_max=max_vals,
            underflow=underflow,
            overflow=overflow,
            sam_below=sam_below,
            sam_above=sam_above,
            include_padding=include_padding,
            bin_width=bin_widths if args.bin_width is None else args.bin_width,
            bin_width_source=bin_width_source,
            log_y=args.log_y,
            hist_percentiles=hist_percentiles,
        )
        all_summaries[label] = summary

        print(f"[{label}] Saved {summary['hist_png']}")
        for c in ("R", "G", "B"):
            s = summary["channels"][c]
            print(
                f"[{label} {c}] count={s['count']} mean={s['mean']:.5f} std={s['std']:.5f} "
                f"full=[{s['full_min']:.5f}, {s['full_max']:.5f}] "
                f"hist=[{s['hist_min']:.5f}, {s['hist_max']:.5f}] bins={s['bins']} "
                f"under={s['underflow']} over={s['overflow']}"
            )
            if s["sam_below"] is not None and s["sam_above"] is not None:
                print(f"[{label} {c}] sam_range_below={s['sam_below']} sam_range_above={s['sam_above']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "amg_input_hist_summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
