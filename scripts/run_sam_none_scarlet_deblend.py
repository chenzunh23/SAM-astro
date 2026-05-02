#!/usr/bin/env python3
"""Run SAM astro none masks and feed them into scarlet deblending.

This experiment intentionally keeps the SAM side narrow:
- three FITS bands are passed through amg_fits_core.build_astro_input(..., mode="none")
- SAM masks are converted to a label map
- each SAM label is treated as a blend region; peaks inside it initialize scarlet sources
- scarlet fits all initialized sources and writes segmentation/deblend diagnostics
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

SCRIPT_DIR = Path(__file__).resolve().parent
SEGMENT_REPO = SCRIPT_DIR.parents[0]
SCARLET_REPO = SCRIPT_DIR.parents[1] / "scarlet"
for path in (SEGMENT_REPO, SCARLET_REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from amg_fits_core import (  # noqa: E402
    build_astro_input,
    build_generator,
    expand_crop_masks,
    filter_large_masks,
    filter_small_masks,
    make_filtered_label_map,
    masks_to_csv,
    read_fits_2d,
    run_generator,
)

import scarlet  # noqa: E402

try:
    from scipy.ndimage import gaussian_filter, maximum_filter
except ImportError:  # pragma: no cover
    gaussian_filter = None
    maximum_filter = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM none-mode masks followed by scarlet deblending.")
    parser.add_argument("--input", nargs=3, required=True, metavar=("R_FITS", "G_FITS", "B_FITS"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hdu", type=int, default=0)

    parser.add_argument("--model-type", default="vit_h", choices=["default", "vit_h", "vit_l", "vit_b"])
    parser.add_argument("--checkpoint", default="/home/chenzunhao/sam_vit_h_4b8939.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=128)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-overlap-ratio", type=float, default=512 / 1500)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=15)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.5)
    parser.add_argument("--overlay-alpha", type=float, default=0.35)
    parser.add_argument("--astro-preprocess-in-model", action="store_true")
    parser.add_argument("--astro-preprocess-clip-sigma", type=float, default=3.0)
    parser.add_argument("--astro-preprocess-sigma-iters", type=int, default=-1)
    parser.add_argument("--astro-preprocess-z-clip", type=float, nargs=2, default=None)
    parser.add_argument("--astro-stats-mode", default="sigmaclip", choices=["bgd", "sigmaclip"])
    parser.add_argument("--astro-crop-size", type=int, default=0, help="SAM crop size; 0 means use the full frame")
    parser.add_argument("--low-percentile", type=float, default=0.1)
    parser.add_argument("--high-percentile", type=float, default=99.5)

    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--psf-sigma", type=float, default=2.0)
    parser.add_argument("--source-thresh", type=float, default=0.1)
    parser.add_argument("--scarlet-min-snr", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--e-rel", type=float, default=1e-5)

    parser.add_argument("--peak-threshold", type=float, default=2.5)
    parser.add_argument("--peak-min-distance", type=int, default=6)
    parser.add_argument("--max-peaks-per-mask", type=int, default=4)
    parser.add_argument("--max-sources", type=int, default=200)
    parser.add_argument("--min-source-area", type=int, default=8)
    parser.add_argument("--min-source-snr", type=float, default=2.0)
    parser.add_argument("--mask-padding", type=int, default=2)
    parser.add_argument("--blend-init-sigma", type=float, default=3.0)
    parser.add_argument(
        "--source-mode",
        choices=["extended", "mask", "blend"],
        default="extended",
        help=(
            "extended uses SAM masks only for source centers; mask uses hard per-source SAM supports; "
            "blend uses each SAM mask as a parent blend region and initializes multiple sources inside it"
        ),
    )
    parser.add_argument("--display-percentiles", type=float, nargs=2, default=[1.0, 99.0])
    return parser.parse_args()


def generator_args(args: argparse.Namespace) -> SimpleNamespace:
    """Build the subset of amg_fits_core args needed by build_generator."""

    return SimpleNamespace(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        scaling_mode="astro_rgb",
        astro_rgb_low_sigma=None,
        astro_preprocess_in_model=args.astro_preprocess_in_model,
        astro_preprocess_clip_sigma=args.astro_preprocess_clip_sigma,
        astro_preprocess_sigma_iters=args.astro_preprocess_sigma_iters,
        astro_preprocess_z_clip=args.astro_preprocess_z_clip,
        device=args.device,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_nms_thresh=args.crop_nms_thresh,
        crop_overlap_ratio=args.crop_overlap_ratio,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
    )


def make_scarlet_norm(cube: np.ndarray, percentiles: Sequence[float]):
    # return scarlet.display.AsinhPercentileNorm(cube, percentiles=percentiles)
    return scarlet.display.AsinhMapping(minimum=0, stretch=0.2, Q=10)


def make_scarlet_rgb(cube: np.ndarray, percentiles: Sequence[float]) -> np.ndarray:
    norm = make_scarlet_norm(cube, percentiles)
    rgb = scarlet.display.img_to_rgb(cube, norm=norm)
    return to_display_rgb(rgb)


def make_scarlet_rgb_with_norm(cube: np.ndarray, norm) -> np.ndarray:
    rgb = scarlet.display.img_to_rgb(cube, norm=norm)
    return to_display_rgb(rgb)


def save_rgb(path: Path, rgb: np.ndarray, title: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.flipud(to_display_rgb(rgb)))
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def label_overlay(rgb: np.ndarray, label_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = np.clip(rgb, 0, 1).copy()
    labels = np.unique(label_map)
    labels = labels[labels > 0]
    if labels.size == 0:
        return out
    rng = np.random.default_rng(11)
    colors = {int(label): rng.uniform(0.15, 1.0, size=3) for label in labels}
    for label in labels:
        mask = label_map == label
        out[mask] = (1 - alpha) * out[mask] + alpha * colors[int(label)]
    return np.clip(out, 0, 1)


def to_display_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if np.issubdtype(rgb.dtype, np.integer) or np.nanmax(rgb) > 1.5:
        rgb = rgb.astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def bbox_to_mask(label_map: np.ndarray, label: int) -> np.ndarray:
    return label_map == label


def flux_weighted_center(mask: np.ndarray, detection: np.ndarray) -> Tuple[float, float]:
    y, x = np.nonzero(mask)
    values = np.clip(detection[y, x], 0, None)
    if values.sum() > 0:
        return float(np.sum(y * values) / np.sum(values)), float(np.sum(x * values) / np.sum(values))
    return float(np.mean(y)), float(np.mean(x))


def estimate_background_sigma(values: np.ndarray) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    bg = float(np.median(finite))
    mad = float(np.median(np.abs(finite - bg)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(finite))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return bg, sigma


def peaks_in_mask(
    mask: np.ndarray,
    detection: np.ndarray,
    threshold_sigma: float,
    min_distance: int,
    max_peaks: int,
) -> List[Tuple[float, float]]:
    if not np.any(mask):
        return []

    work = np.asarray(detection, dtype=np.float32).copy()
    fill = float(np.nanmedian(work[np.isfinite(work)])) if np.any(np.isfinite(work)) else 0.0
    work = np.nan_to_num(work, nan=fill, posinf=fill, neginf=fill)
    if gaussian_filter is not None:
        work = gaussian_filter(work, sigma=1.0)

    bg, sigma = estimate_background_sigma(work[~mask])
    threshold = bg + threshold_sigma * sigma
    masked = np.where(mask, work, -np.inf)

    if maximum_filter is not None:
        local_max = masked == maximum_filter(masked, size=max(3, int(min_distance)))
        candidates = np.argwhere(local_max & mask & (masked >= threshold))
    else:
        candidates = np.argwhere(mask & (masked >= threshold))

    if candidates.size == 0:
        return [flux_weighted_center(mask, detection)]

    values = masked[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(values)[::-1]
    centers: List[Tuple[float, float]] = []
    for idx in order:
        y, x = candidates[idx]
        if all((y - yy) ** 2 + (x - xx) ** 2 >= min_distance**2 for yy, xx in centers):
            centers.append((float(y), float(x)))
        if len(centers) >= max_peaks:
            break
    return centers or [flux_weighted_center(mask, detection)]


def mask_snr(mask: np.ndarray, detection: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    bg, sigma = estimate_background_sigma(detection[~mask])
    signal = np.sum(np.clip(detection[mask] - bg, 0, None))
    return float(signal / (sigma * np.sqrt(np.count_nonzero(mask))))


def centers_from_label_map(
    label_map: np.ndarray,
    detection: np.ndarray,
    threshold_sigma: float,
    min_distance: int,
    max_peaks_per_mask: int,
    max_sources: int,
    min_source_area: int,
    min_source_snr: float,
) -> List[Tuple[float, float]]:
    centers: List[Tuple[float, float]] = []
    skipped_area = 0
    skipped_snr = 0
    for label in np.unique(label_map):
        if label <= 0:
            continue
        mask = bbox_to_mask(label_map, int(label))
        if np.count_nonzero(mask) < min_source_area:
            skipped_area += 1
            continue
        if mask_snr(mask, detection) < min_source_snr:
            skipped_snr += 1
            continue
        centers.extend(
            peaks_in_mask(
                mask,
                detection,
                threshold_sigma,
                min_distance,
                max_peaks_per_mask,
            )
        )
        if len(centers) >= max_sources:
            centers = centers[:max_sources]
            break
    print(
        f"[scarlet] source candidate filter: centers={len(centers)} "
        f"skipped_area={skipped_area} skipped_snr={skipped_snr}"
    )
    return centers


class MaskConstraint(scarlet.Constraint):
    """Keep a source morphology inside its assigned SAM support."""

    def __init__(self, mask: np.ndarray):
        self.mask = np.asarray(mask, dtype=bool)

    def __call__(self, x, step):
        return np.asarray(x) * self.mask


def assign_mask_pixels(mask: np.ndarray, centers: Sequence[Tuple[float, float]]) -> List[np.ndarray]:
    """Split one SAM mask into peak-owned regions with a nearest-center assignment."""

    if len(centers) <= 1:
        return [mask.copy()]
    y, x = np.indices(mask.shape)
    dist2 = np.stack([(y - cy) ** 2 + (x - cx) ** 2 for cy, cx in centers], axis=0)
    owner = np.argmin(dist2, axis=0)
    return [(mask & (owner == idx)) for idx in range(len(centers))]


def source_spectrum_from_morph(observation, bbox, morph: np.ndarray) -> np.ndarray:
    data = observation.data
    weights = observation.weights
    data_cut = np.stack([bbox.extract_from(data[c]) for c in range(data.shape[0])], axis=0)
    weights_cut = np.stack([bbox.extract_from(weights[c]) for c in range(weights.shape[0])], axis=0)
    denom = np.sum(weights_cut * morph[None, :, :] ** 2, axis=(1, 2))
    numer = np.sum(weights_cut * data_cut * morph[None, :, :], axis=(1, 2))
    spectrum = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    noise_rms = np.asarray(np.mean(observation.noise_rms, axis=(1, 2)))
    return np.maximum(spectrum, noise_rms)


def make_mask_sources(
    model_frame,
    observation,
    label_map: np.ndarray,
    detection: np.ndarray,
    threshold_sigma: float,
    min_distance: int,
    max_peaks_per_mask: int,
    max_sources: int,
    min_source_area: int,
    min_source_snr: float,
    padding: int,
) -> list:
    """Use SAM masks as blend regions and per-peak supports for scarlet sources."""

    sources = []
    noise_rms = np.asarray(np.mean(observation.noise_rms, axis=(1, 2)))
    for label in np.unique(label_map):
        if label <= 0:
            continue
        parent_mask = label_map == label
        if np.count_nonzero(parent_mask) < min_source_area:
            continue
        if mask_snr(parent_mask, detection) < min_source_snr:
            continue
        centers = peaks_in_mask(parent_mask, detection, threshold_sigma, min_distance, max_peaks_per_mask)
        support_masks = assign_mask_pixels(parent_mask, centers)

        for center, support in zip(centers, support_masks):
            if len(sources) >= max_sources:
                return sources
            if not np.any(support):
                continue

            bbox = scarlet.Box.from_data(support.astype(np.float32), min_value=0)
            if padding > 0:
                bbox = bbox.grow(padding)
            support_cut = bbox.extract_from(support.astype(np.float32)).astype(bool)
            det_cut = bbox.extract_from(np.clip(detection, 0, None))
            morph = det_cut * support_cut
            if np.max(morph) <= 0:
                morph = support_cut.astype(np.float32)
            morph = morph / np.max(morph)

            spectrum_values = source_spectrum_from_morph(observation, bbox, morph)
            spectrum = scarlet.TabulatedSpectrum(model_frame, spectrum_values, min_step=noise_rms)
            constraint = scarlet.ConstraintChain(
                scarlet.PositivityConstraint(),
                MaskConstraint(support_cut),
                scarlet.NormalizationConstraint("max"),
            )
            parameter = scarlet.Parameter(morph, name="image", step=1e-2, constraint=constraint)
            morphology = scarlet.ImageMorphology(model_frame, parameter, bbox=bbox, resizing=False)
            source = scarlet.FactorizedComponent(model_frame, spectrum, morphology)
            source.center = np.asarray(center, dtype=float)
            sources.append(source)

    if not sources:
        raise RuntimeError("No scarlet sources could be initialized from SAM masks")
    return sources


def make_blend_sources(
    model_frame,
    observation,
    label_map: np.ndarray,
    detection: np.ndarray,
    threshold_sigma: float,
    min_distance: int,
    max_peaks_per_mask: int,
    max_sources: int,
    min_source_area: int,
    min_source_snr: float,
    padding: int,
    init_sigma: float,
) -> list:
    """Initialize multiple sources inside each SAM parent mask without hard child boundaries."""

    sources = []
    noise_rms = np.asarray(np.mean(observation.noise_rms, axis=(1, 2)))
    yy, xx = np.indices(label_map.shape)
    sigma2 = max(float(init_sigma), 1e-3) ** 2

    for label in np.unique(label_map):
        if label <= 0:
            continue
        parent_mask = label_map == label
        if np.count_nonzero(parent_mask) < min_source_area:
            continue
        if mask_snr(parent_mask, detection) < min_source_snr:
            continue

        centers = peaks_in_mask(parent_mask, detection, threshold_sigma, min_distance, max_peaks_per_mask)
        if not centers:
            continue

        bbox = scarlet.Box.from_data(parent_mask.astype(np.float32), min_value=0)
        if padding > 0:
            bbox = bbox.grow(padding)
        parent_cut = bbox.extract_from(parent_mask.astype(np.float32)).astype(bool)
        det_cut = bbox.extract_from(np.clip(detection, 0, None))
        yy_cut = bbox.extract_from(yy)
        xx_cut = bbox.extract_from(xx)

        for center in centers:
            if len(sources) >= max_sources:
                return sources

            cy, cx = center
            seed = np.exp(-0.5 * ((yy_cut - cy) ** 2 + (xx_cut - cx) ** 2) / sigma2)
            morph = det_cut * seed * parent_cut
            if np.max(morph) <= 0:
                morph = seed * parent_cut
            if np.max(morph) <= 0:
                continue
            morph = morph / np.max(morph)

            spectrum_values = source_spectrum_from_morph(observation, bbox, morph)
            spectrum = scarlet.TabulatedSpectrum(model_frame, spectrum_values, min_step=noise_rms)
            constraint = scarlet.ConstraintChain(
                scarlet.PositivityConstraint(),
                MaskConstraint(parent_cut),
                scarlet.NormalizationConstraint("max"),
            )
            parameter = scarlet.Parameter(morph, name="image", step=1e-2, constraint=constraint)
            morphology = scarlet.ImageMorphology(model_frame, parameter, bbox=bbox, resizing=False)
            source = scarlet.FactorizedComponent(model_frame, spectrum, morphology)
            source.center = np.asarray(center, dtype=float)
            sources.append(source)

    if not sources:
        raise RuntimeError("No scarlet sources could be initialized from SAM blend regions")
    return sources


def make_sources(model_frame, observation, centers: Sequence[Tuple[float, float]], thresh: float, min_snr: float) -> list:
    sources, skipped = scarlet.initialization.init_all_sources(
        model_frame,
        list(centers),
        observation,
        max_components=2,
        min_snr=min_snr,
        thresh=thresh,
        fallback=True,
        silent=True,
        set_spectra=False,
    )
    if sources:
        try:
            scarlet.initialization.set_spectra_to_match(sources, observation)
        except ValueError as exc:
            print(f"[WARN] set_spectra_to_match failed: {exc}")

    for center, reason in skipped:
        print(f"[WARN] skipped center={center}: {reason}")

    if not sources:
        print("[WARN] init_all_sources returned no sources; falling back to direct ExtendedSource init")
        for center in centers:
            try:
                sources.append(
                    scarlet.ExtendedSource(
                        model_frame,
                        center,
                        observation,
                        thresh=thresh,
                        shifting=True,
                        resizing=True,
                    )
                )
            except Exception as exc:
                print(f"[WARN] Skipped scarlet source at {center}: {exc}")
    if not sources:
        raise RuntimeError("No scarlet sources could be initialized from SAM masks")
    return sources


def save_deblend_figure(
    output: Path,
    cube: np.ndarray,
    rendered: np.ndarray,
    residual: np.ndarray,
    percentiles: Sequence[float],
    *,
    autoscale: bool = False,
) -> None:
    shared_norm = make_scarlet_norm(cube, percentiles)
    if autoscale:
        images = [
            ("Data", make_scarlet_rgb(cube, percentiles)),
            ("Scarlet model", make_scarlet_rgb(rendered, percentiles)),
            ("Residual", make_scarlet_rgb(residual, percentiles)),
        ]
    else:
        images = [
            ("Data", make_scarlet_rgb_with_norm(cube, shared_norm)),
            ("Scarlet model", make_scarlet_rgb_with_norm(rendered, shared_norm)),
            ("Residual", make_scarlet_rgb_with_norm(residual, shared_norm)),
        ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True)
    for ax, (title, rgb) in zip(axes, images):
        ax.imshow(np.flipud(to_display_rgb(rgb)))
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_axis_off()
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_likelihood_curve(blend, output_dir: Path) -> None:
    log_likelihood = np.asarray(blend.log_likelihood, dtype=np.float64)
    if log_likelihood.size == 0:
        print("[WARN] No scarlet log-likelihood values were recorded")
        return

    iterations = np.arange(log_likelihood.size, dtype=int)
    csv_data = np.column_stack([iterations, log_likelihood])
    np.savetxt(
        output_dir / "scarlet_log_likelihood.csv",
        csv_data,
        delimiter=",",
        header="iteration,log_likelihood",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(iterations, log_likelihood, color="#2f5f8f", linewidth=1.8)
    ax.scatter(iterations[-1], log_likelihood[-1], color="#c43c39", s=28, zorder=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("log-Likelihood")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "scarlet_log_likelihood.png", dpi=180)
    fig.savefig(output_dir / "scarlet_log_likelihood.tif")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(p).expanduser().resolve() for p in args.input]
    arrays = []
    headers = []
    for path in paths:
        image, header = read_fits_2d(path, args.hdu)
        arrays.append(image)
        headers.append(header)
    r, g, b = arrays
    if len({arr.shape for arr in arrays}) != 1:
        raise ValueError(f"Input FITS images must have identical shapes, got {[arr.shape for arr in arrays]}")

    cube = np.stack([r, g, b], axis=0).astype(np.float32)
    detection = cube.sum(axis=0)

    astro_input = build_astro_input(
        r,
        g,
        b,
        mode="none",
        stats_mode=args.astro_stats_mode,
        low_sigma_override=None,
        crop_size=max(r.shape) if args.astro_crop_size <= 0 else args.astro_crop_size,
        low_pct=args.low_percentile,
        high_pct=args.high_percentile,
        preprocess_in_model=args.astro_preprocess_in_model,
    )
    generator = build_generator(generator_args(args), "none")
    sam_config = {
        "scaling_mode": "astro_rgb",
        "astro_rgb_mode": "none",
        "astro_preprocess_in_model": args.astro_preprocess_in_model,
        "astro_preprocess_clip_sigma": args.astro_preprocess_clip_sigma,
        "astro_preprocess_sigma_iters": args.astro_preprocess_sigma_iters,
        "astro_preprocess_z_clip": args.astro_preprocess_z_clip,
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
        "max_mask_area_ratio": args.max_mask_area_ratio,
        "overlay_alpha": args.overlay_alpha,
        "astro_stats_mode": args.astro_stats_mode,
        "astro_crop_size": max(r.shape) if args.astro_crop_size <= 0 else args.astro_crop_size,
    }
    print("[SAM config] " + json.dumps(sam_config, sort_keys=True), flush=True)
    (args.output_dir / "sam_config.json").write_text(json.dumps(sam_config, indent=2), encoding="utf-8")
    masks = run_generator(generator, astro_input.sam_input)
    expand_crop_masks(masks, r.shape, astro_input.sam_input.shape[:2], astro_input.crop_y0, astro_input.crop_x0)
    masks, removed_small = filter_small_masks(masks, r.shape[0], r.shape[1], args.min_mask_region_area)
    masks, removed_large = filter_large_masks(masks, r.shape[0], r.shape[1], args.max_mask_area_ratio)
    label_map, masks, removed_label_small = make_filtered_label_map(masks, r.shape[0], r.shape[1], args.min_mask_region_area)
    removed_small += removed_label_small
    print(f"[SAM] kept={len(masks)} removed_small={removed_small} removed_large={removed_large}")

    fits.writeto(args.output_dir / "sam_labelmap.fits", label_map.astype(np.int32), header=headers[0], overwrite=True)
    masks_to_csv(masks, args.output_dir / "sam_metadata.csv")
    data_rgb = make_scarlet_rgb(cube, args.display_percentiles)
    save_rgb(
        args.output_dir / "sam_segmentation.png",
        label_overlay(data_rgb, label_map, alpha=args.overlay_alpha),
        "SAM segmentation",
    )
    save_rgb(args.output_dir / "scarlet_data_rgb.png", data_rgb, "Scarlet display")

    centers = centers_from_label_map(
        label_map,
        detection,
        args.peak_threshold,
        args.peak_min_distance,
        args.max_peaks_per_mask,
        args.max_sources,
        args.min_source_area,
        args.min_source_snr,
    )
    print(f"[scarlet] initialized centers={len(centers)} from SAM masks")
    for idx, (y, x) in enumerate(centers):
        print(f"  {idx:03d}: y={y:.2f} x={x:.2f}")

    weights = np.full_like(cube, 1.0 / float(args.variance), dtype=np.float32)
    sigma_vals = tuple(args.psf_sigma + 1e-3 * c for c in range(cube.shape[0]))
    model_psf = scarlet.GaussianPSF(sigma=sigma_vals)
    observation_psf = scarlet.GaussianPSF(sigma=sigma_vals)
    model_frame = scarlet.Frame(cube.shape, psf=model_psf, channels=np.asarray(["r", "g", "b"]))
    observation = scarlet.Observation(cube, channels=np.asarray(["r", "g", "b"]), psf=observation_psf, weights=weights).match(model_frame)

    if args.source_mode == "mask":
        sources = make_mask_sources(
            model_frame,
            observation,
            label_map,
            detection,
            args.peak_threshold,
            args.peak_min_distance,
            args.max_peaks_per_mask,
            args.max_sources,
            args.min_source_area,
            args.min_source_snr,
            args.mask_padding,
        )
        print(f"[scarlet] mask-constrained sources={len(sources)}")
    elif args.source_mode == "blend":
        sources = make_blend_sources(
            model_frame,
            observation,
            label_map,
            detection,
            args.peak_threshold,
            args.peak_min_distance,
            args.max_peaks_per_mask,
            args.max_sources,
            args.min_source_area,
            args.min_source_snr,
            args.mask_padding,
            args.blend_init_sigma,
        )
        print(f"[scarlet] blend-region sources={len(sources)}")
    else:
        sources = make_sources(model_frame, observation, centers, args.source_thresh, args.scarlet_min_snr)
        print(f"[scarlet] extended sources={len(sources)}")
    blend = scarlet.Blend(sources, observation)
    it, log_l = blend.fit(args.iterations, e_rel=args.e_rel)
    print(f"[scarlet] iterations={it} logL={log_l}")
    save_likelihood_curve(blend, args.output_dir)

    model = blend.get_model()
    rendered = observation.render(model)
    residual = cube - rendered
    save_deblend_figure(
        args.output_dir / "scarlet_deblend.png",
        cube,
        rendered,
        residual,
        args.display_percentiles,
    )
    save_deblend_figure(
        args.output_dir / "scarlet_deblend_autoscale.png",
        cube,
        rendered,
        residual,
        args.display_percentiles,
        autoscale=True,
    )
    scarlet.display.show_sources(
        sources,
        norm=make_scarlet_norm(cube, args.display_percentiles),
        observation=observation,
        show_model=True,
        show_rendered=True,
        show_observed=False,
        add_markers=True,
        add_boxes=True,
    )
    plt.savefig(args.output_dir / "scarlet_sources.png", dpi=180)
    plt.close("all")

    try:
        with (args.output_dir / "scarlet_sources.sca").open("wb") as f:
            pickle.dump(sources, f)
    except Exception as exc:
        print(f"[WARN] Could not pickle scarlet sources: {exc}")

    # Write scarlet configuration and runtime summary into the output dir
    try:
        scarlet_cfg = {
            "sam_config": sam_config,
            "scarlet": {
                "psf_sigma": list(sigma_vals),
                "variance": float(args.variance),
                "channels": list(model_frame.channels),
                "model_frame_shape": list(cube.shape),
                "weights_shape": list(weights.shape),
                "source_mode": args.source_mode,
                "centers": [[float(y), float(x)] for (y, x) in centers],
                "n_centers": len(centers),
                "n_sources": len(sources) if 'sources' in locals() else None,
                "scarlet_params": {
                    "iterations_requested": int(args.iterations),
                    "iterations_run": int(it) if 'it' in locals() else None,
                    "e_rel": float(args.e_rel),
                    "scarlet_min_snr": float(args.scarlet_min_snr),
                },
                "log_likelihood": float(log_l) if 'log_l' in locals() else None,
            },
        }
        (args.output_dir / "scarlet_config.json").write_text(json.dumps(scarlet_cfg, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote scarlet configuration to {(args.output_dir / 'scarlet_config.json')}")
    except Exception as exc:
        print(f"[WARN] Could not write scarlet_config.json to output directory: {exc}")


if __name__ == "__main__":
    main()
