#!/usr/bin/env python3
"""Visualize SExtractor results on a FITS image.

Creates an RGB/TIFF quicklook with either segmented regions (label map)
or source ellipses (from a SExtractor catalog) overlaid on the original image.

Outputs are standard RGB TIFFs which can be opened in FIJI.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table
from PIL import Image, ImageDraw

from astropy.visualization import ZScaleInterval


def robust_to_uint8(image_f32: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    finite = np.isfinite(image_f32)
    if not np.any(finite):
        raise ValueError("Image has no finite values")
    vals = image_f32[finite]
    lo = float(np.percentile(vals, low_pct))
    hi = float(np.percentile(vals, high_pct))
    # Better visualize dim sources by clipping to zscale limits instead of hard percentiles
    lo, hi = ZScaleInterval().get_limits(image_f32)
    if hi <= lo:
        hi = lo + 1e-6
    clipped = np.clip(image_f32, lo, hi)
    norm = (clipped - lo) / (hi - lo)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    return (norm * 255.0).astype(np.uint8)


def ellipse_polygon(xc, yc, a, b, theta_deg, n=180):
    """Return polygon points approximating a rotated ellipse.

    xc, yc: center (pixel coords)
    a, b: semi-major, semi-minor in pixels (SExtractor's A_IMAGE/B_IMAGE)
    theta_deg: rotation angle in degrees (SExtractor's THETA_IMAGE: deg CCW from +x)
    """
    theta = np.deg2rad(theta_deg)
    t = np.linspace(0, 2 * np.pi, n)
    x = a * np.cos(t)
    y = b * np.sin(t)
    # rotate
    xr = x * np.cos(theta) - y * np.sin(theta)
    yr = x * np.sin(theta) + y * np.cos(theta)
    pts_x = xc + xr
    pts_y = yc + yr
    return list(zip(pts_x.tolist(), pts_y.tolist()))


def overlay_masks_on_rgb(base_rgb: np.ndarray, label_map: np.ndarray, alpha: float = 0.4, seed: int = 1234) -> np.ndarray:
    h, w = label_map.shape
    assert base_rgb.shape[:2] == (h, w)
    out = base_rgb.astype(np.float32).copy()
    rng = np.random.default_rng(seed)
    labels = np.unique(label_map)
    labels = labels[labels > 0]
    for lab in labels:
        mask = label_map == lab
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        out[mask] = (1 - alpha) * out[mask] + alpha * color.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_ellipses_on_rgb(base_rgb: np.ndarray, catalog: Table, alpha: float = 1.0, edge_color: Tuple[int, int, int] = (255, 0, 0), origin: str = "top") -> np.ndarray:
    h, w = base_rgb.shape[:2]
    overlay = Image.fromarray(base_rgb).convert("RGBA")
    draw_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)

    # candidate field names used by SExtractor
    x_names = ["X_IMAGE", "x", "X"]
    y_names = ["Y_IMAGE", "y", "Y"]
    a_names = ["A_IMAGE", "a_image", "A"]
    b_names = ["B_IMAGE", "b_image", "B"]
    theta_names = ["THETA_IMAGE", "theta_image", "THETA"]

    def find_field(tbl, candidates):
        for c in candidates:
            if c in tbl.colnames:
                return c
        return None

    xf = find_field(catalog, x_names)
    yf = find_field(catalog, y_names)
    af = find_field(catalog, a_names)
    bf = find_field(catalog, b_names)
    tf = find_field(catalog, theta_names)

    if None in (xf, yf, af, bf, tf):
        raise ValueError("Catalog is missing one of X_IMAGE/Y_IMAGE/A_IMAGE/B_IMAGE/THETA_IMAGE columns")

    for row in catalog:
        x = float(row[xf]) - 1.0  # convert 1-based -> 0-based
        y = float(row[yf]) - 1.0
        if origin == "bottom":
            y = h - 1 - y
        a = float(row[af])
        b = float(row[bf])
        theta = float(row[tf])
        pts = ellipse_polygon(x, y, a, b, theta, n=180)
        # draw outline
        draw.line(pts + [pts[0]], fill=edge_color + (255,), width=2)

    combined = Image.alpha_composite(overlay, draw_layer)
    return np.array(combined.convert("RGB"))


def ellipse_catalog_from_label_map(label_map: np.ndarray, min_area: int = 3) -> Table:
    rows: List[Tuple[float, float, float, float, float]] = []
    labels = np.unique(label_map)
    labels = labels[labels > 0]
    for lab in labels:
        ys, xs = np.where(label_map == lab)
        if xs.size < min_area:
            continue

        xc = float(xs.mean())
        yc = float(ys.mean())
        x = xs.astype(np.float64) - xc
        y = ys.astype(np.float64) - yc
        cov = np.cov(np.vstack([x, y]), bias=True)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = np.maximum(vals[order], 1e-6)
        vecs = vecs[:, order]

        # For a filled ellipse, coordinate covariance is a^2/4 and b^2/4.
        a = float(2.0 * np.sqrt(vals[0]))
        b = float(2.0 * np.sqrt(vals[1]))
        theta = float(np.rad2deg(np.arctan2(vecs[1, 0], vecs[0, 0])))
        rows.append((xc + 1.0, yc + 1.0, a, b, theta))

    return Table(rows=rows, names=["X_IMAGE", "Y_IMAGE", "A_IMAGE", "B_IMAGE", "THETA_IMAGE"])


def read_label_map(path: Path, hdu: int = 0) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[hdu].data
        if data is None:
            raise ValueError(f"No image data in HDU {hdu} of {path}")
        return np.asarray(data)


def main():
    parser = argparse.ArgumentParser(description="Visualize SExtractor segmentation/ellipses on FITS image for FIJI quicklook.")
    parser.add_argument("--image", required=True, help="Input FITS image (2D)")
    parser.add_argument("--segmentation", help="Label-map FITS (same shape as image). If provided, masks are overlaid.")
    parser.add_argument("--catalog", help="SExtractor catalog (ASCII or FITS) to draw ellipses.")
    parser.add_argument(
        "--label-ellipses",
        action="store_true",
        help="Draw moment-based ellipses computed from the segmentation label map.",
    )
    parser.add_argument(
        "--no-mask-overlay",
        action="store_true",
        help="When --segmentation is set, do not draw filled segmentation masks.",
    )
    parser.add_argument("--hdu", type=int, default=0, help="HDU index for image/segmentation (default 0)")
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay alpha for masks (0-1)")
    parser.add_argument("--edge-color", type=int, nargs=3, default=[255, 0, 0], help="Ellipse edge RGB color")
    parser.add_argument("--origin", choices=["top", "bottom"], default="top", help="Whether catalog Y coords origin is top or bottom (default top)")
    parser.add_argument("--out", required=True, help="Output RGB TIFF/PNG path for FIJI")
    args = parser.parse_args()

    img_path = Path(args.image)
    with fits.open(img_path, memmap=True) as hdul:
        img = hdul[args.hdu].data
        if img is None:
            raise ValueError(f"No image in HDU {args.hdu} of {img_path}")
        img_f32 = np.asarray(img, dtype=np.float32)

    u8 = robust_to_uint8(img_f32, low_pct=0.5, high_pct=99.5)
    base_rgb = np.stack([u8, u8, u8], axis=2)

    out_rgb = base_rgb.copy()

    if args.segmentation and not args.no_mask_overlay:
        seg = read_label_map(Path(args.segmentation), hdu=args.hdu)
        out_rgb = overlay_masks_on_rgb(out_rgb, seg, alpha=args.alpha)

    if args.catalog:
        # read catalog with astropy Table; try SExtractor-specific reader first
        cat_path = Path(args.catalog)
        cat = None
        for fmt in ("ascii.sextractor", "ascii.commented_header", "ascii"):
            try:
                cat = Table.read(cat_path, format=fmt)
                break
            except Exception:
                cat = None
        if cat is None:
            raise RuntimeError(f"Unable to read catalog file: {cat_path}. Tried ascii.sextractor, ascii.commented_header, ascii.")
        out_rgb = overlay_ellipses_on_rgb(out_rgb, cat, edge_color=tuple(args.edge_color), origin=args.origin)

    if args.label_ellipses:
        if not args.segmentation:
            raise ValueError("--label-ellipses requires --segmentation")
        seg = read_label_map(Path(args.segmentation), hdu=args.hdu)
        cat = ellipse_catalog_from_label_map(seg)
        out_rgb = overlay_ellipses_on_rgb(out_rgb, cat, edge_color=tuple(args.edge_color), origin=args.origin)
    
    # Flip the image vertically
    out_rgb = np.flipud(out_rgb)

    out_path = Path(args.out)
    if str(args.out).endswith("/") or (out_path.exists() and out_path.is_dir()):
        out_path = out_path / "sextractor_overlay.tif"
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".tif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_rgb).save(out_path, compression="tiff_lzw")
    print(f"Saved overlay to {out_path}")


if __name__ == "__main__":
    main()
