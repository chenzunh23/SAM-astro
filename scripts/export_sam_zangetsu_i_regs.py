#!/usr/bin/env python3
"""Export SAM-CELLECT Zangetsu lower-right HSC-I seg and shape DS9 regions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader


CELLECT_ROOT = Path("/home/czh23/CELLECT")
sys.path.insert(0, str(CELLECT_ROOT))

from astro_cellect2d import MultiBandAstroCELLECT2D  # noqa: E402
from astro_train_data import AstroCutoutDataset, collate_cutouts, discover_cutout_records  # noqa: E402
from astro_train_ops import build_cellect_style_segmentation, detect_centers, detect_centers_with_en, unwrap_model  # noqa: E402
from sam_backbone import build_sam_cellect2d  # noqa: E402


BAND = "HSC-I"
BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")
TRACT = "9813"
PATCH = "6,1"
TILE = "zangetsu_lower_right_x27366_y6453"
MATCH_RADIUS_PIX = 0.5 / 0.168
DATA_ROOT = (
    CELLECT_ROOT
    / "output/sam_cellect_combination_260611/preprocessing_diagnostics_260611/zangetsu_preprocessed_cutouts_260611"
)
REG_HEADER = [
    "# Region file format: DS9 version 4.1",
    'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" '
    "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
    "image",
]


def _read_config(path: Path) -> dict:
    payload = json.loads(path.read_text())
    args = dict(payload.get("args", {}))
    args["_top"] = payload
    return args


def _checkpoint_epoch(path: Path) -> object:
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        return ""
    return ckpt.get("epoch", "") if isinstance(ckpt, dict) else ""


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def _checkpoint_variant(checkpoint: Path) -> str:
    try:
        ckpt = torch.load(checkpoint, map_location="cpu")
    except Exception:
        return ""
    if isinstance(ckpt, dict):
        variant = ckpt.get("model_variant")
        if variant:
            return str(variant)
        args = ckpt.get("args")
        if isinstance(args, dict) and args.get("model_variant"):
            return str(args["model_variant"])
        state = ckpt.get("model", {})
    else:
        state = ckpt
    if isinstance(state, dict):
        first_key = next(iter(state), "")
        if str(first_key).startswith(("encoder.", "module.encoder.")):
            return "sam_per_band"
        if str(first_key).startswith(("backbone.", "module.backbone.")):
            return "per_band"
    return ""


def _make_model(cfg: dict, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    top = cfg.get("_top", {})
    variant = _checkpoint_variant(checkpoint) or str(top.get("model_variant") or cfg.get("model_variant", "sam_per_band"))
    if variant == "per_band":
        model = MultiBandAstroCELLECT2D(
            num_bands=len(BANDS),
            seg_classes=int(top.get("seg_classes") or cfg.get("seg_classes", 2)),
            confidence_levels=5,
            embedding_dim=int(cfg.get("embedding_dim", 64)),
            base_channels=int(cfg.get("base_channels", 32)),
            shape_channels=3,
            candidate_count=int(cfg.get("matcher_candidate_count", 5)),
            shape_feature_dim=6,
        ).to(device)
    elif variant == "sam_per_band":
        model = build_sam_cellect2d(
            str(top.get("sam_model_type") or cfg.get("sam_model_type", "vit_b")),
            checkpoint=None,
            num_bands=len(BANDS),
            image_size=512,
            patch_size=16,
            seg_classes=int(top.get("seg_classes") or cfg.get("seg_classes", 2)),
            confidence_levels=5,
            embedding_dim=int(cfg.get("embedding_dim", 64)),
            shape_channels=3,
            use_cen=bool(top.get("sam_cen_enabled", not bool(cfg.get("disable_sam_cen", False)))),
            candidate_count=int(cfg.get("matcher_candidate_count", 5)),
            shape_feature_dim=6,
            enable_matchers=True,
            astro_preprocess_in_model=bool(top.get("sam_astro_preprocess_in_model", False)),
        ).to(device)
    else:
        raise ValueError(f"Unsupported checkpoint model_variant={variant!r} for {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(_strip_module_prefix(state))
    if isinstance(ckpt, dict):
        if hasattr(model, "EX") and ckpt.get("EX") is not None:
            model.EX.load_state_dict(_strip_module_prefix(ckpt["EX"]))
        if hasattr(model, "EN") and ckpt.get("EN") is not None:
            model.EN.load_state_dict(_strip_module_prefix(ckpt["EN"]))
    model.eval()
    return model


def _dataset(root: Path, cfg: dict) -> DataLoader:
    records = discover_cutout_records(root, bands=BANDS)
    records = [rec for rec in records if rec.patch == PATCH and rec.tile_name == TILE]
    if len(records) != 1:
        raise RuntimeError(f"Expected one record for {PATCH}/{TILE} under {root}, got {len(records)}")
    ds = AstroCutoutDataset(
        records,
        fits_hdu=int(cfg.get("fits_hdu", 1)),
        confidence_levels=5,
        ellipse_sigma=float(cfg.get("ellipse_sigma", 2.0)),
        core_radius=int(cfg.get("core_radius", 2)),
        shape_source=str(cfg.get("shape_source", "kron")),
        source_filter=str(cfg.get("source_filter", "nchild0")),
        load_eval_ignore_sources=True,
        augment=False,
    )
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_cutouts)


def _band_outputs(outputs: dict[str, torch.Tensor], band_idx: int) -> dict[str, torch.Tensor]:
    selected: dict[str, torch.Tensor] = {}
    for key, value in outputs.items():
        if not torch.is_tensor(value):
            continue
        if value.ndim >= 5:
            selected[key] = value[:, band_idx].float()
        else:
            selected[key] = value.float() if torch.is_floating_point(value) else value
    return selected


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _ellipse(x: float, y: float, major: float, minor: float, theta: float, color: str = "cyan") -> str:
    major = max(abs(_safe_float(major, 1.0)), 1.0)
    minor = max(abs(_safe_float(minor, major)), 1.0)
    theta = _safe_float(theta, 0.0)
    return f"ellipse({x + 1:.3f},{y + 1:.3f},{major:.3f},{minor:.3f},{math.degrees(theta):.3f}) # color={color} width=2"


def _point(x: float, y: float, color: str = "green") -> str:
    return f"circle({x + 1:.3f},{y + 1:.3f},3.000) # color={color} width=2"


def _mask_boundary_loops(mask: np.ndarray) -> list[list[tuple[float, float]]]:
    """Return closed pixel-edge loops for a binary mask in DS9 image coordinates."""

    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def add_edge(a: tuple[int, int], b: tuple[int, int]) -> None:
        edges.add((a, b) if a <= b else (b, a))

    ys, xs = np.where(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        x0 = 2 * x + 1
        x1 = 2 * x + 3
        y0 = 2 * y + 1
        y1 = 2 * y + 3
        if y == 0 or not mask[y - 1, x]:
            add_edge((x0, y0), (x1, y0))
        if x == width - 1 or not mask[y, x + 1]:
            add_edge((x1, y0), (x1, y1))
        if y == height - 1 or not mask[y + 1, x]:
            add_edge((x1, y1), (x0, y1))
        if x == 0 or not mask[y, x - 1]:
            add_edge((x0, y1), (x0, y0))

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    unused = set(edges)
    loops: list[list[tuple[float, float]]] = []

    def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (a, b) if a <= b else (b, a)

    while unused:
        start, nxt = next(iter(unused))
        loop_i: list[tuple[int, int]] = [start]
        prev, curr = start, nxt
        unused.remove(edge_key(prev, curr))
        while True:
            loop_i.append(curr)
            if curr == start:
                break
            candidates = [node for node in adjacency.get(curr, []) if edge_key(curr, node) in unused]
            if not candidates:
                break
            non_backtracking = [node for node in candidates if node != prev]
            nxt = sorted(non_backtracking or candidates)[0]
            unused.remove(edge_key(curr, nxt))
            prev, curr = curr, nxt
        if len(loop_i) >= 4 and loop_i[0] == loop_i[-1]:
            loops.append([(x / 2.0, y / 2.0) for x, y in loop_i[:-1]])
    return loops


def _polygon_line(points: list[tuple[float, float]], *, color: str, width: int = 1, text: str = "") -> str:
    coords = ",".join(f"{x:.2f},{y:.2f}" for x, y in points)
    suffix = f" # color={color} width={width}"
    if text:
        suffix += f" text={{{text}}}"
    return f"polygon({coords}){suffix}"


def _thin_loop(points: list[tuple[float, float]], max_vertices: int) -> list[tuple[float, float]]:
    if len(points) <= max_vertices:
        return points
    step = max(1, int(math.ceil(len(points) / max_vertices)))
    thinned = points[::step]
    return thinned if len(thinned) >= 3 else points[:max_vertices]


def _iter_seg_polygons(mask: np.ndarray, *, max_vertices: int = 160) -> Iterable[str]:
    try:
        from skimage import measure

        contours = measure.find_contours(mask.astype(np.float32), 0.5)
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            step = max(1, int(math.ceil(contour.shape[0] / max_vertices)))
            pts = contour[::step]
            if pts.shape[0] < 3:
                continue
            coords = ",".join(f"{x + 1:.2f},{y + 1:.2f}" for y, x in pts)
            yield f"polygon({coords}) # color=green width=1"
        return
    except Exception:
        pass

    for loop in _mask_boundary_loops(mask):
        if len(loop) >= 3:
            yield _polygon_line(_thin_loop(loop, max_vertices), color="green", width=1)


def _color_for_instance(instance_id: int) -> str:
    colors = ("cyan", "magenta", "yellow", "green", "blue", "red")
    return colors[(int(instance_id) - 1) % len(colors)]


def _iter_instance_polygons(instance: np.ndarray, *, max_vertices: int = 96) -> Iterable[str]:
    labels = [int(label) for label in np.unique(instance) if int(label) > 0]
    try:
        from skimage import measure

        for label in labels:
            mask = instance == label
            contours = measure.find_contours(mask.astype(np.float32), 0.5)
            color = _color_for_instance(label)
            for contour in contours:
                if contour.shape[0] < 3:
                    continue
                step = max(1, int(math.ceil(contour.shape[0] / max_vertices)))
                pts = contour[::step]
                if pts.shape[0] < 3:
                    continue
                coords = ",".join(f"{x + 1:.2f},{y + 1:.2f}" for y, x in pts)
                yield f"polygon({coords}) # color={color} width=1 text={{id={label}}}"
        return
    except Exception:
        pass

    for label in labels:
        color = _color_for_instance(label)
        for loop in _mask_boundary_loops(instance == label):
            if len(loop) >= 3:
                yield _polygon_line(_thin_loop(loop, max_vertices), color=color, width=1, text=f"id={label}")


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _instance_rgb(label: int) -> tuple[float, float, float]:
    palette = (
        (0.000, 0.760, 0.940),
        (0.950, 0.180, 0.650),
        (1.000, 0.820, 0.120),
        (0.160, 0.720, 0.330),
        (0.180, 0.380, 1.000),
        (0.950, 0.220, 0.140),
        (0.900, 0.560, 0.120),
        (0.580, 0.320, 0.900),
    )
    return palette[(int(label) - 1) % len(palette)]


def _write_mask_overlay(path: Path, image: np.ndarray, instance: np.ndarray, *, alpha: float = 0.38) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Overlay image must be 2D, got {image.shape}")
    finite = np.isfinite(image)
    if bool(finite.any()):
        lo, hi = np.nanpercentile(image[finite], [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(image[finite])), float(np.nanmax(image[finite]))
        base = np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        base = np.zeros_like(image, dtype=np.float32)

    rgb = np.repeat(base[..., None], 3, axis=2)
    labels = [int(label) for label in np.unique(instance) if int(label) > 0]
    for label in labels:
        mask = instance == label
        if not bool(mask.any()):
            continue
        color = np.asarray(_instance_rgb(label), dtype=np.float32)
        rgb[mask] = (1.0 - float(alpha)) * rgb[mask] + float(alpha) * color

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 7.2), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, origin="upper", interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def _run_one(
    *,
    model: torch.nn.Module,
    cfg: dict,
    dataset_name: str,
    out_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    loader = _dataset(DATA_ROOT / dataset_name / TRACT, cfg)
    band_idx = BANDS.index(BAND)
    base_model = unwrap_model(model)
    use_en = bool(cfg.get("use_en_postprocess", True)) and hasattr(base_model, "EN")
    threshold = float(cfg.get("confidence_threshold", 2.0))
    nms_radius = int(cfg.get("nms_radius", 1))
    confidence_score = str(cfg.get("confidence_score", "cellect"))
    center_refinement = str(cfg.get("center_refinement", "softargmax"))
    center_refinement_radius = int(cfg.get("center_refinement_radius", 1))
    en_threshold = float(cfg.get("en_postprocess_threshold", 0.6))
    candidate_count = int(cfg.get("matcher_candidate_count", 5))

    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32)
        outputs = model(image)
        image_i = image[0, band_idx].detach().cpu().numpy().astype(np.float32)
        outputs_i = _band_outputs(outputs, band_idx)
        if use_en:
            pred_list = detect_centers_with_en(
                base_model,
                outputs_i,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                match_radius=MATCH_RADIUS_PIX,
                candidate_count=candidate_count,
                en_threshold=en_threshold,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
            )
        else:
            pred_list = detect_centers(
                outputs_i,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
            )
        pred_xy = np.asarray(pred_list[0], dtype=np.float32).reshape(-1, 2)
        seg_mask = (outputs_i["seg_logits"][0].argmax(dim=0).detach().cpu().numpy() > 0)
        shape = outputs_i["shape"][0].detach().cpu().numpy().astype(np.float32)
        break

    prefix = f"{dataset_name}_{PATCH.replace(',', '_')}_{TILE}_{BAND.replace('-', '_')}"
    seg_path = out_dir / dataset_name / f"{prefix}_seg.reg"
    shape_path = out_dir / dataset_name / f"{prefix}_shape.reg"
    centers_path = out_dir / dataset_name / f"{prefix}_centers.reg"
    instances_path = out_dir / dataset_name / f"{prefix}_instances.reg"
    overlay_path = out_dir / dataset_name / f"{prefix}_instances_overlay.png"

    instance_result = build_cellect_style_segmentation(
        outputs_i,
        [pred_xy],
        ellipse_sigma=float(cfg.get("ellipse_sigma", 2.0)),
    )[0]
    instance_mask = np.asarray(instance_result["seg_mask"], dtype=np.int32)

    seg_lines = REG_HEADER + [f"# {dataset_name} {PATCH}/{TILE} {BAND}: predicted foreground segmentation contours"]
    seg_lines.extend(_iter_seg_polygons(seg_mask))
    shape_lines = REG_HEADER + [f"# {dataset_name} {PATCH}/{TILE} {BAND}: predicted shape ellipses at detected centers"]
    center_lines = REG_HEADER + [f"# {dataset_name} {PATCH}/{TILE} {BAND}: detected centers"]
    instance_lines = REG_HEADER + [
        f"# {dataset_name} {PATCH}/{TILE} {BAND}: CELLECT-style instance segmentation contours; text labels are instance ids"
    ]
    instance_lines.extend(_iter_instance_polygons(instance_mask))

    for x, y in pred_xy:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or yi < 0 or xi >= shape.shape[-1] or yi >= shape.shape[-2]:
            continue
        center_lines.append(_point(float(x), float(y)))
        shape_lines.append(_ellipse(float(x), float(y), shape[0, yi, xi], shape[1, yi, xi], shape[2, yi, xi]))

    _write_text(seg_path, seg_lines)
    _write_text(shape_path, shape_lines)
    _write_text(centers_path, center_lines)
    _write_text(instances_path, instance_lines)
    _write_mask_overlay(overlay_path, image_i, instance_mask)
    return {
        "dataset": dataset_name,
        "tile": TILE,
        "band": BAND,
        "detections": int(len(pred_xy)),
        "seg_foreground_pixels": int(seg_mask.sum()),
        "instances": int(len([label for label in np.unique(instance_mask) if int(label) > 0])),
        "seg_reg": str(seg_path),
        "shape_reg": str(shape_path),
        "centers_reg": str(centers_path),
        "instances_reg": str(instances_path),
        "instances_overlay_png": str(overlay_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, default=CELLECT_ROOT / "output/ckpts/SAM_per_band_debug_0612")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help="Checkpoint path to export. Can be passed multiple times. Defaults to best.pt and last.pt under --ckpt-dir.",
    )
    parser.add_argument(
        "--checkpoint-label",
        action="append",
        default=None,
        help="Output subdirectory label for each --checkpoint. Defaults to checkpoint stem.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--datasets", nargs="+", default=["coadd", "noisy", "denoised"])
    args = parser.parse_args()

    cfg = _read_config(args.ckpt_dir / "run_config.json")
    device = torch.device(args.device)
    rows: list[dict[str, object]] = []
    if args.checkpoint:
        ckpt_items = [(Path(path), Path(path).stem) for path in args.checkpoint]
        if args.checkpoint_label:
            if len(args.checkpoint_label) != len(ckpt_items):
                raise ValueError("--checkpoint-label must be passed once per --checkpoint")
            ckpt_items = [(path, str(label)) for (path, _stem), label in zip(ckpt_items, args.checkpoint_label)]
    else:
        ckpt_items = [(args.ckpt_dir / "best.pt", "best"), (args.ckpt_dir / "last.pt", "latest")]
    for ckpt_path, model_name in ckpt_items:
        if not ckpt_path.is_absolute():
            ckpt_path = args.ckpt_dir / ckpt_path
        model = _make_model(cfg, ckpt_path, device)
        model_out = args.out_dir / model_name
        for dataset_name in args.datasets:
            row = _run_one(model=model, cfg=cfg, dataset_name=dataset_name, out_dir=model_out, device=device)
            row["checkpoint"] = str(ckpt_path)
            row["checkpoint_label"] = model_name
            row["checkpoint_epoch"] = _checkpoint_epoch(ckpt_path)
            rows.append(row)
            print(
                f"{model_name} {dataset_name}: detections={row['detections']} "
                f"instances={row['instances']} seg_pixels={row['seg_foreground_pixels']}"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with (args.out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote summary: {args.out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
