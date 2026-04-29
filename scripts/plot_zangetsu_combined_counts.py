#!/usr/bin/env python3
"""Plot combined noisy/denoised six-ROI mask counts."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "results" / "zangetsu_small_sigmaclip_sex"
DEFAULT_CASES = ("denoised", "noisy")
DEFAULT_MODES = ("mode_none", "mode_astro_rgb")
BAR_LABELS = ("SExtractor", "SAM none") #,"SAM astro_rgb"
BAR_COLORS = ("#4a4a4a", "#2f80ed") #, "#f2994a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine denoised/noisy six-ROI count summaries into one grouped bar chart."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def count_sextractor_catalog(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def read_mask_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("mask_count", 0))


def roi_sort_key(path: Path) -> Tuple[int, str]:
    parts = path.name.split("_")
    if len(parts) > 1 and parts[1].isdigit():
        return int(parts[1]), path.name
    return 999, path.name


def roi_label(roi_dir: Path) -> str:
    meta_path = roi_dir / "roi_metadata.json"
    roi_id = roi_dir.name.split("_")[1] if "_" in roi_dir.name else roi_dir.name
    if not meta_path.exists():
        return f"ROI {roi_id}"
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    return f"ROI {roi_id}\n({meta['x_left']}, {meta['y_bottom']})"


def collect_case(root: Path, case: str) -> Tuple[List[str], np.ndarray]:
    case_dir = root / case
    roi_dirs = sorted((p for p in case_dir.glob("roi_*") if p.is_dir()), key=roi_sort_key)
    labels: List[str] = []
    rows: List[List[int]] = []

    for roi_dir in roi_dirs:
        labels.append(roi_label(roi_dir))
        sextractor = count_sextractor_catalog(roi_dir / "sextractor_results" / "result.cat")
        mode_counts = [
            read_mask_count(
                roi_dir / "sam_results" / "astro_rgb_mode" / mode / "experiment_metadata.json"
            )
            for mode in DEFAULT_MODES
        ]
        rows.append([sextractor, *mode_counts])

    if not rows:
        raise FileNotFoundError(f"No ROI result directories found under {case_dir}")
    return labels, np.asarray(rows, dtype=int)


def plot(root: Path, cases: List[str], out: Path, dpi: int) -> None:
    collected: Dict[str, Tuple[List[str], np.ndarray]] = {
        case: collect_case(root, case) for case in cases
    }
    ymax = max(int(values.max()) for _, values in collected.values())

    fig, axes = plt.subplots(
        len(cases),
        1,
        figsize=(13, 4.2 * len(cases)),
        sharex=True,
        constrained_layout=True,
    )
    if len(cases) == 1:
        axes = [axes]

    width = 0.24
    offsets = np.array([-width, 0.0, width])

    for ax, case in zip(axes, cases):
        labels, values = collected[case]
        x = np.arange(len(labels))
        for i, (bar_label, color) in enumerate(zip(BAR_LABELS, BAR_COLORS)):
            bars = ax.bar(x + offsets[i], values[:, i], width, label=bar_label, color=color)
            ax.bar_label(bars, padding=2, fontsize=8)

        ax.set_title(case.capitalize())
        ax.set_ylabel("Mask / source count")
        ax.set_ylim(0, ymax * 1.18 if ymax else 1)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    axes[-1].set_xlabel("Region")
    axes[0].legend(ncols=len(BAR_LABELS), loc="upper right")
    fig.suptitle("Zangetsu noisy/denoised six-ROI mask counts", fontsize=15)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    fig.savefig(out.with_suffix(".tif"), dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out = args.out
    if out is None:
        out = root / "combined_noisy_denoised_roi_counts.png"
    plot(root, args.cases, out.expanduser().resolve(), args.dpi)


if __name__ == "__main__":
    main()
