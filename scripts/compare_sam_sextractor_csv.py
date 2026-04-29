#!/usr/bin/env python3
"""Compare SExtractor catalogs with SAM metadata CSV files.

The main check is whether each SExtractor source centroid falls inside at least
one SAM mask bounding box. SExtractor pixel coordinates are treated as 1-based,
so they are converted to SAM's 0-based pixel coordinates before matching.
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match SExtractor sources against SAM metadata CSV bounding boxes."
    )
    parser.add_argument("--sex-cat", type=Path, help="SExtractor result.cat file.")
    parser.add_argument("--sam-csv", type=Path, help="SAM *_metadata.csv file.")
    parser.add_argument(
        "--root",
        type=Path,
        help=(
            "Root containing ROI result folders. When set, the script compares every "
            "roi_*/sextractor_results/result.cat against SAM CSVs under roi_*/sam_results."
        ),
    )
    parser.add_argument("--sam-glob", default="sam_results/**/*.csv")
    parser.add_argument("--out", type=Path, required=True, help="Output summary CSV.")
    parser.add_argument("--missed-out", type=Path, help="Optional per-source missed CSV.")
    parser.add_argument(
        "--pad",
        type=float,
        default=0.0,
        help="Pad each SAM bbox by this many pixels before centroid-in-box matching.",
    )
    parser.add_argument(
        "--min-sex-area",
        type=float,
        default=0.0,
        help="Ignore SExtractor rows with ISOAREA_IMAGE smaller than this value.",
    )
    return parser.parse_args()


def read_sextractor(path: Path) -> List[Dict[str, float]]:
    columns: List[str] = []
    rows: List[Dict[str, float]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                parts = stripped[1:].split()
                if len(parts) >= 2 and parts[0].isdigit():
                    idx = int(parts[0])
                    while len(columns) < idx:
                        columns.append("")
                    columns[idx - 1] = parts[1]
                continue
            values = stripped.split()
            if not columns:
                raise ValueError(f"No SExtractor header columns found in {path}")
            row: Dict[str, float] = {}
            for name, value in zip(columns, values):
                if name:
                    row[name] = float(value)
            rows.append(row)
    return rows


def read_sam(path: Path) -> List[Dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({key: float(value) for key, value in row.items() if value != ""})
    return rows


def inside_bbox(x: float, y: float, mask: Dict[str, float], pad: float) -> bool:
    x0 = mask["bbox_x0"] - pad
    y0 = mask["bbox_y0"] - pad
    x1 = mask["bbox_x0"] + mask["bbox_w"] + pad
    y1 = mask["bbox_y0"] + mask["bbox_h"] + pad
    return x0 <= x <= x1 and y0 <= y <= y1


def bbox_center(mask: Dict[str, float]) -> Tuple[float, float]:
    return mask["bbox_x0"] + mask["bbox_w"] / 2.0, mask["bbox_y0"] + mask["bbox_h"] / 2.0


def compare_pair(
    sex_cat: Path,
    sam_csv: Path,
    *,
    pad: float,
    min_sex_area: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    sex_rows = [
        row
        for row in read_sextractor(sex_cat)
        if row.get("ISOAREA_IMAGE", 0.0) >= min_sex_area
    ]
    sam_rows = read_sam(sam_csv)
    missed: List[Dict[str, object]] = []
    matched = 0

    for row in sex_rows:
        # SExtractor positions are FITS-style 1-based pixels; SAM metadata uses
        # zero-based image coordinates.
        x = row["X_IMAGE"] - 1.0
        y = row["Y_IMAGE"] - 1.0
        containing = [mask for mask in sam_rows if inside_bbox(x, y, mask, pad)]
        if containing:
            matched += 1
            continue

        nearest_id: Optional[int] = None
        nearest_dist = math.inf
        for mask in sam_rows:
            cx, cy = bbox_center(mask)
            dist = math.hypot(x - cx, y - cy)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = int(mask.get("id", -1))

        missed.append(
            {
                "sex_cat": str(sex_cat),
                "sam_csv": str(sam_csv),
                "sex_number": int(row.get("NUMBER", -1)),
                "x0": f"{x:.3f}",
                "y0": f"{y:.3f}",
                "isoarea": row.get("ISOAREA_IMAGE", ""),
                "flux_auto": row.get("FLUX_AUTO", ""),
                "flags": int(row.get("FLAGS", 0)),
                "nearest_sam_id": nearest_id if nearest_id is not None else "",
                "nearest_sam_center_dist": f"{nearest_dist:.3f}" if math.isfinite(nearest_dist) else "",
            }
        )

    total = len(sex_rows)
    summary = {
        "sex_cat": str(sex_cat),
        "sam_csv": str(sam_csv),
        "sex_count": total,
        "sam_count": len(sam_rows),
        "matched_sex_count": matched,
        "missed_sex_count": total - matched,
        "matched_fraction": f"{matched / total:.6f}" if total else "",
        "pad": pad,
        "min_sex_area": min_sex_area,
    }
    return summary, missed


def iter_pairs(root: Path, sam_glob: str) -> Iterable[Tuple[Path, Path]]:
    for roi_dir in sorted(root.glob("roi_*")):
        sex_cat = roi_dir / "sextractor_results" / "result.cat"
        if not sex_cat.exists():
            continue
        for sam_csv in sorted(roi_dir.glob(sam_glob)):
            yield sex_cat, sam_csv


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.root:
        pairs = list(iter_pairs(args.root.expanduser(), args.sam_glob))
    elif args.sex_cat and args.sam_csv:
        pairs = [(args.sex_cat.expanduser(), args.sam_csv.expanduser())]
    else:
        raise SystemExit("Use either --root or both --sex-cat and --sam-csv.")

    summaries: List[Dict[str, object]] = []
    all_missed: List[Dict[str, object]] = []
    for sex_cat, sam_csv in pairs:
        summary, missed = compare_pair(
            sex_cat,
            sam_csv,
            pad=args.pad,
            min_sex_area=args.min_sex_area,
        )
        summaries.append(summary)
        all_missed.extend(missed)

    write_csv(args.out.expanduser(), summaries)
    if args.missed_out:
        write_csv(args.missed_out.expanduser(), all_missed)

    total_sex = sum(int(row["sex_count"]) for row in summaries)
    total_matched = sum(int(row["matched_sex_count"]) for row in summaries)
    total_missed = sum(int(row["missed_sex_count"]) for row in summaries)
    frac = total_matched / total_sex if total_sex else 0.0
    print(f"pairs={len(summaries)} sex={total_sex} matched={total_matched} missed={total_missed} matched_fraction={frac:.4f}")
    print(f"summary_csv={args.out.expanduser()}")
    if args.missed_out:
        print(f"missed_csv={args.missed_out.expanduser()}")


if __name__ == "__main__":
    main()
