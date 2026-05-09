"""Diagnose SAM false positives by magnitude and parent-mask properties.

This utility is intended for outputs produced by ``scarlet_deblend_from_fits.py``
with ``--detection-mode sam``.  It combines:

* ``centroid_diagnostics/false_positives.csv`` for the FP source ids,
* ``deblend/deepCoadd_deblendedFlux.fits`` for child -> parent mapping,
* ``deblend/deepCoadd_scarletModelData.pickle`` for scarlet source fluxes,
* ``sam/*_sam_metadata.csv`` for parent-mask area, predicted_iou, and
  stability_score.

Example
-------

Run on the current coordinated SAM output and write pie charts:

    python utils/diagnose_sam_false_positives.py \
        --run output/sam_coordinated \
        --output-dir output/sam_coordinated/fp_diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402


DEFAULT_AREA_BINS = (15.0, 20.0, 25.0, 30.0, 50.0, 100.0)
DEFAULT_IOU_BINS = (0.8, 0.84, 0.88, 0.92, 0.96)
DEFAULT_STABILITY_BINS = (0.95, 0.96, 0.97, 0.98, 0.99)


def _load_false_positive_ids(path: Path) -> list[int]:
    with path.open(newline="") as handle:
        return [int(row["id"]) for row in csv.DictReader(handle)]


def _find_sam_metadata(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = sorted((run_dir / "sam").glob("*_sam_metadata.csv"))
    if not candidates:
        raise FileNotFoundError(f"no *_sam_metadata.csv found under {run_dir / 'sam'}")
    multiband = [path for path in candidates if "multiband" in path.name]
    return multiband[0] if multiband else candidates[0]


def _load_parent_metadata(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "area", "predicted_iou", "stability_score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            parent_id = int(row["id"])
            out[parent_id] = {
                "area": float(row["area"]),
                "predicted_iou": float(row["predicted_iou"]),
                "stability_score": float(row["stability_score"]),
            }
    return out


def _load_child_parent_map(path: Path) -> dict[int, int]:
    table = Table.read(path, hdu=1)
    return {int(source_id): int(parent) for source_id, parent in zip(table["id"], table["parent"])}


def _load_parent_child_counts(path: Path) -> dict[int, int]:
    """Return the number of child records owned by each parent source."""
    table = Table.read(path, hdu=1)
    counts: Counter[int] = Counter()
    for parent in table["parent"]:
        parent = int(parent)
        if parent != 0:
            counts[parent] += 1

    # Parent rows usually carry deblend_nChild; use it as a fallback for any
    # parent whose children were not represented in the simple parent column.
    if "deblend_nChild" in table.colnames:
        for source_id, parent, n_child in zip(table["id"], table["parent"], table["deblend_nChild"]):
            if int(parent) == 0 and int(n_child) > 0:
                counts.setdefault(int(source_id), int(n_child))
    return dict(counts)


def _band_to_label(band: str) -> str:
    band = band.strip()
    if band.startswith("HSC-"):
        band = band.split("-", 1)[1]
    return band.lower()


def _load_scarlet_fluxes(path: Path, source_ids: Iterable[int], *, band: str) -> dict[int, float]:
    wanted = {int(source_id) for source_id in source_ids}
    with path.open("rb") as handle:
        model_data = pickle.load(handle)
    if not getattr(model_data, "blends", None):
        return {}

    first_blend = next(iter(model_data.blends.values()))
    bands = [str(value).lower() for value in getattr(first_blend, "bands", [])]
    label = _band_to_label(band)
    if label not in bands:
        raise ValueError(f"band {band!r} not found in scarlet model bands={bands}")
    band_index = bands.index(label)

    out: dict[int, float] = {}
    for blend in model_data.blends.values():
        for source_id, source in blend.sources.items():
            source_id = int(source_id)
            if source_id not in wanted:
                continue
            total = 0.0
            for component in getattr(source, "factorized_components", []):
                spectrum = np.asarray(component.spectrum, dtype=float)
                if band_index < spectrum.size and np.isfinite(spectrum[band_index]):
                    total += float(spectrum[band_index])
            out[source_id] = total
    return out


def _flux_to_mag(flux: float, zeropoint: float) -> float:
    if np.isfinite(flux) and flux > 0:
        return float(zeropoint) - 2.5 * math.log10(float(flux))
    return float("nan")


def _format_edge(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _bin_labels(edges: tuple[float, ...], *, underflow: bool = False) -> list[str]:
    labels: list[str] = []
    if underflow:
        labels.append(f"<{_format_edge(edges[0])}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"{_format_edge(lo)}-{_format_edge(hi)}")
    labels.append(f">={_format_edge(edges[-1])}")
    return labels


def _bin_counts(values: np.ndarray, edges: tuple[float, ...], *, underflow: bool = False) -> tuple[list[str], list[int]]:
    finite = values[np.isfinite(values)]
    labels = _bin_labels(edges, underflow=underflow)
    counts: list[int] = []
    if underflow:
        counts.append(int(np.count_nonzero(finite < edges[0])))
    for lo, hi in zip(edges[:-1], edges[1:]):
        counts.append(int(np.count_nonzero((finite >= lo) & (finite < hi))))
    counts.append(int(np.count_nonzero(finite >= edges[-1])))
    return labels, counts


def _value_bin_label(value: float, edges: tuple[float, ...], *, underflow: bool = False) -> str | None:
    if not np.isfinite(value):
        return None
    if underflow and value < edges[0]:
        return f"<{_format_edge(edges[0])}"
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi:
            return f"{_format_edge(lo)}-{_format_edge(hi)}"
    if value >= edges[-1]:
        return f">={_format_edge(edges[-1])}"
    return None


def _write_pie(ax, labels: list[str], counts: list[int], title: str) -> None:
    positive = [(label, count) for label, count in zip(labels, counts) if count > 0]
    if not positive:
        ax.text(0.5, 0.5, "no sources", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        return
    plot_labels = [f"{label}\n{count}" for label, count in positive]
    plot_counts = [count for _, count in positive]
    ax.pie(plot_counts, labels=plot_labels, autopct="%1.1f%%", startangle=90, counterclock=False)
    ax.set_title(title)


def _child_count_bin(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    value = int(value)
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3-4"
    return ">=5"


def _parent_child_summary(rows: list[dict[str, float]]) -> tuple[list[str], list[int], list[str], dict[str, dict[str, int]]]:
    labels = ["1", "2", "3-4", ">=5", "unknown"]
    fp_counts: Counter[str] = Counter()
    parents_by_label: dict[str, set[int]] = {label: set() for label in labels}
    parent_child_count: dict[int, int] = {}

    for row in rows:
        label = _child_count_bin(row.get("parent_child_count", float("nan")))
        parent = int(row["parent"])
        fp_counts[label] += 1
        parents_by_label[label].add(parent)
        parent_child_count[parent] = int(row.get("parent_child_count", 0))

    counts = [fp_counts[label] for label in labels]
    display_labels = []
    summary: dict[str, dict[str, int]] = {}
    for label in labels:
        child_total = sum(parent_child_count[parent] for parent in parents_by_label[label])
        parent_count = len(parents_by_label[label])
        fp_count = fp_counts[label]
        display_labels.append(f"{label}\nFP/child={fp_count}/{child_total}\nparents={parent_count}")
        summary[label] = {
            "fp_count": fp_count,
            "child_total_for_unique_parents": child_total,
            "parent_count": parent_count,
        }
    return display_labels, counts, labels, summary


def _avg_fp_per_parent_stats(
    rows: list[dict[str, float]],
    *,
    parent_rows: list[dict[str, float]],
    metric: str,
    edges: tuple[float, ...],
    underflow: bool = True,
) -> list[dict[str, float | int | str]]:
    labels = _bin_labels(edges, underflow=underflow)
    fp_counts: Counter[str] = Counter()
    all_parent_counts: Counter[str] = Counter()
    all_parent_child_totals: Counter[str] = Counter()

    for row in parent_rows:
        label = _value_bin_label(float(row[metric]), edges, underflow=underflow)
        if label is None:
            continue
        all_parent_counts[label] += 1
        all_parent_child_totals[label] += int(row.get("parent_child_count", 0))

    # ``rows`` is already restricted to the target FP magnitude group.  The
    # numerator is the number of such FP children whose parent falls in the bin.
    for row in rows:
        label = _value_bin_label(float(row[metric]), edges, underflow=underflow)
        if label is None:
            continue
        fp_counts[label] += 1

    stats: list[dict[str, float | int | str]] = []
    for label in labels:
        fp_count = int(fp_counts[label])
        parent_count = int(all_parent_counts[label])
        child_total = int(all_parent_child_totals[label])
        avg = fp_count / parent_count if parent_count else float("nan")
        avg_total_child = child_total / parent_count if parent_count else float("nan")
        stats.append(
            {
                "metric": metric,
                "bin": label,
                "fp_count": fp_count,
                "parent_count": parent_count,
                "total_child_count_for_all_parents": child_total,
                "avg_fp_children_per_parent": avg,
                "avg_total_children_per_parent": avg_total_child,
            }
        )
    return stats


def _make_avg_fp_per_parent_figure(
    *,
    rows: list[dict[str, float]],
    parent_rows: list[dict[str, float]],
    group_name: str,
    output_path: Path,
    area_bins: tuple[float, ...],
    iou_bins: tuple[float, ...],
    stability_bins: tuple[float, ...],
) -> dict[str, list[dict[str, float | int | str]]]:
    specs = [
        ("area", "Parent mask area", area_bins),
        ("predicted_iou", "SAM predicted_iou", iou_bins),
        ("stability_score", "SAM stability_score", stability_bins),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.8), constrained_layout=False)
    summary: dict[str, list[dict[str, float | int | str]]] = {}

    for ax, (metric, title, edges) in zip(axes, specs):
        stats = _avg_fp_per_parent_stats(
            rows,
            parent_rows=parent_rows,
            metric=metric,
            edges=edges,
            underflow=True,
        )
        summary[metric] = stats
        labels = [str(item["bin"]) for item in stats]
        values = np.asarray([float(item["avg_fp_children_per_parent"]) for item in stats], dtype=float)
        fp_counts = [int(item["fp_count"]) for item in stats]
        parent_counts = [int(item["parent_count"]) for item in stats]
        child_counts = [int(item["total_child_count_for_all_parents"]) for item in stats]
        x = np.arange(len(labels))
        plot_values = np.nan_to_num(values, nan=0.0)
        ax.bar(x, plot_values, color="#4c78a8", edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_ylabel("target FP children / all parents in bin")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ymax = float(np.nanmax(plot_values)) if plot_values.size else 0.0
        ax.set_ylim(0.0, max(0.1, ymax * 1.38))
        for idx, value in enumerate(plot_values):
            if fp_counts[idx] == 0:
                continue
            ax.text(
                idx,
                value + max(0.01, ymax * 0.03),
                f"{value:.2f}\nFP={fp_counts[idx]}\nP={parent_counts[idx]}\nC={child_counts[idx]}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle(f"{group_name}: target FP children per all parents in parent-property bin", fontsize=14)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.18, top=0.82, wspace=0.28)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary


def _make_group_figure(
    *,
    rows: list[dict[str, float]],
    group_name: str,
    output_path: Path,
    area_bins: tuple[float, ...],
    iou_bins: tuple[float, ...],
    stability_bins: tuple[float, ...],
    include_child_count_pie: bool = False,
) -> dict[str, object]:
    area = np.asarray([row["area"] for row in rows], dtype=float)
    iou = np.asarray([row["predicted_iou"] for row in rows], dtype=float)
    stability = np.asarray([row["stability_score"] for row in rows], dtype=float)

    ncols = 4 if include_child_count_pie else 3
    width = 18.5 if include_child_count_pie else 14.5
    fig, axes = plt.subplots(1, ncols, figsize=(width, 4.8), constrained_layout=True)
    summaries: dict[str, object] = {}

    labels, counts = _bin_counts(area, area_bins, underflow=True)
    summaries["area"] = Counter(dict(zip(labels, counts)))
    _write_pie(axes[0], labels, counts, "Parent mask area")

    labels, counts = _bin_counts(iou, iou_bins, underflow=True)
    summaries["predicted_iou"] = Counter(dict(zip(labels, counts)))
    _write_pie(axes[1], labels, counts, "SAM predicted_iou")

    labels, counts = _bin_counts(stability, stability_bins, underflow=True)
    summaries["stability_score"] = Counter(dict(zip(labels, counts)))
    _write_pie(axes[2], labels, counts, "SAM stability_score")

    display_labels, counts, raw_labels, child_summary = _parent_child_summary(rows)
    summaries["parent_child_count"] = {
        "fp_counts": dict(zip(raw_labels, counts)),
        "fp_child_parent_summary": child_summary,
    }
    if include_child_count_pie:
        _write_pie(axes[3], display_labels, counts, "Parent child count")

    fig.suptitle(f"{group_name}: N={len(rows)} false positives", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summaries


def _select_mag_group(rows: list[dict[str, float]], group: str) -> list[dict[str, float]]:
    if group == "gt60":
        return [row for row in rows if np.isfinite(row["mag"]) and row["mag"] > 60.0]
    if group == "24p5_26p5":
        return [row for row in rows if np.isfinite(row["mag"]) and 24.5 <= row["mag"] < 26.5]
    raise ValueError(f"unknown group {group!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="SAM pipeline output directory, e.g. output/sam_coordinated.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--false-positives", type=Path, default=None)
    parser.add_argument("--catalog", type=Path, default=None, help="Deblend catalog FITS. Defaults to RUN/deblend/deepCoadd_deblendedFlux.fits.")
    parser.add_argument("--scarlet-model", type=Path, default=None, help="Scarlet model pickle. Defaults to RUN/deblend/deepCoadd_scarletModelData.pickle.")
    parser.add_argument("--sam-metadata", type=Path, default=None, help="SAM metadata CSV. Defaults to RUN/sam/*_sam_metadata.csv.")
    parser.add_argument("--band", default="HSC-I")
    parser.add_argument("--mag-zero-point", type=float, default=27.0)
    parser.add_argument(
        "--include-child-count-pie",
        action="store_true",
        help="Also draw the parent child-count pie. Child-count statistics are always written to JSON.",
    )
    args = parser.parse_args()

    run_dir = args.run.expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "fp_diagnostics")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fp_path = (args.false_positives or (run_dir / "centroid_diagnostics/false_positives.csv")).expanduser().resolve()
    catalog_path = (args.catalog or (run_dir / "deblend/deepCoadd_deblendedFlux.fits")).expanduser().resolve()
    scarlet_path = (args.scarlet_model or (run_dir / "deblend/deepCoadd_scarletModelData.pickle")).expanduser().resolve()
    metadata_path = _find_sam_metadata(run_dir, args.sam_metadata)

    fp_ids = _load_false_positive_ids(fp_path)
    child_parent = _load_child_parent_map(catalog_path)
    parent_child_counts = _load_parent_child_counts(catalog_path)
    parent_metadata = _load_parent_metadata(metadata_path)
    fluxes = _load_scarlet_fluxes(scarlet_path, fp_ids, band=args.band)
    parent_rows = [
        {
            "parent": float(parent),
            "parent_child_count": float(parent_child_counts.get(parent, 0)),
            **metadata,
        }
        for parent, metadata in parent_metadata.items()
    ]

    rows: list[dict[str, float]] = []
    missing_parent_metadata = 0
    missing_flux = 0
    for source_id in fp_ids:
        parent = child_parent.get(source_id)
        if parent is None or parent not in parent_metadata:
            missing_parent_metadata += 1
            continue
        flux = fluxes.get(source_id, float("nan"))
        if not np.isfinite(flux):
            missing_flux += 1
        mag = _flux_to_mag(flux, args.mag_zero_point)
        rows.append(
            {
                "source_id": float(source_id),
                "parent": float(parent),
                "parent_child_count": float(parent_child_counts.get(parent, 0)),
                "flux": float(flux),
                "mag": float(mag),
                **parent_metadata[parent],
            }
        )

    detail_csv = output_dir / "sam_false_positive_parent_properties.csv"
    with detail_csv.open("w", newline="") as handle:
        fieldnames = [
            "source_id",
            "parent",
            "parent_child_count",
            "flux",
            "mag",
            "area",
            "predicted_iou",
            "stability_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    groups = {
        "mag_gt60": _select_mag_group(rows, "gt60"),
        "mag_24p5_26p5": _select_mag_group(rows, "24p5_26p5"),
    }
    summaries: dict[str, object] = {
        "run": str(run_dir),
        "false_positive_csv": str(fp_path),
        "catalog": str(catalog_path),
        "scarlet_model": str(scarlet_path),
        "sam_metadata": str(metadata_path),
        "band": args.band,
        "mag_zero_point": args.mag_zero_point,
        "fp_count": len(fp_ids),
        "usable_count": len(rows),
        "missing_parent_metadata": missing_parent_metadata,
        "missing_flux": missing_flux,
        "groups": {},
    }
    avg_rows: list[dict[str, object]] = []

    for name, selected in groups.items():
        suffix = "parent_area_iou_stability_childcount_pies" if args.include_child_count_pie else "parent_area_iou_stability_pies"
        figure_path = output_dir / f"{name}_{suffix}.png"
        avg_figure_path = output_dir / f"{name}_avg_fp_children_per_parent_by_parent_property.png"
        group_summary = _make_group_figure(
            rows=selected,
            group_name=name,
            output_path=figure_path,
            area_bins=DEFAULT_AREA_BINS,
            iou_bins=DEFAULT_IOU_BINS,
            stability_bins=DEFAULT_STABILITY_BINS,
            include_child_count_pie=args.include_child_count_pie,
        )
        avg_summary = _make_avg_fp_per_parent_figure(
            rows=selected,
            parent_rows=parent_rows,
            group_name=name,
            output_path=avg_figure_path,
            area_bins=DEFAULT_AREA_BINS,
            iou_bins=DEFAULT_IOU_BINS,
            stability_bins=DEFAULT_STABILITY_BINS,
        )
        for metric, metric_rows in avg_summary.items():
            for row in metric_rows:
                avg_rows.append({"group": name, **row})
        summaries["groups"][name] = {
            "count": len(selected),
            "figure": str(figure_path),
            "avg_fp_children_per_parent_figure": str(avg_figure_path),
            "avg_fp_children_per_parent": avg_summary,
            "counts": {key: dict(value) for key, value in group_summary.items()},
        }

    avg_csv = output_dir / "sam_false_positive_avg_fp_children_per_parent.csv"
    with avg_csv.open("w", newline="") as handle:
        fieldnames = [
            "group",
            "metric",
            "bin",
            "fp_count",
            "parent_count",
            "total_child_count_for_all_parents",
            "avg_fp_children_per_parent",
            "avg_total_children_per_parent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(avg_rows)

    summary_path = output_dir / "sam_false_positive_parent_property_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True))

    print(f"wrote {detail_csv}")
    print(f"wrote {avg_csv}")
    print(f"wrote {summary_path}")
    for name, data in summaries["groups"].items():
        print(f"{name}: {data['count']} -> {data['figure']}")
        print(f"{name}: avg FP children per parent -> {data['avg_fp_children_per_parent_figure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
