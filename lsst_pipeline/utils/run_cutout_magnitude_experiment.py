from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from crop_lsst_catalog import CutoutBox, crop_meas_catalog  # noqa: E402
from evaluate_centroid_matches import (  # noqa: E402
    DEFAULT_PIXEL_SCALE,
    DEFAULT_RADIUS_ARCSEC,
    _load_points,
    match_nearest_unique,
)


BANDS = ("HSC-G", "HSC-R", "HSC-I")
PIXEL_PLANES = ("IMAGE", "MASK", "VARIANCE")
DEFAULT_PREVIOUS_ORIGIN = (18204, 20924)


@dataclass(frozen=True)
class CutoutSpec:
    name: str
    x0: int
    y0: int
    size: int

    @property
    def box(self) -> CutoutBox:
        return CutoutBox(float(self.x0), float(self.y0), float(self.size), float(self.size))


def _origin_from_ltv(header: fits.Header) -> tuple[int, int]:
    if "LTV1" not in header or "LTV2" not in header:
        raise KeyError("IMAGE header needs LTV1/LTV2 to infer parent-patch coordinates")
    return -int(round(float(header["LTV1"]))), -int(round(float(header["LTV2"])))


def _cropped_header(header: fits.Header, *, local_x0: int, local_y0: int) -> fits.Header:
    out = header.copy()
    if "LTV1" in out:
        out["LTV1"] = float(out["LTV1"]) - local_x0
    if "LTV2" in out:
        out["LTV2"] = float(out["LTV2"]) - local_y0
    if "CRPIX1" in out:
        out["CRPIX1"] = float(out["CRPIX1"]) - local_x0
    if "CRPIX2" in out:
        out["CRPIX2"] = float(out["CRPIX2"]) - local_y0
    if "CRVAL1A" in out:
        out["CRVAL1A"] = float(out["CRVAL1A"]) + local_x0
    if "CRVAL2A" in out:
        out["CRVAL2A"] = float(out["CRVAL2A"]) + local_y0
    return out


def _finite_replacement(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else 0.0


def _new_image_hdu_like(hdu, *, data: np.ndarray, header: fits.Header):
    if isinstance(hdu, fits.PrimaryHDU):
        return fits.PrimaryHDU(data=data, header=header)
    return fits.ImageHDU(data=data, header=header, name=hdu.name)


def crop_exposure_cutout(
    *,
    source_path: Path,
    output_path: Path,
    parent_x0: int,
    parent_y0: int,
    size: int,
    clean_nonfinite: bool,
) -> None:
    with fits.open(source_path, memmap=False) as hdul:
        if "IMAGE" not in hdul:
            raise KeyError(f"{source_path} has no IMAGE extension")
        source_origin = _origin_from_ltv(hdul["IMAGE"].header)
        local_x0 = int(parent_x0 - source_origin[0])
        local_y0 = int(parent_y0 - source_origin[1])

        for plane in PIXEL_PLANES:
            if plane not in hdul:
                raise KeyError(f"{source_path} has no {plane} extension")
            data = hdul[plane].data
            if data is None or data.ndim != 2:
                raise ValueError(f"{source_path}[{plane}] is not a 2D image")
            if local_x0 < 0 or local_y0 < 0 or local_x0 + size > data.shape[1] or local_y0 + size > data.shape[0]:
                raise ValueError(
                    f"{source_path}[{plane}] cannot cover parent cutout "
                    f"x={parent_x0}:{parent_x0 + size}, y={parent_y0}:{parent_y0 + size}; "
                    f"source origin={source_origin}, shape={data.shape}"
                )

        out_hdus = []
        for hdu in hdul:
            if hdu.name in PIXEL_PLANES:
                data = np.asarray(hdu.data[local_y0 : local_y0 + size, local_x0 : local_x0 + size]).copy()
                if clean_nonfinite and np.issubdtype(data.dtype, np.floating) and not np.all(np.isfinite(data)):
                    fill = _finite_replacement(data)
                    data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill).astype(data.dtype, copy=False)
                header = _cropped_header(hdu.header, local_x0=local_x0, local_y0=local_y0)
                out_hdus.append(_new_image_hdu_like(hdu, data=data, header=header))
            else:
                # LSST archive tables may contain heap-backed variable-length
                # arrays; avoid Astropy deep-copying those tables.
                out_hdus.append(hdu)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(out_hdus).writeto(output_path, overwrite=True)


def _coadd_path(root: Path, band: str, tract: int, patch: str) -> Path:
    return root / band / f"deepCoadd-{band}-{tract}-{patch}.fits"


def make_cutouts(args: argparse.Namespace, specs: list[CutoutSpec]) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for spec in specs:
        cutout_paths: dict[str, Path] = {}
        for band in BANDS:
            src = _coadd_path(args.coadd_root, band, args.tract, args.patch)
            dst = args.output_root / "cutouts" / spec.name / band / src.name
            if not dst.exists() or not args.skip_existing:
                crop_exposure_cutout(
                    source_path=src,
                    output_path=dst,
                    parent_x0=spec.x0,
                    parent_y0=spec.y0,
                    size=spec.size,
                    clean_nonfinite=not args.no_clean_nonfinite,
                )
            cutout_paths[band] = dst
        result[spec.name] = cutout_paths
    return result


def _run_pipeline(
    *,
    args: argparse.Namespace,
    method: str,
    spec: CutoutSpec,
    cutout_paths: dict[str, Path],
) -> Path:
    outdir = args.output_root / "runs" / spec.name / method
    manifest = outdir / "manifest.json"
    if args.skip_existing and manifest.exists():
        return outdir

    cmd = [
        args.python,
        str(args.pipeline_script),
        "--detection-mode",
        method,
        "--repo",
        str(args.repo),
        "--tract",
        str(args.tract),
        "--patch",
        str(args.patch),
        "--output-dir",
        str(outdir),
        "--clip-sky-sources-to-exposure-bbox",
    ]
    for band in BANDS:
        cmd.extend(["--coadd", f"{band}={cutout_paths[band]}"])
    if method == "lsst" and args.lsst_extra_args:
        cmd.extend(shlex.split(args.lsst_extra_args))
    if method == "sam" and args.sam_extra_args:
        cmd.extend(shlex.split(args.sam_extra_args))

    outdir.parent.mkdir(parents=True, exist_ok=True)
    log_path = outdir.with_suffix(f".{method}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("running:", " ".join(shlex.quote(str(part)) for part in cmd))
    if args.dry_run:
        return outdir
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=args.pipeline_root, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        if args.continue_on_error:
            print(f"WARNING: {method} {spec.name} failed; see {log_path}")
            return outdir
        raise RuntimeError(f"{method} {spec.name} failed with exit code {proc.returncode}; see {log_path}")
    return outdir


def crop_reference_catalog(args: argparse.Namespace, spec: CutoutSpec) -> Path:
    out = args.output_root / "reference_catalogs" / f"{spec.name}_meas.fits"
    if out.exists() and args.skip_existing:
        return out
    table = crop_meas_catalog(
        args.reference_catalog,
        box=spec.box,
        x_col=args.ref_x,
        y_col=args.ref_y,
        margin=float(args.reference_margin),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write(out, format="fits", overwrite=True)
    return out


def _flux_to_mag(flux: np.ndarray, zeropoint: float) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    mag = np.full(flux.shape, np.nan, dtype=float)
    good = np.isfinite(flux) & (flux > 0)
    mag[good] = float(zeropoint) - 2.5 * np.log10(flux[good])
    return mag


def _choose_flux_col(table: Table, preferred: str, fallbacks: tuple[str, ...]) -> str:
    for name in (preferred, *fallbacks):
        if name in table.colnames:
            return name
    raise KeyError(f"none of the flux columns exists: {(preferred, *fallbacks)}")


def _rows_for_bins(
    *,
    cutout: str,
    method: str,
    ref_points,
    pred_points,
    ref_used: np.ndarray,
    pred_used: np.ndarray,
    args: argparse.Namespace,
) -> list[dict]:
    ref_flux_col = _choose_flux_col(
        ref_points.table,
        args.ref_flux_col,
        ("modelfit_CModel_instFlux", "base_PsfFlux_instFlux", "base_SdssShape_instFlux"),
    )
    pred_flux_col = _choose_flux_col(
        pred_points.table,
        args.pred_flux_col,
        ("deblend_scarletFlux", "deblend_peak_instFlux"),
    )
    ref_flux = np.asarray(ref_points.table[ref_flux_col], dtype=float)[ref_points.table_indices]
    pred_flux = np.asarray(pred_points.table[pred_flux_col], dtype=float)[pred_points.table_indices]
    ref_mag = _flux_to_mag(ref_flux, args.mag_zero_point)
    pred_mag = _flux_to_mag(pred_flux, args.mag_zero_point)

    finite = np.concatenate([ref_mag[np.isfinite(ref_mag)], pred_mag[np.isfinite(pred_mag)]])
    if finite.size == 0:
        return []
    mag_min = float(args.mag_min) if args.mag_min is not None else np.floor(np.nanmin(finite) / args.bin_size) * args.bin_size
    mag_max = float(args.mag_max) if args.mag_max is not None else np.ceil(np.nanmax(finite) / args.bin_size) * args.bin_size
    edges = np.arange(mag_min, mag_max + args.bin_size * 0.5, args.bin_size)

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        ref_in = np.isfinite(ref_mag) & (ref_mag >= lo) & (ref_mag < hi)
        pred_in = np.isfinite(pred_mag) & (pred_mag >= lo) & (pred_mag < hi)
        ref_total = int(np.count_nonzero(ref_in))
        pred_total = int(np.count_nonzero(pred_in))
        ref_matched = int(np.count_nonzero(ref_in & ref_used))
        pred_matched = int(np.count_nonzero(pred_in & pred_used))
        rows.append(
            {
                "cutout": cutout,
                "method": method,
                "mag_left": float(lo),
                "mag_right": float(hi),
                "mag_center": float(0.5 * (lo + hi)),
                "reference_flux_col": ref_flux_col,
                "prediction_flux_col": pred_flux_col,
                "reference_total": ref_total,
                "reference_matched": ref_matched,
                "completeness": ref_matched / ref_total if ref_total else np.nan,
                "prediction_total": pred_total,
                "prediction_matched": pred_matched,
                "purity": pred_matched / pred_total if pred_total else np.nan,
            }
        )
    return rows


def evaluate_run(args: argparse.Namespace, *, spec: CutoutSpec, method: str, run_dir: Path, ref_catalog: Path) -> list[dict]:
    pred_catalog = run_dir / "deblend" / "deepCoadd_deblendedFlux.fits"
    if not pred_catalog.exists():
        print(f"WARNING: missing prediction catalog: {pred_catalog}")
        return []
    radius_pix = float(args.match_radius_arcsec) / float(args.pixel_scale)
    ref_points = _load_points(
        ref_catalog,
        x_col=args.ref_x,
        y_col=args.ref_y,
        role="ref",
        hdu=1,
        leaf_only=True,
        drop_flagged_centroids=True,
        require_science_model=False,
    )
    pred_points = _load_points(
        pred_catalog,
        x_col=args.pred_x,
        y_col=args.pred_y,
        role="pred",
        hdu=1,
        leaf_only=True,
        drop_flagged_centroids=True,
        require_science_model=True,
    )
    matches, ref_used, pred_used = match_nearest_unique(ref_points, pred_points, radius_pix)
    summary = {
        "cutout": spec.name,
        "method": method,
        "reference_count": ref_points.n,
        "prediction_count": pred_points.n,
        "matched_count": len(matches),
        "recall": len(matches) / ref_points.n if ref_points.n else np.nan,
        "precision": len(matches) / pred_points.n if pred_points.n else np.nan,
        "match_radius_pix": radius_pix,
        "match_radius_arcsec": float(args.match_radius_arcsec),
    }
    summary_path = run_dir / "magnitude_match_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return _rows_for_bins(
        cutout=spec.name,
        method=method,
        ref_points=ref_points,
        pred_points=pred_points,
        ref_used=ref_used,
        pred_used=pred_used,
        args=args,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cutout",
        "method",
        "mag_left",
        "mag_right",
        "mag_center",
        "reference_flux_col",
        "prediction_flux_col",
        "reference_total",
        "reference_matched",
        "completeness",
        "prediction_total",
        "prediction_matched",
        "purity",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float, float], dict] = {}
    for row in rows:
        key = (row["method"], row["mag_left"], row["mag_right"])
        item = grouped.setdefault(
            key,
            {
                "method": row["method"],
                "mag_left": row["mag_left"],
                "mag_right": row["mag_right"],
                "mag_center": row["mag_center"],
                "reference_total": 0,
                "reference_matched": 0,
                "prediction_total": 0,
                "prediction_matched": 0,
            },
        )
        item["reference_total"] += int(row["reference_total"])
        item["reference_matched"] += int(row["reference_matched"])
        item["prediction_total"] += int(row["prediction_total"])
        item["prediction_matched"] += int(row["prediction_matched"])
    out = []
    for item in grouped.values():
        ref_total = item["reference_total"]
        pred_total = item["prediction_total"]
        item["completeness"] = item["reference_matched"] / ref_total if ref_total else np.nan
        item["purity"] = item["prediction_matched"] / pred_total if pred_total else np.nan
        out.append(item)
    return sorted(out, key=lambda r: (r["method"], r["mag_left"]))


def plot_curves(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted({row["method"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        x = np.array([row["mag_center"] for row in method_rows], dtype=float)
        completeness = np.array([row["completeness"] for row in method_rows], dtype=float)
        purity = np.array([row["purity"] for row in method_rows], dtype=float)
        axes[0].plot(x, completeness, marker="o", linewidth=1.8, label=method)
        axes[1].plot(x, purity, marker="o", linewidth=1.8, label=method)
    for ax, title in zip(axes, ("Completeness by catalog magnitude", "Purity by deblend magnitude")):
        ax.set_title(title)
        ax.set_xlabel("instrumental magnitude")
        ax.set_ylabel("score")
        ax.set_ylim(-0.03, 1.03)
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    if methods:
        axes[0].legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_origins(values: list[str], *, size: int) -> list[CutoutSpec]:
    if not values:
        values = [f"{DEFAULT_PREVIOUS_ORIGIN[0]},{DEFAULT_PREVIOUS_ORIGIN[1]}"]
    specs = []
    seen = set()
    for index, value in enumerate(values, start=1):
        x_text, y_text = value.split(",", 1)
        x0 = int(float(x_text))
        y0 = int(float(y_text))
        key = (x0, y0)
        if key in seen:
            continue
        seen.add(key)
        name = "previous" if key == DEFAULT_PREVIOUS_ORIGIN else f"cutout_{index:03d}_x{x0}_y{y0}"
        specs.append(CutoutSpec(name=name, x0=x0, y0=y0, size=size))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run LSST/SAM deblend experiments on 512x512 cutouts from a large coadd, "
            "then plot completeness and purity as magnitude-binned curves."
        )
    )
    parser.add_argument("--coadd-root", type=Path, default=PIPELINE_ROOT / "fits/projection_cutout")
    parser.add_argument("--reference-catalog", type=Path, default=Path("/home/chenzunhao/2026-05-01_scarlet_deblend_demo/catalog/meas-HSC-I-9813-4,5.fits"))
    parser.add_argument("--repo", type=Path, default=PIPELINE_ROOT / "fits/repo")
    parser.add_argument("--output-root", type=Path, default=PIPELINE_ROOT / "output/cutout_magnitude_experiment")
    parser.add_argument("--pipeline-root", type=Path, default=PIPELINE_ROOT)
    parser.add_argument("--pipeline-script", type=Path, default=PIPELINE_ROOT / "scarlet_deblend_from_fits.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tract", type=int, default=9813)
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--origin", action="append", default=[], help="Parent-patch origin x,y. Defaults to previous x=18204,y=20924.")
    parser.add_argument("--methods", nargs="+", default=["lsst", "sam"], choices=["lsst", "sam"])
    parser.add_argument("--lsst-extra-args", default="", help="Extra args appended only to LSST runs, shell-style string.")
    parser.add_argument("--sam-extra-args", default="", help="Extra args appended only to SAM runs, shell-style string.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Do not crop or run pipelines; evaluate existing outputs.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-clean-nonfinite", action="store_true")
    parser.add_argument("--reference-margin", type=float, default=0.0)
    parser.add_argument("--ref-x", default="base_SdssCentroid_x")
    parser.add_argument("--ref-y", default="base_SdssCentroid_y")
    parser.add_argument("--pred-x", default=None)
    parser.add_argument("--pred-y", default=None)
    parser.add_argument("--ref-flux-col", default="base_PsfFlux_instFlux")
    parser.add_argument("--pred-flux-col", default="deblend_scarletFlux")
    parser.add_argument("--mag-zero-point", type=float, default=27.0)
    parser.add_argument("--bin-size", type=float, default=1.0)
    parser.add_argument("--mag-min", type=float, default=None)
    parser.add_argument("--mag-max", type=float, default=None)
    parser.add_argument("--match-radius-arcsec", type=float, default=DEFAULT_RADIUS_ARCSEC)
    parser.add_argument("--pixel-scale", type=float, default=DEFAULT_PIXEL_SCALE)
    args = parser.parse_args()

    args.coadd_root = args.coadd_root.expanduser().resolve()
    args.reference_catalog = args.reference_catalog.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.pipeline_root = args.pipeline_root.expanduser().resolve()
    args.pipeline_script = args.pipeline_script.expanduser().resolve()

    specs = parse_origins(args.origin, size=args.size)
    if not args.plot_only:
        cutouts = make_cutouts(args, specs)
        for spec in specs:
            for method in args.methods:
                _run_pipeline(args=args, method=method, spec=spec, cutout_paths=cutouts[spec.name])

    all_rows = []
    for spec in specs:
        ref_catalog = crop_reference_catalog(args, spec)
        for method in args.methods:
            run_dir = args.output_root / "runs" / spec.name / method
            all_rows.extend(evaluate_run(args, spec=spec, method=method, run_dir=run_dir, ref_catalog=ref_catalog))

    per_cutout_csv = args.output_root / "magnitude_metrics_per_cutout.csv"
    aggregate_csv = args.output_root / "magnitude_metrics_aggregate.csv"
    plot_path = args.output_root / "magnitude_curves.png"
    write_csv(per_cutout_csv, all_rows)
    aggregate = aggregate_rows(all_rows)
    write_csv(aggregate_csv, aggregate)
    plot_curves(plot_path, aggregate)

    print(f"wrote {per_cutout_csv}")
    print(f"wrote {aggregate_csv}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
