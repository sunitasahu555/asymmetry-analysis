#!/usr/bin/env python3
"""
20_generation_trajectory.py — Plot architectural-allocation trends over time (2021-2026).

Phase-2 WS-20. Reads master_wide.csv and plots:

* fig15_generation_trajectory_params.{pdf,png} — vision-share % over release year, by family
* fig16_generation_trajectory_bandwidth.{pdf,png} — fusion-boundary tokens over release year (log y), by family

Also emits a per-year summary CSV (``generation_trajectory.csv``) and a
Markdown commentary file (``generation_trajectory.md``).

Usage
-----
    python 20_generation_trajectory.py
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import MODELS, ExperimentLogger
from vlm_audit.schema import ModelRecord
from vlm_audit.logging_utils import sanitize_for_filename


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", default="outputs/measurements")
    p.add_argument("--figures-dir",     default="outputs/figures")
    p.add_argument("--log-dir",         default="outputs/logs")
    p.add_argument("--dpi",             type=int, default=300)
    return p.parse_args()


def _to_float(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _read_master_wide(path: Path, log) -> List[dict]:
    if not path.exists():
        log.error(f"master_wide.csv not found: {path}")
        return []
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    log.info(f"Read {len(rows)} rows from master_wide.csv")
    return rows


def _enrich(master_rows, log) -> List[dict]:
    """Add derived columns (vision_pct, fusion_tokens) for plotting."""
    out = []
    for r in master_rows:
        v = _to_float(r.get("vision__params_total")) or 0
        f = _to_float(r.get("fusion__params_total")) or 0
        l = _to_float(r.get("language__params_total")) or 0
        h = _to_float(r.get("head__params_total")) or 0
        total = v + f + l + h
        tokens = (_to_float(r.get("fusion__output_tokens")) or
                  _to_float(r.get("language__output_tokens")) or 0)
        try:
            year = int(r.get("release_year") or 0)
        except (TypeError, ValueError):
            year = 0
        out.append({
            "name":         r.get("name", ""),
            "family":       r.get("family", ""),
            "release_year": year,
            "total_params": total,
            "vision_pct":   (100.0 * v / total) if total > 0 else 0.0,
            "language_pct": (100.0 * l / total) if total > 0 else 0.0,
            "fusion_pct":   (100.0 * f / total) if total > 0 else 0.0,
            "fusion_tokens": tokens,
        })
    return out


def _summarize_by_year(rows, log) -> List[dict]:
    by_year = {}
    for r in rows:
        y = r["release_year"]
        if y <= 0:
            continue
        by_year.setdefault(y, []).append(r)
    out = []
    for y in sorted(by_year.keys()):
        bucket = by_year[y]
        n = len(bucket)
        avg_v_pct = sum(r["vision_pct"] for r in bucket) / n
        avg_l_pct = sum(r["language_pct"] for r in bucket) / n
        tokens = [r["fusion_tokens"] for r in bucket if r["fusion_tokens"] > 0]
        gmean_tokens = (math.exp(sum(math.log(t) for t in tokens) / len(tokens))
                        if tokens else 0)
        models = ", ".join(r["name"] for r in bucket)
        out.append({
            "year":            y,
            "n_models":        n,
            "avg_vision_pct":  avg_v_pct,
            "avg_language_pct": avg_l_pct,
            "gmean_fusion_tokens": gmean_tokens,
            "models":          models,
        })
        log.iteration(f"year {y}", n=n, avg_v_pct=round(avg_v_pct, 1),
                      gmean_tokens=round(gmean_tokens, 1))
    return out


def _write_csv(rows, out_path: Path, log):
    if not rows:
        log.warn(f"No rows to write to {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log.info(f"Wrote {out_path}")


def _write_markdown(year_rows, model_rows, out_path: Path, log):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Generation-trajectory analysis (2021-2026)\n"]
    lines.append("Phase-2 WS-20 output. Tracks how per-family allocation patterns "
                  "and fusion-bandwidth choices evolved across the 5-year window.\n")

    lines.append("\n## Per-year summary\n")
    lines.append("| Year | n | Avg vision % | Avg language % | GMean fusion tokens | Models |")
    lines.append("|:-:|:-:|--:|--:|--:|:--|")
    for r in year_rows:
        lines.append(
            f"| {r['year']} | {r['n_models']} | {r['avg_vision_pct']:.1f} | "
            f"{r['avg_language_pct']:.1f} | {r['gmean_fusion_tokens']:.0f} | {r['models']} |"
        )

    lines.append("\n## Qualitative observations\n")
    if len(year_rows) >= 2:
        first_year = year_rows[0]
        last_year = year_rows[-1]
        v_shift = last_year["avg_vision_pct"] - first_year["avg_vision_pct"]
        t_ratio = (last_year["gmean_fusion_tokens"] / first_year["gmean_fusion_tokens"]
                   if first_year["gmean_fusion_tokens"] > 0 else None)
        lines.append(
            f"* Vision-share trajectory from {first_year['year']} to {last_year['year']}: "
            f"{first_year['avg_vision_pct']:.1f}% → {last_year['avg_vision_pct']:.1f}% "
            f"(Δ {v_shift:+.1f} percentage points)."
        )
        if t_ratio is not None:
            lines.append(
                f"* Fusion-bandwidth trajectory: "
                f"{first_year['gmean_fusion_tokens']:.0f} → "
                f"{last_year['gmean_fusion_tokens']:.0f} tokens "
                f"(geometric-mean ratio {t_ratio:.1f}×)."
            )
        lines.append(
            "* These are descriptive trends; n per year is small so significance "
            "tests are not reported. Interpret as qualitative trajectory, not "
            "scaling law."
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Wrote {out_path}")


def _plot_trajectories(model_rows, out_dir: Path, dpi: int, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        log.error("matplotlib not available; skipping trajectory plots", e)
        return

    pts = [r for r in model_rows if r["release_year"] > 0]
    if not pts:
        log.warn("No models with valid release_year; skipping plots")
        return

    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["font.family"] = "serif"

    families = sorted({r["family"] for r in pts})
    cmap = plt.get_cmap("tab10")
    color_by_family = {fam: cmap(i % 10) for i, fam in enumerate(families)}

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- fig15: vision-share over release year ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for fam in families:
        fam_pts = [r for r in pts if r["family"] == fam]
        if not fam_pts:
            continue
        xs = [r["release_year"] for r in fam_pts]
        ys = [r["vision_pct"] for r in fam_pts]
        ax.scatter(xs, ys, color=color_by_family[fam], edgecolor="black", linewidth=0.7,
                   alpha=0.85, s=120, label=fam.replace("_", " "))
        for r in fam_pts:
            ax.annotate(r["name"], (r["release_year"], r["vision_pct"]),
                        xytext=(6, 5), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Release year")
    ax.set_ylabel("Vision parameter share (%)")
    ax.set_title("Per-family vision-share trajectory (2021-2026)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    plt.tight_layout()
    pdf15 = out_dir / "fig15_generation_trajectory_params.pdf"
    png15 = out_dir / "fig15_generation_trajectory_params.png"
    plt.savefig(pdf15, bbox_inches="tight")
    plt.savefig(png15, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {pdf15}")
    log.info(f"Saved {png15}")

    # --- fig16: fusion bandwidth over release year (log y) ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for fam in families:
        fam_pts = [r for r in pts if r["family"] == fam and r["fusion_tokens"] > 0]
        if not fam_pts:
            continue
        xs = [r["release_year"] for r in fam_pts]
        ys = [r["fusion_tokens"] for r in fam_pts]
        ax.scatter(xs, ys, color=color_by_family[fam], edgecolor="black", linewidth=0.7,
                   alpha=0.85, s=120, label=fam.replace("_", " "))
        for r in fam_pts:
            ax.annotate(r["name"], (r["release_year"], r["fusion_tokens"]),
                        xytext=(6, 5), textcoords="offset points", fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("Release year")
    ax.set_ylabel("Fusion-boundary tokens (log)")
    ax.set_title("Per-family bandwidth trajectory (2021-2026)")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=2)
    plt.tight_layout()
    pdf16 = out_dir / "fig16_generation_trajectory_bandwidth.pdf"
    png16 = out_dir / "fig16_generation_trajectory_bandwidth.png"
    plt.savefig(pdf16, bbox_inches="tight")
    plt.savefig(png16, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {pdf16}")
    log.info(f"Saved {png16}")


def main():
    args = parse_args()
    meas_dir = Path(args.measurement_dir)
    fig_dir = Path(args.figures_dir)

    with ExperimentLogger(args.log_dir, "20_generation_trajectory") as log:
        log.section("Loading inputs")
        master_rows = _read_master_wide(meas_dir / "master_wide.csv", log)
        if not master_rows:
            log.error("No master_wide.csv data; run 06_aggregate_master_table.py first.")
            sys.exit(1)

        log.section("Enriching rows with derived columns")
        try:
            model_rows = _enrich(master_rows, log)
        except Exception as e:
            log.error("Enrichment failed", e)
            sys.exit(2)

        log.section("Summarizing by year")
        try:
            year_rows = _summarize_by_year(model_rows, log)
        except Exception as e:
            log.error("Year summarization failed", e)
            year_rows = []

        log.section("Writing CSV + Markdown")
        try:
            _write_csv(year_rows, meas_dir / "generation_trajectory.csv", log)
        except Exception as e:
            log.error("Writing generation_trajectory.csv failed", e)
        try:
            _write_markdown(year_rows, model_rows, meas_dir / "generation_trajectory.md", log)
        except Exception as e:
            log.error("Writing generation_trajectory.md failed", e)

        log.section("Plotting fig15 + fig16")
        try:
            _plot_trajectories(model_rows, fig_dir, args.dpi, log)
        except Exception as e:
            log.error("Trajectory plot failed", e)

    print(f"Year summary rows: {len(year_rows)}")
    print(f"Outputs in {meas_dir}/ and {fig_dir}/")


if __name__ == "__main__":
    main()
