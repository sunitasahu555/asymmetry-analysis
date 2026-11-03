#!/usr/bin/env python3
"""
19_moe_active_total_analysis.py — Active-vs-total parameter analysis for MoE VLMs.

Phase-2 WS-19. For MoE models (currently: DeepSeek-VL2-Small) we have two
parameter counts: total (~27B for DeepSeek-VL2-Small) and active-per-token
(~2.8B with 6 of 64 experts firing).

This script:
* Tabulates total / active params per component for each MoE model.
* Computes per-benchmark efficiency: ``benchmark_score / billion_active_params``
  and ``benchmark_score / billion_total_params`` for each model in the roster.
* Emits a Pareto-frontier plot (active params vs. average benchmark score)
  showing where MoE sits relative to dense baselines.

Outputs
-------
* outputs/measurements/moe_active_total_analysis.csv
* outputs/measurements/moe_efficiency_per_benchmark.csv
* outputs/measurements/moe_analysis.md
* outputs/figures/fig14_moe_efficiency.{pdf,png}

Usage
-----
    python 19_moe_active_total_analysis.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import MODELS, ExperimentLogger
from vlm_audit.schema import ModelRecord
from vlm_audit.logging_utils import sanitize_for_filename


BENCHMARK_SLUGS = [
    "mmmu_val", "mmmu_pro", "mathvista_mini", "mmbench_en_dev",
    "ocrbench", "ai2d", "chartqa_relaxed", "docvqa_val",
    "mmvet", "hallusionbench", "seedbench_image", "mmstar",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", default="outputs/measurements")
    p.add_argument("--figures-dir",     default="outputs/figures")
    p.add_argument("--log-dir",         default="outputs/logs")
    p.add_argument("--dpi",             type=int, default=300)
    return p.parse_args()


def _load_records(meas_dir: Path, log) -> Dict[str, ModelRecord]:
    out = {}
    for spec in MODELS:
        path = meas_dir / f"{sanitize_for_filename(spec.hf_id)}.json"
        if not path.exists():
            log.warn(f"Missing JSON for {spec.name}; skipping")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                out[spec.hf_id] = ModelRecord.from_dict(json.load(f))
        except Exception as e:
            log.error(f"Could not load {path}", e)
    log.info(f"Loaded {len(out)} records")
    return out


def _load_benchmarks(meas_dir: Path, log) -> Dict[str, Dict[str, Optional[float]]]:
    path = meas_dir / "benchmarks.csv"
    if not path.exists():
        log.warn("benchmarks.csv not found — efficiency analysis will be empty")
        return {}
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            hf_id = row.get("hf_id")
            if not hf_id:
                continue
            d = {}
            for slug in BENCHMARK_SLUGS:
                v = row.get(slug)
                try:
                    d[slug] = float(v) if v not in (None, "", "null") else None
                except (TypeError, ValueError):
                    d[slug] = None
            out[hf_id] = d
    log.info(f"Loaded benchmark scores for {len(out)} models")
    return out


def _analyze_moe_models(records: Dict[str, ModelRecord], log) -> List[dict]:
    rows = []
    for spec in MODELS:
        if not getattr(spec, "is_moe", False):
            continue
        rec = records.get(spec.hf_id)
        if rec is None:
            log.warn(f"No record for MoE model {spec.name}")
            continue
        total = sum((c.params_total or 0) for c in rec.components.values())
        # Active = sum of (params_active or params_total) across components
        active = sum(
            ((c.params_active if c.params_active is not None else c.params_total) or 0)
            for c in rec.components.values()
        )
        row = {
            "name": spec.name,
            "hf_id": spec.hf_id,
            "family": spec.family,
            "n_experts_total": spec.n_experts_total,
            "n_experts_active_per_token": spec.n_experts_active_per_token,
            "total_params": total,
            "active_params": active,
            "active_pct_of_total": (100.0 * active / total) if total > 0 else 0.0,
        }
        # Per-component breakdown
        for cname in ("vision", "fusion", "language", "head"):
            c = rec.components.get(cname)
            if c is None:
                continue
            row[f"{cname}__total"] = c.params_total or 0
            row[f"{cname}__active"] = (c.params_active if c.params_active is not None else c.params_total) or 0
        log.iteration(f"moe {spec.name}",
                      total=total, active=active,
                      active_pct=round(row["active_pct_of_total"], 2))
        rows.append(row)
    return rows


def _compute_efficiency(records, benchmarks, log) -> List[dict]:
    """For every model, compute benchmark / total_params and benchmark / active_params."""
    rows = []
    for spec in MODELS:
        rec = records.get(spec.hf_id)
        if rec is None:
            continue
        total = sum((c.params_total or 0) for c in rec.components.values())
        if getattr(spec, "is_moe", False):
            active = sum(
                ((c.params_active if c.params_active is not None else c.params_total) or 0)
                for c in rec.components.values()
            )
        else:
            active = total
        if total <= 0:
            continue
        bench = benchmarks.get(spec.hf_id, {})
        total_b = total / 1e9
        active_b = active / 1e9
        # Average benchmark score across non-null benchmarks
        valid = [v for v in bench.values() if v is not None]
        avg_bench = (sum(valid) / len(valid)) if valid else None
        row = {
            "name": spec.name,
            "hf_id": spec.hf_id,
            "family": spec.family,
            "is_moe": getattr(spec, "is_moe", False),
            "total_params_b": total_b,
            "active_params_b": active_b,
            "avg_benchmark": avg_bench,
            "score_per_total_param_b": (avg_bench / total_b) if avg_bench is not None and total_b > 0 else None,
            "score_per_active_param_b": (avg_bench / active_b) if avg_bench is not None and active_b > 0 else None,
            "n_benchmarks_filled": len(valid),
        }
        # Per-benchmark efficiency
        for slug in BENCHMARK_SLUGS:
            v = bench.get(slug)
            row[f"{slug}_per_active_b"] = (v / active_b) if v is not None and active_b > 0 else None
            row[f"{slug}_per_total_b"] = (v / total_b) if v is not None and total_b > 0 else None
        rows.append(row)
        log.iteration(
            f"eff {spec.name}",
            total_b=round(total_b, 2),
            active_b=round(active_b, 2),
            avg_bench=round(avg_bench, 2) if avg_bench else None,
        )
    return rows


def _write_csv(rows, out_path: Path, log):
    if not rows:
        log.warn(f"No rows to write to {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r.keys()})
    # Push identity columns to the front
    for front in reversed(["name", "hf_id", "family", "is_moe"]):
        if front in cols:
            cols.remove(front)
            cols.insert(0, front)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log.info(f"Wrote {out_path}")


def _write_markdown(moe_rows, eff_rows, out_path: Path, log):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MoE active-vs-total analysis (Phase-2 WS-19)\n",
        "Mixture-of-Experts VLMs report two different parameter counts: total "
        "(all expert weights resident in memory) and active (per-token computed). "
        "Treating MoE as if it were a dense model with `total_params` "
        "would either understate its capacity or overstate its compute cost. "
        "This document reports both.\n",
    ]

    lines.append("\n## MoE models — total vs active\n")
    if not moe_rows:
        lines.append("_No MoE records loaded._")
    else:
        lines.append("| Model | n_experts (active/total) | Total params (B) | Active params (B) | Active % of total |")
        lines.append("|:--|:--|--:|--:|--:|")
        for r in moe_rows:
            tb = (r["total_params"] or 0) / 1e9
            ab = (r["active_params"] or 0) / 1e9
            lines.append(
                f"| {r['name']} | {r['n_experts_active_per_token']}/{r['n_experts_total']} | "
                f"{tb:.2f} | {ab:.2f} | {r['active_pct_of_total']:.1f}% |"
            )

    lines.append("\n## Efficiency frontier (benchmark / active params)\n")
    if not eff_rows:
        lines.append("_No efficiency rows (benchmarks.csv may be missing or empty)._")
    else:
        # Sort by score / active param for the headline efficiency ranking
        sorted_eff = sorted(
            [r for r in eff_rows if r.get("score_per_active_param_b") is not None],
            key=lambda r: -r["score_per_active_param_b"],
        )
        lines.append("| Model | Family | MoE? | Active (B) | Avg bench | Bench / active-B | Bench / total-B |")
        lines.append("|:--|:--|:-:|--:|--:|--:|--:|")
        for r in sorted_eff:
            ab = r["active_params_b"]
            avg = r["avg_benchmark"]
            spa = r["score_per_active_param_b"]
            spt = r["score_per_total_param_b"]
            moe = "✓" if r["is_moe"] else ""
            lines.append(
                f"| {r['name']} | {r['family']} | {moe} | {ab:.2f} | "
                f"{avg:.2f} | {spa:.2f} | {spt:.2f} |"
            )

    lines.append("\n## Interpretation\n")
    lines.append(
        "* When an MoE model outscores dense peers on `benchmark / active-B`, "
        "it shows the sparse-expert design buys per-token efficiency. "
        "If it scores comparably on `benchmark / total-B` to dense peers, "
        "the design is paying its way: total-VRAM cost is similar to its "
        "active-compute capacity."
    )
    lines.append(
        "* If `bench / active-B` is high but `bench / total-B` is low, the model is "
        "efficient at inference but wastes VRAM — a deployment concern."
    )
    lines.append(
        "* n=1 (DeepSeek-VL2-Small) is the only MoE entry today. The "
        "efficiency-frontier claim becomes more robust when Qwen3-VL-30B-A3B "
        "or another MoE-VLM is added."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Wrote {out_path}")


def _plot_fig14(eff_rows, out_dir: Path, dpi: int, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        log.error("matplotlib not available; skipping fig14", e)
        return

    pts = [r for r in eff_rows if r["avg_benchmark"] is not None and r["active_params_b"] > 0]
    if not pts:
        log.warn("No valid efficiency points; skipping fig14")
        return

    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots(figsize=(10, 7))
    families = sorted({r["family"] for r in pts})
    cmap = plt.get_cmap("tab10")
    for i, fam in enumerate(families):
        fam_pts = [r for r in pts if r["family"] == fam]
        xs = [r["active_params_b"] for r in fam_pts]
        ys = [r["avg_benchmark"] for r in fam_pts]
        is_moe = [r["is_moe"] for r in fam_pts]
        sizes = [200 if m else 90 for m in is_moe]
        ax.scatter(xs, ys, color=cmap(i % 10), edgecolor="black", linewidth=0.8,
                   alpha=0.8, s=sizes, label=fam.replace("_", " "))
        for r in fam_pts:
            ax.annotate(
                r["name"],
                (r["active_params_b"], r["avg_benchmark"]),
                xytext=(6, 4), textcoords="offset points", fontsize=7,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Active parameters (log B)")
    ax.set_ylabel("Average benchmark score (across filled benchmarks)")
    ax.set_title("MoE efficiency vs dense baselines (Finding F14)\n"
                  "marker size: large = MoE (sparse), small = dense")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig14_moe_efficiency.pdf"
    png = out_dir / "fig14_moe_efficiency.png"
    plt.tight_layout()
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {pdf}")
    log.info(f"Saved {png}")


def main():
    args = parse_args()
    meas_dir = Path(args.measurement_dir)
    fig_dir = Path(args.figures_dir)

    with ExperimentLogger(args.log_dir, "19_moe_active_total_analysis") as log:
        log.section("Loading measurement records")
        records = _load_records(meas_dir, log)

        log.section("Loading benchmarks (if present)")
        benchmarks = _load_benchmarks(meas_dir, log)

        log.section("Analyzing MoE models")
        try:
            moe_rows = _analyze_moe_models(records, log)
        except Exception as e:
            log.error("MoE analysis failed", e)
            moe_rows = []

        log.section("Computing per-model efficiency")
        try:
            eff_rows = _compute_efficiency(records, benchmarks, log)
        except Exception as e:
            log.error("Efficiency computation failed", e)
            eff_rows = []

        log.section("Writing outputs")
        try:
            _write_csv(moe_rows, meas_dir / "moe_active_total_analysis.csv", log)
        except Exception as e:
            log.error("Writing moe_active_total_analysis.csv failed", e)
        try:
            _write_csv(eff_rows, meas_dir / "moe_efficiency_per_benchmark.csv", log)
        except Exception as e:
            log.error("Writing moe_efficiency_per_benchmark.csv failed", e)
        try:
            _write_markdown(moe_rows, eff_rows, meas_dir / "moe_analysis.md", log)
        except Exception as e:
            log.error("Writing moe_analysis.md failed", e)

        log.section("Plotting fig14")
        try:
            _plot_fig14(eff_rows, fig_dir, args.dpi, log)
        except Exception as e:
            log.error("fig14 plotting failed", e)

    print(f"MoE models analyzed:    {len(moe_rows)}")
    print(f"Efficiency rows:        {len(eff_rows)}")
    print(f"Outputs in {meas_dir}/")


if __name__ == "__main__":
    main()
