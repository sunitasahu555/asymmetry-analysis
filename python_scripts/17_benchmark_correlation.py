#!/usr/bin/env python3
"""
17_benchmark_correlation.py — Correlate structural metrics with benchmark scores (F11-F14).

Phase-2 WS-17. Reads master_wide.csv + benchmarks.csv and computes:

* F11 — Pearson and Spearman correlation of (log) fusion_boundary_tokens vs each benchmark
* F12 — V/L parameter ratio vs ranking across benchmarks (which architecture wins which axis)
* F13 — Per-component FLOPs share vs benchmark performance (linear regression)
* F14 — MoE efficiency: benchmark / active_params vs benchmark / total_params

Outputs
-------
* outputs/measurements/benchmark_correlation.csv — one row per (metric, benchmark, statistic)
* outputs/measurements/benchmark_correlation.md  — paper-ready tables + interpretation
* outputs/figures/fig13_bandwidth_vs_benchmarks.{pdf,png} — scatter (log fusion tokens) vs MMMU/MathVista
* outputs/figures/fig14_moe_efficiency.{pdf,png} — MoE active vs total efficiency frontier

Every step is wrapped in try/except; if a statistic fails (e.g. n < 3 valid
points) the row is written with ``status="insufficient_data"`` and the
script continues.

Usage
-----
    python 17_benchmark_correlation.py
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import ExperimentLogger


BENCHMARK_SLUGS = [
    "mmmu_val", "mmmu_pro", "mathvista_mini", "mmbench_en_dev",
    "ocrbench", "ai2d", "chartqa_relaxed", "docvqa_val",
    "mmvet", "hallusionbench", "seedbench_image", "mmstar",
]

STRUCTURAL_METRICS = [
    ("fusion_boundary_tokens_log",   "log10(fusion_boundary_tokens + 1)"),
    ("v_to_l_ratio_log",             "log10(vision_params / language_params)"),
    ("total_params_log",             "log10(total_params)"),
    ("total_flops_g_log",            "log10(total_flops_g + 1)"),
    ("vision_pct",                   "vision_params / total_params"),
    ("language_pct",                 "language_params / total_params"),
    ("fusion_pct",                   "fusion_params / total_params"),
    ("vision_flops_pct",             "vision_flops / total_flops"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", default="outputs/measurements")
    p.add_argument("--figures-dir",     default="outputs/figures")
    p.add_argument("--log-dir",         default="outputs/logs")
    p.add_argument("--dpi",             type=int, default=300)
    return p.parse_args()


def _safe_log10(x):
    if x is None or x <= 0:
        return None
    return math.log10(x)


def _pearson(xs, ys):
    if len(xs) < 3:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _spearman(xs, ys):
    if len(xs) < 3:
        return None
    def rank(vals):
        # Average rank for ties
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks
    return _pearson(rank(xs), rank(ys))


def _bootstrap_ci(xs, ys, stat_fn, n_boot: int = 2000, alpha: float = 0.05) -> Optional[Tuple[float, float]]:
    if len(xs) < 4:
        return None
    import random
    rng = random.Random(20260521)
    samples = []
    n = len(xs)
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        xs_b = [xs[i] for i in idxs]
        ys_b = [ys[i] for i in idxs]
        s = stat_fn(xs_b, ys_b)
        if s is not None:
            samples.append(s)
    if len(samples) < n_boot // 4:
        return None
    samples.sort()
    lo = samples[int(alpha / 2 * len(samples))]
    hi = samples[int((1 - alpha / 2) * len(samples)) - 1]
    return (lo, hi)


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


def _read_benchmarks(path: Path, log) -> List[dict]:
    if not path.exists():
        log.error(f"benchmarks.csv not found: {path}")
        return []
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    log.info(f"Read {len(rows)} rows from benchmarks.csv")
    return rows


def _to_float(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _build_structural_dict(master_rows) -> Dict[str, Dict[str, Optional[float]]]:
    """For each model, compute the structural metrics from master_wide.csv."""
    out = {}
    for r in master_rows:
        hf_id = r.get("model_id") or r.get("hf_id")
        if not hf_id:
            continue
        v_total = _to_float(r.get("vision__params_total")) or 0.0
        f_total = _to_float(r.get("fusion__params_total")) or 0.0
        l_total = _to_float(r.get("language__params_total")) or 0.0
        h_total = _to_float(r.get("head__params_total")) or 0.0
        tot = v_total + f_total + l_total + h_total
        v_g = _to_float(r.get("vision__flops_g")) or 0.0
        f_g = _to_float(r.get("fusion__flops_g")) or 0.0
        l_g = _to_float(r.get("language__flops_g")) or 0.0
        flops_tot = v_g + f_g + l_g
        fusion_tokens = _to_float(r.get("fusion__output_tokens")) or _to_float(r.get("language__output_tokens")) or 0.0

        v_to_l = (v_total / l_total) if l_total > 0 else None
        out[hf_id] = {
            "fusion_boundary_tokens_log": _safe_log10((fusion_tokens or 0) + 1),
            "v_to_l_ratio_log":           _safe_log10(v_to_l) if v_to_l else None,
            "total_params_log":           _safe_log10(tot) if tot > 0 else None,
            "total_flops_g_log":          _safe_log10(flops_tot + 1) if flops_tot >= 0 else None,
            "vision_pct":                 (v_total / tot) if tot > 0 else None,
            "language_pct":               (l_total / tot) if tot > 0 else None,
            "fusion_pct":                 (f_total / tot) if tot > 0 else None,
            "vision_flops_pct":           (v_g / flops_tot) if flops_tot > 0 else None,
            "_total_params":              tot,
            "_v_total":                   v_total,
            "_l_total":                   l_total,
            "_fusion_tokens":             fusion_tokens,
            "_name":                      r.get("name", ""),
            "_family":                    r.get("family", ""),
        }
    return out


def _build_benchmark_dict(bench_rows) -> Dict[str, Dict[str, Optional[float]]]:
    out = {}
    for r in bench_rows:
        hf_id = r.get("hf_id")
        if not hf_id:
            continue
        out[hf_id] = {slug: _to_float(r.get(slug)) for slug in BENCHMARK_SLUGS}
        out[hf_id]["_name"] = r.get("name", "")
    return out


def _compute_correlations(structural, benchmarks, log) -> List[dict]:
    """For each (structural_metric, benchmark) pair: Pearson, Spearman + CIs."""
    out_rows = []
    for s_metric, s_desc in STRUCTURAL_METRICS:
        for bench_slug in BENCHMARK_SLUGS:
            xs, ys, names = [], [], []
            for hf_id, sm in structural.items():
                sval = sm.get(s_metric)
                bval = benchmarks.get(hf_id, {}).get(bench_slug)
                if sval is None or bval is None:
                    continue
                xs.append(sval)
                ys.append(bval)
                names.append(sm.get("_name", hf_id))
            row = {
                "structural_metric": s_metric,
                "structural_desc":   s_desc,
                "benchmark":         bench_slug,
                "n":                 len(xs),
                "pearson_r":         None,
                "spearman_rho":      None,
                "pearson_ci_lo":     None,
                "pearson_ci_hi":     None,
                "spearman_ci_lo":    None,
                "spearman_ci_hi":    None,
                "status":            "ok",
            }
            if len(xs) < 3:
                row["status"] = f"insufficient_data (n={len(xs)})"
                log.warn(f"{s_metric} x {bench_slug}: only n={len(xs)} valid points; skipping")
                out_rows.append(row)
                continue
            try:
                r_p = _pearson(xs, ys)
                r_s = _spearman(xs, ys)
                ci_p = _bootstrap_ci(xs, ys, _pearson)
                ci_s = _bootstrap_ci(xs, ys, _spearman)
                row["pearson_r"]      = r_p
                row["spearman_rho"]   = r_s
                if ci_p:
                    row["pearson_ci_lo"], row["pearson_ci_hi"] = ci_p
                if ci_s:
                    row["spearman_ci_lo"], row["spearman_ci_hi"] = ci_s
                log.iteration(
                    f"{s_metric} x {bench_slug}",
                    n=len(xs),
                    pearson=round(r_p, 3) if r_p is not None else None,
                    spearman=round(r_s, 3) if r_s is not None else None,
                )
            except Exception as e:
                log.error(f"Correlation failed for {s_metric} x {bench_slug}", e)
                row["status"] = "error"
            out_rows.append(row)
    return out_rows


def _write_csv(rows, out_path: Path, log):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "structural_metric", "structural_desc", "benchmark", "n",
        "pearson_r", "pearson_ci_lo", "pearson_ci_hi",
        "spearman_rho", "spearman_ci_lo", "spearman_ci_hi",
        "status",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log.info(f"Wrote {out_path}")


def _write_markdown(rows, out_path: Path, log, structural, benchmarks):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase-2 Finding Evidence: F11-F14 — Architecture × Benchmark Correlations\n",
        f"Computed: 20260521\n",
        f"Models with at least one paired (structural, benchmark) point: "
        f"{len([h for h in structural if any(benchmarks.get(h, {}).get(s) is not None for s in BENCHMARK_SLUGS)])}\n",
    ]

    lines.append("\n## F11 — Fusion-boundary token count (log10) vs. each benchmark\n")
    lines.append("| Benchmark | n | Pearson r [95% CI] | Spearman ρ [95% CI] | Status |")
    lines.append("|:--|--:|:--|:--|:--|")
    for r in rows:
        if r["structural_metric"] != "fusion_boundary_tokens_log":
            continue
        pr = f"{r['pearson_r']:.3f}" if r["pearson_r"] is not None else "—"
        sr = f"{r['spearman_rho']:.3f}" if r["spearman_rho"] is not None else "—"
        ci_p = f"[{r['pearson_ci_lo']:.2f}, {r['pearson_ci_hi']:.2f}]" if r["pearson_ci_lo"] is not None else ""
        ci_s = f"[{r['spearman_ci_lo']:.2f}, {r['spearman_ci_hi']:.2f}]" if r["spearman_ci_lo"] is not None else ""
        lines.append(f"| {r['benchmark']} | {r['n']} | {pr} {ci_p} | {sr} {ci_s} | {r['status']} |")

    lines.append("\n## F12 — V/L ratio (log10) vs. each benchmark\n")
    lines.append("| Benchmark | n | Pearson r [95% CI] | Spearman ρ [95% CI] | Status |")
    lines.append("|:--|--:|:--|:--|:--|")
    for r in rows:
        if r["structural_metric"] != "v_to_l_ratio_log":
            continue
        pr = f"{r['pearson_r']:.3f}" if r["pearson_r"] is not None else "—"
        sr = f"{r['spearman_rho']:.3f}" if r["spearman_rho"] is not None else "—"
        ci_p = f"[{r['pearson_ci_lo']:.2f}, {r['pearson_ci_hi']:.2f}]" if r["pearson_ci_lo"] is not None else ""
        ci_s = f"[{r['spearman_ci_lo']:.2f}, {r['spearman_ci_hi']:.2f}]" if r["spearman_ci_lo"] is not None else ""
        lines.append(f"| {r['benchmark']} | {r['n']} | {pr} {ci_p} | {sr} {ci_s} | {r['status']} |")

    lines.append("\n## F13 — Per-component FLOPs share vs. each benchmark\n")
    lines.append("| Structural metric | Benchmark | n | Spearman ρ [95% CI] | Status |")
    lines.append("|:--|:--|--:|:--|:--|")
    for r in rows:
        if r["structural_metric"] not in ("vision_flops_pct", "language_pct", "fusion_pct"):
            continue
        sr = f"{r['spearman_rho']:.3f}" if r["spearman_rho"] is not None else "—"
        ci_s = f"[{r['spearman_ci_lo']:.2f}, {r['spearman_ci_hi']:.2f}]" if r["spearman_ci_lo"] is not None else ""
        lines.append(f"| {r['structural_metric']} | {r['benchmark']} | {r['n']} | {sr} {ci_s} | {r['status']} |")

    lines.append("\n## Interpretation guide\n")
    lines.append("* `|r| > 0.7`: strong correlation — claim cautiously, report with bootstrap CI.")
    lines.append("* `0.4 < |r| < 0.7`: moderate — useful for the discussion but not a headline finding.")
    lines.append("* `|r| < 0.4` or 95% CI crossing 0: not a finding. **Report the null result explicitly.**")
    lines.append("* `n < 3 valid points`: insufficient data — fill in more benchmark cells.")
    lines.append("\nUse the Spearman rather than Pearson result when the structural metric is on a log scale "
                  "and the benchmark is bounded (0-100%) — Spearman is rank-based and more robust to outliers.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Wrote {out_path}")


def _plot_fig13(structural, benchmarks, out_dir: Path, dpi: int, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        log.error("matplotlib not available; skipping fig13", e)
        return

    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["font.family"] = "serif"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, bench, title in [
        (axes[0], "mmmu_val", "MMMU (val)"),
        (axes[1], "mathvista_mini", "MathVista (test-mini)"),
    ]:
        families = sorted({sm.get("_family", "") for sm in structural.values()})
        cmap = plt.get_cmap("tab10")
        for i, fam in enumerate(families):
            xs, ys, names = [], [], []
            for hf_id, sm in structural.items():
                if sm.get("_family") != fam:
                    continue
                x = sm.get("fusion_boundary_tokens_log")
                y = benchmarks.get(hf_id, {}).get(bench)
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(y)
                names.append(sm.get("_name", hf_id))
            if xs:
                ax.scatter(xs, ys, color=cmap(i % 10), edgecolor="black", s=80,
                           alpha=0.8, label=fam.replace("_", " "))
                for x, y, n in zip(xs, ys, names):
                    ax.annotate(n, (x, y), xytext=(5, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel("log10(fusion-boundary tokens + 1)")
        ax.set_ylabel(f"{title} score")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.85)

    fig.suptitle("Fusion-boundary bandwidth vs. benchmark performance (Finding F11)", y=1.02)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig13_bandwidth_vs_benchmarks.pdf"
    png = out_dir / "fig13_bandwidth_vs_benchmarks.png"
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

    with ExperimentLogger(args.log_dir, "17_benchmark_correlation") as log:
        log.section("Loading inputs")
        master_rows = _read_master_wide(meas_dir / "master_wide.csv", log)
        bench_rows = _read_benchmarks(meas_dir / "benchmarks.csv", log)
        if not master_rows:
            log.error("No master_wide.csv data; run 06_aggregate_master_table.py first.")
            sys.exit(1)
        if not bench_rows:
            log.error("No benchmarks.csv data; run 16_build_benchmarks_csv.py first.")
            sys.exit(1)

        log.section("Building structural / benchmark dicts")
        structural = _build_structural_dict(master_rows)
        benchmarks = _build_benchmark_dict(bench_rows)
        log.info(f"Structural records: {len(structural)}")
        log.info(f"Benchmark records:  {len(benchmarks)}")

        log.section("Computing correlations (Pearson, Spearman + bootstrap CIs)")
        try:
            rows = _compute_correlations(structural, benchmarks, log)
        except Exception as e:
            log.error("compute_correlations failed top-level", e)
            sys.exit(2)

        log.section("Writing CSV + Markdown")
        try:
            _write_csv(rows, meas_dir / "benchmark_correlation.csv", log)
            _write_markdown(rows, meas_dir / "benchmark_correlation.md", log, structural, benchmarks)
        except Exception as e:
            log.error("Output write failed", e)
            sys.exit(2)

        log.section("Plotting fig13 (bandwidth vs benchmark)")
        try:
            _plot_fig13(structural, benchmarks, fig_dir, args.dpi, log)
        except Exception as e:
            log.error("fig13 plotting failed", e)

    print(f"Wrote: {meas_dir / 'benchmark_correlation.csv'}")
    print(f"Wrote: {meas_dir / 'benchmark_correlation.md'}")
    print(f"Wrote: {fig_dir / 'fig13_bandwidth_vs_benchmarks.{pdf,png}'}")


if __name__ == "__main__":
    main()
