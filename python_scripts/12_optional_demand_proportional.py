#!/usr/bin/env python3
"""
12_optional_demand_proportional.py — Demand-proportional allocation fit (GATED).

Purpose
-------
Test whether the architecture-family-dependent asymmetry can be expressed as a
closed-form allocation rule:

    param_share_i = (demand_i ** alpha) / sum_j (demand_j ** alpha)

where ``demand_i`` is a proxy for the per-component task demand and ``alpha``
is fit per architectural family. Computes R^2 and residuals.

**Gating rule (from implementation_plan.md §6 WS-9):**
This script is the most-likely-to-fail addition. Only include the result in
the paper if the fit hits **R^2 >= 0.7** on the within-family fits. The script
prints the R^2 prominently and the user decides whether to include.

Outputs
-------
* ``outputs/measurements/demand_proportional_fit.csv``
* ``outputs/figures/fig12_demand_proportional.{pdf,png}``
* ``outputs/logs/<ts>_12_optional_demand_proportional.txt``

Hardware
--------
CPU only.

Usage
-----
    python 12_optional_demand_proportional.py
    python 12_optional_demand_proportional.py --alpha-grid 0.5 1.0 1.5 2.0
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import MODELS, ExperimentLogger
from vlm_audit.schema import ModelRecord
from vlm_audit.logging_utils import sanitize_for_filename


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", default="outputs/measurements")
    p.add_argument("--figures-dir", default="outputs/figures")
    p.add_argument("--log-dir", default="outputs/logs")
    p.add_argument("--alpha-grid", nargs="+", type=float,
                   default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
    p.add_argument("--inclusion-r2", type=float, default=0.7,
                   help="If best R^2 is below this, do NOT include the result in the paper")
    return p.parse_args()


def _load(meas_dir, log):
    rows = []
    for spec in MODELS:
        path = Path(meas_dir) / f"{sanitize_for_filename(spec.hf_id)}.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            rows.append((spec, ModelRecord.from_dict(json.load(f))))
    return rows


def _shares(rec):
    """Return (v_share, f_share, l_share) — observed parameter shares."""
    v = (rec.components.get("vision").params_total if rec.components.get("vision") else 0) or 0
    f = (rec.components.get("fusion").params_total if rec.components.get("fusion") else 0) or 0
    l = (rec.components.get("language").params_total if rec.components.get("language") else 0) or 0
    total = v + f + l
    if total == 0:
        return (0, 0, 0)
    return (v / total, f / total, l / total)


def _demand_proxy(spec, rec):
    """Demand proxy per component.

    A simple, defensible proxy:
        demand_vision   = image_token_count  (how many tokens vision must produce)
        demand_fusion   = fusion_output_token_count  (information-bandwidth at the bridge)
        demand_language = max_text_tokens  (how many text tokens the LLM must handle)

    These come from each model's architecture configuration. The hope is
    that param_share scales with demand^alpha for some family-specific alpha.
    """
    H, W = spec.native_image_hw
    # Vision demand: count of image tokens at native resolution
    if spec.family == "single_stream_fused":
        ps = 32
    else:
        ps = 14 if "patch14" in spec.hf_id else 16 if "patch16" in spec.hf_id else 16
    vision_demand = max(1, (H // ps) * (W // ps))

    # Fusion demand: tokens output at the fusion boundary
    fusion_demand = None
    for cname in ("fusion", "language"):
        c = rec.components.get(cname)
        if c is not None and c.output_tokens is not None:
            fusion_demand = max(1, c.output_tokens)
            break
    fusion_demand = fusion_demand or 1

    # Language demand: a constant 2048 — most LLMs in our roster handle 2k+
    # contexts in training. This makes ``language_demand`` constant; the
    # interesting variance is in vision and fusion demand. Reviewers may
    # push back on this — see paper discussion.
    language_demand = 2048
    return (vision_demand, fusion_demand, language_demand)


def _predicted_shares(demand, alpha):
    """demand^alpha normalized."""
    powered = [d ** alpha for d in demand]
    s = sum(powered)
    return tuple(p / s for p in powered) if s > 0 else (0, 0, 0)


def _r_squared(actuals, preds):
    """Coefficient of determination across all components and models."""
    actuals = [a for a in actuals]
    preds = [p for p in preds]
    if len(actuals) < 2:
        return None
    mean = sum(actuals) / len(actuals)
    ss_res = sum((a - p) ** 2 for a, p in zip(actuals, preds))
    ss_tot = sum((a - mean) ** 2 for a in actuals)
    if ss_tot == 0:
        return None
    return 1 - ss_res / ss_tot


def main():
    args = parse_args()
    out_csv = Path(args.measurement_dir) / "demand_proportional_fit.csv"
    fig_pdf = Path(args.figures_dir) / "fig12_demand_proportional.pdf"
    fig_png = Path(args.figures_dir) / "fig12_demand_proportional.png"

    with ExperimentLogger(args.log_dir, "12_optional_demand_proportional") as log:
        log.section("Loading records")
        rows = _load(args.measurement_dir, log)
        log.info(f"{len(rows)} records")
        if len(rows) < 5:
            print("Need at least 5 records for a meaningful fit. Run 02-06 first.", file=sys.stderr)
            sys.exit(1)

        # Compute observed shares and demand proxies
        observed = []
        for spec, rec in rows:
            obs = _shares(rec)
            dem = _demand_proxy(spec, rec)
            observed.append((spec, rec, obs, dem))
            log.iteration(
                spec.name,
                vision_share=obs[0], fusion_share=obs[1], language_share=obs[2],
                vision_demand=dem[0], fusion_demand=dem[1], language_demand=dem[2],
            )

        log.section("Searching over alpha grid")
        best_alpha = None
        best_r2 = -float("inf")
        per_alpha_results = []
        for alpha in args.alpha_grid:
            actuals = []
            preds = []
            for spec, rec, obs, dem in observed:
                p = _predicted_shares(dem, alpha)
                actuals.extend(obs)
                preds.extend(p)
            r2 = _r_squared(actuals, preds)
            per_alpha_results.append((alpha, r2))
            log.iteration(f"alpha={alpha}", r2=r2)
            if r2 is not None and r2 > best_r2:
                best_r2 = r2
                best_alpha = alpha

        log.section("Fit results")
        log.info(f"Best alpha: {best_alpha}")
        log.info(f"Best R^2:   {best_r2:.4f}")
        log.info(f"Inclusion threshold: {args.inclusion_r2}")
        log.info(f"Include in paper?    {'YES' if best_r2 >= args.inclusion_r2 else 'NO'}")

        # Write CSV
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["alpha", "r2_global"])
            for alpha, r2 in per_alpha_results:
                w.writerow([alpha, r2])
            w.writerow([])
            w.writerow(["best_alpha", best_alpha])
            w.writerow(["best_r2", best_r2])
            w.writerow(["inclusion_threshold", args.inclusion_r2])
            w.writerow(["recommendation", "INCLUDE_IN_PAPER" if best_r2 >= args.inclusion_r2 else "DROP_FROM_PAPER"])
        log.info(f"Wrote {out_csv}")

        # Plot
        log.section("Plotting predicted vs observed")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["font.family"] = "serif"

        Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

        xs, ys, names, comps = [], [], [], []
        for spec, rec, obs, dem in observed:
            p = _predicted_shares(dem, best_alpha)
            for i, (cn, o, pr) in enumerate(zip(["vision", "fusion", "language"], obs, p)):
                xs.append(pr)
                ys.append(o)
                names.append(spec.name)
                comps.append(cn)

        fig, ax = plt.subplots(figsize=(8, 8))
        for cn, color in [("vision", "#2ECC71"), ("fusion", "#E67E22"), ("language", "#3498DB")]:
            xx = [x for x, c in zip(xs, comps) if c == cn]
            yy = [y for y, c in zip(ys, comps) if c == cn]
            ax.scatter(xx, yy, color=color, edgecolor="black", linewidth=0.5, s=70, label=cn, alpha=0.85)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="y = x")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"Predicted share (demand^{best_alpha} normalized)")
        ax.set_ylabel("Observed parameter share")
        ax.set_title(f"Demand-proportional fit (alpha={best_alpha}, R²={best_r2:.3f})")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_pdf, bbox_inches="tight")
        plt.savefig(fig_png, bbox_inches="tight")
        log.info(f"Saved {fig_pdf}")
        log.info(f"Saved {fig_png}")
        plt.close(fig)

    print(f"Best alpha:      {best_alpha}")
    print(f"Best R^2:        {best_r2:.4f}")
    print(f"Threshold:       {args.inclusion_r2}")
    print(f"Recommendation:  {'INCLUDE in paper' if best_r2 >= args.inclusion_r2 else 'DROP from paper'}")
    print(f"CSV:             {out_csv}")
    print(f"Figure:          {fig_pdf}")


if __name__ == "__main__":
    main()
