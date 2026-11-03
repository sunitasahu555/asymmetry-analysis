#!/usr/bin/env python3
"""
07_within_family_consistency.py — Tables showing how the 2 members of each family agree.

Purpose
-------
For each architectural family with multiple representatives, compute the
percent difference between the first and second model on each metric.
This supports the supervisor's "within-family generalization" claim — that
the patterns hold across multiple models per family, not just the original
single representative.

Outputs
-------
* ``outputs/measurements/within_family_consistency.csv`` — one row per
  (family, metric) showing pair-wise % difference.
* ``outputs/measurements/family_summary.csv`` — one row per family with
  mean / median / range of each metric across the family's members.
* ``outputs/logs/<ts>_07_within_family_consistency.txt``

Hardware
--------
CPU only.

Usage
-----
    python 07_within_family_consistency.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict
from statistics import median

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import MODELS, ExperimentLogger
from vlm_audit.schema import ModelRecord
from vlm_audit.logging_utils import sanitize_for_filename


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", default="outputs/measurements")
    p.add_argument("--log-dir", default="outputs/logs")
    return p.parse_args()


def _load_all_records(meas_dir, log):
    records = []
    for spec in MODELS:
        path = Path(meas_dir) / f"{sanitize_for_filename(spec.hf_id)}.json"
        if not path.exists():
            log.warn(f"Missing JSON for {spec.name}; skipping")
            continue
        with open(path, "r", encoding="utf-8") as f:
            records.append((spec, ModelRecord.from_dict(json.load(f))))
    return records


def _total(rec, field):
    """Sum a numeric component field across V/F/L/H, treating None as 0."""
    return sum((getattr(c, field, None) or 0) for c in rec.components.values())


def main():
    args = parse_args()
    out_pairs = Path(args.measurement_dir) / "within_family_consistency.csv"
    out_summary = Path(args.measurement_dir) / "family_summary.csv"

    with ExperimentLogger(args.log_dir, "07_within_family_consistency") as log:
        log.section("Loading records")
        records = _load_all_records(args.measurement_dir, log)
        log.info(f"Loaded {len(records)} records")

        # Group by family
        by_family = defaultdict(list)
        for spec, rec in records:
            by_family[spec.family].append((spec, rec))

        log.section("Family pairings")
        for fam, members in by_family.items():
            names = [s.name for s, _ in members]
            log.iteration(f"family {fam}", n_members=len(members), members=names)

        # --- Pair-wise comparison ---
        log.section("Computing pair-wise differences")
        pair_rows = []
        metric_fields = [
            ("params_total", "params"),
            ("flops_g", "flops_g"),
            ("peak_mem_mb", "peak_mem_mb"),
        ]
        for fam, members in by_family.items():
            if len(members) < 2:
                log.info(f"family {fam}: n=1, skipping pair-wise")
                continue
            # All pairs (typically just one)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a_spec, a_rec = members[i]
                    b_spec, b_rec = members[j]
                    for field, label in metric_fields:
                        a_val = _total(a_rec, field)
                        b_val = _total(b_rec, field)
                        if a_val == 0 and b_val == 0:
                            continue
                        avg = (a_val + b_val) / 2 if (a_val and b_val) else max(a_val, b_val)
                        pct = ((b_val - a_val) / avg * 100) if avg else None
                        pair_rows.append({
                            "family": fam,
                            "metric": label,
                            "model_a": a_spec.name,
                            "model_b": b_spec.name,
                            "a_value": a_val,
                            "b_value": b_val,
                            "abs_diff": abs(b_val - a_val),
                            "pct_diff_vs_avg": pct,
                        })
                        log.iteration(
                            f"pair {label} {a_spec.name} vs {b_spec.name}",
                            a=a_val, b=b_val, pct=pct,
                        )

        with open(out_pairs, "w", newline="", encoding="utf-8") as f:
            cols = ["family", "metric", "model_a", "model_b", "a_value", "b_value", "abs_diff", "pct_diff_vs_avg"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(pair_rows)
        log.info(f"Wrote {out_pairs}")

        # --- Family summary ---
        log.section("Computing per-family summary statistics")
        summary_rows = []
        for fam, members in by_family.items():
            for field, label in metric_fields:
                vals = [_total(rec, field) for _spec, rec in members]
                vals = [v for v in vals if v > 0]
                if not vals:
                    continue
                row = {
                    "family": fam,
                    "n_models": len(vals),
                    "metric": label,
                    "min": min(vals),
                    "median": median(vals),
                    "max": max(vals),
                    "range": max(vals) - min(vals),
                    "mean": sum(vals) / len(vals),
                }
                summary_rows.append(row)
                log.iteration(f"family-summary {fam} {label}", **{k: v for k, v in row.items() if k not in ("family", "metric")})

        with open(out_summary, "w", newline="", encoding="utf-8") as f:
            cols = ["family", "n_models", "metric", "min", "median", "max", "range", "mean"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(summary_rows)
        log.info(f"Wrote {out_summary}")

    print(f"Wrote: {out_pairs}")
    print(f"Wrote: {out_summary}")


if __name__ == "__main__":
    main()
