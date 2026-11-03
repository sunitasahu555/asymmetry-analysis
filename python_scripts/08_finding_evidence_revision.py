#!/usr/bin/env python3
"""
08_finding_evidence_revision.py — Re-state each of the 10 findings under n=11 evidence.

Purpose
-------
Take each of the 10 findings from the Corrected_Findings document and compute,
from the measurements collected in scripts 02-05, the n=11 numerical evidence
that supports / softens / contradicts the finding. Produces both a CSV
(machine-readable) and a Markdown table (paper-ready).

This is the data backbone for the paper's revised Findings section.

Outputs
-------
* ``outputs/measurements/finding_evidence.csv`` — one row per finding x model
* ``outputs/measurements/finding_evidence.md`` — Markdown summary tables
* ``outputs/logs/<ts>_08_finding_evidence_revision.txt``

Hardware
--------
CPU only.

Usage
-----
    python 08_finding_evidence_revision.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import MODELS, ExperimentLogger
from vlm_audit.schema import ModelRecord
from vlm_audit.logging_utils import sanitize_for_filename


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", default="outputs/measurements")
    p.add_argument("--log-dir", default="outputs/logs")
    return p.parse_args()


# Each finding is encoded as (id, title, function-that-extracts-evidence)
# Functions take a list of (spec, rec) tuples and return a list of dict rows.

def _total(rec, field):
    return sum((getattr(c, field, None) or 0) for c in rec.components.values())


def _by(rec, cname, field):
    c = rec.components.get(cname)
    return getattr(c, field, None) if c else None


def finding_1_equal_allocation(records):
    """F1: Equal resource allocation does not work well — show V:T ratios."""
    rows = []
    for spec, rec in records:
        v = _by(rec, "vision", "params_total") or 0
        l = _by(rec, "language", "params_total") or 0
        ratio = (v / l) if l else None
        rows.append({
            "finding": "F1_equal_allocation",
            "model": spec.name,
            "family": spec.family,
            "vision_params": v,
            "language_params": l,
            "vision_to_language_ratio": ratio,
            "interpretation": "vision-heavy" if ratio and ratio > 1.5 else ("LLM-heavy" if ratio and ratio < 0.67 else "balanced"),
        })
    return rows


def finding_2_family_dependent(records):
    """F2: Each modality scales per its role — show direction by family."""
    rows = []
    by_fam = defaultdict(list)
    for spec, rec in records:
        by_fam[spec.family].append((spec, rec))
    for fam, members in by_fam.items():
        for spec, rec in members:
            v = _by(rec, "vision", "params_total") or 0
            f = _by(rec, "fusion", "params_total") or 0
            l = _by(rec, "language", "params_total") or 0
            total = v + f + l
            rows.append({
                "finding": "F2_family_dependent",
                "model": spec.name,
                "family": fam,
                "vision_pct": (v / total * 100) if total else None,
                "fusion_pct": (f / total * 100) if total else None,
                "language_pct": (l / total * 100) if total else None,
            })
    return rows


def finding_4_fusion_under_allocated(records):
    """F4: Fusion modules are systematically under-allocated."""
    rows = []
    for spec, rec in records:
        v = _by(rec, "vision", "params_total") or 0
        f = _by(rec, "fusion", "params_total") or 0
        l = _by(rec, "language", "params_total") or 0
        total = v + f + l
        rows.append({
            "finding": "F4_fusion_under_allocated",
            "model": spec.name,
            "family": spec.family,
            "fusion_params": f,
            "fusion_pct_of_total": (f / total * 100) if total else None,
        })
    return rows


def finding_5_freezing_compute_asymmetry(records):
    """F5: Freezing decouples parameters from effective compute."""
    rows = []
    for spec, rec in records:
        v_total = _by(rec, "vision", "params_total") or 0
        v_train = _by(rec, "vision", "params_trainable") or 0
        l_total = _by(rec, "language", "params_total") or 0
        l_train = _by(rec, "language", "params_trainable") or 0
        f_total = _by(rec, "fusion", "params_total") or 0
        f_train = _by(rec, "fusion", "params_trainable") or 0
        all_total = v_total + l_total + f_total
        all_train = v_train + l_train + f_train
        rows.append({
            "finding": "F5_freezing_compute_asymmetry",
            "model": spec.name,
            "family": spec.family,
            "total_params": all_total,
            "trainable_params": all_train,
            "trainable_fraction": (all_train / all_total) if all_total else None,
            "vision_trainable_fraction": (v_train / v_total) if v_total else None,
            "language_trainable_fraction": (l_train / l_total) if l_total else None,
        })
    return rows


def finding_5b_flops_distribution(records):
    """F5 supplementary: how FLOPs distribute across components."""
    rows = []
    for spec, rec in records:
        v = _by(rec, "vision", "flops_g") or 0
        f = _by(rec, "fusion", "flops_g") or 0
        l = _by(rec, "language", "flops_g") or 0
        total = v + f + l
        rows.append({
            "finding": "F5b_flops_distribution",
            "model": spec.name,
            "family": spec.family,
            "vision_g": v,
            "fusion_g": f,
            "language_g": l,
            "vision_pct": (v / total * 100) if total else None,
            "language_pct": (l / total * 100) if total else None,
        })
    return rows


def finding_6_image_token_count(records):
    """F6: image-token count at fusion boundary is a distinct scaling dimension."""
    rows = []
    for spec, rec in records:
        f_tokens = _by(rec, "fusion", "output_tokens")
        l_tokens = _by(rec, "language", "output_tokens")
        rows.append({
            "finding": "F6_image_token_count",
            "model": spec.name,
            "family": spec.family,
            "fusion_output_tokens": f_tokens,
            "language_output_tokens": l_tokens,
            "fusion_boundary_tokens": f_tokens if f_tokens is not None else l_tokens,
        })
    return rows


def finding_8_capacity_to_function(records):
    """F8: capacity matched to function per family."""
    rows = []
    for spec, rec in records:
        v = _by(rec, "vision", "params_total") or 0
        l = _by(rec, "language", "params_total") or 0
        rows.append({
            "finding": "F8_capacity_to_function",
            "model": spec.name,
            "family": spec.family,
            "release_year": spec.release_year,
            "v_to_l_ratio": (v / l) if l else None,
            "dominant_component": "vision" if v > l else ("language" if l > v else "balanced"),
        })
    return rows


def finding_10_vilt_limit_case(records):
    """F10: equal scaling is suboptimal — ViLT as the symmetric limit case."""
    rows = []
    for spec, rec in records:
        v = _by(rec, "vision", "params_total") or 0
        l = _by(rec, "language", "params_total") or 0
        rows.append({
            "finding": "F10_symmetric_limit_case",
            "model": spec.name,
            "family": spec.family,
            "is_symmetric_design": (spec.family == "single_stream_fused"),
            "vision_params": v,
            "language_params": l,
        })
    return rows


FINDINGS = [
    ("F1", "Equal allocation does not work", finding_1_equal_allocation),
    ("F2", "Family-dependent scaling direction", finding_2_family_dependent),
    ("F4", "Fusion under-allocation", finding_4_fusion_under_allocated),
    ("F5", "Freezing-induced compute asymmetry", finding_5_freezing_compute_asymmetry),
    ("F5b", "FLOPs distribution by component", finding_5b_flops_distribution),
    ("F6", "Image-token count at fusion boundary", finding_6_image_token_count),
    ("F8", "Capacity matched to functional role", finding_8_capacity_to_function),
    ("F10", "Symmetric allocation limit case", finding_10_vilt_limit_case),
]


def _md_table(rows, fields):
    if not rows:
        return "_(no data)_\n"
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for r in rows:
        cells = []
        for f in fields:
            v = r.get(f)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}")
            elif isinstance(v, int) and v > 1_000_000:
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    out_csv = Path(args.measurement_dir) / "finding_evidence.csv"
    out_md = Path(args.measurement_dir) / "finding_evidence.md"

    with ExperimentLogger(args.log_dir, "08_finding_evidence_revision") as log:
        log.section("Loading records")
        records = []
        for spec in MODELS:
            path = Path(args.measurement_dir) / f"{sanitize_for_filename(spec.hf_id)}.json"
            if not path.exists():
                log.warn(f"Missing JSON for {spec.name}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                records.append((spec, ModelRecord.from_dict(json.load(f))))
        log.info(f"Loaded {len(records)} records")
        if not records:
            print("No records. Run scripts 02-05 first.", file=sys.stderr)
            sys.exit(1)

        all_rows = []
        md_chunks = ["# Finding Evidence — n=" + str(len(records)) + " models\n"]

        for fid, title, fn in FINDINGS:
            log.section(f"{fid}: {title}")
            rows = fn(records)
            for r in rows:
                log.iteration(f"{fid} {r.get('model', '?')}", **{k: v for k, v in r.items() if k not in ("finding", "model")})
            all_rows.extend(rows)
            md_chunks.append(f"## {fid}: {title}\n")
            if rows:
                fields = [k for k in rows[0].keys() if k != "finding"]
                md_chunks.append(_md_table(rows, fields))
            md_chunks.append("\n")

        # CSV — all rows together
        if all_rows:
            all_cols = sorted({k for r in all_rows for k in r.keys()})
            # put 'finding' and 'model' first
            order = [c for c in ["finding", "model", "family", "release_year"] if c in all_cols]
            order += [c for c in all_cols if c not in order]
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=order)
                w.writeheader()
                w.writerows(all_rows)
            log.info(f"Wrote {out_csv}")

        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_chunks))
        log.info(f"Wrote {out_md}")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
