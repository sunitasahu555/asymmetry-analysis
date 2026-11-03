#!/usr/bin/env python3
"""
06_aggregate_master_table.py — Build the master CSV from the per-model JSONs.

Purpose
-------
After scripts 02-05 have run, this consolidates every
``outputs/measurements/<model>.json`` into a single long-format CSV with one
row per (model, component). This is the table the paper's headline figures
and per-finding analysis read from.

Outputs
-------
* ``outputs/measurements/master.csv`` — long-format (one row per model-component)
* ``outputs/measurements/master_wide.csv`` — wide-format (one row per model,
  with V/F/L/H columns)
* ``outputs/logs/<ts>_06_aggregate_master_table.txt``

Hardware
--------
CPU only.

Usage
-----
    python 06_aggregate_master_table.py
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import MODELS, ExperimentLogger
from vlm_audit.schema import ModelRecord
from vlm_audit.logging_utils import sanitize_for_filename


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", default="outputs/measurements")
    p.add_argument("--log-dir", default="outputs/logs")
    return p.parse_args()


LONG_COLS = [
    "model_id", "name", "family", "release_year", "component",
    "params_total", "params_trainable",
    "flops_g", "peak_mem_mb", "output_tokens",
    "module_path", "notes",
]

WIDE_COLS_BASE = ["model_id", "name", "family", "release_year"]


def main():
    args = parse_args()
    meas_dir = Path(args.measurement_dir)
    out_long = meas_dir / "master.csv"
    out_wide = meas_dir / "master_wide.csv"

    with ExperimentLogger(args.log_dir, "06_aggregate_master_table") as log:
        log.section("Scanning measurement dir")
        log.info(f"Dir: {meas_dir.resolve()}")

        # Collect records — prefer the canonical order from MODELS list
        records = []
        for spec in MODELS:
            path = meas_dir / f"{sanitize_for_filename(spec.hf_id)}.json"
            if not path.exists():
                log.warn(f"Missing JSON for {spec.name} ({path.name})")
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                rec = ModelRecord.from_dict(data)
                records.append(rec)
                log.iteration(f"loaded {spec.name}", file=path.name, components=list(rec.components.keys()))
            except Exception as e:
                log.error(f"Failed to load {path.name}", e)

        log.info(f"Loaded {len(records)} records")
        if not records:
            print("No records to aggregate. Run scripts 02-05 first.", file=sys.stderr)
            sys.exit(1)

        # Long-format
        log.section("Writing long-format CSV")
        with open(out_long, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LONG_COLS)
            w.writeheader()
            for rec in records:
                for cname, c in rec.components.items():
                    w.writerow({
                        "model_id": rec.model_id,
                        "name": rec.name,
                        "family": rec.family,
                        "release_year": rec.release_year,
                        "component": cname,
                        "params_total": c.params_total,
                        "params_trainable": c.params_trainable,
                        "flops_g": c.flops_g,
                        "peak_mem_mb": c.peak_mem_mb,
                        "output_tokens": c.output_tokens,
                        "module_path": c.module_path,
                        "notes": c.notes,
                    })
        log.info(f"Wrote {out_long}")

        # Wide-format
        log.section("Writing wide-format CSV")
        metric_keys = ["params_total", "params_trainable", "flops_g", "peak_mem_mb", "output_tokens"]
        comp_names = ["vision", "fusion", "language", "head"]
        cols = list(WIDE_COLS_BASE)
        for cn in comp_names:
            for mk in metric_keys:
                cols.append(f"{cn}__{mk}")

        with open(out_wide, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for rec in records:
                row = {
                    "model_id": rec.model_id,
                    "name": rec.name,
                    "family": rec.family,
                    "release_year": rec.release_year,
                }
                for cn in comp_names:
                    c = rec.components.get(cn)
                    for mk in metric_keys:
                        row[f"{cn}__{mk}"] = getattr(c, mk, None) if c is not None else None
                w.writerow(row)
        log.info(f"Wrote {out_wide}")

        log.section("Summary")
        for rec in records:
            total_params = sum((c.params_total or 0) for c in rec.components.values())
            total_flops = sum((c.flops_g or 0) for c in rec.components.values())
            total_mem = sum((c.peak_mem_mb or 0) for c in rec.components.values())
            log.iteration(
                f"summary {rec.name}",
                params=f"{total_params:,}",
                flops_g=f"{total_flops:.4f}",
                peak_mem_mb=f"{total_mem:.2f}",
            )

    print(f"Wrote: {out_long}")
    print(f"Wrote: {out_wide}")
    print(f"Records: {len(records)}")


if __name__ == "__main__":
    main()
