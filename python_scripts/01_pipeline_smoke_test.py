#!/usr/bin/env python3
"""
01_pipeline_smoke_test.py — Load every model briefly and verify component extraction.

Purpose
-------
Catch model-id typos, missing HF tokens, attribute-path mismatches, and OOM
risks BEFORE the long measurement runs. For each model in the roster this
script:

  1. Downloads / loads the model (with quantization if requested or auto).
  2. Calls ``extract_components`` and confirms V / F / L / H slots populate
     as expected (per family).
  3. Counts params per component as a low-cost sanity check.
  4. Frees the model and moves on.

Outputs
-------
* One verbose log per model under ``outputs/logs/``.
* A summary CSV ``outputs/measurements/smoke_test_summary.csv``.

Hardware
--------
GPU recommended (load is faster). Works on CPU but very slow.

Usage
-----
    python 01_pipeline_smoke_test.py
    python 01_pipeline_smoke_test.py --models CLIP-ViT-L/14 SigLIP-L/16
    python 01_pipeline_smoke_test.py --quantization 4bit
"""

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import (
    MODELS, get_model_list,
    ExperimentLogger,
    extract_components, COMPONENT_NAMES,
    count_params,
    get_quantization_config, describe_quantization,
)
from vlm_audit.runner import _resolve_hf_class, _load_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=None, help="Subset of model NAMES (see vlm_audit/models.py)")
    p.add_argument("--quantization", default=None, choices=[None, "int8", "4bit"], help="bitsandbytes quantization level")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--log-dir", default="outputs/logs")
    p.add_argument("--measurement-dir", default="outputs/measurements")
    return p.parse_args()


def smoke_test_one(spec, quantization, device, log):
    import torch
    log.section(f"Smoke test: {spec.name}  ({spec.hf_id})")
    log.info(f"Family: {spec.family}")
    log.info(f"Quantization: {describe_quantization(quantization)}")

    # Resolve class
    log.section("Resolving HF class")
    try:
        cls = _resolve_hf_class(spec.family, spec.hf_id)
        log.info(f"Class: {cls.__name__}")
    except Exception as e:
        log.error("Class resolution failed", e)
        return {"name": spec.name, "loaded": False, "stage": "class_resolution"}

    # Load
    log.section("Loading")
    dtype = torch.float16
    t0 = time.time()
    try:
        model = _load_model(spec, dtype=dtype, quantization=quantization, logger=log)
        log.info(f"Loaded in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error("Load failed", e)
        return {"name": spec.name, "loaded": False, "stage": "load"}

    # Extract components
    log.section("Extracting components")
    try:
        comps = extract_components(model, spec.family)
    except Exception as e:
        log.error("Component extraction failed", e)
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return {"name": spec.name, "loaded": True, "stage": "extract"}

    row = {"name": spec.name, "hf_id": spec.hf_id, "family": spec.family, "loaded": True, "stage": "ok"}
    for cname in COMPONENT_NAMES:
        mod = comps.get(cname)
        if mod is None:
            log.iteration(f"component {cname}", present=False)
            row[f"{cname}_params"] = 0
            row[f"{cname}_type"] = "absent"
        else:
            n = count_params(mod)
            tname = type(mod).__name__
            log.iteration(f"component {cname}", present=True, type=tname, params=n)
            log.metric("params", n)
            row[f"{cname}_params"] = n
            row[f"{cname}_type"] = tname

    # Cleanup
    del model, comps
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Cleanup done")
    return row


def main():
    args = parse_args()

    if args.models:
        specs = get_model_list(names=args.models)
    else:
        specs = MODELS
    if not specs:
        print("No models matched the filter.", file=sys.stderr)
        sys.exit(1)

    rows = []
    print(f"Smoke testing {len(specs)} models with quantization={args.quantization}")
    for spec in specs:
        run_name = f"01_pipeline_smoke_test__{spec.hf_id.replace('/', '_')}"
        with ExperimentLogger(args.log_dir, run_name) as log:
            try:
                row = smoke_test_one(spec, args.quantization, args.device, log)
            except Exception as e:
                log.error("Unhandled exception", e)
                row = {"name": spec.name, "loaded": False, "stage": "exception"}
        rows.append(row)
        print(f"  [{row.get('stage','?'):<14s}] {spec.name}")

    # Summary CSV
    if rows:
        Path(args.measurement_dir).mkdir(parents=True, exist_ok=True)
        csv_path = Path(args.measurement_dir) / "smoke_test_summary.csv"
        all_cols = sorted({k for r in rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote summary: {csv_path}")

    # Exit code
    failed = [r for r in rows if r.get("stage") != "ok"]
    if failed:
        print(f"\n{len(failed)}/{len(rows)} models FAILED:")
        for r in failed:
            print(f"  - {r['name']:<30s} (stage: {r.get('stage','?')})")
        sys.exit(2)
    print(f"\nAll {len(rows)} models passed.")


if __name__ == "__main__":
    main()
