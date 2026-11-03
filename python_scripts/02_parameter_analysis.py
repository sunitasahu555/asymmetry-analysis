#!/usr/bin/env python3
"""
02_parameter_analysis.py — Per-component parameter counts (total + trainable).

Purpose
-------
Run the parameter-counting metric on every model in the roster. Writes /
merges into the per-model JSON record so this can be re-run safely if some
models fail and you need to retry just those.

Outputs
-------
* ``outputs/logs/<ts>_02_parameter_analysis__<model>.txt`` — verbose log per model
* ``outputs/measurements/<sanitized_model_id>.json`` — per-model record
  with ``components[*].params_total`` and ``components[*].params_trainable``
  filled in.

Hardware
--------
GPU helps (faster model load) but CPU works. Params are counted by iterating
through the loaded module; no forward pass is required.

Usage
-----
    python 02_parameter_analysis.py
    python 02_parameter_analysis.py --models CLIP-ViT-L/14 BLIP-2-OPT-2.7B
    python 02_parameter_analysis.py --quantization 4bit
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import MODELS, get_model_list, run_full_audit, InputConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--quantization", default=None, choices=[None, "int8", "4bit"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-dir", default="outputs/logs")
    p.add_argument("--measurement-dir", default="outputs/measurements")
    return p.parse_args()


def main():
    args = parse_args()
    specs = get_model_list(names=args.models) if args.models else MODELS
    if not specs:
        print("No models matched the filter.", file=sys.stderr)
        sys.exit(1)

    ic = InputConfig()  # default: 224x224, 32 tokens, fp16
    print(f"02_parameter_analysis — running params on {len(specs)} model(s)")
    print(f"  quantization: {args.quantization}")
    print(f"  device:       {args.device}")

    overall_t = time.time()
    failed = []
    for i, spec in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}] {spec.name}  ({spec.hf_id})")
        t0 = time.time()
        try:
            rec = run_full_audit(
                spec,
                metrics={"params"},
                ic=ic,
                device=args.device,
                quantization=args.quantization,
                measurement_dir=args.measurement_dir,
                log_dir=args.log_dir,
                notebook_label="02_parameter_analysis",
            )
            total_params = sum((c.params_total or 0) for c in rec.components.values())
            print(f"  total params: {total_params:,}  ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(spec.name)

    dt = time.time() - overall_t
    print(f"\nDone in {dt:.1f}s.  {len(specs) - len(failed)}/{len(specs)} succeeded.")
    if failed:
        print("Failures:", failed, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
