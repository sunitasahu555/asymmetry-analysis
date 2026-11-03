#!/usr/bin/env python3
"""
03_flops_analysis.py — Per-component forward FLOPs (G-FLOPs).

Purpose
-------
Measure forward FLOPs at the standardized input config (224x224 image, 32-token
text, batch=1) for every component of every model. Uses the calflops → fvcore
→ ptflops → manual fallback chain documented in
``vlm_audit/metrics.py``.

The supervisor flagged FLOPs measurement as "the single highest-value addition"
in their feedback. Per-component FLOPs operationalize Finding 5 (freezing
creates a compute asymmetry distinct from parameter asymmetry).

Outputs
-------
* ``outputs/logs/<ts>_03_flops_analysis__<model>.txt`` — verbose log per model;
  notes which counter actually produced the number, including when the manual
  fallback was used.
* Updates ``outputs/measurements/<model>.json`` with ``components[*].flops_g``.

Hardware
--------
FLOPs counting is analytical (fvcore / ptflops) or trace-based (calflops);
GPU not strictly required, but model load is much faster on GPU.

Usage
-----
    python 03_flops_analysis.py
    python 03_flops_analysis.py --models LLaVA-1.5-7B
    python 03_flops_analysis.py --image-hw 336 336 --text-tokens 32
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
    p.add_argument("--image-hw", nargs=2, type=int, default=[224, 224],
                   metavar=("H", "W"), help="Image resolution for FLOPs measurement")
    p.add_argument("--text-tokens", type=int, default=32)
    p.add_argument("--log-dir", default="outputs/logs")
    p.add_argument("--measurement-dir", default="outputs/measurements")
    return p.parse_args()


def main():
    args = parse_args()
    specs = get_model_list(names=args.models) if args.models else MODELS
    if not specs:
        print("No models matched the filter.", file=sys.stderr)
        sys.exit(1)

    ic = InputConfig(
        image_hw=tuple(args.image_hw),
        text_tokens=args.text_tokens,
        batch_size=1,
        precision="fp32",  # FLOPs counting is precision-independent
    )
    print(f"03_flops_analysis — running FLOPs on {len(specs)} model(s)")
    print(f"  image_hw:    {ic.image_hw}")
    print(f"  text_tokens: {ic.text_tokens}")
    print(f"  quantization: {args.quantization}")

    overall_t = time.time()
    failed = []
    for i, spec in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}] {spec.name}  ({spec.hf_id})")
        t0 = time.time()
        try:
            rec = run_full_audit(
                spec,
                metrics={"flops"},
                ic=ic,
                device=args.device,
                quantization=args.quantization,
                measurement_dir=args.measurement_dir,
                log_dir=args.log_dir,
                notebook_label="03_flops_analysis",
            )
            total_g = sum((c.flops_g or 0) for c in rec.components.values())
            print(f"  total FLOPs (sum of components): {total_g:.4f} G  ({time.time() - t0:.1f}s)")
            for cname, c in rec.components.items():
                if c.flops_g is not None:
                    print(f"    {cname:<10s} {c.flops_g:.4f} G")
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
