#!/usr/bin/env python3
"""
05_token_count_analysis.py — Fusion-boundary token count per model (static).

Purpose
-------
Static introspection of how many tokens cross from the visual side into the
language model. These are not measured at runtime — they're determined by
each architecture and reported in the literature. The number is captured here
so the master CSV (notebook 06) has a complete row per model.

This implements the supervisor's "image-token count is a first-class scaling
dimension" point (corrected from the earlier "576 patches for 224x224" mistake).

Outputs
-------
* ``outputs/logs/<ts>_05_token_count_analysis__<model>.txt`` — one entry per model
* Updates ``outputs/measurements/<model>.json`` with
  ``components[*].output_tokens`` (attached to fusion when present, else language).

Hardware
--------
CPU only. No model loading required.

Usage
-----
    python 05_token_count_analysis.py
    python 05_token_count_analysis.py --image-hw 336 336   # tile-1 LLaVA-1.5
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
    p.add_argument("--image-hw", nargs=2, type=int, default=[224, 224],
                   metavar=("H", "W"), help="Resolution for ViLT patch-count calculation")
    p.add_argument("--log-dir", default="outputs/logs")
    p.add_argument("--measurement-dir", default="outputs/measurements")
    return p.parse_args()


def main():
    args = parse_args()
    specs = get_model_list(names=args.models) if args.models else MODELS

    ic = InputConfig(image_hw=tuple(args.image_hw), text_tokens=32, batch_size=1)
    print(f"05_token_count_analysis — {len(specs)} model(s)")
    print(f"  image_hw: {ic.image_hw}")

    for i, spec in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] {spec.name}")
        rec = run_full_audit(
            spec,
            metrics={"tokens"},
            ic=ic,
            device="cpu",
            measurement_dir=args.measurement_dir,
            log_dir=args.log_dir,
            notebook_label="05_token_count_analysis",
        )
        # Print which component holds the token count
        for cname, c in rec.components.items():
            if c.output_tokens is not None:
                print(f"  {cname:<10s} output_tokens: {c.output_tokens}")
    print("\nDone.")


if __name__ == "__main__":
    main()
