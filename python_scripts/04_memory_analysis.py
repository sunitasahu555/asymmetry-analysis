#!/usr/bin/env python3
"""
04_memory_analysis.py — Per-component peak activation memory (MB).

Purpose
-------
Measure peak GPU memory during one forward pass per component, at the
standardized input config, fp16. ``repeats=3`` forward passes per component;
the median is recorded so cuDNN auto-tuning and fragmentation don't poison
the number. The supervisor's "essential" item #2.

Reference hardware
------------------
Single L40S (48 GB). All n=11 models fit fp16 on one L40S; the second L40S is
reserved as a fallback for any future model that doesn't.

Outputs
-------
* ``outputs/logs/<ts>_04_memory_analysis__<model>.txt`` — every forward pass
  is logged as its own iteration with the peak-MB measurement.
* Updates ``outputs/measurements/<model>.json`` with
  ``components[*].peak_mem_mb``.

Hardware
--------
GPU required. Sets ``CUDA_VISIBLE_DEVICES`` if ``--gpu`` is provided.

Usage
-----
    python 04_memory_analysis.py
    python 04_memory_analysis.py --gpu 0
    python 04_memory_analysis.py --gpu 1 --repeats 5
    python 04_memory_analysis.py --models IDEFICS-9B --quantization 4bit
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--quantization", default=None, choices=[None, "int8", "4bit"])
    p.add_argument("--gpu", type=int, default=None,
                   help="Which GPU index to use (sets CUDA_VISIBLE_DEVICES). Default: 0.")
    p.add_argument("--repeats", type=int, default=3, help="Forward passes per component; median is recorded")
    p.add_argument("--image-hw", nargs=2, type=int, default=[224, 224])
    p.add_argument("--text-tokens", type=int, default=32)
    p.add_argument("--log-dir", default="outputs/logs")
    p.add_argument("--measurement-dir", default="outputs/measurements")
    return p.parse_args()


def main():
    args = parse_args()

    # Set the GPU early — must happen before any torch.cuda import side-effect
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Now safe to import torch + vlm_audit
    import torch
    if not torch.cuda.is_available():
        print("CUDA is not available — 04_memory_analysis requires a GPU. Aborting.", file=sys.stderr)
        sys.exit(1)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from vlm_audit import MODELS, get_model_list, run_full_audit, InputConfig

    specs = get_model_list(names=args.models) if args.models else MODELS
    if not specs:
        print("No models matched the filter.", file=sys.stderr)
        sys.exit(1)

    ic = InputConfig(
        image_hw=tuple(args.image_hw),
        text_tokens=args.text_tokens,
        batch_size=1,
        precision="fp16",
    )
    print(f"04_memory_analysis — running memory on {len(specs)} model(s)")
    print(f"  repeats:      {args.repeats}")
    print(f"  image_hw:     {ic.image_hw}")
    print(f"  text_tokens:  {ic.text_tokens}")
    print(f"  quantization: {args.quantization}")

    overall_t = time.time()
    failed = []
    for i, spec in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}] {spec.name}  ({spec.hf_id})  fp16~{spec.fits_in_gb_fp16}GB")
        # Sanity: warn if the model probably won't fit
        free_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if spec.fits_in_gb_fp16 > free_gb - 4 and args.quantization is None:
            print(f"  WARNING: {spec.name} is ~{spec.fits_in_gb_fp16}GB fp16 vs {free_gb:.1f}GB VRAM. "
                  f"Consider --quantization 4bit.")

        t0 = time.time()
        try:
            rec = run_full_audit(
                spec,
                metrics={"memory"},
                ic=ic,
                device="cuda",
                quantization=args.quantization,
                measurement_dir=args.measurement_dir,
                log_dir=args.log_dir,
                notebook_label="04_memory_analysis",
                memory_repeats=args.repeats,
            )
            total_mb = sum((c.peak_mem_mb or 0) for c in rec.components.values())
            print(f"  peak (sum of components): {total_mb:.1f} MB  ({time.time() - t0:.1f}s)")
            for cname, c in rec.components.items():
                if c.peak_mem_mb is not None:
                    print(f"    {cname:<10s} {c.peak_mem_mb:.2f} MB")
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}", file=sys.stderr)
            print(f"  Try re-running just this model with --quantization 4bit.", file=sys.stderr)
            failed.append(spec.name)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(spec.name)

        # Clean between models
        torch.cuda.empty_cache()

    dt = time.time() - overall_t
    print(f"\nDone in {dt:.1f}s.  {len(specs) - len(failed)}/{len(specs)} succeeded.")
    if failed:
        print("Failures:", failed, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
