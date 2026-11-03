#!/usr/bin/env python3
"""
00_environment_check.py — Verify the runtime environment.

Purpose
-------
Confirm that every dependency is installed and that CUDA + GPU(s) are visible.
Write a verbose log so a single ``cat outputs/logs/*_00_environment_check*.txt``
tells you whether the box is ready for the rest of the pipeline.

Outputs
-------
* ``outputs/logs/<timestamp>_00_environment_check.txt`` — verbose log
* Prints PASS / FAIL summary to stdout

Hardware
--------
None required (works on CPU). GPU is reported if present.

Usage
-----
    python 00_environment_check.py
"""

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path

# Make the vlm_audit package importable when running from this folder.
sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import ExperimentLogger


REQUIRED_PKGS = [
    # name imported in code, name displayed in log
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("huggingface_hub", "huggingface_hub"),
    ("calflops", "calflops"),
    ("fvcore", "fvcore"),
    ("ptflops", "ptflops"),
    ("bitsandbytes", "bitsandbytes"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("tqdm", "tqdm"),
    ("PIL", "pillow"),
    ("yaml", "pyyaml"),
    ("einops", "einops"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-dir", default="outputs/logs", help="Where to write the run log")
    return p.parse_args()


def main():
    args = parse_args()
    failures = []

    with ExperimentLogger(args.log_dir, "00_environment_check") as log:
        log.section("System information")
        log.info(f"Platform: {platform.platform()}")
        log.info(f"Machine: {platform.machine()}")
        log.info(f"Processor: {platform.processor()}")
        log.info(f"Python executable: {sys.executable}")
        log.info(f"Python version: {sys.version.split()[0]}")

        log.section("Required packages")
        for import_name, display_name in REQUIRED_PKGS:
            try:
                mod = importlib.import_module(import_name)
                ver = getattr(mod, "__version__", "?")
                log.iteration(f"check {display_name}", installed=True, version=ver)
                log.metric(f"{display_name}_version", ver)
            except Exception as e:
                log.iteration(f"check {display_name}", installed=False, error=str(e))
                failures.append(display_name)

        # GPU
        log.section("CUDA / GPU")
        try:
            import torch
            log.info(f"PyTorch CUDA built: {torch.version.cuda}")
            log.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                log.info(f"Device count: {n}")
                for i in range(n):
                    name = torch.cuda.get_device_name(i)
                    cap = torch.cuda.get_device_capability(i)
                    mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
                    log.iteration(f"GPU {i}", name=name, compute_capability=cap, memory_gb=round(mem_gb, 1))
                    log.metric(f"gpu_{i}_memory_gb", round(mem_gb, 1))
            else:
                log.warn("CUDA is not available — memory measurements (04) will be skipped")
        except Exception as e:
            log.error("GPU check failed", e)
            failures.append("cuda_check")

        # HF token
        log.section("HuggingFace credentials")
        token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        log.info(f"HF token in env: {token_present}")
        if not token_present:
            log.warn("No HF token in env — LLaMA-based models (LLaVA, IDEFICS) may fail to download")

        # Output folders
        log.section("Output directories")
        for d in ("outputs/logs", "outputs/measurements", "outputs/figures"):
            p = Path(d)
            p.mkdir(parents=True, exist_ok=True)
            log.info(f"{d} → {p.resolve()}")

        log.section("Summary")
        if failures:
            log.error(f"FAILED packages/checks: {failures}")
        else:
            log.info("All checks PASSED")

    # stdout summary
    if failures:
        print(f"FAIL — missing or broken: {failures}", file=sys.stderr)
        sys.exit(1)
    print("PASS — environment ready")


if __name__ == "__main__":
    main()
