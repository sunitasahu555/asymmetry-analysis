#!/usr/bin/env python3
"""
11_figure_freezing_analysis.py — Trainable vs total parameters per model.

Purpose
-------
Visualize Finding 5: freezing creates a compute asymmetry distinct from
parameter asymmetry. For each model, show:
  * Total params per component (light bar)
  * Trainable params per component (dark overlay)
…so the gap between the two shows what freezing buys you.

BLIP-2, IDEFICS, and the LLaVA-stage-1 numbers should show the most
dramatic gaps; CLIP/SigLIP and ViLT should look fully solid (end-to-end).

Outputs
-------
* ``outputs/figures/fig11_freezing_analysis.pdf``
* ``outputs/figures/fig11_freezing_analysis.png``

Hardware
--------
CPU only.

Usage
-----
    python 11_figure_freezing_analysis.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import MODELS, ExperimentLogger
from vlm_audit.schema import ModelRecord
from vlm_audit.logging_utils import sanitize_for_filename


COMPONENTS = ["vision", "fusion", "language"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", default="outputs/measurements")
    p.add_argument("--figures-dir", default="outputs/figures")
    p.add_argument("--log-dir", default="outputs/logs")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["figure.dpi"] = args.dpi
    plt.rcParams["savefig.dpi"] = args.dpi
    plt.rcParams["font.family"] = "serif"

    figs_dir = Path(args.figures_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    with ExperimentLogger(args.log_dir, "11_figure_freezing_analysis") as log:
        log.section("Loading records")
        rows = []
        for spec in MODELS:
            path = Path(args.measurement_dir) / f"{sanitize_for_filename(spec.hf_id)}.json"
            if not path.exists():
                log.warn(f"Missing JSON for {spec.name}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                rows.append((spec, ModelRecord.from_dict(json.load(f))))

        rows.sort(key=lambda r: (r[0].family, r[0].release_year))
        names = [s.name for s, _ in rows]
        x = np.arange(len(names))

        fig, ax = plt.subplots(figsize=(14, 6))

        # Bar offsets per component
        width = 0.22
        offsets = {comp: (i - 1) * (width + 0.04) for i, comp in enumerate(COMPONENTS)}
        comp_colors = {"vision": "#2ECC71", "fusion": "#E67E22", "language": "#3498DB"}

        for comp in COMPONENTS:
            totals = []
            trainables = []
            for spec, rec in rows:
                c = rec.components.get(comp)
                t = (c.params_total if c else 0) or 0
                r = (c.params_trainable if c else 0) or 0
                totals.append(t / 1e6)
                trainables.append(r / 1e6)
                log.iteration(f"{spec.name} {comp}", total_m=t/1e6, trainable_m=r/1e6)

            ax.bar(x + offsets[comp], totals, width=width,
                   color=comp_colors[comp], alpha=0.35,
                   edgecolor="black", linewidth=0.4,
                   label=f"{comp} (total)" if comp == "vision" else None)
            ax.bar(x + offsets[comp], trainables, width=width,
                   color=comp_colors[comp], alpha=1.0,
                   edgecolor="black", linewidth=0.5,
                   label=f"{comp} (trainable)" if comp == "vision" else None)

        # Build a clean legend
        from matplotlib.patches import Patch
        handles = []
        for comp in COMPONENTS:
            handles.append(Patch(color=comp_colors[comp], alpha=0.35, ec="black", label=f"{comp} total"))
            handles.append(Patch(color=comp_colors[comp], alpha=1.0, ec="black", label=f"{comp} trainable"))
        ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=3, framealpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_yscale("log")
        ax.set_ylabel("Parameters (millions, log scale)")
        ax.set_title("Trainable vs total parameters by component\nfreezing creates a compute asymmetry distinct from parameter asymmetry")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        pdf = figs_dir / "fig11_freezing_analysis.pdf"
        png = figs_dir / "fig11_freezing_analysis.png"
        plt.tight_layout()
        plt.savefig(pdf, bbox_inches="tight")
        plt.savefig(png, bbox_inches="tight")
        log.info(f"Saved {pdf}")
        log.info(f"Saved {png}")
        plt.close(fig)

    print(f"Wrote: {pdf}")
    print(f"Wrote: {png}")


if __name__ == "__main__":
    main()
