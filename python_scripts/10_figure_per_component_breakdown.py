#!/usr/bin/env python3
"""
10_figure_per_component_breakdown.py — Stacked-bar breakdown per model.

Purpose
-------
For each model, show how parameters / FLOPs / memory split across the four
components (V / F / L / H). Three subplots side by side, grouped by family
on the x-axis.

This is the figure that makes the family-dependent asymmetry visible at a
glance — CLIP/SigLIP bars are vision-heavy; LLaVA/Flamingo bars are
language-heavy; BLIP-2 has a noticeable fusion slice; ViLT is uniform.

Outputs
-------
* ``outputs/figures/fig10_per_component_breakdown.pdf``
* ``outputs/figures/fig10_per_component_breakdown.png``
* ``outputs/logs/<ts>_10_figure_per_component_breakdown.txt``

Hardware
--------
CPU only.

Usage
-----
    python 10_figure_per_component_breakdown.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import MODELS, ExperimentLogger
from vlm_audit.schema import ModelRecord
from vlm_audit.logging_utils import sanitize_for_filename


COMPONENTS = ["vision", "fusion", "language", "head"]
COMP_COLORS = {
    "vision":   "#2ECC71",  # green
    "fusion":   "#E67E22",  # orange
    "language": "#3498DB",  # blue
    "head":     "#95A5A6",  # gray
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", default="outputs/measurements")
    p.add_argument("--figures-dir", default="outputs/figures")
    p.add_argument("--log-dir", default="outputs/logs")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _load_per_model(meas_dir, log):
    rows = []
    for spec in MODELS:
        path = Path(meas_dir) / f"{sanitize_for_filename(spec.hf_id)}.json"
        if not path.exists():
            log.warn(f"Missing JSON for {spec.name}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            rec = ModelRecord.from_dict(json.load(f))
        rows.append((spec, rec))
    return rows


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

    with ExperimentLogger(args.log_dir, "10_figure_per_component_breakdown") as log:
        log.section("Loading records")
        rows = _load_per_model(args.measurement_dir, log)
        log.info(f"{len(rows)} records")
        if not rows:
            print("No records. Run 02–06 first.", file=sys.stderr)
            sys.exit(1)

        # Sort by family then by release year for nice grouping
        rows.sort(key=lambda r: (r[0].family, r[0].release_year, r[0].name))
        names = [s.name for s, _ in rows]
        x = np.arange(len(names))

        # --- 3 subplots ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
        metrics = [
            ("params_total", "Parameters (millions)", lambda v: v / 1e6),
            ("flops_g", "FLOPs (G)", lambda v: v),
            ("peak_mem_mb", "Peak memory (MB)", lambda v: v),
        ]

        for ax, (field, ylabel, scale) in zip(axes, metrics):
            log.section(f"Subplot: {ylabel}")
            bottom = np.zeros(len(names))
            for comp in COMPONENTS:
                vals = []
                for spec, rec in rows:
                    c = rec.components.get(comp)
                    v = getattr(c, field, None) if c else None
                    vals.append(scale(v) if v else 0)
                vals = np.array(vals)
                ax.bar(
                    x, vals, bottom=bottom,
                    color=COMP_COLORS[comp], edgecolor="black", linewidth=0.4,
                    label=comp,
                )
                log.iteration(f"{ylabel} bar {comp}", values=vals.tolist())
                bottom += vals

            ax.set_ylabel(ylabel)
            ax.set_title(ylabel.split(" ")[0])
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        # Shared x-tick labels at the bottom of the figure
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        for ax in axes[1:]:
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)

        # One legend for the whole figure
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=COMP_COLORS[c], ec="black")
            for c in COMPONENTS
        ]
        fig.legend(handles, COMPONENTS, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle("Per-component breakdown across architectural families", y=1.06)

        pdf = figs_dir / "fig10_per_component_breakdown.pdf"
        png = figs_dir / "fig10_per_component_breakdown.png"
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
