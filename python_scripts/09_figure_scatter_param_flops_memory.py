#!/usr/bin/env python3
"""
09_figure_scatter_param_flops_memory.py — Headline triple-axis scatter.

Purpose
-------
The paper's headline figure: parameter count (x) vs FLOPs (y), with marker
size encoding peak memory and marker color encoding architectural family.
One marker per model.

Outputs
-------
* ``outputs/figures/fig09_scatter_param_flops_memory.pdf`` (paper)
* ``outputs/figures/fig09_scatter_param_flops_memory.png`` (presentation)
* ``outputs/logs/<ts>_09_figure_scatter_param_flops_memory.txt``

Hardware
--------
CPU only.

Usage
-----
    python 09_figure_scatter_param_flops_memory.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import MODELS, ExperimentLogger
from vlm_audit.schema import ModelRecord
from vlm_audit.logging_utils import sanitize_for_filename


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
    matplotlib.use("Agg")  # non-interactive
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["figure.dpi"] = args.dpi
    plt.rcParams["savefig.dpi"] = args.dpi
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10

    figs_dir = Path(args.figures_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    with ExperimentLogger(args.log_dir, "09_figure_scatter_param_flops_memory") as log:
        log.section("Loading records")
        rows = []
        for spec in MODELS:
            path = Path(args.measurement_dir) / f"{sanitize_for_filename(spec.hf_id)}.json"
            if not path.exists():
                log.warn(f"Missing JSON for {spec.name}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                rec = ModelRecord.from_dict(json.load(f))
            total_p = sum((c.params_total or 0) for c in rec.components.values())
            total_f = sum((c.flops_g or 0) for c in rec.components.values())
            total_m = sum((c.peak_mem_mb or 0) for c in rec.components.values())
            rows.append({
                "name": spec.name,
                "family": spec.family,
                "params": total_p,
                "flops_g": total_f,
                "mem_mb": total_m,
                "recent_gen": spec.recent_gen,
            })
            log.iteration(spec.name, params=total_p, flops_g=total_f, mem_mb=total_m)
        log.info(f"Loaded {len(rows)} records")

        if not rows:
            print("No records. Run 02–06 first.", file=sys.stderr)
            sys.exit(1)

        # --- Plot ---
        log.section("Plotting")
        families = sorted({r["family"] for r in rows})
        cmap = plt.get_cmap("tab10")
        color_by_family = {fam: cmap(i % 10) for i, fam in enumerate(families)}

        fig, ax = plt.subplots(figsize=(10, 7))

        # Marker size from peak memory (clipped + sqrt for area-proportional)
        mem_values = [r["mem_mb"] for r in rows]
        max_mem = max(mem_values) if max(mem_values) > 0 else 1
        size_for = lambda mb: 80 + 1200 * (max(mb, 0) / max_mem)

        for fam in families:
            xs, ys, sz, names, recent = [], [], [], [], []
            for r in rows:
                if r["family"] != fam:
                    continue
                if r["params"] <= 0 or r["flops_g"] <= 0:
                    continue
                xs.append(r["params"])
                ys.append(r["flops_g"])
                sz.append(size_for(r["mem_mb"]))
                names.append(r["name"])
                recent.append(r["recent_gen"])
            if not xs:
                continue
            ax.scatter(
                xs, ys, s=sz,
                color=color_by_family[fam],
                edgecolor="black", linewidth=0.7,
                alpha=0.75,
                label=fam.replace("_", " "),
            )
            for x, y, name in zip(xs, ys, names):
                ax.annotate(name, (x, y), xytext=(7, 4), textcoords="offset points", fontsize=8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total parameters (log scale)")
        ax.set_ylabel("Total forward FLOPs (G, log scale)")
        ax.set_title(
            "Architectural asymmetry across families\n"
            "(marker area ∝ peak activation memory; n = "
            + str(len(rows)) + " models)"
        )
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

        pdf = figs_dir / "fig09_scatter_param_flops_memory.pdf"
        png = figs_dir / "fig09_scatter_param_flops_memory.png"
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
