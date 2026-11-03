#!/usr/bin/env python3
"""
16_build_benchmarks_csv.py — Build/validate benchmarks.csv from a YAML/JSON source.

This script does NOT do automatic data entry. The user fills in benchmark
scores into a structured YAML file (one block per model) drawn from each
model's original paper or technical report; the script validates the file
against the canonical schema, fills in nulls for missing cells, and emits
``outputs/measurements/benchmarks.csv`` + a provenance Markdown
``outputs/measurements/benchmarks_provenance.md``.

Schema (per model):
    hf_id: "openai/clip-vit-large-patch14"
    name: "CLIP-ViT-L/14"
    family: "dual_encoder_contrastive"
    release_year: 2021
    benchmarks:
      mmmu_val:
        score: 36.0
        source: "Radford et al. 2021, n/a — CLIP does not report MMMU"
      mathvista_mini:
        score: null
        source: ""
      mmbench_en_dev:
        score: ...
        source: "..."
      # etc — 12 benchmarks per model

If the input YAML doesn't exist, the script writes a TEMPLATE file with
one empty block per model in the current roster, so the user can begin
filling it in (Phase-2 WS-16).

All operations are logged. Every model × benchmark cell either has a
``score`` + ``source`` or is null. The provenance .md groups by source
so the user can audit at a glance.

Usage
-----
    # First run: writes outputs/measurements/benchmarks_template.yaml
    python 16_build_benchmarks_csv.py --emit-template

    # User fills in the template (manually, by reading model papers), then:
    python 16_build_benchmarks_csv.py --input outputs/measurements/benchmarks_filled.yaml

    # Validates schema, writes benchmarks.csv + benchmarks_provenance.md.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import MODELS, ExperimentLogger


# The 12 canonical benchmarks for Phase 2. Each is identified by a short
# slug used as the column key in benchmarks.csv.
BENCHMARKS = [
    ("mmmu_val",          "MMMU (val)",            "Multi-discipline knowledge"),
    ("mmmu_pro",          "MMMU-Pro",              "Robust MMMU (no text-only)"),
    ("mathvista_mini",    "MathVista (test-mini)", "Math + visual reasoning"),
    ("mmbench_en_dev",    "MMBench-v1.1 (EN dev)", "Broad multimodal capability"),
    ("ocrbench",          "OCRBench",              "OCR / document understanding"),
    ("ai2d",              "AI2D",                  "Diagram understanding"),
    ("chartqa_relaxed",   "ChartQA (relaxed)",     "Chart understanding"),
    ("docvqa_val",        "DocVQA (val)",          "Document VQA"),
    ("mmvet",             "MMVet",                 "Open-ended VLM evaluation"),
    ("hallusionbench",    "HallusionBench",        "Hallucination evaluation"),
    ("seedbench_image",   "SEED-Bench (image)",    "Image perception"),
    ("mmstar",            "MMStar",                "Vision-essential benchmark"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emit-template", action="store_true",
                   help="Write benchmarks_template.yaml with empty blocks for all models in roster.")
    p.add_argument("--input", default="outputs/measurements/benchmarks_filled.yaml",
                   help="Path to the filled YAML file to convert to CSV.")
    p.add_argument("--measurement-dir", default="outputs/measurements")
    p.add_argument("--log-dir", default="outputs/logs")
    return p.parse_args()


def _emit_template(out_path: Path, log) -> None:
    """Write a blank template with one block per model in MODELS.

    Format is YAML for human readability; users may also fill in JSON
    (load_input accepts both).
    """
    lines = [
        "# benchmarks_template.yaml — Phase 2 benchmark integration",
        "#",
        "# Fill in scores from each model's original paper / technical report.",
        "# For unavailable benchmarks, leave score: null with a brief note.",
        "# Source field: cite the paper + table number, e.g.",
        '#   source: "Wang et al. 2024 Qwen2-VL Tech Report, Table 4"',
        "#",
        "# When you finish, run:",
        "#   python 16_build_benchmarks_csv.py --input <this file>",
        "",
    ]
    for spec in MODELS:
        lines.append(f"- hf_id: {spec.hf_id}")
        lines.append(f"  name: {spec.name}")
        lines.append(f"  family: {spec.family}")
        lines.append(f"  release_year: {spec.release_year}")
        lines.append(f"  benchmarks:")
        for slug, _label, _axis in BENCHMARKS:
            lines.append(f"    {slug}:")
            lines.append(f"      score: null   # TODO")
            lines.append(f'      source: ""    # TODO')
        lines.append("")  # blank line between models

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Template written: {out_path}")
    print(f"Template written: {out_path}")
    print(f"  Fill in the TODO fields, then run:")
    print(f"  python 16_build_benchmarks_csv.py --input {out_path}")


def _load_input(path: Path, log):
    """Load the filled YAML or JSON file. Returns a list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    text = path.read_text(encoding="utf-8")
    # Try YAML first; fall back to JSON
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            log.error("pyyaml not installed but --input is a YAML file. pip install pyyaml")
            raise
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Input must be a YAML/JSON list of model blocks; got {type(data).__name__}")
    return data


def _validate_block(block, idx, log) -> dict:
    """Sanity-check one model block; fill in missing keys with null."""
    if not isinstance(block, dict):
        log.warn(f"Block #{idx}: not a dict, skipping")
        return None
    hf_id = block.get("hf_id")
    if not hf_id:
        log.warn(f"Block #{idx}: missing hf_id, skipping")
        return None
    bm = block.get("benchmarks") or {}
    if not isinstance(bm, dict):
        log.warn(f"Block {hf_id}: 'benchmarks' is not a dict; ignoring")
        bm = {}
    cleaned = {
        "hf_id": hf_id,
        "name": block.get("name") or hf_id,
        "family": block.get("family") or "",
        "release_year": block.get("release_year") or 0,
        "benchmarks": {},
    }
    for slug, _label, _axis in BENCHMARKS:
        cell = bm.get(slug) or {}
        score = cell.get("score") if isinstance(cell, dict) else None
        source = cell.get("source", "") if isinstance(cell, dict) else ""
        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                log.warn(f"Block {hf_id}: {slug} score {score!r} not numeric; setting null")
                score = None
        cleaned["benchmarks"][slug] = {"score": score, "source": source or ""}
    return cleaned


def _write_csv(rows, out_path: Path, log):
    """Write benchmarks.csv with one row per model."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["hf_id", "name", "family", "release_year"]
    for slug, _label, _axis in BENCHMARKS:
        cols.append(slug)
        cols.append(f"{slug}_source")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            flat = {
                "hf_id": row["hf_id"],
                "name": row["name"],
                "family": row["family"],
                "release_year": row["release_year"],
            }
            for slug, _label, _axis in BENCHMARKS:
                cell = row["benchmarks"].get(slug, {})
                flat[slug] = cell.get("score")
                flat[f"{slug}_source"] = cell.get("source", "")
            w.writerow(flat)
    log.info(f"Wrote {out_path}")


def _write_provenance(rows, out_path: Path, log):
    """Write benchmarks_provenance.md grouping cells by their source."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_source = {}
    nulls = []
    for row in rows:
        for slug, _label, _axis in BENCHMARKS:
            cell = row["benchmarks"].get(slug, {})
            if cell.get("score") is None:
                nulls.append((row["name"], slug))
                continue
            src = cell.get("source", "(no source)")
            by_source.setdefault(src, []).append((row["name"], slug, cell["score"]))

    lines = ["# benchmarks.csv provenance summary\n"]
    lines.append(f"Models: {len(rows)} · Benchmarks: {len(BENCHMARKS)} · Total cells: {len(rows) * len(BENCHMARKS)}\n")
    n_filled = sum(len(v) for v in by_source.values())
    lines.append(f"Filled cells: {n_filled} · Null cells: {len(nulls)}\n")
    lines.append(f"Coverage: {100.0 * n_filled / (len(rows) * len(BENCHMARKS)):.1f}%\n")

    lines.append("\n## Cells by source\n")
    for src in sorted(by_source.keys()):
        lines.append(f"\n### {src}")
        for name, slug, score in by_source[src]:
            lines.append(f"- {name} → {slug} = {score}")

    lines.append("\n## Null cells (to fill in)\n")
    null_by_model = {}
    for name, slug in nulls:
        null_by_model.setdefault(name, []).append(slug)
    for name in sorted(null_by_model.keys()):
        lines.append(f"\n### {name}")
        for slug in null_by_model[name]:
            lines.append(f"- {slug}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Wrote {out_path}")


def main():
    args = parse_args()
    meas_dir = Path(args.measurement_dir)

    with ExperimentLogger(args.log_dir, "16_build_benchmarks_csv") as log:
        log.section("Configuration")
        log.info(f"emit_template: {args.emit_template}")
        log.info(f"input: {args.input}")
        log.info(f"measurement_dir: {meas_dir.resolve()}")

        if args.emit_template:
            log.section("Emitting template")
            try:
                _emit_template(meas_dir / "benchmarks_template.yaml", log)
            except Exception as e:
                log.error("Template emission failed", e)
                sys.exit(2)
            return

        log.section("Loading input")
        try:
            blocks = _load_input(Path(args.input), log)
            log.info(f"Loaded {len(blocks)} blocks")
        except FileNotFoundError as e:
            log.error("Input file not found", e)
            print(f"ERROR: {e}", file=sys.stderr)
            print(f"If you haven't started yet, run: python {Path(__file__).name} --emit-template", file=sys.stderr)
            sys.exit(2)
        except Exception as e:
            log.error("Input load failed", e)
            sys.exit(2)

        log.section("Validating blocks")
        rows = []
        for i, b in enumerate(blocks, 1):
            cleaned = _validate_block(b, i, log)
            if cleaned is None:
                continue
            log.iteration(f"validated #{i}", name=cleaned["name"], family=cleaned["family"])
            rows.append(cleaned)
        log.info(f"Validated {len(rows)} of {len(blocks)} blocks")

        if not rows:
            log.error("No valid rows; aborting")
            sys.exit(2)

        log.section("Writing benchmarks.csv + provenance.md")
        try:
            _write_csv(rows, meas_dir / "benchmarks.csv", log)
        except Exception as e:
            log.error("CSV write failed", e)
            sys.exit(2)
        try:
            _write_provenance(rows, meas_dir / "benchmarks_provenance.md", log)
        except Exception as e:
            log.error("Provenance .md write failed", e)

        # Coverage check
        total_cells = len(rows) * len(BENCHMARKS)
        filled = sum(
            1 for r in rows for slug, _l, _a in BENCHMARKS
            if r["benchmarks"][slug]["score"] is not None
        )
        log.section("Coverage summary")
        log.info(f"Filled cells: {filled} / {total_cells} = {100.0 * filled / total_cells:.1f}%")

    print(f"Wrote: {meas_dir / 'benchmarks.csv'}")
    print(f"Wrote: {meas_dir / 'benchmarks_provenance.md'}")
    print(f"Coverage: {filled} / {total_cells} cells filled ({100.0 * filled / total_cells:.1f}%)")


if __name__ == "__main__":
    main()
