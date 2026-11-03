#!/usr/bin/env python3
"""
13_optional_bottleneck_lit_synthesis.py — Bottleneck-from-ablations literature template.

Purpose
-------
For each model in the roster, surface a placeholder where the user (Sunita)
fills in "what does the original paper's ablation section say is the
bottleneck?" The script doesn't *automate* the lit work — it generates a
structured Markdown template with one section per model, pre-filled with the
arxiv search query and a reference to the original paper, leaving
fields for:
  * Which component the original ablation isolates as the bottleneck
  * The quantitative drop from ablating it
  * The page / section reference in the original paper

Once the user fills these in, the script can be re-run with ``--render`` to
produce the final synthesis table comparing the literature-reported
bottlenecks against the structural measurements from scripts 02-06.

Outputs
-------
* ``outputs/measurements/bottleneck_lit_template.md`` — initial template
* When ``--render`` is passed and the template is filled: a final
  ``outputs/measurements/bottleneck_synthesis.md`` comparing literature
  to measurements.

Hardware
--------
CPU only.

Usage
-----
    python 13_optional_bottleneck_lit_synthesis.py            # generate template
    python 13_optional_bottleneck_lit_synthesis.py --render   # render final table
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import MODELS, ExperimentLogger
from vlm_audit.schema import ModelRecord
from vlm_audit.logging_utils import sanitize_for_filename


# Pre-curated references — fill more in as needed.
REFERENCES = {
    "CLIP-ViT-L/14":          "Radford et al. 2021, 'Learning Transferable Visual Models From Natural Language Supervision', ICML",
    "SigLIP-L/16":            "Zhai et al. 2023, 'Sigmoid Loss for Language Image Pre-Training', ICCV",
    "BLIP-large":             "Li et al. 2022, 'BLIP: Bootstrapping Language-Image Pre-training', ICML",
    "BLIP-2-OPT-2.7B":        "Li et al. 2023, 'BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models', ICML",
    "InstructBLIP-Vicuna-7B": "Dai et al. 2023, 'InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning', NeurIPS",
    "LLaVA-1.5-7B":           "Liu et al. 2023, 'Improved Baselines with Visual Instruction Tuning', arxiv",
    "LLaVA-NeXT-7B":          "Liu et al. 2024, 'LLaVA-NeXT: Improved reasoning, OCR, and world knowledge', blog post + technical report",
    "IDEFICS-9B":             "Laurencon et al. 2023, 'OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents', NeurIPS",
    "Idefics2-8B":            "Laurencon et al. 2024, 'What matters when building vision-language models?', arxiv",
    "ViLT-B/32":              "Kim et al. 2021, 'ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision', ICML",
}


TEMPLATE_HEADER = """# Bottleneck Literature Synthesis Template

For each model below, read the **ablation section** of the original paper and
fill in the four fields. Once every model has data, re-run this script with
``--render`` to produce the final cross-model synthesis table.

The eventual paper paragraph reads roughly:
> *"Models X, Y, Z all report in their own ablations that <component> is the
> bottleneck. This is consistent with our structural finding that
> <component> is allocated <observation>."*

---

"""


TEMPLATE_PER_MODEL = """## {model_name}

**Original paper:** {reference}

**Search hint:** Look for "ablation", "removing", or "without <component>" in the paper.

| Field | Value |
| :--- | :--- |
| Component identified as bottleneck | _TODO: vision / fusion / language / none_ |
| Quantitative impact of ablating it | _TODO: e.g. "-12.4 pts on VQAv2"_ |
| Reference (section / page / table) | _TODO: e.g. "Table 6, Sec 4.3"_ |
| Notes / caveats | _TODO_ |

"""


SYNTHESIS_HEADER = """# Bottleneck Synthesis — Literature vs Structural Measurements

This is the final paper-ready table. Each row pairs the original paper's
self-reported bottleneck with our structural measurement of the same component.
A consistent pattern (e.g. fusion-bottleneck claims matching small fusion
parameter share) is the kind of synthesis that distinguishes a measurement
study from a measurement dump.

| Model | Self-reported bottleneck | Measured vision share | Measured fusion share | Measured language share | Consistent with our finding? |
| :--- | :--- | ---: | ---: | ---: | :--- |
"""


def _read_template(path):
    """Return a dict per model of {field: value} parsed from a filled template."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    # Very lightweight parser: split on "## " sections; within each, read
    # the table rows of the form "| Field | Value |".
    sections = text.split("\n## ")
    out = {}
    for section in sections[1:]:
        title, body = section.split("\n", 1)
        model_name = title.strip()
        fields = {}
        for line in body.splitlines():
            if line.startswith("| ") and " | " in line and "---" not in line:
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) >= 2:
                    fields[parts[0]] = parts[1]
        out[model_name] = fields
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", default="outputs/measurements")
    p.add_argument("--log-dir", default="outputs/logs")
    p.add_argument("--render", action="store_true",
                   help="Read filled template and emit the final synthesis table")
    args = p.parse_args()

    template_path = Path(args.measurement_dir) / "bottleneck_lit_template.md"
    synthesis_path = Path(args.measurement_dir) / "bottleneck_synthesis.md"
    Path(args.measurement_dir).mkdir(parents=True, exist_ok=True)

    with ExperimentLogger(args.log_dir, "13_optional_bottleneck_lit_synthesis") as log:
        if not args.render:
            # Generate template
            log.section("Generating empty template")
            chunks = [TEMPLATE_HEADER]
            for spec in MODELS:
                chunks.append(TEMPLATE_PER_MODEL.format(
                    model_name=spec.name,
                    reference=REFERENCES.get(spec.name, "_TODO: add reference_"),
                ))
                log.iteration(f"template {spec.name}")
            template_path.write_text("\n".join(chunks), encoding="utf-8")
            log.info(f"Wrote {template_path}")
            print(f"Template written: {template_path}")
            print("Now fill in the TODOs in that file, then re-run with --render.")
            return

        # Render mode
        log.section("Rendering final synthesis")
        filled = _read_template(template_path)
        if not filled:
            print("Template is empty — generate it first (without --render).", file=sys.stderr)
            sys.exit(1)

        rows = []
        for spec in MODELS:
            path = Path(args.measurement_dir) / f"{sanitize_for_filename(spec.hf_id)}.json"
            if not path.exists():
                log.warn(f"Missing JSON for {spec.name}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                rec = ModelRecord.from_dict(json.load(f))
            v = (rec.components.get("vision").params_total if rec.components.get("vision") else 0) or 0
            fu = (rec.components.get("fusion").params_total if rec.components.get("fusion") else 0) or 0
            l = (rec.components.get("language").params_total if rec.components.get("language") else 0) or 0
            tot = v + fu + l
            row = {
                "model": spec.name,
                "claimed": filled.get(spec.name, {}).get("Component identified as bottleneck", "_TODO_"),
                "v": v / tot * 100 if tot else 0,
                "f": fu / tot * 100 if tot else 0,
                "l": l / tot * 100 if tot else 0,
            }
            rows.append(row)
            log.iteration(spec.name, **row)

        lines = [SYNTHESIS_HEADER]
        for r in rows:
            consistent = "_TODO_"  # user fills based on judgment
            lines.append(
                f"| {r['model']} | {r['claimed']} | {r['v']:.1f}% | {r['f']:.1f}% | {r['l']:.1f}% | {consistent} |"
            )
        synthesis_path.write_text("\n".join(lines), encoding="utf-8")
        log.info(f"Wrote {synthesis_path}")
        print(f"Synthesis written: {synthesis_path}")


if __name__ == "__main__":
    main()
