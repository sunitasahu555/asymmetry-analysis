#!/usr/bin/env python3
"""
18_native_multimodal_analysis.py — Separate analysis for native-fusion VLMs.

Phase-2 WS-18. Chameleon and Janus-Pro do not fit the V/F/L/H decomposition
that the rest of the master table uses. This script:

* Pulls the native-family records (Chameleon-7B, Janus-Pro-7B) from the
  measurement JSONs.
* Reports their parameter / FLOPs / memory breakdown under the EXTENDED
  schema (fused_transformer, embeddings, vision_und, vision_gen).
* Computes a "native vs adapter" comparison: for each native model, find
  the closest-sized adapter-LLM model and tabulate the structural difference.
* Computes the image-token-vocab share for Chameleon (fraction of total vocab
  that's image-tokens).
* Computes the decoupled-encoder split for Janus-Pro (params in vision_und
  vs. vision_gen vs. shared LLM).

Outputs
-------
* outputs/measurements/native_multimodal_analysis.csv
* outputs/measurements/native_multimodal_analysis.md
* outputs/measurements/native_vs_adapter_pairs.csv

Usage
-----
    python 18_native_multimodal_analysis.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import MODELS, ExperimentLogger
from vlm_audit.schema import ModelRecord
from vlm_audit.logging_utils import sanitize_for_filename


NATIVE_FAMILIES = {"native_early_fusion", "native_decoupled_encoder"}
ADAPTER_FAMILIES = {
    "adapter_llm", "dynamic_tile_adapter_llm", "adapter_llm_own_encoder",
    "bridge_module", "bridge_module_adapter_llm", "perceiver_adapter_llm",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", default="outputs/measurements")
    p.add_argument("--log-dir",         default="outputs/logs")
    return p.parse_args()


def _load_records(meas_dir: Path, log) -> Dict[str, ModelRecord]:
    out = {}
    for spec in MODELS:
        path = meas_dir / f"{sanitize_for_filename(spec.hf_id)}.json"
        if not path.exists():
            log.warn(f"Missing JSON for {spec.name}; skipping")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                out[spec.hf_id] = ModelRecord.from_dict(json.load(f))
        except Exception as e:
            log.error(f"Could not load {path}", e)
    log.info(f"Loaded {len(out)} records")
    return out


def _total_params(rec: ModelRecord) -> int:
    return sum((c.params_total or 0) for c in rec.components.values())


def _analyze_native(records: Dict[str, ModelRecord], log) -> List[dict]:
    rows = []
    for spec in MODELS:
        if spec.family not in NATIVE_FAMILIES:
            continue
        rec = records.get(spec.hf_id)
        if rec is None:
            log.warn(f"No measurement record for {spec.name}; skipping")
            continue

        row = {
            "name": spec.name,
            "hf_id": spec.hf_id,
            "family": spec.family,
            "release_year": spec.release_year,
            "total_params": _total_params(rec),
            "fused_transformer_params": (rec.components.get("fused_transformer").params_total
                                          if rec.components.get("fused_transformer") else 0),
            "embeddings_params": (rec.components.get("embeddings").params_total
                                   if rec.components.get("embeddings") else 0),
            "vision_und_params": (rec.components.get("vision_und").params_total
                                   if rec.components.get("vision_und") else 0),
            "vision_gen_params": (rec.components.get("vision_gen").params_total
                                   if rec.components.get("vision_gen") else 0),
            "vision_params":   (rec.components.get("vision").params_total
                                 if rec.components.get("vision") else 0),
            "language_params": (rec.components.get("language").params_total
                                 if rec.components.get("language") else 0),
            "fusion_params":   (rec.components.get("fusion").params_total
                                 if rec.components.get("fusion") else 0),
            "head_params":     (rec.components.get("head").params_total
                                 if rec.components.get("head") else 0),
            "image_token_vocab_size": None,
        }
        # Chameleon: image-token vocab size is on the "embeddings" component
        # as output_tokens (we put it there in runner.py).
        emb = rec.components.get("embeddings")
        if emb and emb.output_tokens is not None:
            row["image_token_vocab_size"] = emb.output_tokens

        log.iteration(f"native {spec.name}", **{k: v for k, v in row.items()
                                                 if k not in ("hf_id", "family")})
        rows.append(row)
    return rows


def _native_vs_adapter_pairs(records: Dict[str, ModelRecord], log) -> List[dict]:
    """For each native model, find the closest-total-param adapter-LLM model."""
    adapter_specs = [m for m in MODELS if m.family in ADAPTER_FAMILIES]
    native_specs = [m for m in MODELS if m.family in NATIVE_FAMILIES]

    pairs = []
    for nspec in native_specs:
        nrec = records.get(nspec.hf_id)
        if nrec is None:
            continue
        n_total = _total_params(nrec)
        # Pick the adapter model with the closest total parameter count
        best = None
        best_diff = None
        for aspec in adapter_specs:
            arec = records.get(aspec.hf_id)
            if arec is None:
                continue
            a_total = _total_params(arec)
            if a_total <= 0:
                continue
            diff = abs(a_total - n_total)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best = (aspec, arec)
        if best is None:
            log.warn(f"Could not find adapter peer for {nspec.name}")
            continue
        aspec, arec = best
        n_v = (nrec.components.get("vision").params_total if nrec.components.get("vision") else 0) or 0
        n_l = (nrec.components.get("language").params_total if nrec.components.get("language") else 0) or 0
        a_v = (arec.components.get("vision").params_total if arec.components.get("vision") else 0) or 0
        a_l = (arec.components.get("language").params_total if arec.components.get("language") else 0) or 0
        pair = {
            "native_name":    nspec.name,
            "native_family":  nspec.family,
            "native_total":   n_total,
            "native_v":       n_v,
            "native_l":       n_l,
            "adapter_name":   aspec.name,
            "adapter_family": aspec.family,
            "adapter_total":  _total_params(arec),
            "adapter_v":      a_v,
            "adapter_l":      a_l,
            "total_diff_pct": (best_diff / max(n_total, _total_params(arec))) * 100 if best_diff else 0.0,
        }
        log.iteration(f"pair {nspec.name} vs {aspec.name}",
                      total_diff_pct=round(pair["total_diff_pct"], 2))
        pairs.append(pair)
    return pairs


def _write_csv(rows, out_path: Path, log, cols=None):
    if not rows:
        log.warn(f"No rows to write to {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if cols is None:
        cols = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log.info(f"Wrote {out_path}")


def _write_markdown(native_rows, pair_rows, out_path: Path, log):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Native-multimodal analysis (Phase-2 WS-18)\n",
        "Native VLMs (Chameleon, Janus-Pro) do not decompose cleanly into V/F/L/H. "
        "This document reports them under the extended schema and contrasts them "
        "with their closest-sized adapter-LLM peers.\n",
    ]

    lines.append("\n## Native models — extended-schema parameter breakdown\n")
    if not native_rows:
        lines.append("_No native records loaded — run scripts 02/05 on the native-family models first._")
    else:
        lines.append("| Model | Family | Total M | Vision M | Language M | Fusion M | Embeddings M | Vision-und M | Vision-gen M | Img-tok vocab |")
        lines.append("|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
        for r in native_rows:
            tm = lambda x: f"{(x or 0) / 1e6:.2f}"
            vocab = r.get("image_token_vocab_size")
            vocab_str = str(vocab) if vocab is not None else "—"
            lines.append(
                f"| {r['name']} | {r['family']} | {tm(r['total_params'])} | "
                f"{tm(r['vision_params'])} | {tm(r['language_params'])} | "
                f"{tm(r['fusion_params'])} | {tm(r['embeddings_params'])} | "
                f"{tm(r['vision_und_params'])} | {tm(r['vision_gen_params'])} | {vocab_str} |"
            )

    lines.append("\n## Native vs. adapter pairings (closest-total-param adapter peer)\n")
    if not pair_rows:
        lines.append("_No pairings — need both native and adapter records loaded._")
    else:
        lines.append("| Native model | Native V M | Native L M | Adapter peer | Adapter V M | Adapter L M | Δ total % |")
        lines.append("|:--|--:|--:|:--|--:|--:|--:|")
        for p in pair_rows:
            tm = lambda x: f"{(x or 0) / 1e6:.2f}"
            lines.append(
                f"| {p['native_name']} | {tm(p['native_v'])} | {tm(p['native_l'])} | "
                f"{p['adapter_name']} | {tm(p['adapter_v'])} | {tm(p['adapter_l'])} | "
                f"{p['total_diff_pct']:.1f} |"
            )

    lines.append("\n## Methodological note\n")
    lines.append(
        "The V/F/L/H decomposition was designed for connected-paradigm VLMs where a "
        "vision encoder, a fusion module, and a language model are physically separate "
        "submodules. Native-fusion models (Chameleon) and decoupled-encoder unified "
        "models (Janus-Pro) violate this assumption in different ways. We report them "
        "here under the extended schema and do not include them in the main "
        "V/F/L/H aggregate figures; they appear in separate panels in the paper."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Wrote {out_path}")


def main():
    args = parse_args()
    meas_dir = Path(args.measurement_dir)

    with ExperimentLogger(args.log_dir, "18_native_multimodal_analysis") as log:
        log.section("Loading measurement records")
        records = _load_records(meas_dir, log)

        log.section("Analyzing native models")
        try:
            native_rows = _analyze_native(records, log)
        except Exception as e:
            log.error("Native-model analysis failed", e)
            native_rows = []

        log.section("Computing native vs adapter pairings")
        try:
            pair_rows = _native_vs_adapter_pairs(records, log)
        except Exception as e:
            log.error("Native-vs-adapter pairing failed", e)
            pair_rows = []

        log.section("Writing outputs")
        try:
            _write_csv(native_rows, meas_dir / "native_multimodal_analysis.csv", log)
        except Exception as e:
            log.error("Writing native_multimodal_analysis.csv failed", e)
        try:
            _write_csv(pair_rows, meas_dir / "native_vs_adapter_pairs.csv", log)
        except Exception as e:
            log.error("Writing native_vs_adapter_pairs.csv failed", e)
        try:
            _write_markdown(native_rows, pair_rows, meas_dir / "native_multimodal_analysis.md", log)
        except Exception as e:
            log.error("Writing native_multimodal_analysis.md failed", e)

    print(f"Native records analyzed: {len(native_rows)}")
    print(f"Pairings computed:       {len(pair_rows)}")
    print(f"Outputs in {meas_dir}/")


if __name__ == "__main__":
    main()
