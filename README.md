# Architectural and Computational Asymmetry in Vision-Language Models

**Measurement code, data and figures for a census of 23 open-weight vision-language models**

**Author:** Sunita Sahu

This repository contains the code and analysis behind the paper *Architectural and Computational Asymmetry in Vision-Language Models*. The paper is a measurement-methodology and resource contribution. It defines a validated protocol for attributing parameters and forward compute to the components of a vision-language model (vision encoder, fusion, language model, output head), and applies it to 23 released models from 2021 to 2026.

Everything is measured from the released model files:
- **Parameters by component:** an explicit tensor-level rule table with a zero residual against the checkpoint total.
- **Forward compute:** a tensor-level cost model validated against PyTorch's op-level FLOP counter.
- **Visual tokens:** counted by running each model's released preprocessing on a common image.
- **Trainable shares:** taken from primary training descriptions and from checkpoint comparisons.
- **Benchmark scores:** traced to their primary sources.
- **Latency:** measured on one NVIDIA A100 80 GB GPU.

> **Status.** The manuscript is under revision. The numbers in `code_active/outputs/` are those of the revision (October 2026). No model weights are stored here: the code reads configurations, checkpoint headers and, for the GPU runs, the released weights from Hugging Face.

---

## Contents

1. [What the code measures](#what-the-code-measures)
2. [Repository layout](#repository-layout)
3. [Model roster](#model-roster)
4. [Environments](#environments)
5. [Reproducing the results](#reproducing-the-results)
6. [Outputs: where each table and figure comes from](#outputs-where-each-table-and-figure-comes-from)
7. [Main results in brief](#main-results-in-brief)
8. [Conventions](#conventions)
9. [Earlier pipeline (phases 1-3)](#earlier-pipeline-phases-1-3)
10. [Notes](#notes)

---

## What the code measures

| Quantity | How it is obtained | Validation | Code |
|---|---|---|---|
| Parameters per component (V, F, L, Head) | Tensor names and shapes from each checkpoint (weightless model or checkpoint header), assigned by an explicit rule table | Residual against the checkpoint total is 0 for all 23 models | `vlm_audit/parameter_layouts.py`, `attribution_rules.py`, `checkpoint_headers.py` |
| Visual tokens at the fusion boundary | Each model's released processor run on synthetic images (448², 1,024², maximum over 36 shapes); remote-code preprocessing re-implemented from the shipped files | Gemma 4 transcription equals both released processors on 63 shapes | `vlm_audit/token_accounting.py` |
| Forward compute (FLOPs) | Every weight matrix charged for the tokens it multiplies, plus attention products; intrinsic and executed (padded) work kept apart | `torch.utils.flop_counter.FlopCounterMode`: 64 comparisons on reduced-depth copies of 21 architectures and 168 on the full-depth released weights of all 23, largest difference 0.008% | `vlm_audit/tensor_cost_model.py`, `flop_counter_validation.py` |
| Element-wise work omitted by the cost model | Every non-matrix operation counted at the PyTorch dispatcher | 0.07–0.91% of FLOPs (reduced depth), 0.02–0.37% (full depth) | `flop_counter_validation.py` |
| Trainable share in the final training stage | Primary training descriptions; where silent, single tensors compared between related released checkpoints | Sources and confidence recorded per model | `attribution_rules.py` (`FINAL_STAGE_UPDATED`) |
| Benchmark scores | First-party sources, with split, metric, prompting, subset and table per score | Every score traced to its source | `outputs/measurements/benchmarks_v2.csv` |
| Latency | Prefill (whole model and per component), decode per token, memory and full-depth FLOPs, bfloat16, batch size 1, on one A100-SXM4-80GB | Same GPU type and software stack for all runs; environment manifests saved | `vlm_audit/latency_measurement.py`, `gpu_colab/` |
| Compute as a function of the input | Every model priced on 16 square sizes (224–4,096 px), 11 aspect ratios and Gemma 4's token budgets | Agrees with the cost model at the two GPU-measured inputs | `vlm_audit/input_size_sweep.py` |
| Operator calls per decode step | Kernel-launching operator calls of one decode step, counted on reduced-depth copies and scaled to full depth by layer type | Same count under transformers 4.57 and 5.18 (Qwen3-VL) | `vlm_audit/operator_dispatch.py` |

---

## Repository layout

```
.
├── README.md                        this file
├── requirements.txt                 CPU analysis environment (transformers 4.57.6)
├── requirements-transformers5.txt   Gemma 4 environment (transformers 5.18.0), notebooks 33 and 34
├── requirements-remote-code.txt     Molmo and DeepSeek-VL2 environment (transformers 4.46.3), notebook 30
│
└── code_active/
    ├── vlm_audit/                   the measurement package (imported by every notebook and script)
    │   ├── parameter_layouts.py       parameter names and shapes of every model, without weights
    │   ├── checkpoint_headers.py      tensor shapes from safetensors headers / .bin indexes (HTTP range reads)
    │   ├── attribution_rules.py       tensor-level V/F/L/Head rule table; final-stage trainable sets
    │   ├── token_accounting.py        visual tokens from each model's released preprocessing
    │   ├── tensor_cost_model.py       the validated cost model (intrinsic and executed work)
    │   ├── flop_counter_validation.py validation against FlopCounterMode; element-wise work
    │   ├── correlation_grid.py        exact permutation tests, BH and BY, exact two-group test
    │   ├── latency_measurement.py     GPU harnesses for all 23 models (prefill, components, decode, memory)
    │   ├── latency_analysis.py        statistics of the latency runs (notebook 31)
    │   ├── input_size_sweep.py        compute over image sizes, shapes and token budgets (notebook 32)
    │   ├── operator_dispatch.py       operator calls per decode step (the batch-1 latency floor)
    │   ├── roster_metadata.py         display names, families, release years, table order
    │   ├── revised_figures.py         every figure of the revision (notebook 35)
    │   └── ...                        earlier-phase modules (runner, metrics, components, analytic_compute, ...)
    ├── notebooks/                   numbered notebooks; each NN_*.ipynb is generated by _build_notebook_NN.py
    │   ├── 24–28                      parameter audit, attribution, token counts, cost model, statistics (CPU)
    │   ├── 29, 30, 34                 GPU latency (Colab A100 80 GB)
    │   ├── 31                         latency analysis and the per-pass floor (CPU)
    │   ├── 32                         compute versus input size (CPU)
    │   ├── 33                         Gemma 4 checks in transformers 5.18 (CPU)
    │   ├── 35                         all revised figures and draft captions (CPU)
    │   └── 00–23                      earlier phases (see below)
    ├── gpu_colab/                   upload folder for the GPU runs: notebooks 29–31 and 34, code bundle, run guide
    ├── python_scripts/              phase-1/2 batch scripts 00–20 (see below)
    ├── outputs/
    │   ├── measurements/              every table the notebooks write (nbNN_*.csv), benchmark provenance,
    │   │   └── gpu_latency/           raw GPU results (JSON per model), logs, environment manifests, executed notebooks
    │   └── figures/revised/           revised figures (PDF and 300-dpi PNG) and draft captions
    └── README.md, PHASE_2_RUN_GUIDE.md, RUNPOD_SETUP.md, requirements.txt   phase-1/2 documentation
```

---

## Model roster

23 models; † marks the three added in the revision. Parameters are checkpoint totals. The commit is the Hugging Face revision measured; no measured weights or configurations have changed since. Full identity table: `code_active/outputs/measurements/nb26_model_identity.csv`.

| Model | Family | Year | Parameters (M) | Hugging Face repository | Commit measured |
|---|---|---|---:|---|---|
| CLIP-ViT-L/14 | Dual-encoder contrastive | 2021 | 427.6 | `openai/clip-vit-large-patch14` | `32bd64288804` |
| SigLIP-L/16 | Dual-encoder contrastive | 2023 | 652.5 | `google/siglip-large-patch16-384` | `ce005573a409` |
| SigLIP2-So400m/14 | Dual-encoder contrastive | 2025 | 1,136.0 | `google/siglip2-so400m-patch14-384` | `e8e487298228` |
| ViLT-B/32 | Single-stream fused | 2021 | 117.6 | `dandelin/vilt-b32-finetuned-vqa` | `d0a1f6ab8852` |
| BLIP-large | Bridge-module precursor | 2022 | 446.3 | `Salesforce/blip-image-captioning-large` | `353689b859fc` |
| BLIP-2-OPT-2.7B | Bridge-module | 2023 | 3,744.8 | `Salesforce/blip2-opt-2.7b` | `59a1ef6c1e51` |
| InstructBLIP-Vicuna-7B | Bridge-module | 2023 | 7,913.7 | `Salesforce/instructblip-vicuna-7b` | `19103d0c5b52` |
| LLaVA-1.5-7B | Adapter-LLM | 2023 | 7,063.4 | `llava-hf/llava-1.5-7b-hf` | `b234b804b114` |
| LLaVA-NeXT-Mistral-7B | Adapter-LLM | 2024 | 7,566.7 | `llava-hf/llava-v1.6-mistral-7b-hf` | `2424fdd47412` |
| Molmo-7B-D | Adapter-LLM | 2024 | 8,021.0 | `allenai/Molmo-7B-D-0924` | `cab33fb7f1a4` |
| IDEFICS-9B | Cross-attention LLM | 2023 | 8,929.7 | `HuggingFaceM4/idefics-9b` | `4986969d084d` |
| Llama-3.2-11B-Vision | Cross-attention LLM | 2024 | 10,670.2 | `meta-llama/Llama-3.2-11B-Vision-Instruct` | `9eb2daaa8597` |
| Idefics2-8B | Perceiver adapter-LLM | 2024 | 8,402.8 | `HuggingFaceM4/idefics2-8b` | `2c42686c57fe` |
| Qwen2-VL-7B | Dynamic-tile adapter-LLM | 2024 | 8,291.4 | `Qwen/Qwen2-VL-7B-Instruct` | `eed13092ef92` |
| Qwen2.5-VL-7B | Dynamic-tile adapter-LLM | 2025 | 8,292.2 | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f` |
| Qwen3-VL-8B † | Dynamic-tile adapter-LLM | 2025 | 8,767.1 | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed8` |
| InternVL3-8B | Dynamic-tile adapter-LLM | 2025 | 7,944.4 | `OpenGVLab/InternVL3-8B` | `853e3a797a66` |
| Pixtral-12B | Adapter-LLM, own encoder | 2024 | 12,682.7 | `mistral-community/pixtral-12b` | `c2756cbbb942` |
| DeepSeek-VL2-Small | MoE adapter-LLM | 2024 | 16,148.3 | `deepseek-ai/deepseek-vl2-small` | `6033e16432a1` |
| Gemma-4-26B-A4B † | MoE adapter-LLM | 2026 | 25,805.9 | `google/gemma-4-26B-A4B-it` | `4d7ae4984b7d` |
| Chameleon-7B | Native early-fusion | 2024 | 7,043.1 | `facebook/chameleon-7b` | `55361b71ee52` |
| Gemma-4-12B † | Encoder-free patch projection | 2026 | 11,959.7 | `google/gemma-4-12B-it` | `707f0a3b8a3c` |
| Janus-Pro-7B | Native decoupled-encoder | 2025 | 7,420.4 | `deepseek-ai/Janus-Pro-7B` | `5c3eb3fb2a3b` |

Two repositories are gated: Llama-3.2-Vision and Chameleon. For CPU work the code reads their configurations from public mirrors that were verified identical (`parameter_layouts.CONFIG_MIRRORS`). Measuring their latency needs access to the weights.

---

## Environments

The analysis used three environments, because no single `transformers` release supports every model:

| Environment | File | Python / PyTorch | `transformers` | Used for |
|---|---|---|---|---|
| CPU analysis | `requirements.txt` | 3.12, torch 2.14.1 (CPU) | 4.57.6 | notebooks 24–28, 31, 32, 35 |
| Gemma 4 | `requirements-transformers5.txt` | 3.12, torch 2.14.1 (CPU) | 5.18.0 | notebook 33 (CPU); notebook 34 (GPU) |
| Remote code | `requirements-remote-code.txt` | — | 4.46.3 | notebook 30 (GPU): Molmo-7B-D and DeepSeek-VL2-Small |

The GPU notebooks (29, 30, 34) install their own pinned versions in a fresh Colab runtime. The runs reported here used Python 3.13.15, torch 2.11.0+cu130 and CUDA 13.0 on an NVIDIA A100-SXM4-80GB. Each run's full `pip freeze` is in `code_active/outputs/measurements/gpu_latency/environment/`.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

A Hugging Face token is optional (every repository except the two gated ones is public). If you have one, set it as the `HF_TOKEN` environment variable; never write it into a notebook.

---

## Reproducing the results

### Without a GPU

Everything except the latency measurements runs on a laptop CPU, without downloading model weights. Run the notebooks from `code_active/notebooks/`, in this order. Measured run times are on a laptop CPU:

| Step | Notebook | Environment | Time | Writes |
|---|---|---|---|---|
| 1 | `24_parameter_residual_audit` | CPU analysis | minutes | `nb24_*` |
| 2 | `25_a13_downstream_recomputation` | CPU analysis | minutes | `nb25_*` |
| 3 | `26_attribution_identity_recipes` | CPU analysis | ≈ 2 min | `nb26_*` (parameters, identity, trainable shares) |
| 4 | `33_transformers5_validation` | **Gemma 4** | ≈ 6 min | `nb33_*` (read by 27 and 31) |
| 5 | `27_compute_protocol_revision` | CPU analysis | ≈ 25 min | `nb27_*` (tokens, compute, validation) |
| 6 | `28_benchmarks_and_statistics` | CPU analysis | ≈ 3 min | `nb28_*` (benchmarks, test grid, efficiency) |
| 7 | `31_latency_analysis` | CPU analysis | ≈ 9 min | `nb31_*` (reads the GPU results already in `outputs/`) |
| 8 | `32_compute_versus_input_size` | CPU analysis | ≈ 4 min | `nb32_*` (reads `nb31_*`) |
| 9 | `35_revised_figures` | CPU analysis | ≈ 2 min | `outputs/figures/revised/` |

Every notebook is generated from its builder, so change the builder rather than the notebook:

```bash
cd code_active/notebooks
python _build_notebook_27.py                                             # regenerate the notebook
jupyter nbconvert --to notebook --execute --inplace 27_compute_protocol_revision.ipynb
```

Notebook 31 counts operator calls (step 7) only when it runs under the `transformers` release of the GPU measurement (4.57.6). Otherwise it reuses `nb31_operator_calls.csv`.

### With a GPU (latency)

The latency results are already in `code_active/outputs/measurements/gpu_latency/`. To measure them again, follow `code_active/gpu_colab/README_GPU_RUN.md`:
- notebook 29 (19 models), then notebook 30 in a fresh runtime (Molmo, DeepSeek-VL2), then notebook 34 in a fresh runtime (Gemma 4);
- all three on the same GPU type: an A100 80 GB, which Gemma 4 26B-A4B needs;
- in total about one hour of A100 time, mostly checkpoint downloads.

The upload bundle is rebuilt with `python _build_gpu_bundle.py` (from `code_active/notebooks/`). Before changing a harness, dry-run it on the CPU with `build_harness(name, mode="reduced", device="cpu")`.

---

## Outputs: where each table and figure comes from

| Manuscript item | Source files (`code_active/outputs/measurements/`) |
|---|---|
| Table 1, roster | `nb26_parameter_components_v2.csv`, `nb26_model_identity.csv` |
| Table 2, parameters by component | `nb26_parameter_components_v2.csv` |
| Table 3, forward compute | `nb27_table3_v2_1024.csv`, `nb27_table3_v2_448.csv` |
| Table 4, visual tokens | `nb27_table4_tokens_v2.csv` |
| Table 5, trainable share | `nb26_trainable_share_v2.csv` |
| Table 6, benchmark scores | `nb28_table6_v2.csv`, `benchmarks_v2.csv` |
| Table 7, vision parameter vs compute share | `nb27_table7_v2.csv`, `nb27_vision_share_correlation.csv` |
| Table 8, correlation grid (supplement) | `nb28_table8_v2.csv`, `nb28_lineage_sensitivity.csv` |
| Tables 9–10, compute–performance efficiency | `nb28_efficiency_*_{1024,448}.csv` |
| Model identity, latency, input-size growth, token budget, latency floor | `nb26_model_identity.csv`, `nb31_*`, `nb32_*` |
| Cost-model validation, benchmark provenance | `nb27_flop_counter_validation.csv`, `nb31_flop_validation_full_depth.csv`, `benchmarks_v2.csv` |

Figures are in `code_active/outputs/figures/revised/`, with draft captions in `figure_captions.md`. Figures 1–9 correspond to the submitted figures (Fig. 9 is split into 9a/9b). There are three new figures:
- `figN1`: compute versus input size;
- `figN2`: latency versus FLOPs;
- `figN3`: the per-pass latency floor.

Figure 8 is supplementary.

---

## Main results in brief

Numbers are from the outputs in this repository (23 models, common 1,024 × 1,024 input unless stated).

- **Attribution:** the vision share of parameters runs from 70.9% (CLIP-ViT-L/14) to 0.29% (Gemma 4 12B, which has no vision encoder). Every parameter of every model is attributed exactly once.
- **Compute:** forward compute runs from 31.9 GFLOPs (ViLT-B/32) to 108.2 TFLOPs (Pixtral-12B). The vision encoder's share of compute tracks its share of parameters across the roster (Spearman ρ = +0.72, n = 22). It departs from it by factors of up to 28.7 in cross-attention, token-compressing and mixture-of-experts designs.
- **The input is a first-order variable:**
  - 43 of 253 model pairs change compute order between 448² and 1,024²;
  - Qwen3-VL's cost grows 244-fold from 448² to its 4,096² limit;
  - Gemma 4's token-budget setting alone changes cost 12–24-fold.
- **Latency:** prefill latency on an A100 follows the executed FLOPs (ρ = +0.87 at 1,024², +0.81 at 448²). At batch size one there is a per-pass floor: every dense language-model prefill of at most 334 tokens takes about one decode step (18 of 18). Decode time per token follows the number of operator calls per step (Pearson r = 0.995, n = 17) more closely than the resident weights (r = 0.80).
- **Training regimes:** trainable shares form four groups (bridge-only, frozen-language-model cross-attention, frozen vision encoder, full training), not two.
- **No structure–performance association** survives multiple-testing correction in the 49-test grid (smallest Benjamini–Hochberg-adjusted p = 0.18).

---

## Conventions

- **Components:** V = vision encoder (including patch and position embeddings), F = fusion (connectors, resamplers, projections into the language model), L = language model (including inserted cross-attention layers), Head = untied output projections. Chameleon (discrete image tokens) reports its image tokenizer as V. Janus-Pro's generation modules are aggregated into V/F/Head.
- **Common input:** compute and tokens use one 1,024 × 1,024 image through each released preprocessor, with 448 × 448 as a sensitivity check and the released maximum for bandwidth. The prompt has 32 text tokens. The LM head is charged at the one position that yields the first token.
- **Intrinsic versus executed work:** tables report intrinsic work. Padding that reference implementations execute is tracked separately (for example Llama-3.2-Vision's four tile slots, Idefics2's padded sub-images, Gemma 4's padded encoder slots). DeepSeek-VL2's latent attention is costed in its decompressed form, the minimum arithmetic.
- **Mixture of experts:** routed experts are charged for the active experts per token (DeepSeek-VL2 6 of 64, Gemma 4 26B-A4B 8 of 128).
- **Statistics:** Spearman correlations with exact or Monte Carlo permutation p-values, Benjamini–Hochberg and Benjamini–Yekutieli adjustment, and sensitivity over every subset that keeps one member of each lineage group.

---

## Earlier pipeline (phases 1-3)

The submitted version of the paper was produced with an earlier pipeline, kept here for the record:
- the batch scripts `code_active/python_scripts/00–20`, documented in `code_active/README.md`, `PHASE_2_RUN_GUIDE.md` and `RUNPOD_SETUP.md`;
- the notebooks `00–23`.

The revision notebooks (24–35) re-derive every published number and supersede the earlier values where they differ. The per-model JSON records and `master*.csv` in `outputs/measurements/` belong to that earlier pipeline.

---

## Notes

- **Third-party models:** all models are third-party releases under their own licences. This repository contains no model weights.
- **Run records:** logs, executed notebooks and environment manifests of every GPU session are in `outputs/measurements/gpu_latency/`. Results that were superseded, with the reason, are in its `superseded/` folder.
- **Citation:** the manuscript is under review; a citation will be added on publication.
- **Licence:** no licence has been chosen yet. Until one is added, all rights are reserved.
