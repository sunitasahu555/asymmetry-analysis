# GPU runs for the revision: latency validation (reviewer item A5 iii, decision D10)

**Author:** Sunita Sahu

These are the only parts of the revision that need a GPU. Everything else (notebooks 24–28) runs on a laptop CPU without model weights.

## What is in this folder

| File | Runs on | What it does | Time on an A100 (measured) |
|---|---|---|---|
| `ThreadB_GPU_bundle.zip` | — | The code (`vlm_audit` package) that the notebooks import, with a manifest of file hashes | — |
| `29_gpu_latency_native.ipynb` | **GPU** | 19 transformers-native models (transformers 4.57.6) | about 35 min for all; 1–3 min per model; about 5 min for one model |
| `30_gpu_latency_remote_code.ipynb` | **GPU**, fresh runtime | Molmo-7B-D and DeepSeek-VL2-Small (transformers 4.46.3) | about 10 min |
| `34_gpu_latency_transformers5.ipynb` | **GPU 80 GB**, fresh runtime | Gemma 4 12B and Gemma 4 26B-A4B (transformers 5.18.0; added under decision D9) | about 15 min |
| `31_latency_analysis.ipynb` | CPU (Colab or laptop) | Merges the results of 29, 30 and 34 and computes the statistics | 1 min |

The times are those of the runs of 2026-10-03 and 2026-10-04 (Colab A100-SXM4-80GB), setup included; each run-status table lists the minutes per model. Checkpoints are fetched one at a time and deleted after each model.

### What each GPU notebook measures

The notebooks measure every released checkpoint in bfloat16 at batch size 1. Each one sees a 448 × 448 and a 1024 × 1024 test image passed through its released processor.

- **Prefill latency:** the whole model, median of 10 timed passes after 3 warm-up passes.
- **Prefill latency per component:** vision encoder, fusion, language model and LM head, timed with CUDA events.
- **Decode latency per generated token.**
- **Memory:** resident weights and the peak of one prefill pass.
- **Full-depth FLOPs:** counted op by op on the released weights, next to the cost model's prediction for the same input. Notebook 27 checked the cost model only on reduced-depth copies; this repeats the check on the real checkpoints. Molmo and DeepSeek-VL2 had no op-level check before.
- **Element-wise share** of the FLOPs (reviewer item A5 i) at full depth.

## Which GPU

| GPU (Colab runtime) | Use? | Why |
|---|---|---|
| **A100 (40 GB)** | **Recommended** | Every model fits in bfloat16 (largest: DeepSeek-VL2 at 32 GB, Pixtral at 25 GB). It is also the most common reference GPU for reported latencies. |
| H100 (80 GB) | Also good | Faster, with more headroom. |
| RTX PRO 6000 / G4 (96 GB) | Works | Fine if it is the only large GPU available. |
| L4 (24 GB) | No | Pixtral-12B does not fit. |
| T4, V100 | No | No bfloat16 support. |

**Run notebooks 29, 30 and 34 on the same GPU type.** Latencies can only be compared within one GPU. Notebook 31 flags results from mixed GPUs.

Choose High-RAM if Colab offers it. You need at least 80 GB of free disk; the notebooks print it.

## Roster extension run (decision D9, three models)

The 20 original models were measured on 2026-10-03 (A100-SXM4-80GB). The three models added under D9 need one more Colab session **on the same GPU type, A100 80 GB** (Gemma 4 26B-A4B holds 52 GB of weights, so a 40 GB A100 is not enough):

1. Replace `ThreadB_GPU_bundle.zip` and the notebooks in `MyDrive/ThreadB_GPU` with the ones in this folder.
2. **Notebook 29**, A100 80 GB: set `MODELS_TO_RUN = ['Qwen3-VL-8B']`, then Run all (about 5 minutes).
3. **Notebook 34** in a fresh runtime (Runtime > Disconnect and delete runtime), A100 80 GB: Run all (about 15 minutes). The GPU cell must read NVIDIA A100-SXM4-80GB.
4. Download the output archives (or the `outputs` folder) and run notebook 31 on the laptop, as in step 4 below.

This run was done on 2026-10-04 (bundle `736d0f20…`); its results are in `outputs/measurements/gpu_latency/`.

### Qwen3-VL-8B re-run (bundle `2a552a41…`)

The full-depth FLOP check of that run listed two of Qwen3-VL's three DeepStack mergers under fusion; the third was counted as vision (decision log N26). Totals, latencies and the cost model are unaffected. The fixed bundle names the merger list as a whole. To replace the result:

1. Replace `ThreadB_GPU_bundle.zip` and `29_gpu_latency_native.ipynb` in `MyDrive/ThreadB_GPU` with the ones in this folder.
2. **Notebook 29**, A100 80 GB: set `MODELS_TO_RUN = ['Qwen3-VL-8B']` and `FORCE_RERUN = True`, then Run all (about 5 minutes).
3. Download the output archive.

Done on 2026-10-04 (session 20261004_063048): every component within 3×10⁻⁶ %. The first result is kept in `outputs/measurements/gpu_latency/superseded/`. **All GPU runs of the revision are complete (23 models).**

## Steps (first run, 20 models)

1. **Upload to Drive.** In Google Drive, create the folder `MyDrive/ThreadB_GPU`. Upload into it:
   - `ThreadB_GPU_bundle.zip`;
   - the three notebooks.
2. **Run notebook 29.** Open `29_gpu_latency_native.ipynb` from Drive with Google Colab.
   1. Runtime > Change runtime type > **A100**.
   2. Runtime > **Run all**.
   3. Allow Drive access when asked.
3. **Run notebook 30 in a fresh runtime.** Open `30_gpu_latency_remote_code.ipynb`.
   1. Runtime > **Disconnect and delete runtime**. This is needed because notebook 30 uses a different transformers release.
   2. Choose the same GPU type as for notebook 29.
   3. **Run all.**
4. **Run notebook 31** to analyse. Either:
   - in Colab (any runtime, CPU is enough): open it and Run all;
   - on the laptop: unzip the output archives and copy `outputs/results/*.json` into `code_active/outputs/measurements/gpu_latency/results/`. Keep `logs/`, `tables/` and `environment/` next to it in `gpu_latency/`. Archive the executed notebooks from `outputs/logs/` (account block removed), not the notebook files Colab keeps at the top of the Drive folder: those record the Google account name in every cell's metadata. Then run the notebook from `code_active/notebooks/`.

A Hugging Face token is **optional**: every repository used is ungated. If you have one, add it in Colab under the key icon > `HF_TOKEN` and allow notebook access. It only raises download rate limits.

## Where everything is saved

Everything is saved under `MyDrive/ThreadB_GPU/outputs/`, written as the run goes:

| Folder | Contents |
|---|---|
| `results/` | One JSON per model: every raw timing, FLOP count, token count, memory figure, and each error with its traceback. It is saved after each input size, so partial runs survive. |
| `logs/` | - One timestamped log per model.<br>- The **session transcript** (every printed line, including library warnings).<br>- **Crash traces** (Python stacks from hard crashes).<br>- The **pip install log**.<br>- The **executed notebook** with its outputs. |
| `tables/` | Summary CSVs rebuilt from the JSON results, and a run-status table per session. |
| `environment/` | GPU, driver, CUDA, library versions and the full `pip freeze` of each session. |

The last cell of notebooks 29 and 30 also writes `MyDrive/ThreadB_GPU/ThreadB_GPU_outputs_<label>_<time>.zip`, which contains all of the above. Download that file, or keep the folder on Drive for notebook 31.

## If something goes wrong

| What you see | What to do |
|---|---|
| The session disconnects mid-run | Run all again. Finished models are skipped. A model interrupted half-way restarts from its first unfinished input size. |
| "transformers X is already loaded in this runtime" | Runtime > Restart session (notebook 29), or Disconnect and delete runtime (notebook 30), then Run all. |
| "No GPU" or "no bfloat16 support" | Runtime > Change runtime type > A100. |
| One model fails (for example out of memory) | The run continues. The error and its traceback are in that model's JSON result and log, and in the run-status table. To retry only that model, set `MODELS_TO_RUN = ['<name>']`. |
| You want to redo a model that already finished | Set `FORCE_RERUN = True` and `MODELS_TO_RUN = ['<name>']`. |
| Disk full | Keep `DELETE_WEIGHTS_AFTER_EACH_MODEL = True` (the default). |

## Provenance

These checks were run before the notebooks were handed over. Every code path was dry-run on a CPU with reduced-depth copies of all 20 architectures and random weights. Notebook 30's stack was dry-run under transformers 4.46.3 with the DeepSeek-VL2 package at commit `ef9f91e`.

After the fixes below, all 72 component comparisons (20 models) agree with the op-level FLOP count within 0.008%. Notebooks 29 and 30 were also executed end to end from their own cells on the CPU, and notebook 31 on the dry-run results.

The dry run found three problems. All three are now handled; see `paper/Revision_Decision_Log.md`, entries N17–N19.

1. **DeepSeek-VL2's attention form.** Its released code computes latent attention in the absorbed form, which is more arithmetic than the cost model assumed.
2. **Molmo's positional table.** Its ViT positional table was charged as a matmul.
3. **Qwen2-VL's and Chameleon's LM heads.** Their forwards compute LM-head logits at every position, not just the last.
