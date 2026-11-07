# FinNum

Adapting [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) to numeric reasoning over financial filings: serving benchmarks, LoRA SFT, GRPO and continued pretraining on FinQA, with every stage measured against its parent.

<!-- TODO: one result line once v0.1 is done, e.g. "LoRA SFT: X% → Y% FinQA test accuracy (95% CI ...)" -->

**Status:** 🚧 v0.1 in progress (serving benchmark, baselines, LoRA SFT). Full version planned for Nov 23, 2026.

---

## Contents

- [FinNum](#finnum)
  - [Contents](#contents)
  - [The task](#the-task)
  - [Pipeline](#pipeline)
  - [Results](#results)
    - [Accuracy ladder](#accuracy-ladder)
    - [Serving benchmark](#serving-benchmark)
    - [SFT details](#sft-details)
  - [Setup](#setup)
  - [Data](#data)
  - [Usage](#usage)
  - [Evaluation protocol](#evaluation-protocol)
  - [Repository layout](#repository-layout)
  - [What worked, what didn't](#what-worked-what-didnt)
  - [Limitations](#limitations)
  - [Licence and citation](#licence-and-citation)

---

## The task

Given one page of a company's annual report (text before a table, the table, text after it) and a question, the model must find the right numbers, choose the calculation and give one final number.

<!-- TODO: replace with a short real example from FinQA (question, the numbers used, gold program, answer) -->

```
Question: <question>
Program:  <gold program, e.g. subtract(a, b), divide(#0, b)>
Answer:   <number>
```

Answers are scored by parsing the final `Answer: <number>` line and checking it is within **0.5% relative tolerance** of the gold answer.

## Pipeline

Each arrow is one evaluation against the parent model.

```
Base ──► SFT (LoRA) ──────────────► GRPO ──► Final (merge, AWQ + GGUF)
  └──► CPT ──► SFT from CPT ──┘   (feeds GRPO only if it beats SFT from base)
```

| Stage             | What it does                                                          | Version |
| ----------------- | --------------------------------------------------------------------- | ------- |
| Serving benchmark | Base model in 5 serving/quantization configs: speed, memory, accuracy | v0.1    |
| Baselines         | Zero-shot and 3-shot FinQA accuracy                                   | v0.1    |
| SFT (LoRA)        | Fine-tune on gold programs rewritten as natural-language steps        | v0.1    |
| GRPO              | Reinforcement learning with a verifiable correctness reward           | Full    |
| CPT               | Continued pretraining on 10-K filings, with general-text replay       | Full    |
| Final model       | Merge best adapter, quantize to AWQ and GGUF Q4_K_M                   | Full    |

## Results

All accuracies are on the FinQA **test** set (evaluated once per finished model) with 95% bootstrap confidence intervals (1,000 resamples). Every decision was made on **dev**.

**Base model:** [`Qwen/Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base) @ revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` (Apache-2.0)
**Hardware:** <!-- TODO: e.g. 1× H100 80GB (Google Colab) --> `<GPU>`

### Accuracy ladder

| Model           | Dev acc. | Test acc. (95% CI) | Δ vs parent | TAT-QA arith. |
| --------------- | -------- | ------------------ | ----------- | ------------- |
| Base, zero-shot | –        | –                  | –           | –             |
| Base, 3-shot    | –        | –                  | –           | –             |
| SFT (LoRA)      | –        | –                  | –           | –             |
| GRPO            | –        | –                  | –           | –             |
| CPT → SFT       | –        | –                  | –           | –             |
| Final (4-bit)   | –        | –                  | –           | –             |

<!-- A stage whose result falls within its parent's CI is reported as "no improvement". -->

### Serving benchmark

200 fixed FinQA prompts, greedy decoding, max 256 output tokens.

| Config                 | Peak memory | Prompt tok/s | Gen tok/s | p50 / p95 latency (c=1) | p50 / p95 (c=8) | p50 / p95 (c=32) | Accuracy |
| ---------------------- | ----------- | ------------ | --------- | ----------------------- | --------------- | ---------------- | -------- |
| vLLM bf16              | –           | –            | –         | –                       | –               | –                | –        |
| vLLM AWQ 4-bit         | –           | –            | –         | –                       | –               | –                | –        |
| llama.cpp Q8_0 (GPU)   | –           | –            | –         | –                       | –               | –                | –        |
| llama.cpp Q4_K_M (GPU) | –           | –            | –         | –                       | –               | –                | –        |
| llama.cpp Q4_K_M (CPU) | –           | –            | –         | –                       | –               | –                | –        |

<!-- TODO: context-length sweep (2k/8k/16k/32k): measured KV-cache memory vs the formula; note the GQA saving -->

### SFT details

| Training data | Dev acc. | Train-dev gap |
| ------------- | -------- | ------------- |
| 25%           | –        | –             |
| 50%           | –        | –             |
| 100%          | –        | –             |

**Memorization test** (table numbers perturbed, gold programs re-executed): original dev `–` → perturbed dev `–`.

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<user>/FinNum.git
cd FinNum
uv sync
```

<!-- TODO: GPU requirements, CUDA version, how to install vLLM / llama.cpp, any env vars (.env: HF_TOKEN, WANDB_API_KEY) -->

## Data

| Source                                    | Use                        | Notes                                                             |
| ----------------------------------------- | -------------------------- | ----------------------------------------------------------------- |
| [FinQA](https://github.com/czyssrs/FinQA) | SFT, GRPO, main evaluation | Official splits only: 6,251 train / 883 dev / 1,147 test          |
| TAT-QA (arithmetic subset)                | Transfer test only         | Never used for training or tuning                                 |
| EDGAR-CORPUS 10-K MD&A                    | CPT (20–50M tokens)        | FinQA company-years removed; 20 companies held out for perplexity |
| General web text (e.g. FineWeb)           | CPT replay (25%)           | Forgetting control                                                |

FinQA is loaded directly from the JSON files in the original GitHub repo. The Hugging Face copy (`ibm-research/finqa`) only contains a loading script, which `datasets` ≥ 4 no longer runs.

Fields used from each example: `id`, `filename`, `pre_text`, `post_text`, `table`, and `qa.question`, `qa.program`, `qa.exe_ans`.

The base model, Qwen3-1.7B-Base, is released under Apache-2.0. Licences for each dataset are listed in [DATA.md](DATA.md). <!-- TODO: create DATA.md -->

## Usage

<!-- TODO: fill in real commands as each stage is implemented -->

```bash
# Run tests
uv run pytest

# Baselines (zero-shot and 3-shot on dev)
uv run python -m <module> <args>

# LoRA SFT
uv run python -m finnum.train_sft <hydra overrides>

# Evaluate a checkpoint
uv run python -m <module> <args>

# Serving benchmark
uv run python <serving script> <args>
```

## Evaluation protocol

- **Splits:** all decisions (epochs, checkpoints, learning rates, prompts) are made on dev. The test set is used once per finished model.
- **Scorer:** one unit-tested function used by every stage. It parses the last `Answer:` line, normalizes `%`, `$`, commas and scale words, and accepts answers within 0.5% relative tolerance.
- **Confidence intervals:** 95% bootstrap CIs (1,000 resamples) on every reported accuracy.
- **Seeds:** final SFT and GRPO configs run with 3 seeds; mean and spread reported.
- **Contamination:** FinQA source company-years removed from the CPT corpus, then a 13-gram overlap check of every training corpus against FinQA test documents.
- **No teacher labels:** SFT and GRPO targets come only from FinQA gold programs.
- **Error analysis:** 50 errors per stage, tagged as wrong cell, arithmetic slip, wrong operation, unit/scale error or format failure.

## Repository layout

```
FinNum/
├── configs/      Hydra configs, one per stage
├── data/         Processed data (tracked with DVC)
├── src/finnum/   data, prompts, programs, scoring, train_sft, train_grpo, train_cpt
├── serving/      Serving and quantization benchmark
├── evals/        Baseline and per-stage evaluation
├── reports/      One report per stage (config, data hash, seed, commit)
└── tests/        Scorer and data tests
```

Every run logs its config, dataset hash, seed and git commit. <!-- TODO: link to W&B / MLflow project -->

## What worked, what didn't

<!-- TODO: fill in after each stage. Include negative results and their likely cause. -->

- **Worked:** –
- **Didn't work:** –
- **Surprises:** –

## Limitations

<!-- TODO: e.g. label noise in FinQA (annotated answers that disagree with the gold program), single base model, single GPU type, prompts truncated at 2048 tokens -->

## Licence and citation

<!-- TODO: choose a licence (e.g. MIT) and add a LICENSE file -->

This project uses FinQA:

```bibtex
@inproceedings{chen2021finqa,
  title     = {FinQA: A Dataset of Numerical Reasoning over Financial Data},
  author    = {Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang},
  booktitle = {Proceedings of EMNLP},
  year      = {2021}
}
```
