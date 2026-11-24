# Name That LLM: Project Documentation

**Authorship attribution of machine-generated text: identifying which of 18 LLMs wrote a passage**

| | |
|---|---|
| **Author** | Somansh Goel (SAP ID 500123667) |
| **Course** | NLP Lab, Lab 9 / Project 2 |
| **Competition** | [Kaggle: Name That LLM](https://www.kaggle.com/competitions/name-that-llm) |
| **Compute** | Local CPU, UPES HPC (1× H100 per job), Kaggle 2× T4 |
| **Best offline result** | 0.653 balanced accuracy (8B + 14B LoRA readers + TF-IDF, unseen prompts) |
| **Best leaderboard result** | 0.686 public (public 0.704 stack + our 8B reader) |

> `README.md` is the short summary. `NOTES.md` is the dated lab log with every run.
> This document explains the whole project end to end: the problem, the reasoning behind each design choice, how to reproduce it, what we found, and what carries over to other tasks.

---

## Table of contents
1. [Overview](#1-overview)
2. [Problem definition](#2-problem-definition)
3. [Data](#3-data)
4. [Repository layout](#4-repository-layout)
5. [Evaluation methodology](#5-evaluation-methodology)
6. [System architecture](#6-system-architecture)
7. [Component 1: Multi-view TF-IDF + stylometry](#7-component-1-multi-view-tf-idf--stylometry)
8. [Component 2: External data (Open Pool)](#8-component-2-external-data-open-pool)
9. [Component 3: Fine-tuned LLM readers (LoRA)](#9-component-3-fine-tuned-llm-readers-lora)
10. [Component 4: Kaggle deployment and stack integration](#10-component-4-kaggle-deployment-and-stack-integration)
11. [Results](#11-results)
12. [Error analysis](#12-error-analysis)
13. [How to reproduce](#13-how-to-reproduce)
14. [Lessons learned (general takeaways)](#14-lessons-learned-general-takeaways)
15. [Limitations and future work](#15-limitations-and-future-work)
16. [Glossary](#16-glossary)
17. [Credits](#17-credits)

---

## 1. Overview

### What we did, in one paragraph
We built a classifier that reads a short passage of text and predicts which of 18 large language models wrote it. We worked in three iterations. First, a CPU baseline based on character and word n-grams plus hand-made style features. Second, the same baseline trained on a public corpus of 231k extra labelled passages. Third, fine-tuned open LLMs (Qwen3-8B and Qwen3-14B) with LoRA adapters and a head that reads several intermediate layers. We trained them on a university GPU cluster and ported them to Kaggle's T4 GPUs. Combining these models took balanced accuracy on unseen prompts from 0.389 to 0.653 (chance is 0.056). We then added our reader to the strongest public Kaggle ensemble. That lowered its leaderboard score from 0.704 to 0.686, and we explain why.

### Progression at a glance
```
 Iteration 1          Iteration 2a            Iteration 2b / 3                 Ensemble
 TF-IDF + stylometry  + 231k pool passages    Qwen3 LoRA "l24" readers         8B + 14B + TF-IDF
 (train only, CPU)    (CPU)                   (H100 training, T4 inference)
     0.389      ──►       0.514 (5-fold) ──►    0.623 (8B, fold 0)        ──►   0.653 (fold 0)
                          0.523 (fold 0)         0.617 (14B, fold 0)
```

### General problem this project is an instance of
This is **authorship attribution under topic shift**. The labels are authors, which here are models. The test topics never appear in training. Any method that learns *what* is said instead of *how* it is said will look good on a naive split and fail on the real test. Most of the decisions below follow from that one fact.

---

## 2. Problem definition

| Aspect | Detail |
|---|---|
| Input | One English passage, 40 to 250 words |
| Output | One of 18 model labels |
| Metric | **Balanced accuracy**: the mean of per-class recall. It rewards getting every class right, not just the frequent ones. Chance = 1/18 ≈ 0.056 |
| Constraint | Kaggle code competition: the notebook must run within 9 h on 2× T4 with **internet off**. Public datasets and models may be attached |

**The 18 classes, by lab:**

| Lab | Models |
|---|---|
| OpenAI (5) | gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra |
| Google (4) | gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash |
| Anthropic (5) | claude-fable-5, claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5 |
| Qwen (1) | qwen3.8-max |
| ZAI (2) | glm-5.3, glm-5.3-flash |
| Moonshot (1) | kimi-k3 |

Several labels are *siblings*: versions or sizes of the same model family. They share training data and alignment style, so they are much harder to tell apart than models from different labs.

---

## 3. Data

### 3.1 Competition data (`data/`)
| File | Rows | Columns | Notes |
|---|---|---|---|
| `train.csv` | 2,699 | `id, prompt_id, text, model` | 150 prompts; each model answers each prompt about once, so there are ~150 rows per class |
| `test.csv` | ~4,000 hidden (local copy is a 3-row sample) | `id, text` | **All prompts are unseen in train**; the class mix differs between the public and private leaderboards |
| `sample_submission.parquet` | — | `id, model` | Output format |

Each prompt is answered by all 18 models, so `prompt_id` is a **group** variable: rows that share it share a topic.

### 3.2 External data: Open Pool v16 (`data/pool/ntl_open_pool.parquet`)
- A public Kaggle dataset (`crtproduction/name-that-llm-open-pool`) with **231,036 labelled passages** from the same 18 models.
- Competition rules (§2.6) allow publicly available external data.
- **Leakage audit before use:**
  - Prompt overlap: the highest similarity from any pool prompt to any train prompt is 0.40, so no pool prompt is a near-duplicate of a train prompt.
  - Text overlap: no passage appears in both the pool and train.
- The pool is ~85× the size of train. It became the single biggest source of improvement (see §11).

---

## 4. Repository layout

```
project 2/
├── README.md                  Short summary of the project and results
├── DOCUMENTATION.md           This file
├── NOTES.md                   Dated lab log: every run, number and decision
├── ntl_baseline.py            Iteration 1 + 2a: multi-view TF-IDF + stylometry (CPU)
├── confusion_matrix.csv       18×18 OOF confusion matrix from `ntl_baseline.py --cv`
├── submission.parquet         Latest local submission file
├── data/                      Competition files + Open Pool (git-ignored, ~114 MB)
│   ├── train.csv, test.csv, sample_submission.parquet
│   └── pool/ntl_open_pool.parquet
├── hpc/                       Iteration 2b: LoRA reader training on the UPES HPC
│   ├── train_lora.py          Training script (two-stage, l24 head)
│   ├── train_lora.pbs         8B, 70k pool sample: fold 0 then full fit
│   ├── train_lora_fullpool.pbs        8B, full pool, fold 0
│   ├── train_lora_fullpool_full.pbs   8B, full pool, full fit (submission)
│   ├── train_lora_14b.pbs             14B (27 layers), full pool, fold 0
│   └── train_lora_14b_full.pbs        14B, full pool, full fit
├── kaggle_kernel/             Kaggle script kernel for the TF-IDF + pool model
│   ├── ntl_baseline.py        (identical copy of ../ntl_baseline.py)
│   └── kernel-metadata.json
├── kaggle_lora/               Kaggle GPU kernel: T4 inference + H100 parity check
│   ├── infer_lora.py
│   └── kernel-metadata.json
└── kaggle_stack/              Fork of the public 0.704 stack with our reader added
    ├── add_ours.py            Patches stack.ipynb: inserts our reader + merge cells
    ├── stack.ipynb
    └── kernel-metadata.json
```

---

## 5. Evaluation methodology

### 5.1 Why a random split is wrong here
With a random split, the 17 other models' answers to the same prompt land in training. A model can then score well by recognising the *topic* and using the other answers to narrow down the author. The test set has only new prompts, so that score is misleading.

### 5.2 What we used
- **`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`**, grouped by `prompt_id` and stratified by label.
- Every validation prompt is unseen in training, as on the test set.
- The same split is used in `ntl_baseline.py` and `hpc/train_lora.py`, so the numbers are comparable.
- **5-fold OOF** (out-of-fold) scores are reported for the cheap CPU models.
- **Fold 0 only** is used for the GPU readers (539 passages, 30 prompts), because each run takes hours. TF-IDF is also scored on those same fold-0 rows so the comparison is like-for-like.
- **External pool data is only ever on the training side.** Folds are cut from `train.csv` alone.

### 5.3 Reliability caveats we tracked
- Blend weights chosen on the same holdout they are scored on are **optimistic**. We marked them as such.
- With extra pool data, grouped CV overstates the leaderboard gain by roughly 0.035 (community observation).
- Leaderboard noise: resubmitting identical code moves the public score by up to ~0.008. Differences smaller than ~0.01 are noise.
- **sklearn version drift:** Kaggle's sklearn produces a *different* fold 0 for the same seed (540 rows vs 539). Re-splitting on Kaggle leaked training rows and showed a fake 0.767. The fix is to save the exact holdout indices (`holdout_idx.npy`) at training time and reuse them everywhere.

---

## 6. System architecture

```
                       ┌──────────────── train.csv (2.7k) ───────────────┐
                       │                + Open Pool (231k)                │
                       ▼                                                  ▼
       ┌──────────────────────────────┐              ┌──────────────────────────────────┐
       │ Component 1/2 (CPU)          │              │ Component 3 (GPU, H100)          │
       │ 4 TF-IDF views ─► SGD/LR     │              │ Qwen3-8B (24 layers) + LoRA      │
       │ 35 stylo feats ─► LR         │              │ Qwen3-14B (27 layers) + LoRA     │
       │ log-linear blend             │              │ multi-layer mean-pool head       │
       └──────────────┬───────────────┘              │ stage 1: pool  → stage 2: train  │
                      │ log-probs (n×18)             └────────────────┬─────────────────┘
                      │                                               │ adapters + head.pt
                      │                                               ▼
                      │                              ┌──────────────────────────────────┐
                      │                              │ Component 4 (Kaggle, 2×T4)        │
                      │                              │ hand-merge LoRA into fp16 base   │
                      │                              │ bf16 fallback on overflow        │
                      │                              └────────────────┬─────────────────┘
                      ▼                                               ▼
               ┌─────────────────────────────────────────────────────────────────┐
               │ Ensemble: sum of log-probabilities (8B + 14B + w·TF-IDF)        │
               │ → argmax → submission.parquet (id, model)                       │
               └─────────────────────────────────────────────────────────────────┘
```

All components output an `(n, 18)` matrix of **log-probabilities** over the same sorted label list. Ensembling is a weighted sum of these matrices (a product of experts), followed by `argmax`.

---

## 7. Component 1: Multi-view TF-IDF + stylometry

**File:** `ntl_baseline.py` · **Runtime:** ~1 min per fit (train only), ~12 min (with pool)

### 7.1 Four TF-IDF views
Each view transforms the text, vectorizes it with `TfidfVectorizer(sublinear_tf=True, lowercase=False)` and is scaled by a weight. All views are stacked horizontally into one sparse matrix.

| View | Transform | Analyzer / n-grams | Max features | Weight | What it captures |
|---|---|---|---|---|---|
| Character | none | char 1–5 | 300k | 1.0 | Spelling, morphology, punctuation habits |
| Word | none, punctuation kept as tokens | word 1–3 | 200k | 0.7 | Phrase choices, favourite connectives |
| **Shape** | every letter run → `X`/`x` | char 2–6 | 60k | 0.5 | Pure layout: punctuation, spacing, capitalisation, sentence rhythm |
| **Function-word** | content words → `C` | word 1–3 | 50k | 0.5 | Syntax skeleton with topic removed |

The shape and function-word views exist **to remove topic**. For example, `"The model, however, failed."` becomes `"X x, x, x."` in the shape view and `"the C , however , C ."` in the function-word view. Neither keeps anything about the subject matter.

`lowercase=False` is deliberate: capitalisation is a style signal.

### 7.2 Stylometric features (35)
Computed by `stylo()`: log word and character counts; sentence count; mean, std and max sentence length; mean and std word length; type-token ratio; capitalised-word rate; digit rate; function-word rate; non-ASCII rate; paragraph breaks; whether the text ends with terminal punctuation; and per-100-word rates of 20 characters (em dash, en dash, semicolon, colon, curly quotes, ellipsis, `*`, `#`, backtick, newline, …). They are standardised and fed to a separate logistic regression (`C=0.1`).

### 7.3 Classifiers and blending
- **Train only:** multinomial `LogisticRegression(C=30)` on the TF-IDF matrix.
- **With pool (230k rows):** `SGDClassifier(loss='log_loss', alpha=2e-6)` instead of lbfgs, for speed. Pool rows get `sample_weight=0.6`.
- **Blend:** `log p = log p_tfidf + w · log p_stylo`, with `w = 0.3` (train only) or `0.2` (with pool).

---

## 8. Component 2: External data (Open Pool)

Integrated in `ntl_baseline.py` (`find_pool()`); used by default and disabled with `--no-pool`.

- The TF-IDF vocabularies are fit on **train + pool** (`min_df=3`), so the n-gram space covers far more styles.
- Pool weight sweep (grouped OOF): 0.3 → 0.496, **0.6 → 0.503**, 1.0 → 0.502. The result is flat, so the weight matters little.
- The dataset's own `match_weight` column added nothing (0.502 / 0.504).
- Stylometry stays trained on train only.

**Result:** 0.389 → 0.514 (5-fold OOF). That is the largest jump of any single change in the CPU model, larger than any feature engineering we did.

---

## 9. Component 3: Fine-tuned LLM readers (LoRA)

**File:** `hpc/train_lora.py` · **Hardware:** 1× H100 per PBS job on the UPES HPC

### 9.1 Idea
A pretrained LLM has already learned a lot about the fine-grained statistics of text. Fine-tuning it lightly (LoRA) to *classify* the author lets it use that knowledge. We call such a model a **reader**.

### 9.2 Architecture ("l24" reader)
1. **Backbone:** Qwen3-8B-Base, **truncated to its first 24 layers** (`--keep 24`). For the 14B variant: Qwen3-14B-Base, 27 layers.
   - Why truncate: early-to-middle layers carry more surface/style signal than the final layers, which specialise in next-token semantics. Truncation also cuts memory and time.
2. **LoRA:** rank 16, alpha 32, dropout 0.05, on all attention and MLP projections (`q,k,v,o,gate,up,down`).
3. **Head:** for each layer in {2, 4, 6, 8, 12, 16, 20, 24}:
   - masked mean-pool over tokens, **skipping the first token** (it acts as an attention sink and carries little content);
   - LayerNorm per layer;
   - concatenate → dropout 0.1 → linear layer to 18 classes.
4. **Precision:** base in bf16; LoRA and head weights kept in **fp32** as master weights, under bf16 autocast. With bf16 weights, small updates at lr 5e-5 mostly round away.
5. **Memory:** gradient checkpointing; batch size 16; max 384 tokens.

### 9.3 Two-stage training
| Stage | Data | Epochs | LR | Purpose |
|---|---|---|---|---|
| 1 | Open Pool: class-balanced sample (≤4,000 per class ≈ 70k) or the **full 231k** (`--pool_per_class 0`) | 1 | 1e-4 | Learn general author style from many topics |
| 2 | Train folds (competition data) | 2 | 5e-5 | Adapt to the competition's distribution |

Both stages use AdamW (weight decay 0.01), 5% linear warm-up, cosine decay and gradient clipping at 1.0. When `--fold ≥ 0`, the holdout is evaluated after each epoch, and `holdout_logp.npy` / `holdout_idx.npy` are saved for later parity checks and ensembling.

### 9.4 Saved artefacts (per run directory)
```
runs/<fold0|full><tag>/
├── adapter_stage1/            LoRA + head after stage 1
├── adapter/                   final LoRA (adapter_model.safetensors, adapter_config.json)
│   ├── head.pt                head weights
│   └── reader.json            labels, kept layers, layers read, max_len, base path
├── holdout_logp.npy           fold-0 log-probs (H100 reference)
└── holdout_idx.npy            exact fold-0 row indices
```

### 9.5 Runs performed
| PBS script | Backbone | Stage-1 data | Fold | Wall time |
|---|---|---|---|---|
| `train_lora.pbs` | 8B / 24 layers | 70k sample | 0, then full | 93 min + 74 min |
| `train_lora_fullpool.pbs` | 8B / 24 | full 231k | 0 | 223 min |
| `train_lora_fullpool_full.pbs` | 8B / 24 | full 231k | full (submission) | — |
| `train_lora_14b.pbs` | 14B / 27 | full 231k | 0 | — |
| `train_lora_14b_full.pbs` | 14B / 27 | full 231k | full | — |

**HPC practicalities** (in the `.pbs` scripts): PBS does not isolate GPUs on a shared node, so each job picks the GPU with the least memory in use. MPS is disabled with per-user pipe directories. `HF_HUB_OFFLINE=1` is set because models are pre-downloaded.

A `--smoke` flag builds a tiny random 4-layer model and runs every code path on CPU in seconds. We used it to catch bugs before spending GPU hours.

---

## 10. Component 4: Kaggle deployment and stack integration

### 10.1 T4 inference (`kaggle_lora/infer_lora.py`)
Kaggle's T4s have no native bf16 and the notebook has no internet. The inference script therefore:
- loads the truncated base in **fp16** from the attached Kaggle model `qwen-lm/qwen-3/Transformers/8b-base/1`;
- **merges the LoRA by hand** (`W ← W + (α/r)·B·A`) from the safetensors file, so `peft` is not needed;
- reimplements the head with plain tensor ops from `head.pt`;
- checks every batch's hidden states for inf/NaN and **recomputes overflowing batches in bf16** (emulated, slower but finite);
- for a `fold0` reader, **re-scores the saved holdout rows** and compares with the H100 predictions (parity check). Otherwise it writes `test_logp_<name>.npy` and `submission.parquet`.

**Parity result:** T4 fp16 0.6248 vs H100 bf16 0.6229 on fold 0; 98.9% identical argmax; max |Δ log p| 0.21; 0 overflow batches; ~44 s for 539 passages, so ~6 min for the 4k test set.

### 10.2 Stack integration (`kaggle_stack/add_ours.py`)
The best public notebook (0.704) is a large ensemble of 26 datasets and 9 models: l24 LoRA readers on Qwen3-14B/8B and Llama-8B, encoders, Sinkhorn balancing, GLM likelihood re-scoring via llama.cpp, family-level reranking and typography rules. We forked it unchanged and wrote `add_ours.py` to patch it:
1. Find the ensemble cell and insert two cells after it.
2. Cell 1 writes `infer_lora.py` to `/kaggle/working`.
3. Cell 2 runs each of our readers in its **own subprocess**, so GPU memory is freed between them, and merges:
   `z ← log_softmax(z_stack) + OURS_W · mean(our log-probs)` with `OURS_W = 0.35`.
4. On Kaggle's fast "commit" path, the merge is exercised on dummy logits as a smoke test.
5. The kernel metadata is updated to attach our private adapter dataset.

The GLM re-scoring and post-processing stages of the stack then run as before.

A bug found along the way: the base-model lookup matched Llama's `8b-base` directory too, causing a shape error. It now matches whole path segments under `/qwen-3/`.

---

## 11. Results

### 11.1 Offline (grouped CV, balanced accuracy, unseen prompts)
| # | Model | Score |
|---|---|---|
| 1 | Stylometry LR, train only (5-fold OOF) | 0.284 |
| 2 | TF-IDF LR, train only (5-fold OOF) | 0.375 |
| 3 | TF-IDF + stylometry (5-fold OOF) | 0.389 |
| 4 | TF-IDF + full pool (5-fold OOF) | 0.503 |
| 5 | TF-IDF + pool + stylometry (5-fold OOF) | 0.514 |
| 6 | TF-IDF + pool (fold 0) | 0.523 |
| 7 | Qwen3-8B l24, stage 1 on 70k pool sample (fold 0) | 0.575 |
| 8 | **Qwen3-8B l24, stage 1 on full pool (fold 0)** | **0.623** |
| 9 | Qwen3-14B l24 (27 layers), full pool (fold 0) | 0.617 |
| 10 | 8B + 14B, sum of log-probs (fold 0) | 0.638 |
| 11 | **8B + 14B + TF-IDF (fold 0)** | **0.653** |

Rows 10–11 use blend weights picked on fold 0, so they are slightly optimistic.

### 11.2 What moved the needle
| Change | Gain |
|---|---|
| Adding pool data to TF-IDF | **+0.128** (0.375 → 0.503) |
| TF-IDF → LLM reader (same fold) | **+0.100** (0.523 → 0.623) |
| 70k → 231k pool for the reader | +0.048 |
| Ensembling 8B + 14B | +0.015 |
| Adding TF-IDF to the two readers | +0.015 |
| Stylometry on top of TF-IDF | +0.011 to +0.014 |
| 8B → 14B backbone alone | −0.006 (none) |
| Stage 2 after full-pool stage 1 | 0.000 |

### 11.3 Leaderboard
| Submission | Public score |
|---|---|
| Public 0.704 stack (forked, unchanged) | 0.704 (author's score) |
| 0.704 stack + our 8B reader, `OURS_W = 0.35` | **0.686** |
| Top of leaderboard (20 Nov 2026) | 0.776 |

**Why adding our reader hurt (−0.018, about 2× noise):**
1. Offline CV overstates LoRA readers on the leaderboard. A public ablation logged a nested-CV 0.687 LoRA blend that scored 0.559; a two-stage 8B LoRA similar to ours went from CV 0.576 to LB 0.549.
2. The stack already contains stronger readers of the same type (14B/8B/Llama l24 LoRAs trained on more data), so ours adds little new information.
3. Our merge ran **after** the stack's Sinkhorn class-balancing step and partly undid it, which hurts a balanced-accuracy metric.
4. The weight could not be tuned offline: several stack members were trained on all of `train.csv`, so there is no clean holdout for the stack.

**Decision:** keep our readers out of the leaderboard stack, and report them as the class-project result.

---

## 12. Error analysis
From the iteration-1 model (TF-IDF + stylometry, 5-fold OOF); see `confusion_matrix.csv`.

**Accuracy grows with passage length.** Longer passages give more evidence.
| Words | ≤60 | 60–120 | 120–180 | >180 |
|---|---|---|---|---|
| Accuracy | 0.30 | 0.36 | 0.45 | 0.48 |

**Most errors stay inside one lab.** Balanced accuracy is 0.603 at lab level vs 0.389 at model level. Identifying the lab is comparatively easy; telling siblings apart is the hard part.

**Systematic confusions:**
- GLM → Claude: 136 GLM passages predicted as Claude vs 123 correct. GLM's style seems to imitate Claude's.
- glm-5.3 vs glm-5.3-flash; the gpt-5.6 variants; the Gemini flash versions.

**Per-class recall extremes:**
| Hardest | Recall | Easiest | Recall |
|---|---|---|---|
| claude-fable-5 | 0.12 | gemini-3.1-pro-preview | 0.79 |
| glm-5.3 | 0.19 | | |
| gpt-5.6-terra | 0.20 | | |

---

## 13. How to reproduce

### 13.1 Environment
- **CPU baseline:** Python 3.10+, `numpy pandas scipy scikit-learn joblib pyarrow`.
- **GPU training:** additionally `torch transformers peft safetensors` and a CUDA GPU with bf16 (A100/H100). Qwen3-8B-Base / 14B-Base downloaded locally.
- **Kaggle:** no install needed; attach the datasets and models listed in each `kernel-metadata.json`.

### 13.2 Data
Put the competition files in `data/` and the Open Pool parquet at `data/pool/ntl_open_pool.parquet`. On Kaggle both are found automatically by glob under `/kaggle/input/`.

### 13.3 CPU baseline
```bash
python ntl_baseline.py --cv        # grouped 5-fold CV (train only): scores, accuracy by length,
                                   # per-model recall, writes confusion_matrix.csv
python ntl_baseline.py             # fit on train + pool, write submission.parquet
python ntl_baseline.py --no-pool   # iteration 1: train only
```

### 13.4 LoRA readers (HPC)
```bash
python hpc/train_lora.py --smoke                       # CPU sanity check of every code path
python hpc/train_lora.py --fold 0                      # 8B, 70k pool sample, fold-0 holdout
python hpc/train_lora.py --fold 0  --pool_per_class 0 --tag _fullpool    # 8B, full pool
python hpc/train_lora.py --fold -1 --pool_per_class 0 --tag _fullpool    # full fit for submission
python hpc/train_lora.py --fold 0  --pool_per_class 0 \
       --model ~/ntl/models/Qwen3-14B-Base --keep 27 --tag _14b_fullpool # 14B
# or submit the matching job:  qsub hpc/train_lora_fullpool.pbs
```
Useful flags: `--keep` (layers kept), `--layers` (layers read by the head), `--r` (LoRA rank), `--epochs2`, `--lr1/--lr2`, `--bs`, `--max_len`.

### 13.5 Kaggle
1. Upload each run's `adapter/` folder (renamed, e.g. `q8_fold0_fullpool`, `q8_full_fullpool`) as a private dataset.
2. `kaggle_lora/`: push with `kaggle kernels push`. It validates `fold0*` readers against the H100 numbers and predicts the test set with the others.
3. `kaggle_stack/`: run `python add_ours.py` inside the folder to patch `stack.ipynb`, then push and submit.
4. `kaggle_kernel/`: a CPU script kernel for the TF-IDF + pool model.

---

## 14. Lessons learned (general takeaways)
These apply beyond this competition, to most text classification problems with topic shift or small labelled data.

1. **Make validation match the test distribution first.** Group by whatever the test set will not share with train (here, the prompt). Otherwise every later decision is tuned to a misleading number.
2. **More relevant data beats cleverer models.** The 231k pool helped both the n-gram model (+0.13) and the LLM reader (+0.05) more than any architecture change. Audit external data for leakage before using it.
3. **Remove the confounder explicitly.** Topic-masking views (shape text, function words) and intermediate-layer reading both push the model towards style rather than content.
4. **Pretrained LLMs are strong feature extractors for style.** Intermediate layers, LoRA and a light head beat a well-tuned n-gram model by +0.10. Bigger is not automatically better: 14B ≈ 8B.
5. **Diversity matters for ensembles.** Two readers that disagree on 21% of passages gained more together than one bigger model. Even the weaker TF-IDF model still added +0.015.
6. **Verify each deployment target.** Precision (fp16 vs bf16), library versions (sklearn fold splits) and file lookup paths all changed results silently. Save exact holdout indices and reference predictions, and check parity on the target hardware.
7. **A member that is good alone can still hurt an ensemble.** Gains are not additive. Where you insert a model matters (before or after calibration/balancing steps), and without a clean holdout you cannot tune its weight.
8. **Build cheap smoke tests for expensive jobs.** `--smoke` and the Kaggle fast-path checks caught bugs in minutes instead of GPU hours.

---

## 15. Limitations and future work

**Limitations**
- GPU readers were evaluated on fold 0 only (539 passages, 30 prompts), so their scores have wider error bars than the 5-fold CPU scores.
- Ensemble weights were picked on the same holdout and are optimistic.
- Error analysis was done for the iteration-1 model only, not for the LoRA readers.
- The leaderboard contribution of our reader was negative at the single weight we tried.

**Future work**
- Full 5-fold CV for the readers to get trustworthy ensemble weights.
- Merge our readers *before* the stack's Sinkhorn balancing instead of after.
- **Hierarchical classification:** predict the lab first (0.60 already), then a sibling-specific model within the lab.
- Specialised pairwise models for the hard pairs (GLM vs Claude, glm-5.3 vs flash, gpt-5.6 variants).
- Try the other public corpora (18-model wide response corpus, extended dataset) and larger or different backbones.
- Error analysis of the readers: do they fix the same classes TF-IDF misses?

---

## 16. Glossary
| Term | Meaning |
|---|---|
| Balanced accuracy | Mean of per-class recall; unaffected by class imbalance |
| Grouped CV | Cross-validation in which all rows of a group (prompt) are in the same fold |
| OOF | Out-of-fold predictions: each row is predicted by the model that did not train on it |
| TF-IDF | Term frequency × inverse document frequency weighting of n-grams |
| Stylometry | Hand-crafted measurements of writing style (sentence length, punctuation rates, …) |
| LoRA | Low-Rank Adaptation: trains small rank-r update matrices instead of the full weights |
| Reader | A fine-tuned LLM used as a classifier |
| l24 head | Head that mean-pools hidden states from several layers of a 24-layer truncated LLM |
| Log-linear blend | Weighted sum of log-probabilities from several models (product of experts) |
| Sinkhorn balancing | Rescaling predictions so each class gets a roughly equal share of the test set |
| Stack | A large ensemble combining many models' outputs |
| Parity check | Confirming a model gives the same predictions on new hardware as during training |

---

## 17. Credits
- **Open Pool dataset:** Carrot Production (`crtproduction/name-that-llm-open-pool`).
- **Public 0.704 stack:** lightningv08, building on Alex Chilton, lumibee, areebbb, feiyu26 and others.
- **Grouped-CV analysis and baselines:** Adisak Sukul and GoPain (discussion 750390).
- **Multi-view TF-IDF ideas** (shape and function-word views): Ahmed Berat Özer, "Teach the Model, Remove the Prompt".
- **Compute:** UPES HPC (H100), Kaggle (T4).
