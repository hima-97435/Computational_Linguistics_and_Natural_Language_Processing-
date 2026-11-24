# Lab 9: Name That LLM — Authorship Attribution of Machine-Generated Text

## Objective
Given a 40–250 word English passage, predict which of **18 large language models** authored it. The task is framed as multi-class authorship attribution across six model families: OpenAI, Google, Anthropic, Qwen, ZAI, and Moonshot.

- **Competition:** [Kaggle: Name That LLM](https://www.kaggle.com/competitions/name-that-llm)
- **Metric:** Balanced accuracy (mean per-class recall). Chance = `1/18 ≈ 0.056`.
- **Constraint:** Kaggle code competition — 9 hours, 2×T4 GPUs, internet disabled.

## Dataset
| Split | Description |
|---|---|
| **Train** | 2,699 passages across 150 prompts; ~150 passages per class. |
| **Test** | ~4,000 hidden passages on prompts not present in train. |
| **External (Open Pool v16)** | 231,036 labeled passages from the same 18 models; no prompt/text leakage into train. |

The key challenge is recognizing **writing style**, not topic, because test prompts are unseen.

## Evaluation Protocol
- **Grouped 5-fold cross-validation** by `prompt_id` (`StratifiedGroupKFold`, seed 42).
- Grouping ensures validation prompts are unseen, matching the competition test setting.
- LoRA reader holdout results below use fold 0: 539 passages from 30 unseen prompts.

## Method

### 1. Multi-View TF-IDF + Stylometry Baseline (`ntl_baseline.py`, CPU)
- **Four TF-IDF views** fed into one multinomial logistic regression:
  - Character 1–5-grams
  - Word 1–3-grams with punctuation tokens
  - Shape view: letters mapped to `X/x`, preserving punctuation/spacing skeleton
  - Function-word view: content words masked to reduce topic leakage
- **35 stylometric features** fed into a second logistic regression (sentence/word length stats, type-token ratio, punctuation rates, curly quotes, em dashes, markdown characters).
- **Blend:** log-linear combination of TF-IDF and stylometry models.
- **External data:** TF-IDF trained on train + Open Pool with pool sample weight `0.6` (SGD logistic for scale).

### 2. Fine-Tuned LLM Readers (`hpc/train_lora.py`, HPC H100)
- **Backbones:** Qwen3-8B-Base (truncated to 24 layers) and Qwen3-14B-Base (truncated to 27 layers).
- **Head:** mean-pool hidden states of layers 2, 4, 6, 8, 12, 16, 20, 24 (skip first token), layer-norm each, concatenate, dropout 0.1, linear classifier to 18 classes.
- **LoRA:** rank 16, alpha 32, applied to all attention and MLP projections; LoRA/head in fp32 under bf16 autocast.
- **Two-stage training:**
  1. 1 epoch on balanced Open Pool (231k passages), lr `1e-4`
  2. 2 epochs on competition train folds, lr `5e-5`
- **Kaggle inference (`kaggle_lora/infer_lora.py`):** manual LoRA merge into fp16 truncated base for T4 inference; fp16-overflow batches recomputed in bf16.

### 3. Ensemble / Stack Integration
- Component log-probabilities combined by sum/weighted blend.
- Best offline ensemble: **8B + 14B LoRA readers + TF-IDF**.
- Kaggle stack integration handled in `kaggle_stack/`.

## Results

### Offline (Grouped CV, Balanced Accuracy)
| # | Model | Score |
|---|---|---|
| 1 | Stylometry LR, train only (5-fold OOF) | 0.284 |
| 2 | TF-IDF LR, train only (5-fold OOF) | 0.375 |
| 3 | TF-IDF + stylometry (5-fold OOF) | 0.389 |
| 4 | TF-IDF + full Open Pool (5-fold OOF) | 0.503 |
| 5 | TF-IDF + pool + stylometry (5-fold OOF) | 0.514 |
| 6 | TF-IDF + pool (fold 0) | 0.523 |
| 7 | Qwen3-8B l24 LoRA, stage 1 on 70k pool sample (fold 0) | 0.575 |
| 8 | Qwen3-8B l24 LoRA, stage 1 on full pool (fold 0) | 0.623 |
| 9 | Qwen3-14B l24 LoRA (27 layers), full pool (fold 0) | 0.617 |
| 10 | 8B + 14B, sum of log-probs (fold 0) | 0.638 |
| 11 | **8B + 14B + TF-IDF (fold 0)** | **0.653** |

> Rows 10–11 use blend weights selected on fold 0 and are slightly optimistic.

### Key Observations
- **External data gave the largest gain:** TF-IDF improved from 0.375 to 0.503 with the Open Pool; the 8B reader improved from 0.575 to 0.623.
- **Model size alone was not enough:** 8B → 14B gave no standalone gain (-0.006), but the readers disagree on 21% of passages and ensemble well.
- **Grouped CV is essential:** random splits leak prompt/topic information and inflate scores.
- **Hardest distinctions:** sibling models within the same lab/family and GLM-vs-Claude confusion.
- **Offline gain does not always transfer to a strong leaderboard stack:** adding our 8B reader to the public 0.704 stack reduced public score to 0.686.

### Error Analysis (TF-IDF + Stylometry Baseline)
- Accuracy increases with passage length: 0.30 (≤60 words), 0.36 (60–120), 0.45 (120–180), 0.48 (>180).
- Lab-level balanced accuracy (0.603) is much higher than model-level (0.389): most errors stay within the same lab.
- Hardest classes: `claude-fable-5` (0.12), `glm-5.3` (0.19), `gpt-5.6-terra` (0.20).
- Easiest class: `gemini-3.1-pro-preview` (0.79).

## Repository Structure

```text
Lab9/
├── README.md                          # This file
├── DOCUMENTATION.md                   # Full project documentation
├── NOTES.md                           # Detailed run log and analysis
├── ntl_baseline.py                    # TF-IDF + stylometry baseline (grouped CV)
├── confusion_matrix.csv               # Iteration-1 confusion matrix
├── Name_That_LLM_Documentation.pdf    # Project documentation PDF
├── submission.parquet                 # Sample/submission artifact
├── data/                              # Competition/pool data files
├── hpc/                               # LoRA training scripts and PBS jobs
│   ├── train_lora.py
│   ├── train_lora.pbs
│   ├── train_lora_14b.pbs
│   ├── train_lora_14b_full.pbs
│   ├── train_lora_fullpool.pbs
│   └── train_lora_fullpool_full.pbs
├── kaggle_kernel/                     # Baseline Kaggle kernel files
│   ├── kernel-metadata.json
│   └── ntl_baseline.py
├── kaggle_lora/                       # LoRA Kaggle inference
│   ├── infer_lora.py
│   └── kernel-metadata.json
└── kaggle_stack/                      # Stack/ensemble integration
    ├── add_ours.py
    ├── kernel-metadata.json
    └── stack.ipynb
```

## How to Run

### Baseline (CPU)
```bash
# Grouped 5-fold CV
python ntl_baseline.py --cv

# Fit on all train and generate submission
python ntl_baseline.py
```

### LoRA Reader (GPU/HPC)
Use the scripts in `hpc/` (e.g., `train_lora.py`) with the corresponding PBS job files for cluster execution.

### Kaggle Inference
Use `kaggle_lora/infer_lora.py` for T4 inference/parity checks and `kaggle_stack/add_ours.py` for stack integration.

## References
- Open Pool data: Carrot Production.
- Grouped-CV analysis and baselines: Adisak Sukul and GoPain (Kaggle discussion 750390).
- Multi-view TF-IDF ideas: Ahmed Berat Özer ("Teach the Model, Remove the Prompt").
- Public 0.704 stack: lightningv08, building on Alex Chilton, lumibee, and others.
