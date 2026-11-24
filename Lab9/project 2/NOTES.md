# Name That LLM: competition analysis + iteration 1

Kaggle: https://www.kaggle.com/competitions/name-that-llm (final submission deadline 13 Jan 2027, team merger deadline 6 Jan 2027)

## Task
- **Goal:** attribute each passage (40 to 250 words) to one of **18 LLMs** from 6 labs:
  - OpenAI ×5
  - Google ×4
  - Anthropic ×5
  - Qwen ×1
  - ZAI ×2
  - Moonshot ×1
- **Metric:** balanced accuracy (mean per-class recall). Chance is 0.056.
- **Train:** 2,699 passages over 150 prompts. Every model answers every prompt once, so there are about 150 rows per class.
- **Test:** about 4,000 hidden passages, all on prompts that are **not in train**. The class mix differs between the public and private leaderboards.
- **Format:** code competition.
  - The notebook must run within 9 hours with internet off.
  - It reads `/kaggle/input/competitions/name-that-llm/test.csv` and writes `submission.parquet` with columns `id` and `model`.
  - Public datasets and models may be attached.

## What the field does (17 Nov 2026)
- **Leaderboard:** the top public score is 0.776. About the top 50 are above 0.726.
- **Public notebooks:** the best reach about 0.70. They are large stacks with 30+ attached datasets.

| Approach | Grouped CV | Public LB | Source |
|---|---|---|---|
| char+word TF-IDF + LR | 0.343 | 0.353 | discussion 750390 (Adisak Sukul) |
| 4-view n-gram SVM, train only | 0.370 | 0.361 | same thread, comment by GoPain |
| Qwen3-14B-Base LoRA, train only | 0.443 (fold 0) | 0.466 | 750390 |
| same SVM + public "pool" of extra passages | 0.467 | 0.423 | GoPain |
| two-stage Qwen3-8B LoRA (pool, then train) + Sinkhorn balancing | 0.576 | 0.549 | GoPain |
| TF-IDF + stylometry + frozen-LLM layer probes + Qwen3-8B LoRA + test-time prompt grouping | ~0.50 | 0.600 | notebook "Teach the Model, Remove the Prompt" (Ahmed Berat Özer) |
| huge stack: LoRA heads, encoders, SVC pool, GLM likelihood re-scoring via llama.cpp | n/a | 0.70 | areebbb's notebook (built on feiyu26's); only its intro was read |
| (not read) | n/a | 0.701 | romantamrazov's notebook |

Main findings:
1. **Group folds by `prompt_id`.** A random split puts other models' answers to the same prompt in train, and that misleads the score. Only grouped CV tracks the leaderboard, and only when training on train alone. With extra pool data, CV overstates the gain by about 0.035.
2. **Extra data is the biggest lever.**
   - The community "Open Pool" is large (one thread title says 120k passages; not verified), and v14 is cleaned.
   - Other public sets: the "18-model wide response corpus" and "extended-name-that-llm-dataset".
   - Rules §2.6 allow publicly available external data.
   - Self-generated API data is still unresolved by the hosts (discussion 751011).
3. **Fine-tuned LLM readers beat n-grams.** Common choices are Qwen3-8B/14B with LoRA and a classification head. Early-to-mid layers carry more author signal than late layers. Inference fits on Kaggle's 2×T4 in fp16.
4. **Test-time tricks are risky.** Clustering test passages by prompt, then Hungarian assignment or per-cluster centring, gained about +0.10 in CV but *lowered* the public leaderboard (0.353 → 0.344 / 0.329). The test probably has few passages per prompt. Class-balancing predictions (Sinkhorn) went 0.538 → 0.549 in a single submission, so it is possibly a small gain; differences under about 0.01 are noise.
5. **Hard pairs:** glm-5.3 vs glm-5.3-flash; GLM vs Claude; the Gemini flash versions; the gpt-5.6 variants.

## Iteration 1 (this repo): `ntl_baseline.py`
CPU only, about 1 minute per fit, no external data.
- 4-view TF-IDF:
  - char 1-5
  - word 1-3 with punctuation tokens
  - "shape text", where letters become X/x so only the punctuation and spacing skeleton remains
  - function-word text, where content words are masked to remove topic
- These four views feed one multinomial logistic regression.
- 35 stylometric features (sentence and word length statistics, type-token ratio, punctuation rates, curly quotes, em dashes, markdown characters) feed a second logistic regression.
- The two are combined as a log-linear blend: `tfidf + 0.3·stylo`.

```
python ntl_baseline.py --cv   # grouped 5-fold CV, per-length accuracy, per-model recall, confusion_matrix.csv
python ntl_baseline.py        # fit on all train, write submission.parquet
```
To submit, paste the script into a Kaggle notebook cell, attach the competition data and turn internet off. It finds the data by glob.

### Results (5-fold StratifiedGroupKFold by prompt, seed 42)
| Model | Balanced accuracy |
|---|---|
| TF-IDF LR | 0.375 |
| stylometry LR | 0.284 |
| **blend (w=0.3)** | **0.389** (w was preset, then checked on OOF, where it was also the best of the grid) |

- **Not yet submitted to Kaggle.** With train-only data, grouped CV tracked the leaderboard within about 0.01 in thread 750390, so we *expect* about 0.36–0.39 on the public leaderboard. That is an expectation, not a result.

- **By length:** ≤60 words scores 0.30; 60-120 words 0.36; 120-180 words 0.45; >180 words 0.48.
- **Provider-level balanced accuracy is 0.603**, against 0.389 at model level, so roughly half the errors stay within one lab.
- **GLM is mostly classed as Claude:** 136 of the GLM passages are predicted as Claude, against 123 predicted correctly. Discussion threads report the same.
- **Worst recall:**
  - claude-fable-5: 0.12
  - glm-5.3: 0.19
  - gpt-5.6-terra: 0.20
- **Best recall:** gemini-3.1-pro-preview at 0.79.

## Iteration 2: Open Pool + LLM reader
### 2a. TF-IDF + Open Pool v16 (`ntl_baseline.py`, now the default)
- **Data:** Open Pool v16 (`crtproduction/name-that-llm-open-pool`, 231,036 labelled passages).
  - None of its prompts are flagged as near a train prompt; the highest similarity to any train prompt is 0.40.
  - No passage text overlaps train.
- **Training:** the TF-IDF views are fit on train + pool (`min_df=3`). An SGD logistic model is trained on all rows, with pool rows at sample weight 0.6. Stylometry is still trained on train alone and blended at w=0.2.
- **CV setup:** folds stay on train only, and the pool is always in the training side.
  - The pool results come from the cached-feature sweep in `scratchpad/exp_cv.py`.
  - `--cv` still runs the train-only model, because a full pool CV takes about 20 min per fold.

| Setting (grouped 5-fold OOF) | Balanced accuracy |
|---|---|
| train only (iteration 1 TF-IDF) | 0.375 |
| + 60k pool rows, weight 0.3 | 0.458 |
| + full pool, weight 0.3 / 0.6 / 1.0 | 0.496 / **0.503** / 0.502 |
| + full pool, weight 0.3 / 0.6 × `match_weight` | 0.502 / 0.504 (no gain) |
| **full pool (w 0.6) + stylometry blend (w 0.2)** | **0.514** |

- **Pool weight is flat between 0.3 and 1.0**, and the dataset's `match_weight` column adds nothing.
- **Expected leaderboard:** with pool data, CV overstates the gain by about 0.035, so we expect roughly 0.47.
- **Kaggle notebook:** private notebook `somanshgoel/ntl-tf-idf-open-pool`, version 1.
  - It ran successfully in 24.7 min, with internet off and the pool loaded.
  - It is not yet submitted: the account first needs Kaggle identity verification. After that, use Submit Prediction and pick this notebook, version 1, output `submission.parquet`.
- **Cost:** the full fit with the pool took 12.4 min locally (4 parallel views), mostly vectorizing. That fits Kaggle's 9 h easily.
- **Kaggle setup:** the notebook needs the Open Pool dataset attached. It finds `ntl_open_pool.parquet` by glob; `--no-pool` restores iteration 1.

### 2b. Qwen3-8B-Base LoRA reader (`hpc/train_lora.py`, UPES HPC)
- **Model:** sequence classification head, LoRA r=16 on all linear layers, bf16, max 384 tokens.
- **Stage 1:** a class-balanced pool sample of up to 4,000 per class (about 70k rows), 1 epoch at lr 1e-4.
- **Stage 2:** the train folds, 2 epochs at lr 5e-5.
- **Head:** the "l24" design used by the 0.70 stacks.
  - The LLM is truncated to its first 24 layers.
  - The head mean-pools the hidden states of layers 2, 4, 6, 8, 12, 16, 20 and 24, layer-norms each, concatenates them and applies a linear layer.
  - LoRA and head weights are fp32, run under bf16 autocast.
  - This replaced the last-token classification head before any GPU run.
- **Runs:**
  - `train_lora.pbs`: fold 0 (for validation), then the full fit, on one GPU.
  - `train_lora_fullpool.pbs`: fold 0 with stage 1 on all 231k pool rows, to compare against the 70k sample.
- **Status:** both ran on H100s. A `--smoke` run (tiny random 2-layer model, CPU) covers every code path.

Fold-0 holdout results (540 rows, prompts unseen in training):

| Model | Fold-0 balanced accuracy |
|---|---|
| TF-IDF + full pool (grouped OOF, fold-0 rows) | 0.523 |
| LoRA l24, after stage 1 (70k pool only) | 0.565 |
| LoRA l24, stage 2, epoch 1 / epoch 2 (final) | 0.582 / **0.575** |
| LoRA + TF-IDF log-linear blend, w 0.2 / 0.35 / 1.0 | 0.601 / 0.610 / 0.619 (w picked on this holdout, so optimistic) |

| **LoRA l24, stage 1 on the full 231k pool** (after stage 1 / final) | 0.623 / **0.623** |
| full-pool LoRA + TF-IDF blend, w 0.5 / 0.75 | 0.643 / 0.645 |
| both LoRAs + TF-IDF (w 1.0) | 0.647 |

- **Full pool vs sample:** the full pool adds +0.048 over the 70k sample. Stage 2 on train adds about nothing on top of the full-pool stage 1.
- **Time:** the 70k-pool run took 93 min on one H100, the full fit 74 min, and the full-pool run 223 min.
| Qwen3-14B-Base l24 (27 layers), full pool (after stage 1 / final) | 0.606 / 0.617 |
| 8B + 14B (sum of log-probs) | 0.638 |
| **8B + 14B + TF-IDF (w 1.0)** | **0.653** |

- **14B vs 8B:** the 14B alone is no better than the 8B, but the two disagree on 21% of passages, so they ensemble well.
- **Fits for submission:**
  - 8B full-pool full fit: done (`runs/full_fullpool`).
  - 14B full fit: job 47552.

### Kaggle inference for our readers (`kaggle_lora/infer_lora.py`)
- **Assets:**
  - Adapters are in the private dataset `somanshgoel/ntl-lora-readers`: `q8_fold0_fullpool` (with its holdout idx/logp) and `q8_full_fullpool`.
  - The base is the Kaggle model `qwen-lm/qwen-3/Transformers/8b-base/1`.
- **How it runs:** it merges the LoRA by hand into a 24-layer fp16 base, so peft isn't needed. Any batch that overflows fp16 is recomputed in bf16.
- **Parity on 2×T4, private kernel `somanshgoel/ntl-lora-reader-t4`:**
  - Fold-0 holdout balanced accuracy: **T4 fp16 0.6248 vs H100 bf16 0.6229**.
  - 98.9% argmax agreement, max |Δ log p| 0.21.
  - 0 fp16 overflows.
  - 539 passages in 44 s, so about 6 min for the 4,000-row test.
- **Gotcha:** Kaggle's sklearn splits `StratifiedGroupKFold` differently (fold 0 had 540 rows there vs 539 on the HPC). Validation must use the saved `holdout_idx.npy`. Re-splitting leaked training rows and showed a fake 0.767.
- **Comparison:** this matches GoPain's published two-stage 8B LoRA (CV 0.576, leaderboard 0.549).

### How the 0.70 public stacks are built (read from `lightningv08/0-704-glm-fable-routing-qwen3-8-27b-lora`)
- Everything in it is public. It uses 26 attached datasets and 9 models, and runs on Kaggle 2×T4. Anyone can fork it for about 0.70.
- **Readers** (blend weight in brackets), all l24 LoRA readers unless noted:
  - Qwen3-14B (3.0)
  - Qwen3-8B (2.0)
  - Llama-3.1-8B (1.0)
  - a second Qwen3-14B (2.5)
  - Qwen2.5-14B (1.0)
  - DeBERTa (1.0), an encoder rather than an l24 reader
  - Qwen3-Embedding-8B multi-depth, five folds (share 0.40)
  - Qwen3.8-27B adapter (share 0.03)
- **TF-IDF, SVC and distortion members get weight 0.**
- **After the blend:**
  - Sinkhorn class balancing at 0.30.
  - GLM-5.3 and GLM-5.3-Flash token likelihood via llama.cpp GGUF, on the top 15% of texts by GLM probability.
  - A "sibling residual" that reranks within a model family.
  - Soft typography rules: a label the rule forbids is multiplied by 0.05.
- **Leaderboard noise:** resubmitting identical code moves the public score by up to about 0.008 (Alex Chilton's table).
- **Tried and failed:** Hungarian or cluster assignment, extra small members and intra-family gates. Most added members gave 0 to −0.01.
- **Our fork:** the private notebook `somanshgoel/ntl-stack-base-0704`, version 1, is an exact copy with the same attachments and T4 machine.
  - Its normal run took the fast path (`FAST_COMMIT`: asset and smoke checks only) and completed in 3.5 min.
  - The full pipeline runs only on a scoring rerun, up to 9 h.
  - Not yet submitted: this waits on identity verification. The expected score is about 0.70 (the author's 0.704, ± about 0.008 noise).
- **Fork + our readers (version 3, `kaggle_stack/add_ours.py` rebuilds it from the original notebook):**
  - Two cells are inserted after the ensemble cell. One writes `infer_lora.py`; the other runs each of our readers in a subprocess on the test set.
  - The merge is `z ← log_softmax(z_stack) + OURS_W · mean(our log-probs)` on `/kaggle/temp/final-logits.npz`. It also rewrites the interim submission, then GLM likelihood and the apex post-processing run as before.
  - `OURS_W = 0.35`, chosen blind (roughly one stack member's share). It can only be tuned on the leaderboard.
  - Fast-path run: our reader and the merge (tested on dummy logits) work. The total run took 5.7 min.
  - Bug fixed on the way: the base lookup matched Llama's `8b-base`, giving a shape error. It now matches whole path segments under `/qwen-3/`.
- **Submitted (20 Nov 2026):** stack + our 8B reader (`OURS_W` 0.35), notebook version 3, submission ref 58598009. Public score **0.686**, below the 0.704 base by 0.018 (about twice the resubmission noise). At this weight our reader hurts the stack.
- **Implication:** our own LoRA alone is about 0.55-level (GoPain's two-stage 8B LoRA scored 0.549). The route to the top is the public stack as a base, plus stronger HPC-trained readers (more data, bigger backbones) that replace or outweigh its members.

## Roadmap
1. ~~Iteration 2a: Open Pool for TF-IDF.~~ Done, see above. Still to try: the 18-model corpus.
2. **Iteration 2b/3: LLM reader.** Finish the HPC run. Then upload the adapter as a Kaggle dataset and run fp16 inference on 2×T4.
3. Blend the components and add Sinkhorn class-balancing. Avoid Hungarian grouping, which hurt the leaderboard.
4. **Class-project angles:**
   - hierarchical provider → model classification
   - which stylometric cues separate labs (confusion analysis above)
   - the length vs accuracy curve

## Credits
- Adisak Sukul: grouped-CV analysis and TF-IDF baseline (discussion 750390)
- GoPain: CV vs leaderboard table (same thread)
- Ahmed Berat Özer: "Teach the Model, Remove the Prompt" (multi-view TF-IDF, shape and function-word views, probes, grouping)
- areebbb / feiyu26: the 0.70 stack notebooks
- Carrot Production: Open Pool dataset

### Why an offline weight fit isn't possible (20 Nov 2026)
- **No honest holdout for the stack:** several heavy members (Q14 LoRAs, L24 adapters, the 27B adapter, the SVC prefit pool) are single fits on all of `train.csv` plus extra generations, so the stack is in-sample on our fold-0 rows. A weight fit there returns about 0 by construction.
- **Offline CV doesn't carry over to the board:** the public ablation notebook (`name-that-llm-0-702-all-public-what-failed`) logs a five-backbone two-stage LoRA blend at nested CV 0.6865 that scored 0.559 on the board. Blending that kind of model into a stack lost 0.003 to 0.013 each time. Our 0.704 to 0.686 fits the same pattern.
- **Integration caveat:** our merge runs after the stack's Sinkhorn balancing (`final-logits.npz` is saved post-balance), so it also partly undoes the balancing.
- **Decision:** keep our readers out of the leaderboard stack, and report them as the class-project result (fold-0 0.623 for 8B; 0.653 with 14B and TF-IDF).
