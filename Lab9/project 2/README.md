# Project: Name That LLM (Authorship Attribution of LLM Text)

**Name:** Somansh Goel  
**SAP ID:** 500123667  
**Competition:** [Kaggle: Name That LLM](https://www.kaggle.com/competitions/name-that-llm)  
**Full documentation:** [DOCUMENTATION.md](DOCUMENTATION.md)

## Objective
Given a passage of 40 to 250 words, predict which of **18 large language models** wrote it. The 18 models come from 6 labs: OpenAI, Google, Anthropic, Qwen, ZAI and Moonshot. The score is **balanced accuracy** (mean recall per class). Chance is 0.056.

- **Train:** 2,699 passages on 150 prompts. Each model answers each prompt once, so there are about 150 passages per class.
- **Test:** about 4,000 hidden passages, all on prompts that **do not appear in train**. The task is to recognise *style*, not topic.
- **Format:** Kaggle code competition. The notebook must run within 9 h on 2×T4 GPUs with internet off.

## Evaluation protocol
Grouped 5-fold cross-validation by prompt (`StratifiedGroupKFold`, seed 42). A random split puts other models' answers to the same prompt into training, which inflates the score. Grouping means every validation prompt is unseen, as on the test set. The LoRA results below use the fold-0 holdout: 539 passages from 30 unseen prompts.

## Method
The project ran in three iterations. Each iteration fed into the next.

### 1. Multi-view TF-IDF + stylometry (`ntl_baseline.py`, CPU)
- **Four TF-IDF views,** fed into one multinomial logistic regression:
  - character 1-5-grams
  - word 1-3-grams with punctuation tokens
  - a *shape* view: letters mapped to X/x, so only punctuation and spacing remain
  - a *function-word* view: content words masked, to remove topic
- **35 stylometric features,** fed into a second logistic regression: sentence and word length statistics, type-token ratio, punctuation rates, curly quotes, em dashes and markdown characters.
- **Blend:** the two models are combined log-linearly.

### 2. External data: Open Pool v16
- **Data:** a public corpus of 231,036 labelled passages from the same 18 models.
- **Leakage checks:** no pool prompt is close to a train prompt (highest similarity 0.40), and no passage text overlaps train.
- **Use:** TF-IDF is fit on train + pool, with pool rows at sample weight 0.6.

### 3. Fine-tuned LLM readers (`hpc/train_lora.py`, UPES HPC, H100)
- **Backbone:** Qwen3-8B-Base truncated to its first 24 layers. Qwen3-14B-Base truncated to 27 layers was also trained.
- **Head ("l24"):** mean-pool the hidden states of layers 2, 4, 6, 8, 12, 16, 20 and 24, skipping the first token. Layer-norm each, concatenate, apply dropout 0.1 and a linear layer to 18 classes. Early-to-middle layers carry more style signal than the final layer.
- **LoRA:** rank 16, alpha 32, on all attention and MLP projections. LoRA and head are kept in fp32 under bf16 autocast.
- **Two training stages:**
  1. 1 epoch on the full Open Pool (231k passages), lr 1e-4.
  2. 2 epochs on the train folds, lr 5e-5.
- **Kaggle inference (`kaggle_lora/infer_lora.py`):** the LoRA is merged by hand into an fp16 truncated base, so peft isn't needed. Any batch that overflows fp16 is recomputed in bf16.

## Results

### Offline (grouped CV, balanced accuracy)
| Model | Score |
|---|---|
| TF-IDF, train only (5-fold OOF) | 0.375 |
| Stylometry, train only (5-fold OOF) | 0.284 |
| TF-IDF + stylometry blend (5-fold OOF) | 0.389 |
| TF-IDF + full Open Pool (5-fold OOF) | 0.503 |
| TF-IDF + pool + stylometry (5-fold OOF) | 0.514 |
| TF-IDF + pool (fold 0) | 0.523 |
| Qwen3-8B l24 LoRA, stage 1 on a 70k pool sample (fold 0) | 0.575 |
| **Qwen3-8B l24 LoRA, stage 1 on the full pool (fold 0)** | **0.623** |
| Qwen3-14B l24 LoRA (27 layers), full pool (fold 0) | 0.617 |
| 8B + 14B, sum of log-probs (fold 0) | 0.638 |
| **8B + 14B + TF-IDF (fold 0)** | **0.653** |

- **Data mattered more than model size.** Moving from a 70k pool sample to the full pool added +0.048. Moving from the 8B to the 14B added nothing on its own.
- **The readers complement each other.** The 8B and 14B disagree on 21% of passages, and TF-IDF adds a further +0.015 on top of them.
- **Stage 2 adds little.** After stage 1 on the full pool, fine-tuning on train left the score unchanged (0.623 → 0.623).
- **Kaggle T4 matches the H100.** Fold 0 scores 0.6248 on Kaggle's T4 against 0.6229 on the HPC's H100, with 98.9% of predictions identical and no fp16 overflows. The 4k-passage test set takes about 6 min per reader.

### Error analysis (iteration 1, TF-IDF + stylometry)
- **Accuracy grows with length:** 0.30 for passages of ≤60 words, 0.36 for 60-120, 0.45 for 120-180 and 0.48 for >180.
- **Most errors stay within one lab:** balanced accuracy at lab level is 0.603, against 0.389 at model level. Telling sibling models apart is the hard part.
- **GLM is mostly mistaken for Claude:** 136 GLM passages are predicted as Claude, against 123 predicted correctly.
- **Hardest classes:**
  - claude-fable-5: 0.12
  - glm-5.3: 0.19
  - gpt-5.6-terra: 0.20
- **Easiest class:** gemini-3.1-pro-preview at 0.79.

### Leaderboard
| Submission | Public score |
|---|---|
| Public 0.704 stack (forked, unchanged) | 0.704 (author's score) |
| 0.704 stack + our 8B reader, blend weight 0.35 | **0.686** |
| Top of the leaderboard (20 Nov 2026) | 0.776 |

- **Adding our reader lowered the stack's score by 0.018.** That is about twice the noise between identical resubmissions (about 0.008).
- **Why:**
  1. **Offline CV overstates readers like ours on the leaderboard.** A public ablation reports a LoRA blend with nested CV of 0.687 that scored 0.559 on the board. A published two-stage 8B LoRA similar to ours went from CV 0.576 to leaderboard 0.549.
  2. **The stack's own members are already the same kind of model, only stronger.** They are l24 LoRA readers on Qwen3-14B, Qwen3-8B and Llama-8B, trained on more data.
  3. **Our merge partly undid the stack's balancing.** It ran after the Sinkhorn class-balancing step, which hurts a balanced-accuracy metric.
- **The weight couldn't be tuned offline.** Several stack members were trained on all of train, so the stack has already seen every holdout row.

## Conclusions
1. **Grouped evaluation is essential.** With random splits, train and test share prompts, and the score measures topic memory rather than style.
2. **For TF-IDF, extra data beat model changes.** The 231k-passage pool raised TF-IDF from 0.375 to 0.503 and the LoRA reader from 0.575 to 0.623.
3. **LoRA readers on intermediate layers clearly beat n-grams:** 0.623 against 0.523 on the same holdout. Ensembling different readers helps further.
4. **The hardest distinctions are between sibling models,** such as glm-5.3 vs glm-5.3-flash and the gpt-5.6 variants, and between GLM and Claude.
5. **A good offline score doesn't guarantee a gain inside a strong ensemble.** A member that scores well alone can still lower the stack's leaderboard score.

## Files
| Path | Contents |
|---|---|
| `ntl_baseline.py` | TF-IDF + stylometry baseline, with optional Open Pool (`--cv` for grouped CV) |
| `hpc/train_lora.py`, `hpc/*.pbs` | LoRA reader training and PBS job scripts for the UPES HPC |
| `kaggle_lora/infer_lora.py` | T4 inference and parity check for the readers |
| `kaggle_stack/add_ours.py` | builds the forked stack notebook with our readers added |
| `DOCUMENTATION.md` | full project documentation (method, results, reproduction, lessons) |
| `NOTES.md` | detailed run log and analysis of the public stacks |
| `confusion_matrix.csv` | iteration-1 confusion matrix |

## Credits
- **Data:** Open Pool by Carrot Production.
- **Public 0.704 stack:** lightningv08, building on Alex Chilton, lumibee and others.
- **Grouped-CV analysis and baselines:** Adisak Sukul and GoPain (discussion 750390).
- **Multi-view TF-IDF ideas:** Ahmed Berat Özer ("Teach the Model, Remove the Prompt").
