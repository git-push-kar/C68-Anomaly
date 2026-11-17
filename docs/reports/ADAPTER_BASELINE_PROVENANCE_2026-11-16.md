# Adapter baseline provenance — 2026-11-16

**Question.** The 497-item diagnosis table quotes an adapter row. Two values exist in the artefacts —
**0.8330 (414/497)** and **0.7646 (380/497)** — and earlier notes described the difference as a page-vocabulary
effect. This note traces both to their producer and records what the pipeline actually does.

**Answer.** Both are **label-scoring** outputs from `outputs/tep_rca_v2_protocolA_seed42`, on the same
`val.jsonl` prompts, from **different checkpoints**: 0.8330 is checkpoint **300**, 0.7646 is checkpoint **400**.
The vocabulary fix did not touch adapter inputs. A third number, the **48 errors** used in the truncation
analysis, is a **free decode** at checkpoint **500** and is a different instrument again.

---

## 1. The pipeline, in four steps

| step | file | what it does | output |
|---|---|---|---|
| 1 | `llm/benchmark/label_scorer.py` | Closed-set scoring: one prefill per item, then P(candidate) read at the answer prefix `{"fault_id": "` and terminated with `",` (one token, id 497). Normalises over the menu ids plus `UNCLASSIFIED_OPEN_SET`; `pred` = argmax, `probs` = the vector, `probability` = its max. No generation, so no parse step and no truncation risk. | rows with `pred`/`probs`/`probability` |
| 2 | `scripts/eval_protocolA.py` | Swaps each LoRA checkpoint (`checkpoint-100 … checkpoint-540`) into one wrapped model, scores a 300-item stratified validation subsample, selects by **macro-F1** (accuracy, then later step, as tie-breaks), writes `preds_val300_step<step>.jsonl`, then re-scores the full 834-row validation split with the winner | `preds_val_full_<step>.jsonl`, `checkpoint_table.json`, `selected_checkpoint.json` |
| 3 | `scripts/evidence_classifier.py` | Filters the adapter file to the menu variant and drops the unobservable items (faults 3 and 9), so the adapter is scored on exactly the items the feature classifiers are scored on | `preds_adapter_val503.jsonl` (497 rows) + `summary.json` |
| 4 | `scripts/classifier_contrasts.py` | Reads that file via `--adapter-preds`, pairs it with the hybrid and feature-model predictions, computes accuracy, CI and exact McNemar contrasts | `classifier_contrasts.json/.md` |

Step 3 is why the file has a `probs` vector: it is a label-scoring matrix, not a decoded JSON report. A free-decoding
adapter cannot emit probabilities, which is what made the file look unproducible in an earlier note; the producer
was in the repository all along, one directory upstream.

## 2. Which file produced which number

| number | upstream artefact | checkpoint | rows | selection |
|---|---|---|---|---|
| **0.8330** (0.7977–0.8632) | `outputs/benchmark_v2/eval/preds_val_full_300.jsonl` → published `evidence_classifier/preds_adapter_val503.jsonl` | **300** | 414/497 | **selected**: macro-F1 0.8720 on the 300-item subsample (accuracy 0.8833) |
| **0.7646** (0.7253–0.7998) | `outputs/benchmark_v2/eval/preds_val_full_400.jsonl` → `vocabfix/evidence_classifier/preds_adapter_val503.jsonl` | **400** | 380/497 | not selected; accuracy 0.8533 on the subsample |
| 48 errors | `outputs/analysis/phase3a/instrument_check/preds_decode_step500_full.jsonl` (**free decode**, `prompt_budget.py`) | 500 | 48/497 | used only for the truncation/feature-visibility argument in `g1_fit_signal_report.md` |

Verified by recomputation: filtering each full-val file to the same 497 menu, non-unobservable item ids gives
**414** correct for `preds_val_full_300.jsonl` and **380** for `preds_val_full_400.jsonl`. The checkpoint table
(`checkpoint_table.json`) records 0.8833 / 0.8533 / 0.9033 on the 300-item subsample for steps 300 / 400 / 500.

## 3. Why the retrained table carried 400

`scripts/evidence_classifier.py` has `--adapter-preds` defaulting to
`outputs/benchmark_v2/eval/preds_val_full_400.jsonl`, while `selected_checkpoint.json` records step **300** as the
macro-F1 winner (step 500 and 540 score higher on accuracy at 0.9033 but lose on macro-F1). The `vocabfix`
summary was regenerated with that default, so its adapter row is checkpoint-400. Nothing about the vocabulary fix
changes adapter predictions: the fix was in the classifier's feature parser (`XMV_1..11` → `XMV_42..52`), while
the adapter reads the rendered prompt text, which is unchanged in `val.jsonl`.

## 4. Open decision — which number is "the adapter"

- **Quote the selected checkpoint (300):** 0.8330 (0.7977–0.8632), 414/497, abstention precision 0.8723 / recall
  0.6560. This is the checkpoint `selected_checkpoint.json` picked, and the one the older `p2` table used.
- **Quote the file the pipeline points at (400):** 0.7646 (0.7253–0.7998), 380/497, abstention precision 0.9878 /
  recall 0.6480. This is what the retrained table currently quotes, labelled checkpoint-400.

The current documents use the second and label it; switching to the first is one flag:

```
python scripts/classifier_contrasts.py \
  --hybrid-preds outputs/analysis/phase3a/hybrid/vocabfix/preds_protocolA_val.jsonl \
  --evidence-dir outputs/analysis/phase3a/hybrid/vocabfix/evidence_classifier \
  --adapter-preds outputs/benchmark_v2/eval/preds_val_full_300.jsonl \
  --out-json outputs/analysis/phase3a/hybrid/vocabfix/classifier_contrasts.json \
  --out-md   outputs/analysis/phase3a/hybrid/vocabfix/classifier_contrasts.md
```

Either way the paper's conclusion is unchanged: the adapter is **16.5 points** behind the nearest rival
(LightGBM 0.9980) as checkpoint-300, or **23.3 points** as checkpoint-400, and the hybrid-vs-adapter contrast is
dominated by open-set items and `NORMAL_TRANSIENT`.

## 5. Reproduction

```
# 1. score every checkpoint on the subsample, select by macro-F1, score full val
venv_train/Scripts/python scripts/eval_protocolA.py

# 2. filter to the 497 menu items (pin the checkpoint here)
venv_train/Scripts/python scripts/evidence_classifier.py \
    --adapter-preds outputs/benchmark_v2/eval/preds_val_full_300.jsonl   # or _400

# 3. build the table
python scripts/classifier_contrasts.py --adapter-preds <that preds_adapter_val503.jsonl> ...

# 4. the free-decode instrument, for comparison only (not a reproduction of the table)
venv_train/Scripts/python scripts/decode_adapter_val.py --out ...   # 19-40 s/item, resumable
```

Artefacts under `outputs/` are gitignored; the scripts and this note are versioned, so the chain is reproducible
without the files.

## 6. What this note corrects

- **"The 0.8330 row was measured on pre-vocabulary-fix pages."** Wrong. Both rows come from the same `val.jsonl`
  prompts; the difference is checkpoint 300 vs 400. Documents that said this have been corrected.
- **"The adapter files are not adapter output / provenance failure."** Too strong. They are adapter output under a
  different *instrument*: closed-set label scoring rather than free decoding. The instrument is traced; the only
  open item is which checkpoint the paper should quote.
- **"48 items the adapter gets wrong"** in `g1_fit_signal_report.md` refers to the step-500 **free decode**, not to
  either label-scoring baseline; the text now says so.
