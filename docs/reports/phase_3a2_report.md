# Phase 3A-2 — report

**Repo:** `C:\Users\Admin\Desktop\anomaly`, branch `benchmark-upgrade`, `venv_train`
for all GPU work. No full training; one 20-step smoke run, adapter discarded. No
teacher calls. GPU idle before timing (618 MiB used by the desktop, no Ollama model
resident, `ollama ps` empty).

---

## 1. Summary

* **The answer shrank by a quarter and nothing was lost.** The student now emits
  five fields; subsystem, mitigation, table choice, follow-ups and the clarifying
  question are derived in code from the KB and router. Answer tokens go from p50
  **402 → 296** (mean 403.8 → 298.9, **−26%**), and a rendered report from a slim
  answer still carries the KB's own mitigation text and passes the response-layer
  validators.
* **Decode is 5.1× faster batched than forward-only, and the cache is the small
  half of it.** forward-only 33.9 s/item → kv-cache 16.6 → batch16 **6.68 s/item**
  (256-token cap, 20 val items). The remaining cost is the dummy 448×448 image the
  model forces on every step: `image_flags=None` is rejected, so each token re-runs
  the vision tower.
* **Greedy output is not bit-identical between the cached and uncached paths.**
  First-step argmax agrees and the distributions agree to cosine 0.9995, but over
  256 steps the trajectories diverge (1 of 5 short sequences identical). This is
  bf16 accumulation noise, not a plumbing bug, and it is recorded because the live
  path has the same silent fallback.
* **Closed-set scoring costs 2.13 s/item at batch 16**, against 6.68 s/item to
  generate the same answer. That is the whole evaluation budget.
* **Training projects to 2.8 h per config** (20 steps measured, 28.0 s/step,
  17.3 GB peak), so 5 runs ≈ 14 h and 9 runs ≈ 25 h; evaluation adds ~3.6 h per
  run. Full Phase 4 for five runs is ≈ **32 h of GPU**, for nine ≈ **57 h**.
* **The smoke run's loss check is exact**: model 1.2969 vs hand-computed 1.2933
  (bf16 rounding), 349.2 supervised tokens per example against 1,451.8 prompt
  tokens.

## 2. Changes

| commit | contents |
|---|---|
| `48ce803` | slim target: `llm/benchmark/hydrate.py`, exporter guards, `targets.py` single-sources the mitigation, token measurement |
| `0662d0e` | `scripts/build_eval_exports.py`, `llm/benchmark/decode_bench.py`, `llm/benchmark/label_scorer.py` |
| *(this turn)* | batched label scoring, `--batch`; `_step` fallback ladder shared by both GPU paths |

## 3. Slim target

| measure | before | after |
|---|---|---|
| answer keys | 11 | **5** |
| answer tokens p50 / p90 / max | 402 / 460 / 561 | **296 / 350 / 475** |
| answer tokens mean | 403.8 | **298.9 (−26%)** |
| full sequence p50 | 1,510 | **1,406** |
| supervised tokens/example (smoke batch) | 402 | **349.2** |

Guards updated and tested: `DERIVED_AT_RENDER` (six keys) must not appear in an
answer, a missing field is an error rather than a default, and a label outside the
`FaultID` enum is refused. `tests/test_slim_target.py` renders a report from a slim
answer plus a fixture event, asserts the KB's `recommended_action` appears in the
markdown, and checks `mitigation_for(...) == build_target(...)["mitigation"]` for
four labelled faults and both abstention verdicts — one definition, not two.

## 4. Fold exports

| file | rows | held out |
|---|---|---|
| `train.jsonl` (Protocol A) | 1,919 | — |
| `train_fold_1_cooling_loop_holdout.jsonl` | 1,562 | 4, 11, 15 |
| `train_fold_2_valve_sticking_family.jsonl` | 1,681 | 14, 15 |

Whole items are dropped when their fault is held out, and a kept fault's ranking
has held-out entries removed (fault 4's KB differential names fault 15). The menu
still lists every class, so nothing is hidden from the model; what is absent is any
training *answer* that ranks a held-out class. Asserted per row.

## 5. Decode speed

20 validation items, greedy, 256-token cap, one process per mode:

| mode | s/item | tok/s | peak reserved | step form |
|---|---|---|---|---|
| `forward_only` (no cache) | **33.92** | 6.61 | 33.5 GB | — |
| `kv_cache` | **16.58** | 13.83 | 4.9 GB | dummy image |
| `merged` (LoRA merged into a copy) | 16.91 | 13.57 | — | dummy image |
| `batch4` | 8.43 | 23.00 | — | dummy image |
| `batch8` | 7.43 | 29.66 | — | dummy image |
| `batch16` | **6.68** | 33.50 | — | dummy image |

Notes that matter:

1. **`merged` measures the base model**, because no dataset-v2 adapter exists yet.
   The merge step itself is exercised by `PeftModel.from_pretrained(...).merge_and_unload()`
   on a *second* model instance — the adapter on disk is never modified — but its
   effect cannot be measured until training produces one. Re-running this row after
   the first training run is a two-minute job.
2. **`image_flags=None` is rejected** (`'NoneType' object has no attribute
   'squeeze'`) exactly as `llm/inference.py`'s fallback assumes, so every decode
   step carries a dummy image through the vision tower. That is why the cache
   buys 2.0× and not the 50× one might expect. Removing the image from the step
   path is the single largest remaining speedup and is not attempted here.
3. **Identity check: 1 of 5 short sequences identical** between `forward_only` and
   `kv_cache`. The first cached step's argmax and probability vector match the
   full-sequence forward (max |Δp| 0.009, cosine 0.9995), so this is bf16 noise
   amplified by greedy feedback rather than a position/mask bug. Consequence: no
   number may be produced with `forward_only` and compared against a number
   produced with the cache.

## 6. Label-scoring path

Single prefill after `{"fault_id": "`, then each of the 16 menu labels teacher-forced
through the cache; softmax over summed log-probabilities. Not generation, no parser.

| batch | s/item (40 val items) |
|---|---|
| 1 | 6.06 |
| 4 | 2.92 |
| 8 | 2.34 |
| 16 | **2.13** |

**Generation agreement: not established.** With the untrained base model and a
64-token cap, 18 of 20 generations contained no menu label at all, so the
comparison has no signal yet (0/20 agree, 2 comparable). The check is implemented
and must be re-run on the first trained adapter before either path is trusted for
reporting. Accuracy on the base model is 0.35–0.50 depending on batch size — the
spread is bf16/padding non-determinism at the margins and is itself worth knowing
before any batch-size choice is frozen into a report.

## 7. Evaluation budget

Per component, at the measured batch-16 rates (label 2.13 s/item, generation
6.68 s/item):

| component | rows | label scoring | full generation |
|---|---|---|---|
| `A/val` | 834 | 0.49 h | 1.55 h |
| `A/test` | 3,168 | 1.87 h | 5.88 h |
| `fold_1/val` | 637 | 0.38 h | 1.18 h |
| `fold_1/test` | 600 | 0.35 h | 1.11 h |
| `fold_1/test_seen` | 2,528 | 1.50 h | 4.69 h |
| `fold_2/val` | 687 | 0.41 h | 1.27 h |
| `fold_2/test` | 400 | 0.24 h | 0.74 h |
| `fold_2/test_seen` | 2,728 | 1.61 h | 5.06 h |
| **per config** | | **2.2–2.4 h** | **7.0–7.4 h** |

Recommended shape, so the number is affordable: **label scoring on every
component** for the headline metrics, **generation on a stratified 100-item
subsample per component** for schema validity only, and **checkpoint selection on
a 300-item validation subsample** (10.7 min per evaluation, ~4 evaluations per
run) with the full `A/val` scored once at the end.

| plan | training | evaluation | total |
|---|---|---|---|
| 5 runs (A × 3 seeds, each fold × 1) | 14.0 h | ~4.2 h | **≈ 18 h** |
| 9 runs (A × 3, folds × 3) | 25.2 h | ~7.2 h | **≈ 32 h** |

Full generation on every component would make the 5-run plan **≈ 60 h**, which is
the honest reason to keep generation to a subsample.

## 8. Smoke run (slim target)

| measure | value |
|---|---|
| steps | 20 (batch 4 × accum 4) |
| time | 560.7 s → **28.03 s/step** (20–31 s/step observed across the run) |
| loss | 1.4062 → **0.6480** |
| peak GPU | **17.3 GB** |
| hand cross-entropy vs trainer | 1.2933 vs 1.2969 (Δ 0.0035, bf16) |
| supervised / masked tokens per example | **349.2** / 1,451.8 (19.4% supervised) |
| projection | 120 steps/epoch, 360 steps, **2.8 h per config**, 25.2 h for 9 |
| generation after 20 steps | 10 items, 28.88 s/item, 3.32 tok/s at 96 tokens (forward-only, unmerged; outputs not scored) |

The adapter is discarded; nothing was written to any adapter directory.

## 9. Tests

| suite | result |
|---|---|
| `tests/test_slim_target.py` | 15 passed |
| `tests/test_to_sft.py` | 12 passed |
| `tests/test_eval_exports.py` | 9 passed |
| `tests/test_label_scorer.py` (new) | 6 passed |
| `tests/test_to_sft_followup.py` (new, Stage B) | 11 passed |
| full suite, app environment | **463 passed** |

## 10. Gate table

| gate item | verdict | evidence |
|---|---|---|
| Slim-target tokens and rendering equivalence shown | **PASS** | §3, §3 equivalence tests |
| Fold exports counted with asserts | **PASS** | 1,562 / 1,681 rows, per-row asserts |
| Speed table (a)–(d) with identity check | **PASS (with finding)** | §5; identity 1/5, explained |
| Label scorer agreement and cost | **PARTIAL** | cost measured; agreement not established on an untrained base, §6 |
| Eval budget | **PASS** | §7 |
| Smoke metrics | **PASS** | §8 |
| App and venv_train tests pass | **PASS** | 463 passed in the app env; the GPU runs import and exercise the venv stack |

## 11. Deviations

1. **`merged` is the base model**, because no adapter exists yet (§5.1).
2. **Generation agreement is not established** (§6). The check exists; its result
   means nothing until a trained adapter exists, and it is on the list for after
   the first training run.
3. **Batch sizes above 16 were not tried**; 16 is the largest the 24 GB card took
   comfortably alongside the 33.5 GB reserved peak seen in the forward-only run.
4. **The smoke run's per-step time is noisy** (20–31 s), so the 2.8 h projection
   carries ±20% and excludes evaluation/save overhead.

## 12. Open questions

1. **Can the dummy image be removed from the decode step?** It is worth up to ~3×,
   and it is the difference between an 18 h and a 55 h Phase 4. It needs a change
   in how `llm/inference.py` calls the model, which is out of scope here.
2. **Which decode path is normative?** The cache is 2× faster and numerically
   different. Proposal: the cache is normative for evaluation, because the label
   scorer (and therefore the headline metric) uses it; `forward_only` stays as the
   baseline row and is never mixed into a reported number.
3. **Batch size for scoring.** Accuracy moved 0.35 → 0.50 across batch sizes on the
   base model. Fix the batch size in the protocol before the first real number.
