# Pre-training close-out (Phase 2B final / 3A-2 prep)

**Repo:** `C:\Users\Admin\Desktop\anomaly`
**Branch:** `benchmark-upgrade` (HEAD `296c444`, plus this report)
**Environments:** app env `python` (3.11), GPU env `venv_train`
**GPU:** NVIDIA RTX A5000, 24,564 MiB, driver 596.71, max SM clock 2,100 MHz
**Date:** 2026-10-31
**Training performed:** none. One 20-step smoke run on the mixed file, adapter discarded.

---

## 1. Out-of-scope refusals

**What failed.** 50 of the 185 rejected follow-up turns carried `refusal_missing`
(a refusal that does not decline). Five raw answers, straight from
`outputs/benchmark_v2/followup/teacher_cache.jsonl`:

| # | Raw teacher answer |
|---|---|
| 1 | "The condenser cooling water valve is not mentioned in the incident context. The detected issue involves A_Feed_Stream1, Stripper_Temperature, Stripper_Steam_Flow, and related manipulated variables…" |
| 2 | "The condenser valve is not mentioned in the incident context. The context describes changes in A_Feed_Stream1, Stripper_Temperature, Stripper_Steam_Flow, A_Feed_Flow, and Stripper_Steam_Valve, but there is no…" |
| 3 | "The context does not provide information about a purge mechanism or how it affects composition. The incident involves changes in A_Feed_Stream1, Stripper_Pressure, Stripper_Temperature…" |
| 4 | "The purge valve is not mentioned in the incident context. The context describes the stripper steam valve and other variables, but there is no reference to a purge valve or its behavior." |
| 5 | "The incident context does not mention valve stiction or oscillations. The observed trends show a decreasing trend for A_Feed_Flow (XMV_44) and A_Feed_Stream1 (XMEAS_1)…" |

43 of the 50 used `the incident context does not provide … for <sensor>`; the rest used
`is not mentioned` / `does not mention`.

**Was the prompt missing the guidance?** No. `llm/followup/teacher.py:125-129` already
injects the code-side decision — `- this question cannot be answered as asked: <reason>`
and `- answerable alternatives you may point to: <question (intent); …>` — and
`teacher.py:51-52` instructs the model to "hold, say plainly that you cannot confirm it,
then point to an answerable alternative from the list supplied". No prompt change was
needed or made.

**Cause and fix.** The rule's vocabulary was narrower than plain English. `llm/followup/filters.py`
now recognises an absence family (`ABSENCE_SUBJECTS` / `ABSENCE_VERBS` / `BARE_ABSENCE`,
lines 84-95, 206-212) scoped by subject, so "the feed does not contain enough reactant"
still reads as a physical claim and is still rejected. Commit `392ed40`, tests in
`tests/test_followup_filters.py` both ways.

**Acceptance before/after.**

| Pass | Turns | Accepted | Rate |
|---|---|---|---|
| Full train run, rule before the fix | 2,562 calls | 2,377 | 92.8% |
| Re-check build, rule after the fix (`outputs/benchmark_v2/_oos_recheck`) | 19 | 19 | 100% |

**Gap, stated plainly.** The re-check build has no "failed-only" selector: it sampled
8 dialogues (19 turns, 3.588 s/call) from the same seeded stream, not the 50 failed items,
and the corpus was **not** rebuilt — `train.jsonl` still holds 2,377 accepted / 185 rejected.
So the honest statement is *the fixed rule accepts refusals the old one dropped on a 19-turn
sample*, not *50 items were recovered*. Recovering them needs a rebuild (~318 fresh calls at
the measured 3.2 s ≈ 17 min) — see pending decision D1.

---

## 2. Stage B confirmations

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 2a | 2C merged | **VERIFIED** | merge `caf5e32` (parents `4124cde`, `01f617a`), `llm/followup/` present (`cases.py`, `filters.py`, `leak.py`, `paraphrases.py`, `provenance.py`, `teacher.py`). `llm/response/router.py` last touched after the merge (`84cab4f`), so the merged router is what runs. |
| 2b | Mixed-row shape | **VERIFIED** | Report row keys `{item_id, system, question, answer}`; `answer` parses to the slim target `{fault_id, verdict, ranked_hypotheses, physical_reasoning, narrative}` — no mitigation, subsystem, tables, suggestions or clarifying question. Dialogue row has the same four keys (`item_id` = `FD-train-00746#0`), with the earlier turns folded into `question`: one row per assistant turn, because `llm/dataset.py` masks a single assistant span. |
| 2c | Val/test follow-up sets | **VERIFIED** | val 608 turns, test 2,130 turns; `answer: null`, `answer_source: structural`, `teacher_model: null`, expectation carried in `route` / `answerability` / `refusal` / `clarification_trigger`. Zero teacher prose. 14 (val) and 64 (test) deterministic template turns are marked accepted. |
| 2d | Accepted counts | **CORRECTED** — see below | |
| 2e | `cost_estimate` | **FIXED** | `scripts/refresh_followup_manifest.py` rewrote the block: `measured {fresh_calls_this_build: 318, cache_hits_this_build: 2,244, seconds_per_fresh_call: 3.228, cache {n: 2,747, mean_seconds: 2.732}, serial_hours_if_all_fresh: 2.3}`; the 10.9 s constant survives only under `projection_assumption`. Test updated in `tests/test_followup_dataset.py`. |

**2d, the numbers as they reconcile.**

| Quantity | Value | Where it comes from |
|---|---|---|
| Teacher calls (train) | 2,562 | manifest `acceptance.teacher_turns` |
| Rejected | 185 | 92.8% acceptance → 2,377 accepted |
| Accepted turns written to `train.jsonl` | 2,445 | +68 deterministic template turns, which are not teacher calls |
| Rejection codes | 119 + 50 + 14 + 11 = **194** | 9 items carry two codes (`near_duplicate` plus a content code); codes are not a partition |
| Dialogue / turns | 1,139 / 2,630 | 2630 − 2562 = 68 template turns |

* **2,432** appears in no artifact in the tree and is not reproducible from the cache,
  the manifest or `train.jsonl`; it is superseded by **2,377 / 2,562 (92.8%)**.
* Pilot figures, with denominators: **188/191 = 98.4%** (fresh pilot, before the filter
  fixes) and **184/191 = 96.3%** (same 191 turns after the fixes, 4 of them deterministic).
  **90.6%** has no producing artifact either; 90.6% would be 173/191, which no file records.
  The two measured pilot figures are the ones above; 191 is the denominator in both.

---

## 3. Token share and optimizer steps (real tokenizer, `outputs/benchmark_v2/sft/token_share.json`)

Mixed file: **2,878 rows**, `max_sequence_length` 4096, batch 4 × accumulation 4 → effective 16.

| Component | Rows | Prompt mean (p99) | Supervised mean (p99) | Supervised total | Share of supervised |
|---|---|---|---|---|---|
| Report | 1,919 | 1,380.2 (1,434) | 299.6 (396) | 574,980 | **88.69%** |
| Dialogue | 959 | 752.5 (929) | 76.4 (134) | 73,309 | **11.31%** |

Dialogue rows by case type:

| Case | Rows | Prompt mean | Supervised mean | Supervised total |
|---|---|---|---|---|
| valve_actuator | 211 | 740.1 | 71.4 | 15,059 |
| causal_why | 223 | 751.7 | 73.1 | 16,310 |
| mechanism | 130 | 755.5 | 76.3 | 9,925 |
| hypothesis_comparison | 121 | 766.7 | 78.4 | 9,481 |
| ambiguous_question | 79 | 757.8 | 88.4 | 6,980 |
| false_premise | 59 | 751.7 | 69.2 | 4,085 |
| clarification_reply | 48 | 749.2 | 93.4 | 4,485 |
| repeated_question | 47 | 754.2 | 75.0 | 3,523 |
| template_sample | 32 | 766.2 | 87.1 | 2,787 |
| out_of_scope | 9 | 742.9 | 74.9 | 674 |

Truncation: **0** answers cut by the length budget, **0** prompts at the cap.

**Steps:** 180 per epoch, **540 for 3 epochs** at effective batch 16 (one optimizer step per
accumulation window).

**Accepted train turns not used in the mix:** 2,377 accepted → 959 used, **1,418 unused**
(valve_actuator 660, mechanism 342, causal_why 210, ambiguous_question 63,
hypothesis_comparison 55, false_premise 48, clarification_reply 28, repeat 27, out_of_scope 11).
`token_share.json` reports `template` as `accepted 0 / used 26 / unused -26`; the 26 is the
deterministic template-sample turns, whose prose is written by code rather than bought from
the teacher, so the "unused" arithmetic goes negative for that case. It is a reporting
artifact of counting template turns against a teacher-accepted denominator, not a negative
population.

---

## 4. Label-scoring invariance

**Before the fix** (`label_batch_invariance_before_fix.json`, 40 val items):

| Batch | Accuracy | s/item | Argmax vs batch 1 | max abs Δp |
|---|---|---|---|---|
| 1 | 0.475 | 6.034 | — | — |
| 4 | 0.400 | 2.903 | 36/40 | 0.731 |
| 8 | 0.425 | 2.337 | — | — |
| 16 | 0.450 | 2.135 | — | — |

Two causes, both real bugs:

1. With `position_ids=None` this `transformers` numbers tokens from `cache_position`, so
   left-padded rows inherited the longest prompt's width.
2. The prefill's `DynamicCache` is mutated in place and was shared, so each label's steps
   began at the previous label's length.

**Fix** (`llm/benchmark/label_scorer.py`, commit `296c444`): positions derived from the
attention mask, an explicit causal+padding mask built per call (passing `position_ids`
alone sends `create_causal_mask` down its packing path), and a cache copy per label.

**After the fix** (`label_batch_invariance_after.json`):

| Batch | Accuracy | s/item | Argmax vs batch 1 | max abs Δp | p50 abs Δp |
|---|---|---|---|---|---|
| 1 | 0.500 | 6.029 | — | — | — |
| 4 | 0.500 | 2.922 | 40/40 | 0.008324 | 0.002759 |
| 8 | 0.500 | 2.308 | 40/40 | 0.007364 | 0.002976 |
| 16 | 0.500 | 2.104 | 40/40 | 0.007218 | 0.002850 |

Accuracy is batch-invariant at 0.500, argmax is identical everywhere, and the residue
(0.008) is bf16 nondeterminism, not a mask error. **Scoring batch size 16 is recorded in
`docs/PROTOCOL.md` (dated 2026-10-31).** `llm/inference.py` was not modified.

---

## 5. Decode without the vision tower (`outputs/benchmark_v2/sft/decode_lm_direct.json`)

`model.language_model` called directly with `inputs_embeds` from text tokens and a KV cache,
bypassing `extract_feature`.

| Path | s/item | Speedup | Argmax agreement | min cosine | max abs Δ |
|---|---|---|---|---|---|
| Wrapper, cached (`model(...)` with a dummy image) | 2.160 | 1.00× | 20/20 | — | — |
| Language model, direct | 0.489 | **4.42×** | 20/20 | 1.0 | 0.0 |

The probabilities are identical, not merely close: the wrapper spends its time on the
vision tower's forward pass for a dummy image that contributes nothing to a text prompt.

**Generation with a 512-token cap** (8 items, both paths):

| Path | s/item | Mean tokens | Complete JSON in 512 | Hit the cap |
|---|---|---|---|---|
| Wrapper, cached | 21.158 | 294.6 | 5/8 (62.5%) | 2 |
| Language model, direct | 14.197 | 293.1 | 5/8 (62.5%) | 1 |

Peak GPU 13.34 GiB. Two caveats stated rather than buried: generation ran on **8** items
not 20 (time), and **62.5% completion at 512 tokens** means a 512-token cap is too small to
evaluate the JSON contract by generation — either raise the cap (768-1,024) or lean on the
label-scoring path, which is already the recorded closed-set evaluation.

---

## 6. Fold test exports

Held-out faults' items from the `train` dataset (runs 1-300, used by no fold's training)
are appended to `fold_N/test` and marked `source_dataset: train`, with an assert that none
sits in that fold's train keys (`scripts/build_eval_exports.py`, commit `fcfc2da`).

| Export | Component | Events rows | Train-range rows | Total |
|---|---|---|---|---|
| `test_eval.jsonl` | A/test | 3,168 | 0 | **3,168** |
| | fold_1/test | 600 | 357 | **957** |
| | fold_1/test_seen | 2,528 | 0 | **2,528** |
| | fold_2/test | 400 | 238 | **638** |
| | fold_2/test_seen | 2,728 | 0 | **2,728** |
| `val_eval.jsonl` | A/val | 834 | 0 | **834** |
| | fold_1/val | 637 | 0 | **637** |
| | fold_2/val | 687 | 0 | **687** |

`test_eval.jsonl` = 10,019 rows, `val_eval.jsonl` = 2,408; Protocol A rows equal the dataset
item counts (3,168 and 834), and no row carries prose. PROTOCOL now states the actual test
n: **fold_1/test = 957, fold_2/test = 638**.

---

## 7. Smoke step time, with telemetry

20 optimizer steps on `train_mixed.jsonl` (slim target), 10-second `nvidia-smi` sampling
(`outputs/benchmark_v2/sft/smoke_telemetry.csv`, 706 samples):

| Metric | Value |
|---|---|
| s/step | **14.31** (286.3 s / 20) |
| Peak GPU memory | 17.02 GiB |
| Loss step 1 → 20 | 1.2931 → 0.6494 (trainer mean 0.8511) |
| Hand CE vs trainer step-1 loss | 1.3040 vs 1.3047, Δ 6.7e-4 |
| Supervised / masked prompt tokens (one batch) | 1,137 / 6,067 (fraction 0.158) |
| SM clock | median **1,658 MHz**, min 1,455, max 1,800 |
| Temperature | max **80 °C** |
| Power | max **227 W** |
| Projection | 540 steps = **2.15 h/run**, five runs **10.7 h** |

**17.0 vs 28.0 s/step — how far the data goes.** The 3A-2 smoke ran on `train.jsonl`
(report rows only: 1,452 prompt + 349 supervised ≈ 1,801 tokens/row), this one on the mixed
file (1,521 mean, because 959 dialogue rows are ~829 tokens and pull the batch maximum
down). Step cost follows the longest row in the batch of 4, so batch shape explains roughly
17% of the drop. It does not explain 49%. The 3A-2 log drifted from 20 to 31 s/step *within
its own 20 steps*, which is throttling or contention, and no counters were recorded then —
so the split between "shorter batches" and "better clocks" cannot be recovered from existing
data, and I will not invent it. Telemetry today shows 80 °C, close to the point where an
A5000 starts trimming.

**Budget consequence:** plan on 14.3 s/step with 10-20% headroom → **2.4-2.6 h per run**,
**12-13 h for five runs**.

---

## 8. Rebuild and checks

| Check | Result |
|---|---|
| Eval exports rebuilt | val 2,408 rows, test 10,019 rows; Protocol A counts match the dataset; no prose fields |
| Manifest hashes | rehashed 5 files (`scripts/refresh_followup_manifest.py`); measured cost block written |
| Report-prompt open-set scan (`scripts/scan_open_set_leaks.py`) | 1,971 unique items, 487 abstention records (481 accepted), **0** accepted answers naming the removed fault |
| Follow-up split overlap | train/val/test dialogue-id overlap **0 / 0 / 0** (1,139 / 261 / 927) |
| Dataset splits (`scripts/check_splits_v2.py`) | **PASS** — unique keys, no train/val/test overlap, fault-disjoint where required, recorded sha256 matches |
| Tests, app env | **483 passed** (6.95 s) |
| Tests, `venv_train` | **483 passed** (7.14 s) |

---

## Gate

| Gate item | Status | Evidence |
|---|---|---|
| out_of_scope acceptance before/after | **PASS (sample)** | 0/50 before (50 `refusal_missing`), 19/19 after on the re-check build; prompt already carried the decision; corpus not rebuilt (§1) |
| Confirmations answered | **PASS** | 2a-2e in §2; 2,432 and 90.6% shown unreproducible, corrected figures given |
| Token shares and step counts | **PASS** | §3: 11.31% dialogue share, 180/540 steps |
| Scoring invariance result and fix | **PASS** | §4: 0.731 → 0.008 max Δ, 36/40 → 40/40 argmax, batch 16 recorded |
| Decode comparison, 512 completion | **PASS** | §5: 4.42× with identical probabilities, 62.5% complete JSON |
| Fold export counts | **PASS** | §6: fold_1/test 957, fold_2/test 638, asserts in place |
| Step-time explanation | **PASS (bounded)** | §7: batch shape ~17% measured, remainder throttling and unrecoverable without counters |
| Checks clean, tests pass | **PASS** | §8 |
| No training | **PASS** | one 20-step smoke, adapter discarded |

| Stage B gates carried forward | Status |
|---|---|
| Merge clean | PASS (`caf5e32`) |
| Pilot ≥ 80% | PASS (96.3% / 98.4%) |
| Full-run acceptance and drops by case | PASS (92.8%) |
| Ratio 2:1 achieved | PASS (1,919 : 959 = 2.001) |
| Leak / split / open-set clean | PASS |
| Audit sheet written | PASS (100 rows) |

---

## Pending decisions

**D1 — rebuild the follow-up corpus for the refusal fix?** 185 turns (7.2%) were dropped;
up to 50 of them (`refusal_missing`) are expected to pass under the fixed rule. A rebuild
resumes from the cache and needs ~318 fresh calls at 3.2 s ≈ **17 min**. Without it the
prose the model never saw is 7.2% of the dialogue mix. My recommendation: rebuild, because
it is cheap, deterministic by seed, and removes a known false-negative class from the
training corpus before any run is spent.

**D2 — generation cap for evaluation.** 512 tokens completes 5/8 JSON answers. Either raise
the cap to 768-1,024 for the stratified generation subsample (cost grows linearly in tokens,
~+25-50% of the 21 s/item), or keep 512 and treat label scoring as the only scored path with
generation reported as a schema-validity check. My recommendation: 768, recorded in PROTOCOL
before any model result exists.

**D3 — training budget approval.** 540 steps/run at 14.31 s/step = 2.15 h; five runs ≈ 10.7 h,
plan 12-13 h with headroom. Not started; the 20-step adapter is discarded.
