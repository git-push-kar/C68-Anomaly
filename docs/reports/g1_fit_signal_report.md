# 3B-6 G1 DECISION RECORD, 3B-5 CORRECTIONS, FIT-SIGNAL FEASIBILITY

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade` · **Environment:** app env (CPU) for everything; `venv_train` used only to load the training tokenizer
**Date:** 2026-11-01 · **Scope:** PROTOCOL §18 — the G1 decision as a record, the corrections owed to §17's report, whether the adapter sees the evidence the classifier uses, whether a code-computed evidence-fit score can carry a grounding verdict, and a shuffled-menu harness. **No adapter training. No test component opened. No teacher calls. No GPU work** (idle GPU confirmed at 889 MiB used, nothing launched on it).

No job was projected over 30 minutes, so no estimate was needed before starting: the longest was `scripts/hybrid_novelty.py` at **92 s** (13 surrogate refits' novelty plus two fold reproductions of §17's per-item decisions), then `scripts/prompt_budget.py` at **29 s** for all 3,714 rows through the real tokenizer. Two numbers in §4 are stated per variant and per event where the old report stated one; both corrections are in [3b5_corrections.md](3b5_corrections.md).

---

## 0. Gate table

| # | Gate item | Result |
|---|---|---|
| 1 | §18 dated and committed **before** any new number | **PASS** — commit `2de7499`, PROTOCOL only, one file, written before a single figure below existed |
| 2 | Corrections file (a-d) | **PASS** — [3b5_corrections.md](3b5_corrections.md): unobservable agreement re-split per variant (0/6 vs 6/6), "44 of 50" withdrawn with its definition, natural-error populations separated and the `FAULT_11` claim corrected, feature sources confirmed |
| 3 | Truncation counts, by class and for the 48 errors | **PASS** — §2: **0 of 3,714 rows exceed 4,096 tokens** (3,762 measurements); max 1,439; minimum headroom 2,215 tokens; the 48 step-500 errors max 1,166 |
| 4 | Feature-visibility conclusion (i / ii / iii) | **PASS** — §3: **(iii) present and unused**; nothing truncated, and the classifier's gold-class mass on the 48 errors is 50.4 % detector statistics + 28.9 % CVA residual, both printed |
| 5 | Novelty channel on surrogate, folds and validation, with false abstention | **PASS** — §4: τ 18.7678 at 4.94 % OOF false-novel; validation accuracy −1.95 pt with **0 improved / 16 worsened**; fold_1 −3.67 pt for **0** extra declines; fold_2 +2 of 23 |
| 6 | Fit-score AUROC table and feasibility verdict | **PASS (negative result)** — §5: mean paired AUROC **0.6508 < 0.80**, one pair inverted at 0.00; **not feasible**, per the threshold fixed before measuring |
| 7 | Shuffled-menu harness and invariance test | **PASS** — §6: `llm/benchmark/menu_order.py`, `--menu-seed` in `llm/benchmark/decode_bench.py`, 7 tests including "the classifier's feature vector is bit-identical under every permutation" |
| 8 | Tests in both environments | **PASS** — 565 passed (app env); 562 passed, 3 skipped (`venv_train`, no lightgbm) |
| 9 | No test data, no GPU, no teacher calls | **PASS** — no new script opens `test.jsonl` or `test_eval.jsonl`; no model loaded; no network call |
| 10 | Run-1 and earlier hybrid artefacts unchanged | **PASS** — `git status` shows new files only plus two additive edits (`decode_bench.py` gains an optional flag, default behaviour unchanged); no §16/§17 artefact rewritten |

---

## 1. The decision, recorded before it could be informed by the measurements below

§18 fixes the following, and this report implements it rather than revisiting it:

* **G1 = hybrid.** The classifier plus the menu rule diagnose; the adapter writes `ranked_hypotheses`, `physical_reasoning` and `narrative` with `fault_id` and `verdict` forced into the answer prefix. Adapter-only (§14 decode instrument, step 500) remains a **reference row**, not a candidate.
* **No Option-2 retrain** is authorised. §17.6's wrong-label census stays context; nothing here builds training data from it — and §5 of the corrections file shows that pool is 68 events, not 258.
* **Faults 3/9 are a separate stratum, scored two ways** (strict = the dataset's target; acceptable non-diagnosis = `NORMAL_TRANSIENT` or `UNCLASSIFIED_OPEN_SET`), reported side by side, neither in a headline metric.
* **Novelty is the second abstention channel**, with its threshold derived out-of-fold on train events and its rejection criteria written down before measuring (§18.6).
* **Adapter-only numbers are to be reported in KB order and shuffled order** from now on (§18.5).

---

## 2. Does the adapter see the evidence? Truncation, with the tokenizer training used

Token lengths for **every** training row and **every** validation row, through `transformers` 4.57.6 (the `venv_train` build that trained the adapter) and the base model's own tokenizer, in a chat-wrapped prompt exactly as `llm/dataset.py` builds it. `truncation_side = "right"`, so anything that did not fit would cut the **menu, instruction and schema line** — the worst possible end of this particular prompt.

| population | rows | prompt p50 | prompt max | over 4,096 | min headroom (cap − prompt − answer) | zero-budget rows |
|---|---|---|---|---|---|---|
| training, report rows | 1,919 | 1,400 | **1,439** | **0** | 2,215 | 0 |
| training, dialogue rows | 961 | 749 | 995 | **0** | 2,978 | 0 |
| validation, all rows | 834 | 1,112 | 1,167 | **0** | 2,929 | 0 |
| validation, the 48 step-500 errors | 48 | 1,135 | 1,166 | **0** | 2,930 | 0 |
| **total measured** | **3,762** measurements over **3,714** distinct rows (the 48 are a subset of validation) | — | **1,439** | **0** | — | **0** |

Per class the widest are `FAULT_12` 1,439, `FAULT_18` 1,437, `FAULT_05` 1,435, `FAULT_01` 1,434, `FAULT_07` 1,429 (`FAULT_15` 1,420); the widest prompt in the whole corpus uses **35 %** of the cap. So:

* **no row is truncated, no block is ever cut, and no row loses its supervised tokens**;
* the 48 items the adapter gets wrong are not longer than the ones it gets right (p50 1,135 vs 1,112, max 1,166 vs 1,167) — a truncation explanation for them is excluded, not merely unlikely;
* a *previously recorded* number is worth noting here because it looks like a contradiction and is not: `outputs/analysis/phase3a/token_lengths.md` reports `train_with_system` max 1,536 over 1,939 rows. That file measures `dataset/train.jsonl` (the benchmark split) with the **report** system prompt; this table measures `sft/train_mixed.jsonl` as the trainer built it. Both are far under the cap.

**This is the only finding in this report that removes a hypothesis.** Everything below is about what the evidence can support, not about whether it arrived.

---

## 3. Which evidence the classifier leans on, and where the prompt prints it

Per-group absolute SHAP mass for the **gold class** (LightGBM per-class `pred_contrib`, same estimator and seed as the recorded classifier), on the adapter's 48 menu-present errors and on a seeded 100 of its correct items:

| group | printed in the prompt as | share of gold-class mass, 48 errors | share, 100 correct |
|---|---|---|---|
| detector statistics (score, LSTM error + ratio, CVA Q + ratio, lag, slow-drift, trigger) | **in full** — 3 detector lines | **0.504** | 0.331 |
| CVA residual shares | **summarised** — top 5 variables, one line each | **0.289** | 0.347 |
| per-sensor `z` | **summarised** — the 5 sensors listed | **0.040** | 0.166 |
| per-sensor `deviation %` | summarised — same 5 lines | 0.027 | 0.088 |
| per-sensor `trend` | summarised — same 5 lines | 0.081 | 0.041 |
| sensor presence / `n_listed` | implicit — a tag not listed is absent | 0.049 | 0.022 |
| onset minute | summarised — 2–6 lines in *ONSET ORDER* | 0.009 | 0.005 |

Measured prompt shape: **exactly 5** sensor lines, **5** residual lines, **2–6** onset lines in every row.

### Conclusion: (iii) present and unused

* **(i) evidence absent or truncated — rejected.** §2 measured it: nothing is cut, and the per-sensor columns the classifier uses for *unlisted* tags carry information the adapter also has (the absence of a line). The one asymmetry is that the classifier sees all 52 tags as zero-filled columns while the prompt prints 5 lines — but the *listed* five are the same five on both sides, and the classifier's own five-line parse is what produces its features.
* **(ii) summarised too lossily — rejected as an explanation of the gap.** The classifier reads the same summarised text: its `prompt_features` parses these five lines. A lossy summary cannot give one reader an advantage over another reader of that same summary.
* **(iii) present and unused — accepted.** On the items the adapter gets wrong, **79 % of the classifier's gold-class signal is in the detector statistics and the CVA residual block** — three detector lines plus five residual lines that the adapter reads verbatim — and only 4 % is in the per-sensor `z` values. On the items it gets right, the split is flatter (0.33 / 0.35, with `z` at 0.17). The errors are therefore not cases where the discriminating evidence was missing from the page; they are cases where the adapter, holding the same three numbers, answers `FAULT_10` on 20 open-set items, and involves `FAULT_15` on one side or the other in **19** of the 48 (`NORMAL_TRANSIENT` → `FAULT_15` 5, the reverse 4, `UNCLASSIFIED` → `FAULT_15` 4, and six other 15-related swaps).

That reading is consistent with §14's finding that the label field is 3 % of the training signal: the model was mostly trained to write prose, and the detector statistics it needs to weigh are printed as three lines it can restate without using.

---

## 4. The novelty second channel: measured, and rejected by its own rule

Defined in §18.3 and applied without a tuned number. On 1,919 train events, out-of-fold and grouped by event: **τ = 18.7678**, which flags **67 of 1,356** out-of-fold menu-present seen-fault events — **4.94 %** of the 5 % budget. (Novelty recomputed for the validation rows matches the §17 stored values exactly: max |Δ| = **0.0**.)

### 4.1 Validation, 822 observable rows — before → after

| metric | §17 rule alone | with the novelty channel | change |
|---|---|---|---|
| accuracy | 0.9781 | **0.9586** | **−1.95 pt** |
| macro-F1 | 0.9708 | **0.9457** | −2.51 pt |
| abstention precision | 0.9739 | 0.9411 | −3.28 pt |
| abstention recall | 0.9933 | 0.9933 | 0 |
| false-abstention rate (all observable) | 0.0323 | 0.0753 | +4.30 pt |
| false abstention on the named-fault stratum (325) | 0.0363 | 0.0846 | +4.83 pt |
| open-set recall on the menu (125) | 1.0000 | 1.0000 | 0 |
| flagged on `NORMAL_TRANSIENT` rows | — | **0** of 47 | the precedence rule works |

**16 decisions changed, 16 of them from correct to abstention, 0 from wrong to right.** The channel flags 18 named-fault rows; two of those were already abstentions, and the other sixteen were right answers. On the one instrument where per-item truth is fully known, the channel is pure loss.

### 4.2 The held-out instruments — where the channel is supposed to earn its cost

| instrument | held-out items | §17 declines | novelty flags | of the previously **named** | new declines | kept-fault accuracy before → after |
|---|---|---|---|---|---|---|
| 13-fault surrogate (per fault, menu) | 25 each | see below | 0–25 | bounded | — | — |
| `fold_1` (held out 4, 11, 15) | 75 | 50 (66.7 %) | 27 | **0** (none were named) | **50** (no change) | 0.9580 → 0.9213 |
| `fold_2` (held out 14, 15) | 50 | 11 (22.0 %) | 5 | **2 of 23** named `FAULT_11` | **13** (26.0 %) | 0.9532 → 0.9187 |

The two fold reproductions were checked against §17's stored aggregates and **match exactly** (`declined` 50/11, `named` 0/23), so the per-item overlap above is the same run that produced those numbers, not a near-copy.

Surrogate, per removed fault (τ re-derived per refit; the gain bound is `min(named, flagged)` because the stored artefact has counts, not items):

| removed fault | τ | flags of 25 | §17 named | decline-gain bound | false-novel on kept faults |
|---|---|---|---|---|---|
| `FAULT_01` | 18.47 | 25 | 25 | **25** | 0.056 |
| `FAULT_02` | 22.18 | 25 | 11 | 11 | 0.036 |
| `FAULT_04` | 18.43 | 3 | 0 | 0 | 0.056 |
| `FAULT_05` | 19.07 | **0** | 25 | **0** | 0.056 |
| `FAULT_06` | 20.81 | 25 | 9 | 9 | 0.049 |
| `FAULT_07` | 18.77 | **0** | 13 | **0** | 0.056 |
| `FAULT_08` | 17.42 | 14 | 4 | 4 | 0.059 |
| `FAULT_10` | 18.63 | 3 | 0 | 0 | 0.059 |
| `FAULT_11` | 19.10 | 23 | 2 | 2 | 0.062 |
| `FAULT_12` | 20.03 | 13 | 0 | 0 | 0.046 |
| `FAULT_13` | 18.76 | 3 | 0 | 0 | 0.052 |
| `FAULT_14` | 18.65 | **2** | 25 | **2** | 0.056 |
| `FAULT_15` | 17.92 | 7 | 22 | 7 | 0.072 |

Two faults are the whole story: `FAULT_05` (0 flags of 25) and `FAULT_14` (2 of 25) are the surrogate's cleanest absorptions — 25 of 25 named `FAULT_12` and `FAULT_11` respectively — and they are exactly where a distance-to-training signal would have to help. It does not: those events are *close* to `FAULT_12` and `FAULT_11` events in feature space, because they are close in the plant. Novelty separates "nothing like this" from "something like this"; it cannot separate "this thing" from "the thing it resembles".

### 4.3 Verdict, from §18.6's criteria rather than from preference

| criterion | measured | triggered |
|---|---|---|
| (a) seen-fault accuracy drops > 2 pt on an instrument **without** a higher held-out decline rate on the same instrument | validation −1.95 pt, no gain (not over the bar); **`fold_1` −3.67 pt with 0 extra declines** | **YES — `fold_1`** |
| (b) fails to beat the §17 rule on all three instruments | beats it on 1 of 3 (`fold_2` +2 declines; `fold_1` 0; validation −16) | no |

**The channel is rejected.** It fails criterion (a) on `fold_1`: it costs 3.67 points of kept-fault accuracy and converts none of the 25 confidently-normal answers on the held-out faults into declines. The validation drop of 1.95 points is under the 2-point bar, and that is worth saying plainly — one instrument was saved by 0.05 of a point, and the rejection rests on a different instrument, not on a rounding of the one that flatters it. Two implementation notes on my own rule, recorded because they change how the verdict reads:

* **§18.6 was ambiguous about "instrument", and I first implemented it too narrowly** — I computed criterion (a) from validation only, which would have returned "not rejected". Reading the criterion as written ("on an instrument") makes the folds part of the test, and `fold_1` triggers it. The wider reading is the one written down before measuring; the narrow implementation was a bug, and the verdict uses the wide reading.
* The surrogate's per-fault gain is an **upper bound** and is excluded from the verdict for that reason: with per-item decisions it would be smaller (fold_2's `FAULT_11` case measured 2 of 23, while the corresponding bound is 2 — the bound is loose for the faults that name 25 of 25).

**What the hybrid keeps from this:** the §17 rule, as it stands. The open-set arm of the deployed system remains "the classifier's prediction is off the menu, or the calibrated maximum is below 0.9792", which abstains on 12 of 325 unseen-nameable items after the fact — a limitation that is now measured three ways (surrogate absorption, fold_2's 46 % naming rate, and this channel's failure to fix either).

---

## 5. Evidence-fit feasibility: negative, and informative

A code-computed fraction of eight required elements taken from the KB signature — initiating tag listed, its printed trend matching the KB's direction, its presence in the onset block, first and second cascading variables listed and matching, and the initiator's onset not later than the cascades'. No parameter, no threshold, no fitting. Computed on every pair §17's absorption table names.

| pair (true vs lookalike) | train events | fit(true) − fit(lookalike) | paired AUROC |
|---|---|---|---|
| `FAULT_01` vs `FAULT_08` | 120 | +0.010 | 0.542 |
| `FAULT_02` vs `FAULT_08` | 120 | +0.250 | **1.000** |
| `FAULT_06` vs `FAULT_08` | 120 | +0.218 | **1.000** |
| `FAULT_07` vs `FAULT_08` | 120 | +0.206 | **1.000** |
| `FAULT_08` vs `FAULT_01` | 120 | +0.022 | 0.521 |
| `FAULT_08` vs `FAULT_07` | 120 | +0.034 | 0.575 |
| `FAULT_05` vs `FAULT_12` | 119 | +0.039 | 0.576 |
| `FAULT_14` vs `FAULT_11` | 120 | **−0.125** | **0.000** |
| `FAULT_15` vs `FAULT_11` | 118 | +0.014 | 0.644 |
| **mean (9 pairs)** | 1,077 | — | **0.6508** |

* **Share where the score cannot be computed: 0.0 %.** Every fault 1–20 has a resolvable initiating tag, so the score is always defined — including for faults 3 and 9, whose initiator (`XMEAS_2`) is printed but does not move, which is precisely the unobservable case: the score *runs* and comes out near zero.
* **Verdict: not feasible.** §18.4 fixed the bar at mean paired AUROC **≥ 0.80** before measuring; 0.6508 is short of it, and no variant of the score is substituted afterwards. Three of the nine pairs are at 1.000 for a structural reason rather than a discriminating one — `FAULT_08`'s required elements are essentially never printed, so `fit(08)` is 0 and any event with a positive fit beats it. The pairs that are actually hard sit at 0.52–0.64, i.e. **at or barely above chance**, and `FAULT_14`/`FAULT_11` is **inverted**: on fault-14 events the evidence fits *fault 11* better, because valve sticking and the cooling-water inlet temperature move the same variables while the KB's directed distinction (step vs random) is invisible to a presence-and-direction test.
* **On validation the score cannot be tested at all.** Of the 15 errors and false abstentions on the named-fault stratum, **13 are abstentions** — a prediction that names no fault has no candidate to fit — leaving **2 rows** to compare against 310 correct ones. AUROC 0.478 with two negatives is not a measurement, and no conclusion is drawn from it beyond "the error pool is too small and too abstention-shaped to test a grounding signal on".

**Why this matters for the prose target.** A counterfactual-aware target needs the system to know when the evidence does *not* fit the diagnosis it was handed. A presence-and-direction test built from the KB cannot supply that, for the same reason the surrogate cannot be fixed by confidence: **the lookalike pairs differ in shape, not in which variables move.** The distinction the KB actually records (step vs random variation, valve sticking vs the temperature driving it) is a *dynamics* property, and the prompt prints it as one scalar — the slow-drift ratio — which is where 28.9 % of the classifier's gold-class signal already sits. A grounding verdict derived from this score is therefore **not** feasible as specified; a version that compares slow-drift ratio against the KB's expected shape is a different proposal and is not measured here.

---

## 6. Menu order: the harness, and the invariance it can prove

* `llm/benchmark/menu_order.py` — `reorder_menu(prompt, seed, item_id)` permutes only the candidate-menu block, deterministically per item, leaving every byte outside it untouched (asserted). `menu_ids()` reads the order back.
* `llm/benchmark/decode_bench.py` — new optional `--menu-seed`; omitted, behaviour is exactly as before and the prompt is the stored one. A run records `menu_order: "kb" | "shuffled"`.
* **Recorded rule:** adapter-only numbers are henceforth to be reported in **both** KB order and shuffled order (§18.5). The shuffled-order adapter numbers themselves are **not** in this report: producing them needs generation on the GPU, which this task excluded. This is the one item the harness enables but does not yet measure, and it is listed as open below.
* **Order invariance, proved by test rather than argued** (`tests/test_menu_order.py`, 7 tests): the classifier's 323 features are **bit-identical** under five different permutations of the same menu, because `prompt_features` parses evidence lines only; no feature name mentions the menu; the menu rule's decision is unchanged across 12 permutations of the full menu for a menu entry, a removed entry, `NORMAL_TRANSIENT` and `UNCLASSIFIED_OPEN_SET`; and a removed entry stays removed under every order. The last one matters: it is the open-set construction, and an order bug there would silently turn an abstention episode into a nameable one.

---

## 7. Tests and artefacts

* `python -m pytest tests -q` — **565 passed** (app env; 558 before + 7 new).
* `./venv_train/Scripts/python.exe -m pytest tests -q` — **562 passed, 3 skipped** (the lightgbm-dependent tests; `venv_train` has no lightgbm).
* New tests: `tests/test_menu_order.py` (**7 passed**).

New artefacts (new files only):

| file | contents |
|---|---|
| `outputs/analysis/phase3a/fit/prompt_budget.json` | token budget per block, per class, and for the 48 errors |
| `outputs/analysis/phase3a/fit/feature_visibility.json` | per-group gold-class SHAP mass, prompt shape, visibility per group |
| `outputs/analysis/phase3a/fit/fit_score.json` | per-pair AUROC, uncomputable share, validation AUROC and its error pool |
| `outputs/analysis/phase3a/fit/novelty_channel.json` | τ, validation before/after, surrogate per fault, folds per item, verdict |

Code: `scripts/prompt_budget.py`, `scripts/feature_visibility.py`, `scripts/hybrid_fit_score.py`, `scripts/hybrid_novelty.py`, `llm/benchmark/menu_order.py`. Reproduce, in order: `./venv_train/Scripts/python.exe scripts/prompt_budget.py` (**29 s**), `python scripts/feature_visibility.py`, `python scripts/hybrid_fit_score.py`, `python scripts/hybrid_novelty.py` (**92 s**).

---

## 8. What this changes, and what it does not

**Changes.** The novelty channel is **not adopted** — it costs 16 correct validation answers and 3.67 points of kept-fault accuracy in `fold_1` to catch 2 of fold_2's 23 confidently-wrong items, and §18.6's criterion (a) rejects it. A code-computed evidence-fit score is **not feasible** at the pre-registered bar (0.65 vs 0.80) and is **inverted** on the pair that matters most, because the lookalikes differ in dynamics rather than in which sensors move. The corrections file withdraws one number and fixes three statements in §17's report, the most consequential being that the hybrid's own error pool is **68 events**, not 258, and that `FAULT_11` absorbs none of the 8 natural argmax errors.

**Does not change.** The G1 decision (hybrid, classifier diagnoses, adapter explains) stands, and nothing here touches a §16/§17 artefact, the run-1 adapter, the prompts, the datasets or their hashes. No test component was opened.

**Open, and each is a decision rather than a measurement.** (1) The shuffled-order adapter run — the harness exists and the rule is recorded, but the generation needs the GPU. (2) The open-set arm has no working instrument: three lines of evidence now say the same thing (confidence abstains 22 times in 834, novelty flags the already-declined, and both lookalike surrogates absorb 25 of 25). (3) The faults-3/9 labelling question, which §18.2 now scores two ways without resolving. (4) The prose target's grounding verdict, which needs a *dynamics* test — slow-drift ratio against the KB's expected shape — before it can be built.
