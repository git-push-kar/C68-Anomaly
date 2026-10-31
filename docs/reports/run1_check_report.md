# RUN 1 CHECK — instrument repair and the references it needs

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade` · **Environment:** app env (CPU), `venv_train` (GPU)
**Date:** 2026-10-31 / 2026-11-01 · **Scope:** rescore and reselect run 1 under a corrected instrument; add base-model and tabular references; no training, no test component opened, run-1 adapter and checkpoints untouched.

---

## 0. Gate table

| # | Gate item | Status | Evidence |
|---|---|---|---|
| 1 | Candidate set printed; NORMAL_TRANSIENT presence checked | **PASS** | 16 candidates, `NORMAL_TRANSIENT` absent; 111 training rows carry it (§1) |
| 2 | Missing answer value added; test asserting training targets are candidates | **PASS** | 17 candidates; `test_every_training_target_is_a_scorer_candidate`; 515 tests pass (§1, §7) |
| 3 | Dated PROTOCOL amendment recorded before test contact | **PASS** | §13 (candidate set), §13.1 (per-token score), both 2026-10-31 |
| 4 | All 6 checkpoints rescored; reselection by the pre-registered rule; full validation | **PASS** | §1 — step 300 selected (macro-F1 0.8720); 822-row validation §1.4 |
| 5 | NORMAL_TRANSIENT vs FAULT_15 confusion reported before/after | **PASS** | §1.5 — 58 → 22 FAULT_15 predictions, precision 0.241 → 0.591 |
| 6 | 7 generation/scoring disagreements listed with old and new scorer | **PASS** | §2.1 — 6 at step 400, 4 at the selected step, none the generation's fault |
| 7 | 4 enum failures listed with offending strings | **PASS** | §2.2 — all four are `NORMAL_TRANSIENT`, i.e. not failures |
| 8 | Report arithmetic for FAULT_15 corrected | **PASS** | §2.3 |
| 9 | Base model scored on subsample and full validation, same prompt and labels | **PASS** | §3 — base abstains on 100 % of rows; accuracy 0.5474 = open-set share |
| 10 | Follow-up: ≥15 per case type, 25 re-asks, 40 controls | **AMENDED** | 179 turns; floors met where the corpus allows (§4.1) |
| 11 | Refusals on in-scope cases listed | **PASS** | §4.3 (15 listed) |
| 12 | Evidence classifier (GBDT + MLP) trained on train events, adapter on same subset | **PASS** | §5 |
| 13 | 21.7 vs 14.3 s/step explained; allocated vs reserved separated; projections | **PARTIAL** | §6 — what the telemetry rules out and what it cannot settle |
| 14 | Config comment about checkpoint count/size fixed | **PASS** | §6.4 |
| 15 | 540 × 21.71 reconciled with 12,003.5 s | **PASS** | §6.2 — 280.1 s of pre-loop time |
| 16 | Full suite in app env and `venv_train` | **PASS** | 515 passed in each (§7) |
| 17 | No test component opened | **PASS** | only `val` and follow-up `val` read |

**Headline.** The run's reported closed-set numbers were measured with an instrument
that could not answer `NORMAL_TRANSIENT` at all. Corrected, the model is
substantially better than reported: macro-F1 **0.7888 → 0.8689**, accuracy
**0.8406 → 0.8869**, NORMAL_TRANSIENT recall **0.000 → 0.915**, and the "FAULT_15
absorbs everything" defect in the run report is mostly a property of the scorer, not
the model. The selected checkpoint changes from step 400 to step 300, and the
adapter still loses to a gradient-boosted model on the same evidence (§5) — which is
the finding that should shape the next decision.

---

## 1. Scorer candidate set

### 1.1 What the scorer was reading

`label_scorer.candidates()` returned 16 strings, in menu order:

```
FAULT_01 FAULT_02 FAULT_03 FAULT_04 FAULT_05 FAULT_06 FAULT_07 FAULT_08
FAULT_09 FAULT_10 FAULT_11 FAULT_12 FAULT_13 FAULT_14 FAULT_15
UNCLASSIFIED_OPEN_SET
```

`NORMAL_TRANSIENT` is **not** among them, and by design: `menu.py` states that
NORMAL_TRANSIENT is not a fault and must not be a menu entry, so that an open-set
episode cannot point at one. But it is a training target. The mixed SFT file's 1,919
report rows carry 15 distinct `fault_id` values, and one of them is missing from the
scorer:

| fault_id | rows | in candidate set |
|---|---|---|
| FAULT_01 … FAULT_15 (13 classes) | 100–110 each | yes |
| UNCLASSIFIED_OPEN_SET | 452 | yes |
| **NORMAL_TRANSIENT** | **111** | **no** |

Three NORMAL_TRANSIENT training rows, as the model was taught to answer them:

| item_id | fault_id | verdict | ranked_hypotheses |
|---|---|---|---|
| `NORM-train-r001::menu` | `NORMAL_TRANSIENT` | `no_fault` | `[]` |
| `NORM-train-r003::menu` | `NORMAL_TRANSIENT` | `no_fault` | `[]` |
| `NORM-train-r006::menu` | `NORMAL_TRANSIENT` | `no_fault` | `[]` |

Their prose is the same shape as the rest (`"The CVA Q-statistic (195.5) exceeded
threshold (194.6) with a ratio of 1.00x … all top deviating sensors show small
z-scores"`), so nothing about the row is unusable; only the scoring path could not
name it. Verdicts in the file are `classified` 1,356, `unclassified_open_set` 452,
`no_fault` 111, and `None` for the 961 dialogue rows.

### 1.2 A second fault in the same place, found by fixing the first

Adding the 17th candidate produced a *worse* table (accuracy 0.057 at step 100,
0.100 at step 200), with **287 of 300** validation items answering
`NORMAL_TRANSIENT`. The cause is the arithmetic, not the model. The scorer summed
the log-probability of each candidate's tokens after `{"fault_id": "`, and the
suffixes are not the same length:

| candidate | suffix tokens after the 5-token prefix |
|---|---|
| `FAULT_01` | 5 (`_`, `0`, `1`, `"`) |
| `NORMAL_TRANSIENT` | 4 |
| `UNCLASSIFIED_OPEN_SET` | 6 |

A sum charges every candidate one negative term per token, so the shortest string
wins on arithmetic alone. The score is now the **mean log-probability per token**
(`normalised_score`), with the raw sum kept in every prediction row as `total_logp`.
Two tests pin it — one on the arithmetic (equal per-token probability scores equal
across 4 and 6 tokens), one on the call site — and PROTOCOL §13.1 records the
measurement. Note the direction of the old bias: `UNCLASSIFIED_OPEN_SET` being 6
tokens meant abstention was being *penalised* by one term all along, which is why the
corrected abstention recall can only move up.

### 1.3 Checkpoint table, before and after

Same 300-item stratified validation subsample (`val_subsample_ids.json` unchanged),
batch 16, language-model-direct path, seen faults, both menu variants.

| Step | Before: acc | Before: F1 | Before: NORMAL R | After: acc | After: F1 | After: NORMAL R |
|---|---|---|---|---|---|---|
| 100 | 0.2000 | 0.2727 | 0.000 | 0.4833 | 0.2515 | 0.000 |
| 200 | 0.3800 | 0.5605 | 0.000 | 0.7633 | 0.5623 | 0.941 |
| **300** | 0.4633 | 0.5948 | 0.000 | **0.8833** | **0.8720** | **1.000** |
| 400 | 0.8333 | **0.7826** | 0.000 | 0.8533 | 0.7325 | 1.000 |
| 500 | 0.7600 | 0.7659 | 0.000 | 0.9033 | 0.8540 | 1.000 |
| 540 | 0.7733 | 0.7721 | 0.000 | 0.9033 | 0.8545 | 1.000 |

**Selection changes from step 400 to step 300** (macro-F1 0.8720 against 0.8545 at
step 540 and 0.8540 at 500). The rule was not touched: macro-F1, accuracy as
tie-break, later step as second tie-break.

Two things are worth stating because they are easy to misread:

* step 300 wins on macro-F1 while **500 and 540 win on accuracy and top-3**
  (0.9033 / 0.9933 against 0.8833 / 0.9867). The single class that decides it is
  FAULT_15: recall 0.556 at 300, 0.0 at 400/500/540. Macro-F1 is doing exactly what
  it was chosen to do — refusing to let a class die — and the report should say so
  rather than present 300 as universally better.
* the before/after jump is not the model improving. It is 111 training rows becoming
  answerable, and a length bias being removed from the comparison.

### 1.4 Selected checkpoint on the full Protocol A validation (822 observable rows)

| Metric | Before (step 400) | After (step 300) |
|---|---|---|
| Accuracy | 0.8406 | **0.8869** |
| Top-3 | 0.9331 | **0.9866** |
| Macro-F1 | 0.7888 | **0.8689** |
| Abstention rate | 0.4781 | 0.4976 |
| Abstention precision | 0.9975 | 0.9707 |
| Abstention recall | 0.8711 | **0.8822** |
| False abstention | 0.0027 | 0.0323 |
| Missed abstention | 0.1289 | 0.1178 |
| ECE / Brier | 0.0364 / 0.1171 | 0.6667 / 0.5357 *(see §1.6)* |
| Scoring time | 834.7 s | **790.6 s** (0.96 s/item) |

Per class:

| Class | Before P / R / F1 | After P / R / F1 | n |
|---|---|---|---|
| FAULT_01 | 0.962 / 1.000 / 0.980 | 0.962 / 1.000 / 0.980 | 25 |
| FAULT_02 | 1.000 / 1.000 / 1.000 | 0.926 / 1.000 / 0.962 | 25 |
| FAULT_04 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 25 |
| FAULT_05 | 0.923 / 0.960 / 0.941 | 0.889 / 0.960 / 0.923 | 25 |
| FAULT_06 | 0.917 / 0.880 / 0.898 | 0.960 / 0.960 / 0.960 | 25 |
| FAULT_07 | 1.000 / 0.960 / 0.980 | 1.000 / 1.000 / 1.000 | 25 |
| FAULT_08 | 0.688 / 0.880 / 0.772 | 0.952 / 0.800 / 0.870 | 25 |
| FAULT_10 | 0.321 / 1.000 / 0.485 | 0.426 / 0.920 / 0.582 | 25 |
| FAULT_11 | 0.694 / 1.000 / 0.820 | 0.960 / 0.960 / 0.960 | 25 |
| FAULT_12 | 0.962 / 1.000 / 0.980 | 1.000 / 0.800 / 0.889 | 25 |
| FAULT_13 | 0.727 / 0.960 / 0.828 | 0.793 / 0.920 / 0.852 | 25 |
| FAULT_14 | 1.000 / 0.800 / 0.889 | 1.000 / 0.720 / 0.837 | 25 |
| FAULT_15 | **0.241** / 0.520 / **0.329** | **0.591** / 0.520 / **0.553** | 25 |
| NORMAL_TRANSIENT | **0.000 / 0.000 / 0.000** | **0.623 / 0.915 / 0.741** | 47 |
| UNCLASSIFIED_OPEN_SET | 0.998 / 0.871 / 0.930 | 0.971 / 0.882 / **0.924** | 450 |

Unobservable rows (faults 3 and 9, 12 rows): accuracy 0.50 → **0.00**, macro-F1
0.333 → 0.000. The selected checkpoint is now more willing to commit elsewhere on
these rows; with n = 12 and 6 run clusters this is not a finding, but it moved in the
wrong direction and is reported rather than dropped.

### 1.5 The NORMAL_TRANSIENT / FAULT_15 confusion, corrected

| | Before | After |
|---|---|---|
| Rows predicted FAULT_15 | 58 | 22 |
| …correct | 13 | 13 |
| …wrong | 45 | 9 |
| …of which NORMAL_TRANSIENT | **39** | **3** |
| …fault 09 / fault 03 / open-set / other fault | 3 / 1 / 2 / 0 | 0 / 0 / 5 / 1 |
| FAULT_15 precision (observable set) | 0.241 | 0.591 |
| FAULT_15 recall | 0.520 | 0.520 |
| NORMAL_TRANSIENT predictions | 39 → F15, 6 → F10, 1 → F13, 1 → F11 | **43 correct**, 3 → F15, 1 → F02 |

The run report called FAULT_15 "the single largest defect in the run" on the strength
of 39 normal rows being absorbed. With the label answerable, 43 of 47 normal rows are
correct and the absorption is gone. What remains of the FAULT_15 problem is real but
much smaller: 3 normal rows and 5 open-set rows land on it, and its precision (0.591)
is the weakest of the 15 observable classes.

### 1.6 The probability vector is not calibrated, and the ECE says so

ECE moves 0.0364 → 0.6667 and Brier 0.1171 → 0.5357. That is not a regression in the
model; it is the scale of the score changing. The vector is a softmax over
**mean** log-probabilities, whose spread is a few tenths of a nat, so the softmax is
flat and the maximum probability sits near 0.2 while the model is right 89 % of the
time. The old per-sum vector was sharply peaked and accidentally looked calibrated
against 84 % accuracy.

The honest statement is that **no confidence estimate exists yet**, which is what
PROTOCOL §11.3 already says (isotonic calibration on validation, none fitted). ECE
and Brier are kept in the artifact for comparability and flagged here so they are not
quoted as calibration. Everything the run reports about abstention depends on the
argmax only, not on the probability scale.

---

## 2. Generation versus scoring

### 2.1 The disagreements

The 50 generated validation answers were written by **step 400** (the checkpoint
selected under the old scorer). The selection has since moved to step 300, so the
apples-to-apples comparison scores those same 50 items with step 400 itself
(`preds_gen_scored.jsonl`); the comparison against the selected checkpoint is
reported beside it.

| Item | Gold | Generated fault_id | Scorer, new (step 400) | Scorer, old (16 candidates) |
|---|---|---|---|---|
| `EVT-val-f11-r307::menu` | FAULT_11 | FAULT_11 | UNCLASSIFIED_OPEN_SET | FAULT_11 |
| `EVT-val-f11-r317::menu` | FAULT_11 | FAULT_11 | UNCLASSIFIED_OPEN_SET | FAULT_11 |
| `EVT-val-f12-r311::menu` | FAULT_12 | FAULT_12 | UNCLASSIFIED_OPEN_SET | FAULT_12 |
| `EVT-val-f14-r324::menu` | FAULT_14 | FAULT_14 | UNCLASSIFIED_OPEN_SET | FAULT_14 |
| `EVT-val-f14-r310::menu` | FAULT_14 | FAULT_14 | UNCLASSIFIED_OPEN_SET | FAULT_14 |
| `EVT-val-f15-r308::menu` | FAULT_15 | FAULT_15 | NORMAL_TRANSIENT | FAULT_15 |

**6 disagreements at step 400** (was 7 with the old scorer), and the generation is
right in all six: the labelled scorer abstains on five faults it could name and calls
one fault normal. Against the *selected* checkpoint (step 300) there are **4**
disagreements, all of them open-set episodes whose prose correctly abstained and
whose labelled score did not:

| Item | Gold | Generated | Scorer at 300 | Scorer, old |
|---|---|---|---|---|
| `EVT-val-f19-r315::menu` | UNCLASSIFIED_OPEN_SET | UNCLASSIFIED_OPEN_SET | NORMAL_TRANSIENT | FAULT_10 |
| `EVT-val-f16-r317::menu` | UNCLASSIFIED_OPEN_SET | UNCLASSIFIED_OPEN_SET | FAULT_10 | FAULT_10 |
| `EVT-val-f19-r302::menu` | UNCLASSIFIED_OPEN_SET | UNCLASSIFIED_OPEN_SET | FAULT_10 | FAULT_10 |
| `EVT-val-f15-r323::openset` | UNCLASSIFIED_OPEN_SET | UNCLASSIFIED_OPEN_SET | NORMAL_TRANSIENT | UNCLASSIFIED_OPEN_SET |

The pattern is worth naming: the old instrument **over-committed** (it could not
abstain to a normal answer, and it was charged an extra token for abstaining); the new
one **over-abstains on 5 of 50** items at step 400. Neither is calibrated, and
generation remains the more accurate of the two paths on this sample — 49/50 against
43/50 — at roughly 45× the cost per item.

### 2.2 The four "enum failures"

| Item | Gold | Generated fault_id | Legal under corrected set |
|---|---|---|---|
| `EVT-val-f15-r318::menu` | FAULT_15 | `NORMAL_TRANSIENT` | **yes** |
| `NORM-val-r308::menu` | NORMAL_TRANSIENT | `NORMAL_TRANSIENT` | **yes** |
| `NORM-val-r323::menu` | NORMAL_TRANSIENT | `NORMAL_TRANSIENT` | **yes** |
| `NORM-val-r386::menu` | NORMAL_TRANSIENT | `NORMAL_TRANSIENT` | **yes** |

All four are the same bug: `asks_schema` validated the generated `fault_id` against
`candidates()`, so every legal NORMAL_TRANSIENT answer was counted invalid. Corrected:
**enum-valid 50/50, not 46/50**, schema-valid 50/50, generated accuracy 0.98 (49/50,
the single miss being `EVT-val-f15-r318` answering NORMAL_TRANSIENT for FAULT_15).

### 2.3 The FAULT_15 arithmetic in the run report

The report said FAULT_15 "is predicted 25 gold rows + 39 normal rows + others = ~104
times, of which 25 are right". Correct figures from the same prediction file:

* predicted **58** times, not ~104;
* **13** correct (FAULT_15 recall 0.52 on its 25 rows), **45** wrong;
* the 45 are 39 NORMAL_TRANSIENT + 3 FAULT_09 + 1 FAULT_03 + 2 UNCLASSIFIED_OPEN_SET;
* FAULT_15 precision is 0.241 on the observable set (13/54), which is where the
  reported 0.241 came from — it was right, the count beside it was not.

After the fix: 22 predictions, 13 correct, precision 0.591.

---

## 3. Base-model reference (no adapter, same prompt, same 17 labels)

The unadapted base never names a fault. It answers `UNCLASSIFIED_OPEN_SET` on
**100 %** of rows, which makes its accuracy exactly the open-set share of the set:
450/822 = 0.5474. That is the number to keep in mind whenever accuracy is quoted.

| Arm | Items | Accuracy | Top-3 | Macro-F1 | Abstention rate | Abstention P / R |
|---|---|---|---|---|---|---|
| Base, full validation | 822 | 0.5474 | 0.6168 | 0.0472 | 1.000 | 0.547 / 1.000 |
| Adapter (step 300), full validation | 822 | 0.8869 | 0.9866 | 0.8689 | 0.498 | 0.971 / 0.882 |
| **Adapter − base** | | **+0.3395** | **+0.3698** | **+0.8217** | −0.502 | |
| Base, 300-item subsample | 300 | 0.5533 | 0.6200 | 0.0475 | 1.000 | 0.553 / 1.000 |
| Adapter, same 300 | 300 | 0.8833 | 0.9867 | 0.8720 | 0.483 | 0.986 / 0.861 |

Per class the base is 0.000 precision and recall everywhere except
`UNCLASSIFIED_OPEN_SET` (P 0.547, R 1.000). Its macro-F1 of 0.047 is what a constant
answer scores, and it is the honest denominator for every macro-F1 in this project.

---

## 4. Follow-up sample (200 requested, 179 delivered)

### 4.1 Why 179 and not 200

The floors were ≥15 per case type, 25 `repeated_question`, 40 in-scope controls. The
validation follow-up corpus is 261 dialogues, and four case types cannot supply 15:

| case type | val corpus | requested floor | turns scored |
|---|---|---|---|
| valve_actuator | 73 | 15 | 15 |
| causal_why | 42 | 15 | 15 |
| mechanism | 39 | 15 | 15 |
| false_premise | 20 | 15 | 15 |
| hypothesis_comparison | 19 | 15 | 15 |
| clarification_reply | 19 | 15 | 15 |
| **repeated_question** | **17** | **25** | **17** |
| **out_of_scope** | **13** | 15 | **13** |
| **ambiguous_question** | **12** | 15 | **12** |
| **template_sample** | **7** | 15 | **7** |
| in_scope_control | — | 40 | 40 |
| **total** | 261 | 200 | **179** |

The shortfall is supply, not sampling: every dialogue in those four cases was used.
Reported as a corrected denominator rather than padded by duplicating turns.

### 4.2 Results by case type (selected checkpoint, step 300)

| Case | n | Refusal | Grounded clean | Repetition |
|---|---|---|---|---|
| out_of_scope | 13 | **1.000** | 1.000 | 0.000 |
| false_premise | 15 | **1.000** | 1.000 | 0.000 |
| in_scope_control | 40 | **0.050** | 0.900 | 0.000 |
| ambiguous_question | 12 | 0.000 | 1.000 | 0.000 |
| clarification_reply | 15 | 0.000 | 1.000 | 0.000 |
| hypothesis_comparison | 15 | 0.000 | 0.933 | 0.000 |
| template_sample | 7 | 0.143 | 1.000 | 0.143 |
| valve_actuator | 15 | 0.400 | 0.933 | 0.000 |
| mechanism | 15 | 0.467 | 1.000 | 0.000 |
| repeated_question | 17 | **0.529** | 0.941 | 0.000 |
| causal_why | 15 | **0.733** | 0.867 | 0.000 |
| **all** | 179 | 0.306 (in-scope cases) | 0.950 | 0.006 |

Totals: refusal on out_of_scope + false_premise **1.00 (28/28)**, over-refusal on
controls **0.05 (2/40)**, grounding 0.950, repetition 0.006, 9.64 s/item.

**The refusal asymmetry is the result.** The model declines every unanswerable turn
and 2 of 40 answerable controls, but it also declines **53 % of re-asks and 73 % of
causal_why** — both in-scope case types. That 0.306 is much worse than the 60-turn
sample's 0.1875, and two things changed at once: a different checkpoint (300 instead
of 400) and a bigger, harder sample. Do not read the difference as a trend.

### 4.3 Refusals on in-scope cases (first 15 of the artifact)

| Case | Question | Answer (truncated) |
|---|---|---|
| causal_why | What is the chain from the actuator to the analyser reading? | "The context does not provide a chain from the actuator to the analyser reading. The incident context…" |
| causal_why | How would a stuck valve show up two loops later? | "The incident context does not mention any stuck valves or valve behavior. The leading diagnosis stat…" |
| causal_why | What is the chain from the actuator to the analyser reading? | "The context does not provide a chain from the actuator to the analyser reading. The incident context…" |
| causal_why | How does the cooling water flow restriction ripple through the plant? | "The incident context does not mention cooling water flow restrictions or their ripple effects. The l…" |
| causal_why | Walk me through how the condenser restriction reaches the reactor. | "The incident context does not provide any information about condenser restriction or its relationshi…" |
| causal_why | How does closing the purge valve move the separator pressure? | "The context does not mention a purge valve or its effect on separator pressure. The incident context…" |

The remaining nine are the same shape, listed in full in
`outputs/analysis/phase3a/check_run1.json` (`followup_after.refusals_on_in_scope_cases`).
The pattern is legible: these answers are not evasive boilerplate, they name the
missing relationship. Whether that is correct behaviour depends on whether the
`causal_why` template is supposed to be answered from the KB (which the event context
does not carry) or from the event; that is a design decision, not a measurement, and
nothing was tuned on it.

---

## 5. Evidence-feature classifier versus the adapter

Both read the same page: the classifier parses the numbers out of the rendered prompt
(per-tag z, deviation, trend, onset, residual share, plus the detector block), the
adapter reads the prompt. Trained on **train events only** (1,729 menu-variant rows),
faults 16-20 folded into `UNCLASSIFIED_OPEN_SET`, fault-free rows kept as
`NORMAL_TRANSIENT`. Evaluated on the 503 menu-variant validation rows; the adapter has
no prediction for the six unobservable rows, so the comparable table is the 497-item
intersection.

| Model | Items | Accuracy | Top-3 | Macro-F1 |
|---|---|---|---|---|
| GBDT (LightGBM, 400 trees) | 497 | **0.9940** | 1.0000 | **0.9926** |
| MLP (323→128→64→17) | 497 | 0.9698 | 0.9940 | 0.9697 |
| Adapter (step 300, label scoring) | 497 | 0.8330 | 0.9779 | 0.8635 |
| GBDT | 503 | 0.9821 | 0.9881 | 0.8724 |
| MLP | 503 | 0.9583 | 0.9821 | 0.8512 |
| Adapter | 503 | 0.7646 | 0.8893 | 0.7942 |

On menu-variant items a tabular model reading the same numbers makes **3 of 497**
mistakes (0.9940 × 497 = 494 correct) where the adapter makes **83**; the MLP makes
15. The adapter's own full-validation number
(0.8869) is higher than its menu-only number (0.833) because the full set includes 331
removed-menu open-set episodes it abstains on correctly, and those are items the
classifier is not asked.

What this does and does not mean: it does not say the adapter is worthless — it writes
prose, refuses, and handles the dialogue task, none of which the classifier does. It
does say that **on the closed-set question the evidence is sufficient and the 2 B
parameters are not buying accuracy**. A hybrid (classifier for `fault_id`, adapter for
prose) is the obvious next candidate and is not implemented here.

Artifacts: `outputs/analysis/phase3a/evidence_classifier/{summary.json,preds_gbdt_val503.jsonl,mlp,preds_adapter_val503.jsonl}`.

---

## 6. Budget: 21.7 vs 14.3 s/step, memory, and what a run costs

### 6.1 The telemetry

54 logged intervals (every 10 steps, plus the duplicated 540 row):

| Quantity | Value |
|---|---|
| Mean step | 22.40 s (median 21.23, min 17.23, max 47.46 at step 60) |
| Clock | 1,695–1,830 MHz, mean 1,804 |
| Temperature | 70–80 °C (limit 85 for 2 min; never tripped) |
| Power | 154–212 W |
| Device memory | 24,023–24,095 MiB |
| Save steps (100/200/300/400/500) | 17.4 / 25.0 / 20.9 / 20.1 / 20.7 s — no elevation |
| Stalls | step 60: 47.46 s at 154 W, step 70: 44.63 s at 161 W |

What this rules out, item by item:

* **Checkpoint saves.** Six directories at 217 MB each cost **no measurable step
  time** — the five save steps are the same as their neighbours. Adapter-only saves
  are too small to matter.
* **Thermals.** The run never exceeded 80 °C; the smoke reached the same 80 °C.
* **Clocks.** The run averaged **1,804 MHz**, higher than the smoke's 1,658 MHz. The
  real run was clocked *up*, not down.
* **Sequence length.** Both runs read the same file at the same batch shape; the mixed
  file's mean is 1,394.6 tokens/row (report rows 1,679.8, dialogue rows 825.1), so a
  16-row optimizer step is ~22,300 tokens in both.
* **Sustained stalls.** The two steps above 30 s (60 and 70) are a 92 s excursion with
  power dropping to ~154 W — external contention, not the job. They account for ~50 s
  of the 540-step total.

What remains: the 20-step smoke measured 1,559 tokens/s and the run measured
1,028 tokens/s on the same shape, and the telemetry cannot say why. The candidate the
data does not exclude is a difference in allocator behaviour between a fresh 20-step
process and a 3 h 20 m one — but the run's device memory (24.09 of 24.56 GiB, 98.1 %)
is *reserved* memory reported by the driver, not the tensors in flight, so it is not
evidence of pressure either. The gap is **unexplained**, and the practical response is
to plan from the measured run, not from the smoke: the 14.31 s/step projection was
**52 % optimistic**.

### 6.2 The 540 × 21.71 arithmetic

`train_runtime 12,003.5 s` is the wall clock of `Trainer.train()`; the progress bar's
21.71 s/it is computed over the step loop only. The difference is the pre-loop work
(model load, dataset tokenisation, first batch):

```
540 × 21.71 = 11,723.4 s
12,003.5 − 11,723.4 = 280.1 s  (0.52 s/step amortised, 2.3 % of the run)
```

Both numbers are correct; quoting one against the other was an error in the run
report, and the reconciliation is now in `check_run1.json` (`budget`).

### 6.3 Memory: allocated, reserved, device

Three different numbers were being used interchangeably:

| Quantity | Value | What it measures |
|---|---|---|
| `nvidia-smi` device used during training | 24,023–24,095 MiB | everything the process holds, including CUDA context and allocator cache |
| Card total | 24,564 MiB | physical |
| Headroom | **~470–540 MiB (1.9 %)** | the margin before OOM |
| Smoke `torch.cuda.max_memory_allocated()` | 17.02 GiB | tensors handed out by the caching allocator only |

The run report compared the third with the first and called it a 7 GB difference. They
are different measurements of different things: a 17 GiB live set inside a 24 GiB
reserved pool is normal for bf16 at 4,096 tokens with activation checkpointing. The
useful fact is the **1.9 % headroom** — a longer max sequence or a larger batch would
OOM, and nothing logged the run's own `max_memory_allocated`, which is the gap to fix
in the next run's telemetry.

### 6.4 Re-estimate

| | Measured / projected |
|---|---|
| Per run (540 steps) | **3.33 h** (12,003.5 s) |
| Four more runs | **13.3 h** |
| Five runs total | **16.7 h** |
| Five runs, +10 % / +20 % headroom | 18.3 / 20.0 h |
| Nine runs (3 configs × 3 seeds) at this rate | 30.0 h |
| Previous estimate from the smoke | 2.15 h/run, 10.7 h for five — **superseded** |

Config comment fixed in `configs/config_v2_protocolA_seed42.yaml`: `save_total_limit: 0`
now records the measured "6 directories, 217 MB each, 1.30 GB, 74 MB adapter + 143 MB
AdamW state" instead of "14 of them, ~35 MB each".

---

## 7. Tests and artifacts

`python -m pytest -q` — **515 passed** in the app environment; `venv_train` run of the
same suite — **515 passed**. Two new label-scorer tests and one new candidate-coverage
test; the earlier 512 becomes 515.

| Artifact | Path |
|---|---|
| Before-state (old table, old preds) | `outputs/benchmark_v2/eval/before_candidate_fix/` |
| Checkpoint table, after | `outputs/benchmark_v2/eval/checkpoint_table.json` |
| Selection, after | `outputs/benchmark_v2/eval/selected_checkpoint.json` |
| Full-val predictions, step 300 | `outputs/benchmark_v2/eval/preds_val_full_300.jsonl` |
| Generation vs scoring at step 400 | `outputs/benchmark_v2/eval/preds_gen_scored.jsonl` |
| Base reference | `outputs/benchmark_v2/eval/base_reference.json` |
| Follow-up, 179 turns | `outputs/benchmark_v2/eval/followup_sanity_200.json` |
| Classifier | `outputs/analysis/phase3a/evidence_classifier/` |
| Everything above, one file | `outputs/analysis/phase3a/check_run1.json` |

Reproduce: `python scripts/check_run1.py`, `python scripts/evidence_classifier.py
--adapter-preds outputs/benchmark_v2/eval/preds_val_full_300.jsonl`,
`venv_train/Scripts/python.exe scripts/eval_protocolA.py {checkpoints,base,followup}`.

---

## 8. What this changes about run 1, and what it does not

Changes:

* the closed-set headline is **0.8689 macro-F1 / 0.8869 accuracy** on 822 observable
  validation rows, selected at **step 300**;
* NORMAL_TRANSIENT is a working class (P 0.623 / R 0.915), and the reported
  "FAULT_15 absorbs everything" defect is mostly instrument;
* enum validity of generated answers is 50/50, and generated accuracy is 0.98;
* deep-decode failure modes are now visible: the labelled scorer abstains where the
  prose commits (6 of 50).

Does not change:

* no test component has been scored, and none of this is a test result;
* the adapter still loses to gradient-boosted trees on the closed-set question (§5);
* follow-up refusal on in-scope case types is 0.306, and re-asks are the worst of it;
* the 3.33 h/run budget and the 1.9 % memory headroom stand, and the 2.15 h estimate
  was too optimistic by 52 %.

Recommendation, not applied: freeze the scoring instrument here (candidate set,
per-token score, batch 16), reselect once, and do not re-open it between seeds — the
run's numbers have now moved twice for reasons that had nothing to do with the model.
