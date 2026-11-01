# 3B-5 HYBRID HARDENING AND HELD-OUT VALIDATION

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade` · **Environment:** app env (CPU) for everything except the bounded retry, `venv_train` for the tokenizer-free GPU item
**Date:** 2026-11-01 · **Scope:** §17 of PROTOCOL — cross-fitted calibration, a wider leave-one-fault-out surrogate with four uncertainty signals, fold classifiers on their held-out faults, the twelve unobservable rows, the natural wrong-label census, detector transparency and a bounded valve/flow retry. **No adapter training. No test component opened.** The run-1 adapter, its checkpoints and every §16 artifact are unmodified; §16's numbers are reported unchanged and are not replaced.

No job was projected over 30 minutes, so no estimate was needed before starting: the longest was `hybrid_fold_holdout.py` at **26 s**, and the GPU retry (21 items at 14.43 s/item) at about **5 min**. Two pre-registered numbers were wrong when first applied and are corrected in §17.9 before anything was built on them — §6 below is the honest account, and it changes the chosen uncertainty signal.

---

## 0. Gate table

| # | Gate item | Result |
|---|---|---|
| 1 | §17 dated, committed **before** any new number | **PASS** — commit `2b8e19f`, PROTOCOL only, 100 lines added, §16 numbers quoted as the baseline it must not replace |
| 2 | Cross-fitted calibration tables | **PASS** — §2: OOF ECE isotonic 0.00596 vs temperature 0.00638, isotonic chosen, threshold 0.9792 at 4.74 % OOF false abstention |
| 3 | Saturation share, before and after | **PASS** — 828/834 (99.3 %) → **774/834 (92.8 %)**; still high, and stated as such |
| 4 | Wider surrogate, 13 faults, per-signal AUROC, chosen signal | **PASS** with a correction — §3 and §6: all 13 faults run; means margin 0.8808, max_prob 0.8815, entropy 0.8828, novelty 0.8456; the tie rule selects **margin**, not the entropy reported before the orientation fix |
| 5 | Fold classifiers on validation-range held-out faults | **PASS** — §4: fold_1 0 of 75 held-out items named (66.7 % declined), fold_2 23 of 50 named FAULT_11; both make their calibration and threshold from out-of-fold train events only |
| 6 | Unobservable table | **PASS** — §5: all 12 rows, item by item, with evidence strength, probabilities, rule output, gold and the step-500 answer |
| 7 | Wrong-label counts | **PASS** — §5.3: 8 wrong and 254 abstained out-of-fold events of 1,919; the lookalike split is 44 of 50 |
| 8 | Transparency table | **PASS** — §7: 248 sentences that mention another fault, all cleared, 3 by the rejection guard, 20 listed for manual check |
| 9 | Valve/flow result | **PASS** — §8: the report path has no valve rule at all; a single retry fixed 7 of 21 and moved 1 to a different mislabel |
| 10 | Tests in the app environment and `venv_train` | **PASS** — 558 passed (app env); 555 passed, 3 skipped (`venv_train`, no lightgbm there) |
| 11 | No test data touched | **PASS** — no new script references `test.jsonl` or `test_eval.jsonl`; every number here is validation |
| 12 | Run-1 and earlier hybrid artifacts unchanged | **PASS** — `git status` shows only new files plus `docs/RESPONSE_SPEC.md`, `llm/hybrid/generate.py`, both additive; no §16 file rewritten |

---

## 1. What §17 had to fix, in one paragraph

§16 fitted the calibrator on the same 834 validation rows it then scored, and the result saturated: 828 of 834 calibrated maxima came out at exactly 1.0, the pre-registered threshold rule therefore returned 1.0, and the abstention arm abstained **4 times in 497** menu-present items. A hybrid whose abstention never fires is a hybrid with no open-set story. §17 moves every fitted part onto out-of-fold **train** events — classifier folds, then calibrator folds nested inside them, then the threshold — so the 834 validation rows are evaluation only. The cost is visible and is the point: the honest numbers are **lower** than §16's, and they are the ones to use.

---

## 2. Cross-fitted calibration and abstention (§17.1, §17.2)

**Method choice, out-of-fold (5 folds over 1,919 train events, grouped by event):**

| method | OOF ECE | OOF Brier |
|---|---|---|
| isotonic (one-vs-rest, renormalised) | **0.00596** | **0.0036** |
| temperature scaling | 0.00638 | 0.0051 |

Isotonic is chosen by the pre-registered rule (lower OOF ECE). Threshold from OOF train menu-present seen-fault items: **0.9792**, spending 4.74 % of the 5 % false-abstention budget (82 of 1,729 abstained).

**Saturation on validation:** 774 of 834 calibrated maxima are at or above 0.979 (**92.81 %**), down from 828 at exactly 1.0 (§16, 99.28 %). The calibrator no longer memorises the evaluation rows, but a LightGBM classifier on separable sensor signatures is still very confident, and isotonic maps "very confident" to "1" whenever the OOF bin is pure. **A confidence threshold remains a blunt open-set instrument** — which is why §3 exists.

**Validation, 834 rows (clean evaluation — nothing here was fitted on):**

| | §16 (in-sample calibration) | §17.2 (cross-fitted) | change |
|---|---|---|---|
| observable items | 822 | 822 | – |
| accuracy | 0.9964 | **0.9781** | −1.83 pt |
| macro-F1 | 0.9963 | **0.9708** | −2.55 pt |
| abstention precision | 0.9934 | **0.9739** | −1.95 pt |
| abstention recall | 1.0000 | **0.9933** | −0.67 pt |
| false abstention | 0.0081 | **0.0323** | +2.42 pt |
| missed abstention | 0.0000 | **0.0067** | +0.67 pt |
| ECE | 0.0009 | **0.0206** | +1.97 pt |
| Brier | 0.0017 | **0.0217** | +2.00 pt |

Bootstrap (run-clustered, 1,000 resamples, 497 clusters): accuracy 0.9781 [0.9660, 0.9879], macro-F1 0.9708 [0.9532, 0.9841]. For comparison the adapter alone (§14 decode instrument, step 500) scores macro-F1 **0.9263** on the same 822 rows, so the classifier-plus-menu-rule diagnosis is still ahead by 4.5 points — §16.6's rejection bar is cleared, with less room than §16 appeared to show.

**Per stratum:**

| stratum | n | accuracy | macro-F1 | abstain rate | abstain precision | false abst. | missed abst. | ECE |
|---|---|---|---|---|---|---|---|---|
| observable | 822 | 0.9781 | 0.9708 | 0.5584 | 0.9739 | 0.0323 | 0.0067 | 0.0206 |
| menu present | 497 | 0.9698 | 0.9721 | 0.2757 | 0.9124 | 0.0323 | 0.0000 | 0.0285 |
| menu removed | 325 | 0.9908 | 0.9954 | 0.9908 | 1.0000 | – | 0.0092 | 0.0082 |
| fault named in menu | 325 | 0.9538 | 0.9721 | 0.0369 | 0.0000 | 0.0369 | – | 0.0441 |
| NORMAL_TRANSIENT | 47 | 1.0000 | 1.0000 | 0.0000 | – | 0.0000 | – | 0.0000 |
| open-set on menu | 125 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | – | 0.0000 | 0.0046 |
| unobservable (3, 9) | 12 | 0.0000 | 0.0000 | 0.0000 | – | 0.0000 | 1.0000 | 1.0000 |

Rule reasons over 834: on-menu above threshold 429, not on menu 322, no fault predicted 61, below threshold 22. **The 22 "below threshold" rows are the whole abstention arm**, i.e. the calibrated maximum is still doing almost none of the work; the menu rule is doing it by checking membership.

Two per-class weaknesses survive from §16 and are unchanged in kind: FAULT_08 recall 0.80, FAULT_12 recall 0.84 — the same two lookalikes §3 finds absorbing everything else.

---

## 3. Leave-one-fault-out surrogate, widened to 13 faults (§17.3, §17.4)

Each fault is removed from the classifier's train set, the calibrator is cross-fitted without that fault's events, the threshold is re-derived by §17.2 on the remaining train events, and the fault's validation items are scored. 25 menu-present items per fault (the menu-removed twin is reported in the artefact).

### 3.1 Signals, oriented as §17.3 requires

| signal | mean AUROC (13 faults) | menu-present only | |
|---|---|---|---|
| `margin` (top1 − top2) | **0.8808** | 0.8798 | ← **chosen** (tie rule, simplest) |
| `max_prob` (calibrated max) | 0.8815 | 0.8806 | |
| `entropy` (of the calibrated vector) | **0.8828** | 0.8820 | highest mean, inside the 0.005 tie tolerance |
| `novelty` (mean distance to the 10 nearest train events) | 0.8456 | 0.8313 | |

Every confidence-shaped signal separates an unseen fault from the seen ones about equally well; the spread between the top three is 0.002. By §17.3 the highest mean wins unless a tie is within 0.005, in which case the simplest signal wins — margin, which needs no fitted component. **The first run of this table reported 0.12 for `max_prob` and 0.12 for `margin` and chose entropy; that was an orientation error in my own ranking, not a property of the classifier, and §17.9 records the correction and withdraws the earlier choice.** The abstention threshold never used the chosen signal, so no other number moves.

### 3.2 Per fault

| fault | held-out (menu) | declined | named | named as | max_prob | margin | entropy | novelty | false abst. on others |
|---|---|---|---|---|---|---|---|---|---|
| FAULT_01 | 25 | 0 (0.00) | 25 | FAULT_08 ×25 | 0.481 | 0.481 | 0.481 | 0.980 | 0.044 |
| FAULT_02 | 25 | 14 (0.56) | 11 | FAULT_08 ×11 | 0.955 | 0.950 | 0.958 | 0.999 | 0.036 |
| FAULT_04 | 25 | 25 (1.00) | 0 | – | 0.976 | 0.976 | 0.978 | 0.773 | 0.034 |
| FAULT_05 | 25 | 0 (0.00) | 25 | FAULT_12 ×25 | 0.825 | 0.835 | 0.824 | 0.627 | 0.036 |
| FAULT_06 | 25 | 16 (0.64) | 9 | FAULT_08 ×9 | 0.970 | 0.967 | 0.972 | 0.997 | 0.030 |
| FAULT_07 | 25 | 12 (0.48) | 13 | FAULT_08 ×13 | 0.959 | 0.956 | 0.964 | 0.654 | 0.036 |
| FAULT_08 | 25 | 21 (0.84) | 4 | FAULT_01 ×3, FAULT_07 ×1 | 0.915 | 0.915 | 0.915 | 0.883 | 0.040 |
| FAULT_10 | 25 | 25 (1.00) | 0 | – | 0.503 | 0.503 | 0.503 | 0.875 | 0.030 |
| FAULT_11 | 25 | 23 (0.92) | 2 | FAULT_04 ×2 | 0.986 | 0.984 | 0.988 | 0.977 | 0.047 |
| FAULT_12 | 25 | 25 (1.00) | 0 | – | 0.979 | 0.979 | 0.979 | 0.854 | 0.034 |
| FAULT_13 | 25 | 25 (1.00) | 0 | – | 0.995 | 0.995 | 0.996 | 0.739 | 0.044 |
| FAULT_14 | 25 | 0 (0.00) | 25 | FAULT_11 ×25 | 0.948 | 0.946 | 0.950 | 0.764 | 0.038 |
| FAULT_15 | 25 | 3 (0.12) | 22 | NORMAL_TRANSIENT ×22 | 0.966 | 0.964 | 0.967 | 0.872 | 0.038 |

No held-out fault is ever named correctly (by construction — its class does not exist in that fold's training), and per-fault false abstention on the other faults stays at 3.0–4.7 %, inside the 5 % budget at every refit.

### 3.3 Lookalike absorption — the finding that matters

Abstention protects against "nothing I know", not against "something I know that looks like this". Six of thirteen faults are absorbed **whole or nearly whole** by a specific kept fault:

| removed fault | absorbed by | of 25 |
|---|---|---|
| FAULT_01 (A/C feed ratio step) | FAULT_08 (A/B/C composition random variation) | 25 |
| FAULT_05 (condenser cooling water inlet temp step) | FAULT_12 (same, random variation) | 25 |
| FAULT_14 (reactor cooling water valve sticking) | FAULT_11 (reactor cooling water inlet temp random) | 25 |
| FAULT_15 (condenser cooling water valve sticking) | NORMAL_TRANSIENT | 22 |
| FAULT_02 (B composition step) | FAULT_08 | 11 |
| FAULT_07 (C header pressure loss) | FAULT_08 | 13 |
| FAULT_06 (A feed loss) | FAULT_08 | 9 |

The pairs are exactly the ones the KB describes as the same mechanism with a different shape: step versus random variation (1/8, 5/12, 4/11), and valve sticking versus the inlet temperature that moves the same valve (14/11, 15 → normal). **A hidden fault in deployment will therefore be answered as its lookalike with high confidence, not as an abstention.** That is a property of the evidence, not of the calibrator: the classifier's entropy on those rows is low because the features really are similar.

---

## 4. Fold classifiers on their held-out faults (§17.5)

Calibration and threshold are rebuilt per fold from out-of-fold train events (fold_1: 1,562 events, isotonic, threshold 0.9792 at 4.53 %; fold_2: 1,681 events, isotonic, threshold 0.9897 at 4.72 %). **Export check, verified by command rather than assumed:** `val_eval.jsonl` has no `source_dataset` field; `fold_1/test` = 150 rows and `fold_2/test` = 100 rows, **250 total, 200 distinct, 50 shared** (fault 15 is held out in both folds), all in the validation range, **0 in any fold's train file, 0 in `train.jsonl`**. The runs 1-300 additions §17.5 describes are not in this export; §17.9 records that they are dropped rather than retrofitted. Every number below is a validation number.

### fold_1 — held out {4, 11, 15}

| split | n | declined | named a fault | named as | correct held-out name | accuracy vs gold |
|---|---|---|---|---|---|---|
| held-out, menu present | 75 | 50 (66.7 %) | 0 | – | 0 | 0.000 (gold is the held-out fault: impossible by construction) |
| held-out, menu removed | 75 | 50 (66.7 %) | 0 | – | 0 | **0.667** (gold is UNCLASSIFIED: 50 of 75 correct) |
| held-out, either variant, no_fault answer | 25 + 25 | – | – | NORMAL_TRANSIENT | – | – |
| kept faults, menu present | 381 | 132 (34.7 %) | 243 | FAULT_01 ×26, FAULT_02/05/06/07/10 ×25 | 365 | 0.958 |

False abstention on kept faults: **3.12 %**. Signals separating held-out from kept (menu): max_prob 0.9324, margin 0.9315, entropy 0.9349, novelty 0.9269.

### fold_2 — held out {14, 15}

| split | n | declined | named a fault | named as | correct held-out name | accuracy vs gold |
|---|---|---|---|---|---|---|
| held-out, menu present | 50 | 11 (22.0 %) | **23 (46.0 %)** | **FAULT_11 ×23** | 0 | 0.000 |
| held-out, menu removed | 50 | 11 (22.0 %) | 23 | FAULT_11 ×23 | 0 | 0.220 |
| held-out, no_fault answer | 16 remaining | – | – | NORMAL_TRANSIENT | – | – |
| kept faults, menu present | 406 | 136 (33.5 %) | 264 | FAULT_01/04 ×26, FAULT_02/05/06/07 ×25 | 387 | 0.953 |

False abstention on kept faults: **3.91 %**. Signals (menu): max_prob 0.7078, margin 0.7071, entropy 0.7078, novelty 0.8774.

**The two folds disagree, and the disagreement is the result.** The held-out set {4, 11, 15} is genuinely out of reach: zero names, two thirds declined, and the rest answered normal. The set {14, 15} is not: 46 % of its items come back as FAULT_11 — the lookalike §3.3 predicts — because valve sticking and the inlet-temperature random variation move the same valve. On this fold the novelty signal is markedly the best separator (0.877 vs 0.708), which is the one argument in the record for a distance-based open-set test rather than a confidence one.

---

## 5. Unobservable rows and the natural wrong labels (§17.6)

### 5.1 The twelve unobservable rows, item by item

Faults 3 and 9 act on D-feed temperature, a variable with no measurement in this dataset's evidence block, so the event is opened by the detector with nothing to attribute it to. Six rows are the menu-present twin (gold: the named fault) and six the menu-removed twin (gold: `UNCLASSIFIED_OPEN_SET`).

| item | fault | variant | gold | score | top \|z\| | CVA ratio | LSTM ratio | classifier | conf | rule output | adapter @ step 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EVT-val-f03-r302::menu | 3 | menu | FAULT_03 | 0.703 | 0.19 | 0.97 | 0.99 | NORMAL_TRANSIENT | 1.00 | NORMAL_TRANSIENT | NORMAL_TRANSIENT |
| EVT-val-f03-r302::openset | 3 | removed | UNCLASSIFIED_OPEN_SET | 0.703 | 0.19 | 0.97 | 0.99 | NORMAL_TRANSIENT | 1.00 | NORMAL_TRANSIENT | UNCLASSIFIED_OPEN_SET |
| EVT-val-f03-r317::menu | 3 | menu | FAULT_03 | 0.711 | 0.23 | 0.92 | 0.92 | NORMAL_TRANSIENT | 1.00 | NORMAL_TRANSIENT | NORMAL_TRANSIENT |
| EVT-val-f03-r317::openset | 3 | removed | UNCLASSIFIED_OPEN_SET | 0.711 | 0.23 | 0.92 | 0.92 | NORMAL_TRANSIENT | 1.00 | NORMAL_TRANSIENT | UNCLASSIFIED_OPEN_SET |
| EVT-val-f09-r301::menu | 9 | menu | FAULT_09 | 0.696 | 0.36 | 0.95 | 0.92 | NORMAL_TRANSIENT | 1.00 | NORMAL_TRANSIENT | NORMAL_TRANSIENT |
| EVT-val-f09-r301::openset | 9 | removed | UNCLASSIFIED_OPEN_SET | 0.696 | 0.36 | 0.95 | 0.92 | NORMAL_TRANSIENT | 1.00 | NORMAL_TRANSIENT | UNCLASSIFIED_OPEN_SET |
| EVT-val-f09-r302::menu | 9 | menu | FAULT_09 | 0.729 | 0.19 | 0.97 | 0.94 | NORMAL_TRANSIENT | 1.00 | NORMAL_TRANSIENT | NORMAL_TRANSIENT |
| EVT-val-f09-r302::openset | 9 | removed | UNCLASSIFIED_OPEN_SET | 0.729 | 0.19 | 0.97 | 0.94 | NORMAL_TRANSIENT | 1.00 | NORMAL_TRANSIENT | UNCLASSIFIED_OPEN_SET |
| EVT-val-f09-r317::menu | 9 | menu | FAULT_09 | 0.715 | 0.23 | 0.93 | 0.92 | NORMAL_TRANSIENT | 1.00 | NORMAL_TRANSIENT | NORMAL_TRANSIENT |
| EVT-val-f09-r317::openset | 9 | removed | UNCLASSIFIED_OPEN_SET | 0.715 | 0.23 | 0.93 | 0.92 | NORMAL_TRANSIENT | 1.00 | NORMAL_TRANSIENT | UNCLASSIFIED_OPEN_SET |
| EVT-val-f09-r319::menu | 9 | menu | FAULT_09 | 0.697 | 0.22 | 0.97 | 0.90 | NORMAL_TRANSIENT | 1.00 | NORMAL_TRANSIENT | NORMAL_TRANSIENT |
| EVT-val-f09-r319::openset | 9 | removed | UNCLASSIFIED_OPEN_SET | 0.697 | 0.22 | 0.97 | 0.90 | NORMAL_TRANSIENT | 1.00 | NORMAL_TRANSIENT | UNCLASSIFIED_OPEN_SET |

Both halves answer NORMAL_TRANSIENT on 12/12; the menu-removed gold agrees with that on 6/12, and the menu-present gold (`FAULT_03`, `FAULT_09`) never does. Confidence is 1.00 on every row, so **no confidence threshold, at any value, can move these twelve**.

### 5.2 Why a "no trace" rule cannot be a magnitude test

Evidence strength by stratum (validation, menu-present, event-deduplicated):

| stratum | n | top \|z\| min / p10 / median / p90 / max | CVA ratio median | overall score median |
|---|---|---|---|---|
| normal | 47 | 0.11 / 0.16 / 0.32 / 0.54 / 0.83 | 0.99 | 0.71 |
| unobservable (3, 9) | 6 | 0.19 / 0.19 / 0.22 / 0.23 / 0.36 | 0.95 | 0.70 |
| open set (16-20) | 125 | 0.11 / 0.25 / 4.36 / 29.34 / 34.11 | 59.60 | 55.27 |
| seen fault | 325 | 0.08 / 0.20 / 4.10 / 29.14 / 33.01 | 16.05 | 40.20 |

The twelve unobservable rows sit **inside the normal distribution on every signal** — and 10 % of real seen-fault events look equally weak (p10 top |z| = 0.20). So:

* a magnitude threshold cannot separate them from normal events, because they *are* normal events as far as the evidence goes;
* they cannot be separated from weak real faults either, for the same reason;
* and the two halves of the dataset disagree about the right answer (a named fault on one twin, an abstention on the other), which is a labelling question, not a modelling one.

**What a "no trace" rule would need (PROPOSAL only, not implemented):** a *paired* comparison against the fault's expected signature — which sensors the KB says must move for FAULT_03 and FAULT_09 — read as a test rather than as prose, plus a decision about whether "the mechanism happened but left no trace in this sensor set" is a correct answer or a dataset defect. Neither exists, and nothing in this section implements one.

### 5.3 Natural wrong labels from the cross-fit (§17.6, for a possible Option 2)

Of 1,919 out-of-fold train events: **8 wrong**, **254 abstained**, **258 (13.4 %) wrong or abstained**. By true fault (wrong / abstained): 01: 1/19, 02: 1/13, 04: 0/12, 05: 0/14, 06: 0/13, 07: 0/24, 08: 0/34, 10: 1/24, 11: 0/13, 12: 0/27, 13: 1/19, 14: 0/20, 15: 0/19, open set: 4/3.

By predicted class (wrong only): FAULT_08 2, FAULT_10 4, UNCLASSIFIED_OPEN_SET 2. By variant: menu 8 wrong and 64 abstained, removed 190 abstained. Lookalike split of the eight: 01→08 (1), 02→08 (1), 10→OPEN (1), 13→OPEN (1), OPEN→10 (4).

The catalogue is small and entirely made of the pairs §3.3 names: **44 of the 50 surrogate errors and all 8 natural errors are absorbed by FAULT_08, FAULT_11 or the open-set class**. That is the population an Option-2 "learn from natural wrong labels" dataset would be built from, and this section creates no such data.

---

## 6. Correction: the AUROC orientation (§17.9)

Stated plainly because it changed a headline. §17.3 fixes one orientation for all four signals — *higher = more likely the unseen class*. The first surrogate run ranked raw values, so `max_prob` and `margin` were scored at **1 − AUROC** and came out at 0.12, and the summary picked entropy. An unseen class gets a *lower* maximum probability and a *narrower* margin; the complement is the measurement.

* Fix: `llm/hybrid/calibration.py::ORIENTATION` + `orient_values()`, applied before ranking in both scripts, with a test asserting `AUROC(oriented) == 1 − AUROC(raw)` and that an unseen class with lower confidence ranks at 1.0.
* Effect on the choice: means margin 0.8808, max_prob 0.8815, entropy 0.8828, novelty 0.8456 — a 0.002 spread, inside the 0.005 tie tolerance, so §17.3's simplicity order decides and the chosen signal is **margin**. The earlier "entropy" is withdrawn.
* Effect on everything else: none. The abstention threshold is on the calibrated maximum probability (§17.2) and never referenced the chosen signal, so no abstention rate, accuracy or macro-F1 in §2 or §4 moves.
* Recorded in PROTOCOL §17.9 with the fold-export correction, dated 2026-11-01, after §17.1-§17.8 were committed.

The second correction in §17.9: the fold test components are validation-range rows, not the runs 1-300 additions §17.5 anticipated. Verified by command (§4), reported as validation, and the export is left as it is.

---

## 7. Detector transparency (§17.7a)

The contradiction detector fires on 0 of the 242 main-batch answers (§16). That number is only meaningful if a reader can see what it looked at, so: over all 242 answers there are **248 sentences that mention a fault other than the diagnosis**; every one carries verdict "not a contradiction"; **245** were decided by `no_assertion_cue` (the other fault is named without an assertion attached) and **3** by `rejection_guard` (an assertion pattern matched, but a rejection cue in the same sentence cleared it — these are the three "suppressed" rows). No sentence reached `assertion_adjacent`.

Twenty randomly chosen, for manual check (seeded, `random.Random(42)`):

| # | item | diagnosis | mentions | verdict | rule | sentence |
|---|---|---|---|---|---|---|
| 1 | EVT-val-f04-r320::menu | FAULT_04 | 14 | false | no_assertion_cue | "This matches FAULT_04's description … rejecting FAULT_14 (valve sticking) which would show decreasing flow." |
| 2 | EVT-val-f01-r306::menu | FAULT_01 | 06, 07 | false | no_assertion_cue | "FAULT_06 (A feed loss) is rejected because the A_Feed_Flow (XMV_44) is increasing, not decreasing, and FAULT_07 … as secondary effect." |
| 3 | EVT-val-f13-r320::openset | UNCLASSIFIED | 01 | false | no_assertion_cue | "This signature does not match any listed fault: FAULT_01 requires a decrease in A_C_Feed_Flow first, but … no such decrease" |
| 4 | EVT-val-f08-r317::openset | UNCLASSIFIED | 02 | false | no_assertion_cue | "FAULT_02 requires Component_B_Stream6 to increase first (not observed) …" |
| 5 | EVT-val-f07-r317::openset | UNCLASSIFIED | 01, 02 | false | no_assertion_cue | "… contradicting the expected behavior for FAULT_01 (A/C feed ratio step change)" |
| 6 | EVT-val-f04-r320::openset | UNCLASSIFIED | 01 | false | no_assertion_cue | "FAULT_01 (A/C feed ratio step change) is ruled out because the A_C_Feed_Flow is not deviating;" |
| 7 | EVT-val-f06-r317::menu | FAULT_06 | 01, 07 | false | no_assertion_cue | "FAULT_01 and FAULT_07 were rejected because FAULT_01 requires a step decrease …" |
| 8 | EVT-val-f04-r310::openset | UNCLASSIFIED | 14 | false | no_assertion_cue | "FAULT_14 … is ruled out because the valve position is increasing, not decreasing" |
| 9 | EVT-val-f17-r313::menu | UNCLASSIFIED | 01 | false | no_assertion_cue | "FAULT_01 … would not cause a direct cooling water flow increase without corresponding feed flow changes" |
| 10 | EVT-val-f15-r317::openset | UNCLASSIFIED | 01, 05 | false | no_assertion_cue | "… contradicts the step-change patterns of FAULT_01 … and FAULT_05 …" |
| 11 | EVT-val-f12-r317::menu | FAULT_12 | 15 | false | no_assertion_cue | "FAULT_15 was rejected as it would cause a decreasing flow, but the outlet temperature trend is increasing." |
| 12 | EVT-val-f11-r319::menu | FAULT_11 | 14 | false | no_assertion_cue | "FAULT_14 was rejected as it would cause a decreasing flow, but the flow is increasing." |
| 13 | EVT-val-f11-r306::menu | FAULT_11 | 04, 14 | false | no_assertion_cue | "… ruling out FAULT_04 (step change) and FAULT_14 (valve sticking) …" |
| 14 | EVT-val-f19-r317::menu | UNCLASSIFIED | 01 | false | no_assertion_cue | "FAULT_01 requires a decrease in A_C_Feed_Flow first, but no such decrease is observed" |
| 15 | EVT-val-f18-r310::menu | UNCLASSIFIED | 02 | false | no_assertion_cue | "FAULT_02 requires Component_B_Stream6 to increase first, but no such sensor is deviating." |
| 16 | EVT-val-f11-r311::menu | FAULT_11 | 04, 14 | false | no_assertion_cue | "… rejecting FAULT_04 … and FAULT_14 (valve sticking)" |
| 17 | EVT-val-f17-r310::menu | UNCLASSIFIED | 01 | false | no_assertion_cue | "FAULT_01 … would not cause such a large flow increase without corresponding temperature changes" |
| 18 | EVT-val-f07-r313::menu | FAULT_07 | 01 | false | no_assertion_cue | "FAULT_01 was rejected because it requires A_C_Feed_Flow to decrease first …" |
| 19 | EVT-val-f02-r317::openset | UNCLASSIFIED | 01, 03 | false | no_assertion_cue | "This pattern contradicts FAULT_01 … and FAULT_03 …" |
| 20 | EVT-val-f11-r317::openset | UNCLASSIFIED | 04 | false | no_assertion_cue | "This contradicts FAULT_04 … which would require …" |

The three cleared by the rejection guard:

* `EVT-val-f12-r310::menu` — "The rejected FAULT_05 (step change) is ruled out by the slow-drift ratio 0.37 … and FAULT_15 … shows increasing deviation … rather than …"
* `EVT-val-f05-r310::openset` — "… no listed fault matches the observed signature: FAULT_15 … would show a decre…"
* `EVT-val-f12-r320::openset` — "… FAULT_05 requires Condenser_Cooling_Water_Flow increase (not observed), and FAULT_15 requires …"

Read by hand, all 20 are differential reasoning — the model naming the alternative in order to exclude it — and the three guard rows are the same thing with a "would show"/"ruled out" cue inside the sentence. So the detector's 0 % is correct on this sample, but the *rule that carries it* is `no_assertion_cue`: the verdict is mostly "no verb of attribution next to the fault number", which is a weak positive claim. A sentence such as "the fault signature is characteristic of FAULT_11" would be caught; "this is FAULT_11's signature" — with the number placed after the cue phrase — depends on the 32-character window the regex allows.

---

## 8. Valve/flow (§17.7b)

* **Does the report-side validator already catch the 21 mislabels?** No. `llm/response/validators.py` has no valve- or label-kind rule (`report_validator_has_a_valve_rule: false`); the rule exists only on the follow-up path (`llm/followup/filters.py::_valve_mislabel`). The 21 of 242 (8.7 %) went out unflagged by the report path.
* **GPU state at the time:** 618 MiB used, 36 % util, 38 °C — judged idle, so the retry ran as permitted. 21 items, merged-adapter copy (adapter files untouched), one retry each, 512-token cap, **14.43 s/item**.
* **Result:** schema valid 21/21, contradictions 0; **7 fixed**, **13 unchanged** (same mislabel as before), **1 moved to a different mislabel** (it had called a measurement a valve position; it came back calling a valve position a temperature). Before: 10× XMV_50-as-temperature, 3× XMV_51-as-temperature, 2× XMV_45-as-pressure, 2× XMV_49-as-level, 2× XMEAS_19-as-valve, 1× XMEAS_21-as-valve, 1× XMV_47-as-pressure.

A single retry is a **33 % fix** on this failure mode and the retry cannot be trusted to stay put. If the report path ever hands the model a diagnosis, it needs a rejecting validator and a prose-reuse rewrite, not a stronger instruction. `RETRY_MISLABEL` and `retry_question()` are implemented in `llm/hybrid/generate.py` for whoever builds that.

---

## 9. RESPONSE_SPEC §9.1 — proposal updated, live path untouched

`docs/RESPONSE_SPEC.md` §9.1 now says what §17 measured, and it says it differently from its first draft (§6):

1. **Second channel for "I do not recognise this".** All three confidence-shaped signals separate unseen from seen about equally well (0.881-0.883); the problem is the *threshold*, not the ranking — saturation keeps 774 of 834 maxima at or above 0.979, so the threshold abstains 4 times in 497 while fold_2 names FAULT_11 on 23 of 50 never-seen items. What a hybrid needs is a **reference** — nearest train events or the KB's expected signature — not a better confidence number.
2. **A grounding verdict the model is allowed to use.** Forced-diagnosis generation contradicted its prefix on 0 of 242 items and rationalised a deliberately wrong prefix on 58 of 58. A prose writer that cannot disagree is a paraphrase, so the contract needs a system-computed grounding field plus a validator that fails an answer contradicting it.
3. **Tag kinds enforced, not printed.** 21 of 242 mislabelled a tag's kind despite the kind being printed on every tag; one retry fixed 7 of 21 and moved 1.

Still a proposal: nothing in the live path, the prompts, the datasets or their hashes changed, and §9's original text stands.

---

## 10. Tests and artefacts

* `python -m pytest tests -q` — **558 passed** (app env).
* `./venv_train/Scripts/python.exe -m pytest tests -q` — **555 passed, 3 skipped** (the skips are the lightgbm-dependent tests; `venv_train` has no lightgbm).
* New tests: `tests/test_hybrid_calibration.py` (**14 passed**) — event-grouped fold determinism, OOF-only ECE, a threshold that only train sees, AUROC ties and symmetry, **the orientation contract and its complement identity**, the tie rule, novelty scale invariance, and the detector's decision rule.

Artefacts (new files only): `outputs/analysis/phase3a/hybrid/{crossfit.json, crossfit_val_rows.jsonl, preds_protocolA_val.jsonl, surrogate_wide.json, surrogate_shard1.json, surrogate_shard2.json, fold_holdout.json, hardening_checks.json, valve_flow_retry.json}`; code: `llm/hybrid/calibration.py`, `scripts/hybrid_{crossfit,surrogate_wide,surrogate_summary,fold_holdout,unobservable,valve_flow_retry}.py`.

Reproduce: `python scripts/hybrid_crossfit.py`, `python scripts/hybrid_surrogate_wide.py --faults 1,2,4,5,6,7,8,10,11,12,13,14,15` (two shards in parallel), `python scripts/hybrid_surrogate_summary.py`, `python scripts/hybrid_fold_holdout.py`, `python scripts/hybrid_unobservable.py`, `python scripts/hybrid_valve_flow_retry.py --generate`.

---

## 11. Where this leaves the hybrid, and what is still open

**The diagnosis half is good and honestly measured.** Cross-fitted macro-F1 0.9708 on 822 observable validation rows, against 0.9263 for the adapter alone — the hybrid's premise survives removal of the calibration leakage, with 4.5 points rather than 7.

**The open-set half is not solved.** Three independent lines of evidence say the same thing: confidence thresholds abstain almost never (22 of 834), a confidently wrong lookalike is the dominant error mode (44 of 50 surrogate errors, 8 of 8 natural ones), and the held-out fold that contains a lookalike names it 46 % of the time. The strongest single signal for that case is novelty in fold_2 (0.877), which is the only signal that is a distance to the training set rather than a restatement of the classifier's opinion — worth one more look before any deployment decision, and worth nothing on a low-latency path until someone prices the reference-set storage.

**Two dataset questions are now blocking, and neither is a model problem.** The twelve unobservable rows ask for a named fault whose sensors are not in the event, and the two halves of the dataset disagree about what the right answer is; §5.2 shows no signal can find them and no strength threshold can separate them from normal events. Until that is decided, those rows will keep scoring 0 for every model, base or adapted.

**Still open, and not started:** no test-component scoring (that stays closed until the report and follow-up audits pass); no second training seed or fold run; no adapter retrain on the diagnosis-forced target; no deployment calibration on a population support.
