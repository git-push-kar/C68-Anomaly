# HYBRID TRANSITION — the classifier names the fault, the adapter explains it

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade` · **Environments:** app env (CPU, LightGBM), `venv_train` (GPU)
**Date:** 2026-11-01 · **Scope:** pre-registration, classifier, validation tables, forced-diagnosis faithfulness, comparison. **No adapter training. No test component opened. Run-1 adapter and checkpoints unchanged (verified by hash).**

---

## 0. Gate table

| # | Gate item | Status | Evidence |
|---|---|---|---|
| 1 | PROTOCOL §16 dated and written **first** | **PASS** | §1 — committed as `7f35b55` before any classifier existed; §16.7 budget note written before the batch was read |
| 2 | Classifier artifacts hashed into the manifest | **PASS** | §2.3 — booster, calibration, threshold per variant |
| 3 | Validation tables for classifier, adapter-only and hybrid | **PASS** | §3 — seven strata, same rows, paired |
| 4 | Abstention surrogate numbers | **PASS** | §3.4 — faults 5, 8, 12, 13 |
| 5 | Faithfulness and contradiction results with the counterfactual | **PASS** | §5 — 300 generated items (242 main + 58 counterfactual), 0/242 contradictions after the detector was corrected, 58/58 counterfactual rationalisations |
| 6 | Audit CSV, 100 rows, scoring blank | **PASS** | §5.5 — `audit_hybrid_100.csv` |
| 7 | Tests pass in both environments | **PASS** | §7 — 539 (app), 536 + 3 skipped (`venv_train`) |
| 8 | No test data touched | **PASS** | every input is `outputs/benchmark_v2/dataset/val.jsonl` |
| 9 | Run-1 files unchanged | **PASS** | §7.2 — checkpoint-500 `adapter_model.safetensors` matches its recorded hash, differs from every other checkpoint |

**Headline.** On the 822 observable validation rows, replacing the adapter's *diagnosis* with a
LightGBM classifier plus the menu rule moves accuracy from **0.9416 to 0.9964** and macro-F1
from **0.9263 to 0.9963** (McNemar b=1, c=46, p≈0). The gain is not spread evenly and is not
free: it is +26 items where an open-set fault was named on a menu the adapter could see
(0.7920 → 1.0000), +15 fault-identification errors, and it goes **backwards** on the 12
unobservable rows (0.5000 → 0.0000), because calibrated confidence never flags a fault whose
sensors are absent. The abstention arm of the rule is close to inert: isotonic calibration
returns exactly 1.0 on 828 of 834 rows, so the pre-registered "largest threshold with false
abstention ≤ 5 %" is **1.0**, and the confidence path abstains on 4 items. The honest
description is that the hybrid's diagnosis quality is the classifier's *closed-set* quality,
and the abstention is doing almost nothing.

The explanation half comes out clean and empty at the same time. Given the diagnosis in the
prefix, the adapter writes schema-valid, complete answers for all 242 items and never once
contradicts the label (0 of 242). Given a **wrong** label, it does exactly the same thing: 58 of
58 counterfactual answers argue for the wrong fault, and 8 of them explicitly rule out the true
one. Its prose is faithful to the prefix, not to the evidence — which is the finding this
hybrid was built to expose, and the reason it is reported as a viable *proposal* rather than an
improvement to adopt.

---

## 1. What was pre-registered, and when

`docs/PROTOCOL.md` §16 (dated 2026-11-01) was written and committed **before** the classifier
was trained — commit `7f35b55`, whose tree contains no `llm/hybrid/` and no
`scripts/hybrid_classifier.py`. It fixes:

* the definition under test (detector → classifier → menu rule → adapter prose → code);
* LightGBM 400 trees / lr 0.05 / 15 leaves / `min_child_samples` 10 / subsample 0.9 /
  colsample 0.6 / L2 1.0 / seed 42, on the features `scripts/evidence_classifier.py` parses
  from the rendered evidence block, classes = true fault 1-15, `UNCLASSIFIED_OPEN_SET` for
  16-20, `NORMAL_TRANSIENT`;
* calibration by isotonic (one-vs-rest, renormalised) versus temperature, chosen by the lower
  **5-fold cross-validated** ECE within validation;
* the threshold rule: **largest** calibrated max-probability threshold with false abstention
  ≤ 5 % on menu-present seen-fault items;
* what is measured, and the rejection conditions (§16.6).

Two notes on numbering and budget, both dated and recorded before the numbers existed:

* §15 was already taken by the instrument correction earlier today, so the hybrid
  pre-registration is **§16**, not §15 as the task text assumed.
* §16.7 records the budget **as executed** (242 main + 58 counterfactual = 300). The
  unobservable family has only 12 validation rows in the whole split, so no sample can fill
  its 24-item quota; the two rows released went to the counterfactual, and the total stayed
  inside the hard cap.

`docs/RESPONSE_SPEC.md` §9 records the prompt contract **as a proposal only**: nothing in
`llm/inference.py`, the System prompt, the benchmark items, the datasets or their hashes was
changed by it.

---

## 2. The classifier

### 2.1 Training

| variant | held-out faults | train events | features | calibration chosen | CV ECE (isotonic / temperature) |
|---|---|---|---|---|---|
| protocolA | — | 1,919 | 323 | isotonic | **0.02008** / 0.02102 |
| fold_1 | 4, 11, 15 | 1,562 | 323 | isotonic | 0.13212 / 0.18956 |
| fold_2 | 14, 15 | 1,681 | 323 | isotonic | 0.09594 / 0.11462 |

Trained on **train events only**, one row per event: the two menu variants of an episode carry
the same evidence, so they are deduplicated before fitting (a classifier trained on both would
see one event twice *and* learn that it has two different gold answers). `deterministic=True,
force_row_wise=True, n_jobs=1` are set so the surrogate models later are comparable to this
one; a test asserts a re-fit reproduces the predictions bit-for-bit.

The **feature audit** is clean on all three variants: 323 features, no name containing any of
`item_id, run, event, index, order, position_in, seed, split, hash, simulation` (the banned
list is carried in the manifest with the audit's verdict). The features are parsed from the
rendered prompt, so the classifier reads the same page the adapter does.

### 2.2 The menu rule and the threshold

`llm/hybrid/menu_rule.py::menu_rule` maps one prediction to `(diagnosis, verdict, reason)`:

| classifier says | item's menu | result |
|---|---|---|
| `FAULT_13`, p ≥ threshold | contains it | `classified` |
| `FAULT_13`, p < threshold | contains it | `UNCLASSIFIED_OPEN_SET`, reason `below_threshold` |
| `FAULT_07` | does not contain it | `UNCLASSIFIED_OPEN_SET`, reason `not_on_menu` |
| `NORMAL_TRANSIENT` | — | `no_fault`, never an abstention |
| nothing | — | `UNCLASSIFIED_OPEN_SET`, reason `no_prediction` |

Threshold on Protocol A validation: **1.0**, achieved false abstention **0.008048** (4 of the
497 menu-present seen-fault items), budget 0.05. That number is the finding in §3.3: isotonic
calibration saturates, so the "largest threshold inside the budget" is the top of the range and
the rule abstains almost never.

### 2.3 Artifacts and hashes

`outputs/analysis/phase3a/hybrid/manifest.json` (tracked) records, per variant, the SHA-256 of
`classifier.txt`, `calibration.json`, `threshold.json`:

| variant | classifier.txt | calibration.json | threshold.json |
|---|---|---|---|
| protocolA | `273188e2…d3175e6` | `1a7f892f…422f18d` | `f7ea4fe8…5110638` |
| fold_1 | `f721c950…52cbbf3a` | `196ef161…43a43c280` | `cb7987ec…2cffe04515` |
| fold_2 | `30530dd5…6ef1ac51d9f` | `febb2a3a…5e5d2daee8` | `8386710b…c1ba1b159` |

The 5 MB boosters and the per-item prediction files stay ignored; `.gitignore` was extended so
that the small JSONs that the tables come from are tracked (the same convention as
`outputs/benchmark_v2/eval/`).

---

## 3. Protocol A validation — classifier, adapter, hybrid on identical rows

All 834 validation rows, both menu variants. Adapter = §14's frozen decode instrument at step
500 (the reference the task names); classifier = classifier + menu rule. **Hybrid diagnosis is
the classifier column by construction** — the adapter cannot change it — so the hybrid is
reported once, and §5 measures what it contributes instead: the prose.

### 3.1 The table

| stratum | n | adapter acc | classifier+rule acc | adapter macro-F1 | classifier macro-F1 | McNemar (b / c, p) |
|---|---|---|---|---|---|---|
| observable | 822 | 0.9416 | **0.9964** | 0.9263 | **0.9963** | 1 / 46, p≈0 |
| menu-present | 497 | 0.9034 | **0.9940** | 0.9199 | **0.9958** | 1 / 46, p≈0 |
| menu-removed | 325 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 / 0 |
| fault named (menu, gold 1-15) | 325 | 0.9508 | **0.9938** | 0.9593 | **0.9969** | 1 / 15, p=0.0005 |
| `NORMAL_TRANSIENT` | 47 | 0.8723 | **0.9787** | 0.9318 | **0.9892** | 0 / 5, p=0.0625 |
| open-set on a menu | 125 | 0.7920 | **1.0000** | 0.8839 | **1.0000** | 0 / 26, p=1e-6 |
| unobservable (3, 9) | 12 | **0.5000** | 0.0000 | 0.3333 | 0.0000 | 6 / 0, p=0.031 |

`b` = adapter right and classifier wrong, `c` = classifier right and adapter wrong.

Two rows deserve their own sentence, because the aggregate hides them:

* **open-set on a menu (125):** an adapter shown a menu that does not contain the true fault
  still names a menu fault 26 times out of 125 (20.8 %); the classifier predicts
  `UNCLASSIFIED_OPEN_SET` from the evidence and the rule passes it through. This single stratum
  is 56 % of the classifier's entire advantage.
* **unobservable (12):** faults 3 and 9 have no sensor signature in the evidence, so the pair
  cannot be separated by reading sensors — the adapter abstains on half of them (6/12) and the
  classifier abstains on **none**. Calibrated confidence does not encode "this fault leaves no
  trace in the evidence", because the classifier never saw a training example like that. This is
  a concrete reason the hybrid is *not* strictly better, and it is the one place the adapter
  wins.

### 3.2 Abstention, per class, and what top-3 means here

| metric (observable, n=822) | adapter | classifier + rule |
|---|---|---|
| abstention rate | 0.5207 | 0.5511 |
| abstention precision | 0.9907 | **0.9934** |
| abstention recall | 0.9422 | **1.0000** |
| false abstention (answerable items declined) | 0.0108 | **0.0081** |
| missed abstention (unclassifiable items answered) | 0.0578 | **0.0000** |

Per-class recall on the 325 named-fault rows — classifier first, adapter second:
`FAULT_10` 0.96 / 0.92, `FAULT_15` 0.96 / 0.68, `FAULT_08` 1.00 / 0.92,
`FAULT_12` 1.00 / 0.96, `FAULT_13` 1.00 / 0.92, `FAULT_06` 1.00 / 0.96, all other faults
1.00 / 1.00. On `NORMAL_TRANSIENT`: 0.9787 / 0.8723. The classifier's two named-fault misses are
**both false abstentions** (FAULT_10 at p=0.75, FAULT_15 at p=0.50), not misclassifications;
without the rule it is 325/325 on that stratum.

**Top-3 is 1.0000 on menu-present and 0.0 on menu-removed**, which is not a bug in the
classifier: the probability vector is over the 17 *classes of event*, so on a menu-removed row
the gold answer is `UNCLASSIFIED_OPEN_SET` and the vector's own top entry is the true fault. The
aggregate top-3 of 0.6046 is an artefact of mixing those two questions, and the report quotes
top-3 per stratum only.

### 3.3 The threshold is inert, and why

`828 of 834` calibrated maxima are **exactly 1.0**; six rows sit at 0.5 / 0.5 / 0.5 / 0.75 /
0.75 / 0.75. Isotonic regression is a step function, and around a confident classifier its top
step is flat at 1.0, so the calibrated confidence carries almost no information about
correctness. Consequences, stated rather than fixed:

* the rule abstains on **4 of 497** menu-present rows; two were correct faults (false
  abstention), one was the open-set row (still correct), one was a `NORMAL` row the classifier
  had already mislabelled as `FAULT_15`;
* the reported ECE/Brier (0.0009 / 0.0017 on 822 rows) are **in-sample** and must not be quoted
  as calibration quality: the calibrator and the threshold were both fitted on the same
  validation rows they are scored on. The honest calibration figure is the **cross-validated
  ECE, 0.0201**, and even that is optimistic because the *method* was chosen on the same split;
* the pre-registered rule was not changed after seeing this. Changing it now would be tuning on
  the selection set.

### 3.4 Leave-one-fault-out abstention surrogate

A class the model has never seen is the honest test of "does the rule abstain when it should".
Retrained on Protocol A train minus one fault, calibrated on the other faults only, threshold by
the same rule:

| held-out fault | validation items | abstained | rate | mislabelled as | false abstention on the other faults |
|---|---|---|---|---|---|
| `FAULT_05` | 25 | **0** | 0.00 | `FAULT_12` ×25 | 0.0085 |
| `FAULT_08` | 25 | 15 | 0.60 | `FAULT_01` ×8, `FAULT_13` ×2 | 0.0021 |
| `FAULT_12` | 25 | 23 | 0.92 | `UNCLASSIFIED_OPEN_SET` ×2 | 0.0085 |
| `FAULT_13` | 25 | **25** | 1.00 | — | 0.0064 |

The pattern is interpretable and uncomfortable: `FAULT_13` and `FAULT_12` are caught because
their evidence is distinctive, `FAULT_05` is not caught at all — every one of its 25 items is
named `FAULT_12` with full confidence, and fault 5 is the KB's own differential for 12. A
confidence threshold cannot express "I have never seen this class"; it can only express "this
looks unlike my classes", and a lookalike looks like its lookalike.

---

## 5. Faithfulness — what the adapter does when the diagnosis is not its own

### 5.1 The batch

300 generations, one per item, checkpoint-500 adapter merged into an in-memory copy of the base
model (the adapter on disk was never written to), 512-token cap, LM-direct decode (no vision
tower). 242 main + 58 counterfactual, exactly the budget of §16.5/§16.7.

| family | n | schema valid | complete | grounded clean (raw) | valve/flow mislabels | s/item |
|---|---|---|---|---|---|---|
| menu-present fault items | 110 | 110 | 110 | 0.9182 | 11 | 13.68 |
| menu-removed | 50 | 50 | 50 | 0.9200 | 7 | 14.92 |
| `NORMAL_TRANSIENT` | 45 | 45 | 45 | 0.8889 | 0 | 12.51 |
| open-set on a menu | 25 | 25 | 25 | 0.9600 | 2 | 14.53 |
| unobservable (3, 9) | 12 | 12 | 12 | 1.0000 | 1 | 15.27 |
| **main total** | **242** | **242 (1.00)** | **242 (1.00)** | **0.9215** | **21** | **13.89** |
| counterfactual | 58 | 58 | 58 | 0.8966 | — | 13.9 |

No generation failed, nothing hit the cap incomplete: **completion 242/242 at 512 tokens**
(mean 286 tokens), so the §12.2 fallback to 768 did not trigger. Wall clock 69 minutes for
300 items.

The forced diagnosis matched the item's gold on **235 of 242** rows: 110/110 menu-present
faults, 50/50 menu-removed, 25/25 open-set-on-a-menu, 44/45 `NORMAL` (one `NORMAL` row the
classifier mislabels `FAULT_15` and the rule then abstains), and **6/12 unobservable** — where
the 12 rows are not one convention: six have `UNCLASSIFIED_OPEN_SET` as gold and six have the
named fault, so a forced abstention cannot be right on more than six.

### 5.2 The contradiction detector was wrong twice, and both times in the same direction

Reported in this order because the first two readings are what a literal implementation of the
§16.4 checks gives, and they are wrong.

1. **As specified** (cue word anywhere in a sentence that mentions another fault; rank check
   applied to every verdict): 107 of 242 rows flagged (44.2 %), and 19 of 110 in the classified
   stratum (17.3 %). Reading the flags: **88** were `ranked_top_mismatch` on rows whose verdict
   *is* an abstention, where ranking a candidate is the honest way to write the answer;
   **51** were sentences that name another fault in order to **reject** it ("FAULT_14 was
   rejected because it would cause the valve position to decrease"); **2** were hedges of a
   process variable ("the cooling flow does not respond") that merely mention the diagnosis.
2. **Verdict-scoped, with a rejection guard**: 20 rows (8.3 %), all still false positives —
   every one was of the form "FAULT_05 … *would cause* a flow increase, but the evidence shows
   no such change".
3. **Affirmation-scoped** (the asserting cue must sit next to the label, the sentence must not
   be a rejection, and a hedge must deny the diagnosis itself): **0 of 242**. The detector now
   suppresses only 3 sentences that would otherwise have been flags, and all three were read:
   "FAULT_15 … is ruled out", "no listed fault matches … FAULT_15 would sh[ow]", "FAULT_05
   requires … (not observed)". The guard is doing the right thing, and the batch contains **no
   prose that contradicts the diagnosis it was handed**.

This is a measurement correction, not a model change: the 300 answers are the same bytes; only
the detector that reads them changed, and it changed before any conclusion was drawn from the
first two readings. The corrected rules are pinned by tests (`test_rejecting_a_differential_is_
not_a_contradiction`, `test_a_process_description_is_not_a_hedge_about_the_diagnosis`).

### 5.3 Grounding: 27 flags, 5 real ones

`evaluate_v2.grounding_violations` is a rejection rule, not a counting instrument: it names
every number not found verbatim in the store and every tag not in the evidence. Adjudicating all
27 flags on the main batch one by one (`scripts/hybrid_grounding_audit.py`):

| kind | count | example |
|---|---|---|
| digit inside a sensor name | 4 | `9` of `Component_F_Stream9` |
| fragment of a longer number | 1 | `0.33` inside `0.333` |
| stored percentage written to fewer digits | 15 | `223 %` for a stored 222.7 |
| **genuine** number not in the store | **4** | `3.24 %` where the store has 2.40 |
| **genuine** tag not in the evidence | **3** | `XMEAS_33` on three rows of one event |

So **5 of 242 rows (2.1 %)** carry a genuine grounding flag, and the raw rate of 88.8 %
"clean" is an artefact of the matcher. Run-1's 50-item generation (§14 reference) has 2 flags,
**both artefacts and 0 genuine**. 5/242 against 0/50 is not evidence of a difference (at 2.1 %
the chance of seeing zero in 50 rows is ≈ 0.35), so **the §16.6 grounding condition is not
shown to be violated** — and it is not shown to be met either; the honest statement is that both
batches write a small number of ungrounded numbers and the hybrid's sample is too small to
separate them.

**Valve/flow mislabels are the largest prose defect: 21 of 242 rows (8.7 %)** describe a
manipulated variable as a measurement or the reverse — `XMV_45 described as pressure`,
`XMV_50 described as temperature`, `XMEAS_21 described as valve position`. It is the one place
where the prose contradicts the context it was given, and it is not measured on run-1, so it
starts here as a baseline rather than a regression.

### 5.4 Counterfactual: the prose follows the label, not the evidence

58 menu-present items with a **wrong** diagnosis forced — the classifier's own errors first
(there are none in that pool: it is 325/325 on named faults), then the KB's own differential for
the true fault. Result: **58 of 58 rationalised (100 %), 0 contradicted, 0 hedged**.

| forced | true fault | what the prose did |
|---|---|---|
| `FAULT_09` | `FAULT_10` | "This matches FAULT_09's description of C feed temperature random variation, where the C feed temperature would cause a random variation…" |
| `FAULT_05` | `FAULT_12` | "…indicating a cooling water inlet temperature disturbance. … consistent with a condenser cooling water inlet temperature step." |
| `FAULT_04` | `FAULT_11` | "The disturbance 'Reactor cooling water inlet temperature step' matches the observed increase in cooling water flow valve position…" |

**8 of the 58 go further and explicitly reject the true fault**: "FAULT_14 (valve sticking) was
rejected because the valve position trend is increasing, not stuck" — on an event whose true
fault *is* 14, whose KB signature is that increasing valve position. When the model is handed a
label it argues for it and argues against the alternative, including against the correct answer
and sometimes against the evidence itself. This is the failure mode that makes a hybrid
proposal rather than a hybrid default: the prose cannot be read as independent confirmation of
the diagnosis it accompanies.

Nothing was repaired: contradictions are recorded as written, and the fallback (KB-backed,
cannot contradict the diagnosis) was implemented but not used, because no generation failed
validation.

### 5.5 The audit sheet

`outputs/analysis/phase3a/hybrid/audit_hybrid_100.csv` — 100 rows stratified by family, with
`item_id, family, forced_diagnosis, correct_diagnosis (y/n), prose` filled and
`correct, grounded, consistent_with_diagnosis, useful, notes` **blank**. Nothing in it is scored.

---

## 6. The comparison, and what the hybrid is actually worth

| | diagnosis | explanation |
|---|---|---|
| **adapter alone** (§14, step 500) | macro-F1 0.9263 on 822 rows; abstention precision 0.9907, recall 0.9422; **20.8 %** of open-set-on-menu items name a menu fault | writes everything, including the label it may not be able to justify |
| **classifier + menu rule** | macro-F1 0.9963 on the same rows; recall 1.0, false abstention 0.0081; **0 %** open-set-on-menu items name a menu fault | writes nothing |
| **hybrid** (classifier diagnosis + adapter prose) | identical to the classifier column, by construction | 242/242 schema-valid and complete at 512 tokens; **0** contradictions of the supplied diagnosis; **2.1 %** of rows with a genuine grounding flag; **8.7 %** valve/flow mislabels; **58/58** counterfactual rationalisations |

Read against the rejection conditions written in §16.6 before any of this existed:

1. *"rejected if the classifier+menu-rule diagnosis is worse than the adapter's own macro-F1
   0.9263 on the same rows"* — **not triggered**: 0.9963 > 0.9263.
2. *"rejected if the adapter's forced-diagnosis prose contradicts the supplied diagnosis on more
   than 10 % of items"* — **not triggered**: 0 of 242 under the corrected detector (§5.2).
   Under the literal first implementation it would have been 44.2 %, which is why the detector's
   own history is reported rather than only its final number.
3. *"rejected if grounding violations exceed the run-1 rate"* — **not demonstrated**: 5 of 242
   rows (2.1 %) against 0 of 50 in run-1; at 2.1 % the probability of zero flags in 50 rows is
   ≈ 0.35. The condition neither rejects nor clears the hybrid.

The conditions do **not** cover the counterfactual, which is the strongest negative result in
this report: handed a wrong diagnosis, the adapter argues for it in 58 of 58 cases and rejects
the true fault in 8. That is not a rejection criterion because it was not written as one before
the numbers existed; it is the reason the recommendation below is "proposal, not deployment".

What the table cannot settle, and what the next stage would have to:

* **The classifier's 0.9964 is not a 0.9964 on a harder population.** It is measured on 834
  validation rows from the same runs the training data came from, with 55 % of those rows
  answerable by abstaining. `FAULT_15` (17/25 recall with the adapter) and `FAULT_10` are the
  classes
  where both halves of the hybrid are weakest, and they stay the classes to watch.
* **The unobservable rows are a real regression** (0.5 → 0.0 on 12 rows). A deployment would
  need either the adapter's abstention on those faults or a feature that says "nothing here".
* **Calibration is the open defect.** A saturated calibrator makes the confidence field
  decorative and the abstention arm untestable in practice; the surrogate in §3.4 shows what
  the arm *would* do on genuinely unseen classes, and it is not uniform.

Validation is the **selection set** (§15.2): every number here was seen before any decision
about it, and no test component was read, scored or generated from. The instrument, the
candidate set, the classifier configuration and the threshold rule are frozen.

---

## 7. Tests, files and what changed

### 7.1 New code

| file | what it is |
|---|---|
| `llm/hybrid/menu_rule.py` | the decision: prediction + menu + threshold → diagnosis, verdict, reason |
| `llm/hybrid/generate.py` | forced prefix, parse/validate, contradiction signals, KB-backed fallback, LM-direct decode, valve/flow check |
| `scripts/hybrid_classifier.py` | train / evaluate / surrogate for the three variants, with hashes |
| `scripts/hybrid_faithfulness.py` | the 300-item forced-diagnosis batch, counterfactual, audit CSV |
| `scripts/hybrid_comparison.py` | the three-way table from saved prediction files |
| `tests/test_hybrid.py` | 21 tests: menu rule, threshold rule, prefix forcing, contradictions, fallback, determinism, feature audit |

A defect the tests caught before the batch ran: `contradictions()` matched the diagnosis by
stripping the zero from the fault number, so it watched for `FAULT 5` while the diagnosis was
`FAULT 13` — the hedge check could never fire for faults 10-15. Fixed, and the fix is pinned by
a test that walks `FAULT_05`, `FAULT_13`, `FAULT_15`.

### 7.2 Nothing else moved

* `git diff` on `llm/inference.py`, `llm/dataset.py`, prompts, datasets and hashes: **empty**.
* run-1 adapter: `checkpoint-500/tep_rca_v2_protocolA_seed42/adapter_model.safetensors`
  hashes to `9fa37a3d…` which is the value recorded in `training_summary.json`, and differs
  from checkpoints 100/200/300/400/540 and from the final copy — i.e. it is the run-1 file,
  unmodified. `merge_and_unload` was applied to an in-memory copy for generation.
* GPU: idle before the batch (527 MiB, desktop only), used only for the 300 forced-label
  generations (≤ 300 items, hard stop).

### 7.3 Test suites

| environment | result |
|---|---|
| app env (LightGBM present) | **539 passed** in 13.5 s |
| `venv_train` | **536 passed, 3 skipped** in 11.6 s (the three LightGBM tests skip there by design) |

---

## 8. What is decided, and what is not

**Decided (validation only, frozen):** the hybrid's *diagnosis* is the classifier + menu rule;
it is better than the adapter's own on every stratum except the 12 unobservable rows, and the
rejection conditions of §16.6 on diagnosis are not triggered.

**Measured, and worth more than the diagnosis gain:** the adapter's prose is faithful to a
label it did not choose — 0 contradictions in 242 rows — but it is faithful to *any* label: the
58 counterfactual rows are all rationalised, and 8 explicitly reject the true fault with an
argument the KB contradicts. Faithfulness here means consistency with the prefix, not with the
evidence.

**Not decided:** whether to deploy it. The proposal in `RESPONSE_SPEC.md` §9 stays a proposal;
the live path, the prompts and the datasets are untouched. Four things stand between this and a
deployment, in the order they would have to be fixed:

1. **Calibration that means something.** Isotonic saturates (828/834 rows at exactly 1.0) and the
   pre-registered threshold is therefore 1.0; the abstention arm abstains 4 times in 497. Until
   the confidence is informative, the rule is a menu check with a decoration.
2. **A signal that can say "this fault leaves no trace".** The 12 unobservable rows are the only
   stratum where the hybrid is worse than the adapter (0.0 vs 0.5), and the leave-one-fault-out
   surrogate shows `FAULT_05` is never abstained at all — it is renamed `FAULT_12` 25 times out
   of 25.
3. **Prose that can disagree.** If the report is to carry the classifier's label, the model must
   be able to write "the evidence does not fit this well", and today it never does. That is a
   target-level change (a per-item grounding verdict in the prompt), not a decoding setting.
4. **The valve/flow mislabel rate (8.7 %).** It is a wording error with a mechanical fix (the
   prompt already labels each tag's kind), and it should not survive into a second run.

If those are not addressed, the honest deployment is the classifier's diagnosis with a
*deterministic* KB-built explanation — which already exists as `fallback_answer()` — and no
adapter prose at all.
