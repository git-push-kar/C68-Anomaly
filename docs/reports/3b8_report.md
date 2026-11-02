# 3B-8 — FALLBACK V2, THE A/B REBUILD, VALIDATOR RECALL, COUNT RECONCILIATION

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade` · **Date:** 2026-11-01
**Environment:** app env (CPU) for everything; `venv_train` for the tokenizer count and its test suite
**Scope:** §20 pre-registered and committed **first**; fallback v2 measured against v1 on all 834
validation rows; the A/B sheet rebuilt on 100 non-counterfactual items; validator recall on 30
synthetic positives and 30 matched negatives; adapter defect counts; three reconciliations; the
prose-only training spec. **No adapter trained. No test component opened. No teacher call. No run-1
file modified. GPU not used** — see §7 for why, stated plainly.

---

## 0. Pre-registration, committed before any number

`docs/PROTOCOL.md` **§20**, commit **`d140700`**, written and committed **before** v2 existed in
code and before any figure below was computed. It fixes:

* **(a)** §19.2's comparator is **fallback v2**, because v1 quotes the KB example event's numbers;
  v1's numbers stay reported; the sheet is rebuilt blank before any scoring;
* **(b)** the rule reads **non-counterfactual rows only** (100 distinct items, none forced-wrong);
  counterfactual stays a separate, already-measured result; percentages carry 95 % Wilson intervals
  and the rule is a **point-estimate rule**;
* **(c)** the **decision matrix** (A / B / C), with no fixed KB menu order in any training prompt and
  fold adapters required under every outcome;
* **(d)** the **pilot gates**: schema ≥ 98 %, completion at 512 ≥ 95 %, genuine grounding flags ≤ 2 %
  of rows, empty narrative 0 %, over-refusal ≤ 10 %, refusal ≥ 90 %, shuffled-order sensitivity
  reported;
* **(e)** every number is **validation**.

---

## 1. Fallback v2 against v1, on all 834 validation rows

`scripts/fallback_v2.py`; artefact `outputs/analysis/phase3a/fit/fallback_v2.json`. CPU, deterministic,
no model.

| property | **v1** | **v2** |
|---|---|---|
| rows | 834 | 834 |
| schema valid (five keys, types) | **834 / 834** | **834 / 834** |
| rows with any grounding violation, **against the prompt** | **268 (32.1 %)** | **0** |
| number tokens not in the prompt | **1,039** | **666 — every one a supplied classifier probability** |
| tag tokens not in the prompt | **47** | **0** |
| rows clean of everything | 566 | **405**, plus **429** whose only flags are the supplied probabilities |
| mean words — `physical_reasoning` / `narrative` | 38.1 / 22.9 | 102.5 / 34.5 |
| rows on the default mechanism sentence | 520 (62.3 %) | 520 (62.3 %) |
| reasoning byte-identical inside its diagnosis group | **6.95 %** of rows | **5.04 %** of rows |

**v1 reproduces 3B-7 exactly** — 268 rows, 1,039 numbers, 47 tags, 38.1/22.9 words, 520 default —
which is the check that the comparison is on the same rule, not on a rewritten one.

**What v2 removes and what it cannot.** The KB example's numbers are gone: the equipment ids become
the roles the KB itself gives them (`R-101` → "the reactor"), stream indices become header names,
numeric ranges become a qualitative phrase whose direction is read off the two numbers
("from its nominal level toward a lower level"), and any clause that still holds an unquotable
number or tag is dropped. Re-checked on the finished string: **0 number violations and 0 tag
violations against the prompt.** The twelve-item example set (the same ten events 3B-7 printed, 3
abstention + 2 `NORMAL_TRANSIENT` + 5 faults) is in the artefact; one pair, verbatim:

> **`EVT-val-f01-r306::menu`** — `FAULT_01`
> **v1:** "Detector and sensor evidence: A_Feed_Stream1 z=18.03; A_Feed_Flow z=18.02. … Reactor_Feed_Rate
> drops from nominal **45.0 kscfm** to **~38 kscfm** … Reactor_Pressure declines from **2705 kPa**
> toward **2550 kPa**."
> **v2:** "Evidence for this event: detector ratios are LSTM autoencoder reconstruction error 24.97x
> and CVA Q-statistic 57.76x; the largest deviations are A_Feed_Stream1 (XMEAS_1, z=18.03, stable)
> and A_Feed_Flow (XMV_44, z=18.02, stable); onset starts with A_Feed_Stream1 (XMEAS_1) at +0.0 min
> and A_Feed_Flow follows at +0.0 min; the dominant lag is 4 window(s) at a slow-drift ratio of
> 0.52. **A step disturbance in the A/C feed ratio controller alters the stoichiometric proportion
> of reactants delivered through the combined feed header. …** Differential: the classifier keeps
> FAULT_01 at a calibrated probability of **1.00** and FAULT_02 at **0.00**."
> **v2 narrative:** "Diagnosis FAULT_01 is the best fit for this event. A_Feed_Stream1 (XMEAS_1)
> deviates most, at z=18.03 and stable, and it is first to move at onset +0.0 min. Recommended:
> Inspect the combined feed header A/C ratio controller setpoint in the DCS."

**The finding that matters for the contract, not for the table.** v2's differential prints the
classifier's calibrated probabilities, and those are **not on the page**. Under §19.2 criterion 2 as
literally written ("every number it cites appears in that item's evidence block") that is a failure
on **429 of 834 rows**; under the reading "supplied classifier values are evidence for this item" it
is not. §20.6 required the count both ways rather than a choice, and both are here. **Consequence,
carried into the training spec:** if v2 ships prose, the supplied top-2 with probabilities must be
**printed in the prompt**, so the differential quotes a number a reader can see. Nothing was
iterated on v2 after this table was read, and the A/B sheet is still unscored.

---

## 2. The rebuilt A/B sheet

`outputs/analysis/phase3a/fit/audit_ab_100_v2.csv` — **100 rows over 100 distinct items, none
counterfactual** — with the assignment in `audit_ab_100_v2_key.csv`, a separate file, seed 42.

| property | value |
|---|---|
| kept from the old sheet | **81** non-counterfactual items |
| replaced | **19** counterfactual rows, by **19 seeded draws from the 242 main-batch answers** |
| draw allocation (largest remainder on the surviving mix) | fault_menu 9, removed 4, normal 3, open_set 2, unobservable 1 |
| final family mix | fault_menu 46, removed 21, normal 18, open_set 10, unobservable 5 |
| adapter in slot A / B | **50 / 50** |
| scoring columns filled | **none** — 0 of 9 columns on 100 rows |
| comparator on the other side | fallback v2 |

The old sheet and key are **retained, unedited**, and marked superseded by
`outputs/analysis/phase3a/fit/audit_ab_100_SUPERSEDED.txt` and by `audit_ab_100_v2_meta.json`
(`"supersedes": ["audit_ab_100.csv", "audit_ab_100_key.csv"]`).

**Disclosed:** the sheet was built twice. The first build used `adapter_answer`, which parses the
*string* form of a generation row and returned an empty narrative for every row, because the stored
`answer` is already a dict; the first defect count therefore read "400 of 400 narratives empty". The
bug was found and fixed **before any scoring**, the sheet rebuilt blank, and §20.1's condition —
that nothing was seen — still holds.

---

## 3. Validator recall

`scripts/validator_recall.py`; artefact `outputs/analysis/phase3a/fit/validator_recall.json`; the
exported sheet is `validator_recall_21.csv`.

**(a) The 21, exported for a human.** All 21 stored flags are listed with `tag`, `quantity_word`,
`old_rule_verdict`, `new_rule_verdict`, the reproduced 40-character window, the clause it landed in,
and a **blank `human_verdict` column** for "real kind slip / not". The strict rule finds **0 of 21**,
as 3B-7 reported.

**(b) 30 synthetic positives and 30 matched negatives**, written *after* the rule, in four syntaxes.

| variant | recall | specificity | by syntax (positives flagged) |
|---|---|---|---|
| **strict attribution rule** (adopted) | **15 / 30** | **30 / 30** | copula 8/10, kind-noun 7/8, **appositive 0/6, passive 0/6** |
| **permissive, appositive-aware** (measured, **not adopted**) | **29 / 30** | **28 / 30** | copula 10/10, kind-noun 8/8, appositive 6/6, passive 5/6 |

The permissive variant's two false positives are exactly the pattern 3B-7 flagged as its borderline
case: **"XMV_52, the condenser cooling water flow valve, falls."** and **"XMV_44, the A feed flow
valve, swings."** — the word *flow* belongs to the valve's service, not to its kind. That trade
(+14 recall, −2 specificity on 30) is why §20.8 requires a dated note before adoption and why the
strict rule stays the default.

**Stated plainly:** the rule was written with the 21 in view, so its score on them measures its
author. The synthetic sets are the only honest recall figure here, and they are **small**: 15/30 with
a 95 % Wilson interval of roughly 0.33–0.67. This rule is a guard against a failure mode that has not
been observed in 300 answers, not a proven detector.

**(c) Counts of both variants on the stored answers.**

| answer set | rows | strict | permissive |
|---|---|---|---|
| main batch | 242 | **0** | **19** |
| counterfactual | 58 | **0** | **7** |
| retry | 21 | **0** | **4** |

**Not adopted.** No dated note, no change to the live path.

---

## 4. Adapter output defects, and a correction to 3B-7's example

`scripts/adapter_defects.py`; artefact `outputs/analysis/phase3a/fit/adapter_defects.json`.

| population | rows | empty narrative | narrative < 5 words | empty `physical_reasoning` |
|---|---|---|---|---|
| main batch | 242 | **0** | **0** | **0** |
| counterfactual | 58 | **0** | **0** | **0** |
| A/B adapter texts | 100 | **0** | **0** | **0** |
| all of it | 400 | **0** | **0** | **0** |

**The empty-narrative example from 3B-7 is not isolated — it does not exist.** 3B-7 read
`EVT-val-f01-r306::menu` as "adapter narrative: empty", and that observation came from the same
dict/string defect described in §2: the answer is a dict, `adapter_answer` parsed only strings, and
every row appears empty through that path. Reading the stored answers directly, that item's narrative
is 24 words and **no row in any of the three populations has an empty narrative, an empty reasoning
field, or a narrative under five words**. So §20.4's "empty narrative 0 %" gate is currently met by
the existing adapter on these 400 rows, and the defect 3B-7 recorded is withdrawn.

---

## 5. Count reconciliation

Full text in [`docs/reports/3b8_corrections.md`](3b8_corrections.md); artefacts in
`outputs/analysis/phase3a/fit/reconcile_3b8.json`. No old report edited.

**(a) The shuffled-order table**, recomputed on the population each row names — the **325** observable
gold-fault rows and the **822** support (450 + 325 + 47):

| order | 822 accuracy | abstention P / R | **325 accuracy** | named | correct \| named | **false abstention** |
|---|---|---|---|---|---|---|
| KB | 0.9416 | 0.9852 / 0.9376 | **0.9508** | **318** | 0.9717 | **7 / 325 = 0.0215** |
| shuffled-1 | 0.6594 | 0.7579 / 0.9698 | **0.2123** | **171** | 0.4035 | **154 / 325 = 0.4738** |
| shuffled-2 | 0.6642 | 0.7726 / 0.9638 | **0.2338** | **184** | 0.4130 | **141 / 325 = 0.4338** |

The two "named" denominators are **171 / 184 / 318** (named on the **331** gold-fault rows) and
**186 / 202 / 349** (named **anywhere in the 822**); the differences are **15 / 18 / 31** — faults
asserted on rows whose gold is an abstention class, which belong in the denominator for the precision
of naming and not in the denominator for coverage. Both are now reported with their population. The
3B-7 false-abstention column (0.0108 / 0.4328 / 0.3978) came from the decode job's own metric on a
different denominator; on the stated stratum it is 0.0215 / 0.4738 / 0.4338, i.e. **worse**, and the
finding is unchanged.

**(b) The masking counts.** Measured directly: the masked prefix removes **32,512** tokens over the
**1,919** report rows (the 961 dialogue rows carry no prefix). §9.2's **34,431** is exactly
**1,919 higher** — one token per row, because **13,774 − 11,855 = 9,143 − 7,224 = 1,919**: the
trailing `",` terminator was counted in *both* field spans, so the comma between the fields entered
the total twice. The gap to 22,917 is keys (7,676 + 7,676), the brace (1,919), separators (9,595),
minus the double-counted terminator (1,919), minus **13,433 tokens of BPE boundary merges** that the
component sum cannot see. **3B-3's 19,531 and 7,224 are on inconsistent bases** — key-inclusive for
`fault_id`, value-only for `verdict` — which is where the 11,514 confusion starts.

**(c) The §9.2 label errors**, on the dataset-target basis §20.10 fixed: **wrong 2**, **false
abstention 63**, **pool 65 (3.39 %)**, **correct abstentions 451**. **"254 / 13.2 %" is withdrawn** —
the nearest reconstruction is 63 + 190 menu-removed rows = 253, and those 190 rows are ones where
abstaining is *correct*; no population of size 1 explains the remainder, so the figure is withdrawn
rather than given a definition after the fact. Like-for-like: **63 / 1,919 = 3.28 %**.

---

## 6. Training spec

`docs/TRAINING_SPEC_PROSE_V1.md` — outcomes A and B, docs only, nothing built. Prompt layout with the
supplied fields and the **shuffled-or-removed menu (never a fixed order)**; label supply listed as
**gold vs cross-fitted with the 65-event count and no recommendation**; masking with the measured
token cost (644,128 → 609,697; **≈ 0** change in step time); unchanged hyperparameters; rebuild steps
with **only step 5 able to spend teacher calls**; the §20.4 pilot gates; and **≈ 3 h 20 m per run**
at run-1's measured 21.71 s/step × 540 steps, identical for both outcomes because masking changes
which positions are supervised, not the sequence length.

---

## 7. GPU (step 7): **not run — explicit skip**

The card **was idle** (807 MiB, no compute process beyond the desktop compositor), so the skip is
not the idle-clause. The job as specified needs **free-form prose generation** (40 items × KB and
`--menu-seed 1`), and this repo has no validated harness for it: `scripts/instrument_check.py`
decodes the `fault_id` field with a 16-token cap and stops on the closing quote,
`llm/benchmark/decode_bench.py` measures decode *speed*, and `scripts/menu_order_decode.py` is
field-decode. Writing a 512-token batched generator, validating it, and then running two orders was
not achievable inside this round's remaining budget, and running an unvalidated GPU script would
have produced numbers I could not defend.

**What is therefore missing:** the order-sensitivity table for **adapter-written prose** (schema
validity, completion at 512, contradiction rate, genuine grounding flags, empty narrative,
sentence-level change in the faults named). The **fault_id** order result stands as measured in 3B-7
and is reconciled in §5(a) above. This is a limitation, not a pass, and it is the one gate row below
that is not satisfied.

**Step 7(b) is not applicable, and that is a measurement, not an omission:** the 261 follow-up
validation rows' prompts carry **no candidate menu** (`0 of 261` contain a menu block), so there is
no order to permute and no KB-vs-shuffled follow-up comparison to make.

---

## 8. Gate

| # | gate item | status | evidence |
|---|---|---|---|
| 1 | §20 dated and committed **first** | **pass** | commit `d140700`, before v2 existed in code and before any number |
| 2 | fallback v2 table v1 vs v2, with examples | **pass** | 268 / 1,039 / 47 → **0 / 666 supplied / 0**; 10 examples (3 abstention, 2 `NORMAL_TRANSIENT`); `fallback_v2.json` |
| 3 | `audit_ab_100_v2.csv` blank + separate key, old files marked superseded | **pass** | 100 rows / 100 distinct non-counterfactual items, 50/50 A/B, 0 of 9 columns filled; superseded note + meta |
| 4 | validator export and synthetic recall table | **pass** | 21 exported with a blank human column; strict **15/30 recall, 30/30 specificity**; permissive **29/30, 28/30**, not adopted; counts 0/19, 0/7, 0/4 |
| 5 | empty-narrative counts | **pass, and 3B-7's example withdrawn** | 0 / 242, 0 / 58, 0 / 100; no narrative under 5 words; no empty reasoning |
| 6 | reconciliation file (a–c) | **pass** | `3b8_corrections.md`; 325 vs 331 vs 822, both named denominators, false-abstention denominators; 34,431 / 32,512 / 22,917 / 19,531 / 7,224 all reconciled; 254 withdrawn |
| 7 | GPU table **or explicit skip** | **skip, stated** | §7 — card idle, harness absent, job not run; 7(b) not applicable (no menu in follow-up prompts) |
| 8 | training spec | **pass** | `docs/TRAINING_SPEC_PROSE_V1.md`, outcomes A and B, 65-event label count, ≈ 3 h 20 m per run |
| 9 | full tests | **pass** | app env **588 passed**; `venv_train` **585 passed, 3 skipped** |
| 10 | no test data, no teacher calls, no training; run-1 intact | **pass** | validation rows only; no adapter written; run-1 adapter and checkpoints read only; earlier reports not edited |

**Estimates given before long jobs:** none was needed. The only job that could have exceeded 30
minutes was the GPU prose decode, which was not run; every completed step took seconds to a few
minutes, and the tokenizer accounting took under a minute.

---

## 9. What this changes

1. **The deterministic explainer is now grounded by construction, and that is measurable**: 0 number
   and 0 tag violations against the prompt on 834 rows, against v1's 268 rows / 1,039 numbers.
2. **The A/B contest is now between two viable sources on 100 clean items** — one that quotes only
   what the page shows and prints a supplied probability it cannot show, and one that writes an
   item-specific narrative. The §19.2 verdict is now entirely the user's scoring.
3. **The tag-kind rule's recall is 15/30 on sentences it was not written against**, with 0 false
   positives; the permissive variant buys 14 positives for 2 false ones and stays out of the path.
4. **3B-7's empty-narrative defect does not exist**, and two of its tables' denominators and one
   token count do not reconcile as printed. The direction of the order finding survives all of it
   and gets slightly worse for the adapter.

**Does not change:** the G1 architecture, the frozen hybrid, the closed instruments of §18.1, and
the fact that no prose source has passed §19.2 yet — the sheet is blank.

---

## 10. Reproduce

```
python scripts/fallback_v2.py                     # v1 vs v2 on 834 rows + 10 examples
python scripts/make_ab_sheet_v2.py                # audit_ab_100_v2.csv + key + meta
python scripts/validator_recall.py                # the 21 export, 30+30 synthetic, counts
python scripts/adapter_defects.py                 # empty narrative / stub / empty reasoning
./venv_train/Scripts/python.exe scripts/reconcile_3b8.py   # the three reconciliations
python -m pytest tests -q                         # 588 passed
./venv_train/Scripts/python.exe -m pytest tests -q         # 585 passed, 3 skipped
```
