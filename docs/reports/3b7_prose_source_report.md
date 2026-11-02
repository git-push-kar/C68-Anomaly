# 3B-7 — PROSE-SOURCE DECISION PACK, AND THE SEVEN CORRECTIONS THE 3B-6 ROWS DID NOT SUPPORT

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade` · **Date:** 2026-11-01
**Environment:** app env (CPU) for everything except step 5b, `venv_train` for that one GPU run
**Scope:** §19 pre-registered before any number; seven corrections to the 3B-6 report; the tag-kind
validator and the deterministic-explainer baseline; the blinded A/B sheet; adapter-only decode in one
KB order and two shuffled orders; a prose-only training contract as a proposal only.

No adapter training. No test component opened. No teacher calls. Run-1 adapter and checkpoints read
only. GPU used for one decode job: **617.5 s end to end, three orders, ~10.3 min** — under the
30-minute estimate, so no pre-approval was needed.

---

## 0. Pre-registration (committed first)

`docs/PROTOCOL.md` **§19**, committed as `a7fd3f8` *("Fix the prose-source rule before the sheet that
decides it is scored")* **before** the A/B sheet existed and before any 3B-7 number was computed.
It fixes, verbatim:

* **(a) architecture status** — hybrid frozen; the novelty channel and the hand-coded fit score are
  closed as negative results;
* **(b) the prose-source rule** — the adapter's report prose is kept for the report path **only if**
  adapter prose is ≥ 85 % correct **and** ≥ 90 % grounded **and** preferred over the deterministic
  fallback in ≥ 60 % of non-tie pairs; otherwise reports use the deterministic explanation and the
  adapter is limited to follow-ups;
* **(c)** faults 3/9 under acceptable-non-diagnosis scoring are reported as **uninformative** if every
  system scores 12/12;
* **(d)** adapter-only numbers are reported in KB order **and** in two shuffled orders;
* **(e)** every number here is **validation** (the selection set); the test component stays closed.

The rule in (b) cannot be evaluated yet: it is decided on the user's scoring of `audit_ab_100.csv`,
which is blank by construction. What this round delivers is the sheet and everything the rule will be
read against.

---

## 1. Seven corrections to the 3B-6 report

Full text: [`docs/reports/3b6_corrections.md`](3b6_corrections.md). The old report was not edited.

| # | 3B-6 said | the rows say | why it happened |
|---|---|---|---|
| **a** | adapter scores 6/12 on faults 3/9 ("acceptable non-diagnosis") | **12/12 for all three systems** (classifier, rule, adapter). Under §19.3 that reading is **uninformative** and enters no headline; the strict column (0 / 0 / 6) is the one that discriminates | `NORMAL_TRANSIENT` is acceptable on a 3/9 row, and the classifier answers it on all twelve — the scoring, not the systems, was measured |
| **b** | "natural error pool 68 events (3.5 %); 4 of the 8 label errors become correct diagnoses" | **pool = 65 events, one row per event** (2 named wrong + 63 false abstentions on seen-fault rows); the two events named in the question (`f10-r091`, `f13-r086`) sit **inside** the 63, not outside them; **3**, not 4, label errors are rescued by the rule; **4 of the 8 are predicted FAULT_10** (not absorbed by FAULT_08/11/open-set) | one of §17's 64 menu abstentions is on an open-set-gold row, where abstaining is *correct*, so it belongs with the 451 correct abstentions; and `EVT-train-f16-r040::menu` answers `FAULT_10`, which **is** on that episode's menu, so the rule cannot withhold it |
| **c** | two threshold populations, 1,729 items / 82 abstained (§17) and 1,356 events / 67 flagged (§18.3), left unreconciled | both are the same 1,919 out-of-fold train events under **different filters**: §17's is **all menu-present rows** — 1,356 seen-fault + **262** open-set-gold + **111** `NORMAL_TRANSIENT` (= 1,729, 82 abstained, 4.74 %) — while §18.3's is **seen-fault menu rows only** (1,356, 67 flagged, 4.94 %). Difference **373**. Neither is an arithmetic error, but §17's 4.74 % is a rate over a set in which abstaining is right for 373 rows, so it is **stricter than it reads**; §18.3's population is the one the rule names | §17's extra filter `true_label in training_classes` is a no-op (every label is a class), so it did not narrow the set the way the rule's wording intends |
| **d** | named-stratum false abstention 0.0363 → 0.0846, denominator "325" | the function was passed 331 rows; on the 325-row stratum it is **12/325 = 0.0369 → 28/325 = 0.0862**. The 331 was six menu-present 3/9 rows that the list did not filter | before/after lists passed unfiltered |
| **e** | fold table column "new declines"; fold_2 11 | it is **total declines after the channel**; `fold_1` 50 → **50** (27 flagged, all already declined), `fold_2` 11 → **13** | the column was the OR-ed total, not new declines |
| **f** | fit score "mean paired AUROC 0.6508" | **0.4762** over the 6 informative pairs (below chance); 0.6508 is carried by three pairs that are 1.000 by construction. `FAULT_15` vs `NORMAL_TRANSIENT` is **not computable** (`NORMAL_TRANSIENT` has no KB signature) — stating it is not an omission | the structurally-decided pairs were never excluded from the mean |
| **g** | novelty channel "rejected" | restated with **both** bases: (a) criterion on `fold_1` — 3.67 points of kept-fault accuracy lost, **zero** declines converted (impossible by construction, so weak evidence); (b) the validation result — **16 decisions changed, 16 correct → abstention, 0 improved**. Disclosed that (a) was first implemented on validation only and the wider reading of "on an instrument" was adopted **after** seeing that it did not fire | post-hoc widening of a pre-registered criterion, recorded rather than smoothed over |

**§19.3 consequence, now on the record:** the 3/9 stratum's acceptable-non-diagnosis column is
saturated for all three systems, so it cannot separate them and is not used anywhere in what follows.

---

## 2. The tag-kind validator, and the expectation it failed

`llm/hybrid/tag_kind.py` — deterministic, attribution-based, no model. A sentence is rejected when a
**physical quantity class** (temperature, pressure, level, flow, composition, valve position) is
predicated **of a tag whose kind contradicts it** (an `XMV_*` tag described as a measurement, an
`XMEAS_*` tag described as a valve position). Four guards keep it off legitimate prose: sensor names
and bracketed kinds are masked before scanning; the tag and the quantity must share a clause; the
quantity must be **predicated** of the tag (copula forward, kind-noun `of/from/on/at` backward,
associative `for` excluded); negated or contrastive attributions are not violations. Locked by 23
tests in `tests/test_tag_kind.py`. Live path untouched.

| answer set | rows | flagged by the new rule | expected |
|---|---|---|---|
| `main_242` (§17.7 main batch) | 242 | **0** | "the 21 known mislabels should be found" |
| counterfactual (§18.5 forced-wrong) | 58 | **0** | — |
| retry_21 (§17.7 retry-once) | 21 | **0** | — |

**The expectation is wrong, and the task's premise with it: the 21 items were never mislabels.**
Each one reproduces as a 40-character window that read past the clause — the flagged word belongs to
a **neighbouring sensor's name** (`Stripper_Steam_Valve (XMV_50) and Stripper_Temperature
(XMEAS_18)`), to a **fault name**, to **another tag's own kind**, or to a **downstream consequence**.
The old rule matched a window containing both a tag and a quantity word and did not check who the
quantity was said of. So: **0 of the 21 are found, because 0 of the 21 are violations.**

Re-judged retry results: of the 21 answers, 7 changed under the retry, 14 stayed as stored, and 1
changed differently — the new rule finds **0 violations before and after**, i.e. the retry rewrote
prose that already had none. **No retry budget should be spent on this failure mode.** A rejecting
validator stays in the contract as a guard (§6), but it guards nothing that has been observed.

**False positives on the rest — 10 near-misses, for the manual check.** Every one is phrasing the
loose rule would flag and the attribution rule does not; the reason is the same in all ten: the
quantity is not predicated of the tag.

| # | tag | quantity | clause | human verdict needed |
|---|---|---|---|---|
| 1 | XMV_50 | temperature | "The Stripper_Steam_Valve (XMV_50) trend increasing aligns with a C feed temperature change…" | other sensor's name — not a violation |
| 2 | XMV_51 | flow | "The reactor cooling water flow valve (XMV_51) is opening more…" | name of the valve — not a violation |
| 3 | XMV_51 | flow | "The reactor cooling water flow valve (XMV_51) is increasing" | as #2 |
| 4 | XMV_45 | flow | "The A/C feed flow valve (XMV_45) is increasing" | as #2 |
| 5 | XMV_52 | flow | "The condenser cooling water flow valve (XMV_52) is decreasing" | as #2 |
| 6 | XMV_51 | flow | "but the flow is a manipulated variable (XMV_51) trending upward" | two facts, joined — not predication |
| 7 | XMV_45 | flow | "…would cause the A/C feed valve to open more (increasing XMV_45) to maintain flow" | consequence + `to maintain flow` |
| 8 | XMV_52 | flow | "flow measurement for XMV_52 first." | associative `for` — excluded by guard 3 |
| 9 | XMV_51 | temperature | "…inconsistent with the observed trend in XMV_51… rather than a temperature change" | contrastive — guard 4 |
| 10 | XMV_44 | flow | "The A feed flow (XMV_44)" | **the borderline one**: an appositive that names the valve as a flow with no copula. A human could rule it sloppy; the rule's clause test does not fire |

Item 10 is the one place the attribution rule is arguably **too permissive**, and it is recorded as
such rather than tuned after seeing it.

---

## 3. The deterministic explainer, measured on all 834 validation rows

`fallback_answer()` with the classifier-plus-menu-rule diagnosis supplied — no model anywhere.

| property | result |
|---|---|
| rows | **834** |
| schema valid (five keys, types) | **834 / 834** |
| `ranked_hypotheses[0]` == supplied diagnosis | **834 / 834** |
| `ranked_hypotheses` contains an off-menu entry | **0** |
| rows with **any** grounding violation | **268 / 834 (32.1 %)** — pass 566 (67.9 %) |
| violations by kind | **1,039 numbers**, **47 tags** |
| mean words — `physical_reasoning` / `narrative` | **38.1 / 22.9** |
| rows where prose is byte-identical in shape | 834 / 834 start "Detector and sensor evidence"; 834 / 834 narrative starts "Diagnosis …" |
| rows whose reasoning falls back to the default mechanism sentence | **520 (62.3 %)** — 459 `UNCLASSIFIED_OPEN_SET` + 61 `NORMAL_TRANSIENT`, i.e. every row where the KB has no entry, which then says "the stored evidence is **consistent with this fault family**" |

**The grounding failure is structural, not sloppy:** the fallback quotes the KB's mechanism verbatim,
and the mechanism carries numbers from *the KB's example event*, not from this one —
`Reactor_Feed_Rate drops from nominal 45.0 kscfm to ~38 kscfm`, `2705 kPa toward 2550 kPa`,
`120.4`, `-101`, `2550`. Those are exactly the 1,039 flagged numbers. A fallback that is grounded *by
construction* is only grounded if it never quotes the KB's numbers, and it does.

**What it cannot write** (recorded, not fixed): the per-item **differential** — which alternatives
were excluded and why — is absent, the ranked list being the KB's lookalike list rather than a
judgement about this evidence; and the **narrative** is one template with `recommended_action`
substituted, naming no sensor, no onset and no trend. Two events of `FAULT_07` get byte-identical
reasoning.

**Ten side-by-side examples** are in `outputs/analysis/phase3a/fit/fallback_baseline.json`
(`examples`), and **five of the ten are abstention/NORMAL rows** (3 `UNCLASSIFIED_OPEN_SET` + 2
`NORMAL_TRANSIENT`), which is more than the 3 + 2 asked for. One, verbatim:

> **`EVT-val-f01-r306::menu`** — diagnosis `FAULT_01`, family `fault_menu`
> **fallback reasoning:** "Detector and sensor evidence: A_Feed_Stream1 z=18.03; A_Feed_Flow
> z=18.02. A step disturbance in the A/C feed ratio controller alters the stoichiometric proportion of
> reactants delivered through Stream 4. As A_C_Feed_Flow drops, the total molar feed rate to R-101
> decreases (Reactor_Feed_Rate drops from nominal **45.0 kscfm** to **~38 kscfm**)… Reactor_Pressure
> declines from **2705 kPa** toward **2550 kPa**."
> **fallback narrative:** "Diagnosis FAULT_01 is the best fit for this event. Recommended: Inspect the
> Stream 4 A/C ratio controller setpoint in the DCS. Stroke control valve XMV_45…"
> **adapter reasoning:** the sensor-level story with z-scores, deviations and onset times that the
> page actually shows; **adapter narrative: empty** in this generation row.

That single pair is the whole trade: the adapter is grounded and item-specific but writes no
recommendation; the fallback writes a recommendation and quotes a different event's numbers.

---

## 4. The blinded A/B sheet — and a defect in its first build, fixed

`outputs/analysis/phase3a/fit/audit_ab_100.csv` (scoring columns blank) with the assignment in
`audit_ab_100_key.csv`, a **separate file**: `item_id, A, B, seed=42`.

| property | value |
|---|---|
| rows | **100** |
| **distinct items** | **100** |
| adapter in slot A / B | **48 / 52** |
| families | fault_menu 37, counterfactual 19, removed 17, normal 15, open_set 8, unobservable 4 |
| scoring columns filled | **none** |

**Defect, disclosed rather than quietly rebuilt.** The first build copied the 100 rows of
`audit_hybrid_100.csv` straight through, and that sheet is **94 distinct items over 100 rows**: the
six items that appear both as `fault_menu` rows and as `counterfactual` rows were counted in both
(`f04-r313`, `f05-r320`, `f07-r319`, `f10-r313`, `f10-r317`, `f12-r319`, all `::menu`). In an A/B
sheet that is the same item shown twice — once with its own diagnosis, once with a forced wrong one —
i.e. a repeated measure counted as two independent judgements. `scripts/make_ab_sheet.py` rebuilds it:
the 94 distinct items plus six more `fault_menu` items drawn seeded from the generated set, **100
rows over 100 distinct items**, same two sources, same one presentation format, same seeded
assignment. `audit_ab_100_meta.json` records the dropped duplicates and the seed.

---

## 5. Adapter-only decode in three menu orders (GPU, step 500, `<= 20 min`)

GPU was idle at start (5,181 MiB, 19 %, 34 °C — desktop compositing). One process, three orders,
`decode_first_field` with the language-model-direct path, `batch 8`, `cap 16`, adapter read-only.
**617.5 s total, 194.9 / 195.0 / 195.2 s per order.** The classifier-plus-menu-rule row is invariant:
`tests/test_menu_order.py` proves it cannot see the order (7 tests, including "a permuted menu cannot
change the rule").

| instrument | accuracy | macro-F1 | abstention P / R | false abstention | of the 331 gold-fault rows: named / correct |
|---|---|---|---|---|---|
| **classifier + menu rule** (order-invariant) | **0.9781** | **0.9708** | 0.9739 / 0.9933 | **0.0323** | **312 / 310** (0.937) |
| adapter-only, **KB order** | 0.9416 | 0.9263 | 0.9907 / 0.9422 | 0.0108 | **318 / 309** (0.933) |
| adapter-only, **shuffled-1** | **0.6594** | **0.3324** | 0.7299 / 0.9667 | **0.4328** | **171 / 69** (0.208) |
| adapter-only, **shuffled-2** | **0.6642** | **0.3508** | 0.7448 / 0.9600 | **0.3978** | **184 / 76** (0.230) |

*(all rows: 822 observable of 834; support 450 abstention-class, 331 gold-fault, 47 `NORMAL_TRANSIENT`)*

**This is the finding of the round, and it is not a small one.** Permuting the sixteen menu lines —
contents identical, only the order changed — costs the adapter **28 points of accuracy and 60 points
of macro-F1**, and turns a 1.1 % false-abstention rate into **43 %**.

Where the loss comes from, measured:

* **It is not slot-copying.** If the adapter read "the entry in the gold's old position", it would be
  right ~100 % of the time; the prediction equals the entry now sitting in the gold's KB slot for
  **29 of 331** rows in both shuffled orders. It is not recency or primacy either: of the faults it
  does name, only 22 (shuffled-1) / 18 (shuffled-2) are the first menu entry and 133 / 143 are in the
  middle — the same spread as KB order (26 first, 270 middle).
* **It withdraws, and when it does not, it is wrong.** It names a fault on **171 / 184** of the 331
  gold-fault rows (318 in KB order). When it does name one under a permuted menu, it is right
  **69 / 186 = 0.371** (shuffled-1) and **76 / 202 = 0.376** (shuffled-2) — against **309 / 349 =
  0.885** in KB order. On the 331 gold-fault rows outright: **0.933 KB → 0.208 / 0.230 shuffled**.
* **The two shuffled orders agree with each other more than either agrees with KB**: 622 of 834
  identical predictions between shuffled-1 and shuffled-2, against 533 between KB and shuffled-1.
  Two different permutations agree with *each other* at least as well as each agrees with the training
  order — what they share is "unfamiliar order", and the model has learned to treat that as "no
  match".

**How this must be read.** The adapter-only reference row used throughout §14–§18 was produced in KB
order, which is the order it was trained on, and **its 0.9263 macro-F1 is largely a property of that
order**. Per §19(d) the row is now always reported three ways; the KB-order number stands as the
row it always was, and the two shuffled numbers are what the same instrument does when the layout it
memorised is changed. It does not touch the G1 architecture: the diagnosis in the frozen hybrid comes
from the classifier and the menu rule, which are order-invariant by construction, and this result
makes that choice look considerably better than it looked yesterday.

---

## 6. Prose-only training contract — proposal only, nothing built

`docs/RESPONSE_SPEC.md` **§9.2** (dated 2026-11-01). `llm/train_adapter.py`, `llm/dataset.py`, the
prompts, `train_mixed.jsonl` and every hash are untouched; no adapter was trained.

The contract: `fault_id` and `verdict` **supplied in the prompt** (classifier + menu rule) together
with the classifier's top-2 and calibrated probabilities; loss masked to `ranked_hypotheses` /
`physical_reasoning` / `narrative` / follow-up answers; shuffled menu in training and evaluation; the
tag-kind validator with retry as a guard.

Counts, from the existing `train_mixed.jsonl` (tokenizer of the trained model, counting only, no new
files):

| quantity | value |
|---|---|
| supervised tokens today (report 573,061 + dialogue 71,067) | **644,128** |
| with `fault_id` + `verdict` masked | **609,697** |
| removed | **22,917 tokens = 5.35 %** (`fault_id` 13,774, `verdict` 9,143) |
| expected change in step time | **≈ 0** |
| events whose supplied label would be **wrong** vs gold (cross-fitted prediction) | **8 / 1,919 = 0.42 %** |
| events whose supplied label would be an **abstention where gold is a named fault** | **254 / 1,919 = 13.2 %** |
| events where supplied ≠ gold at all | **262 / 1,919 = 13.7 %** |

* **Step time does not change and saying otherwise would be false.** Masking changes which positions
  contribute to the loss, not the sequence length.
* **13.2 % is the number to worry about.** Supplying the cross-fitted prediction hands the model an
  abstention on one event in eight where the truth is a named fault — the §17 lookalike failure moved
  into the training set.
* **Rebuild steps needing teacher calls:** only step 5 — regenerating dialogue answers *if* their
  prompts change (≈961 rows, ≈250–330 calls at the observed 3–4 rows/call). Steps 1–4 (supply the
  fields, mask, shuffle the menu, rebuild manifest/hashes and re-run the checks) are **programmatic**.

---

## 7. Gate

| # | gate item | status | evidence |
|---|---|---|---|
| 1 | §19 dated and committed **first** | **pass** | commit `a7fd3f8`, before the sheet and before any 3B-7 number |
| 2 | corrections file (a–g) | **pass** | `docs/reports/3b6_corrections.md`; seven statements replaced, each with the rows recomputed |
| 3 | tag-kind validator with counts and a 10-item false-positive list | **pass, with the task's expectation refuted** | `llm/hybrid/tag_kind.py` + 23 tests; 0 / 242, 0 / 58, 0 / 21; the 21 are window artefacts; 10 near-misses listed |
| 4 | fallback baseline table and 10 examples | **pass** | 834 rows, 834 schema-valid, 268 grounding-violating, 1,039 numbers + 47 tags, 38.1/22.9 words; 10 examples, 5 of them abstention/NORMAL |
| 5 | `audit_ab_100.csv` blank, key separate | **pass (rebuilt)** | 100 rows / **100 distinct items**, 48 A / 52 B, all scoring blank, key in `audit_ab_100_key.csv`; first build's 6 duplicates dropped and disclosed |
| 6 | shuffled-order adapter table, or an explicit skip | **pass** | three orders, 617.5 s GPU total; 0.9416 → 0.6594 / 0.6642 accuracy; classifier row beside it, order-invariant by test |
| 7 | contract proposal with counts | **pass** | `RESPONSE_SPEC.md` §9.2; 644,128 → 609,697 tokens; 8/254/262 event counts; teacher-call table |
| 8 | full tests, both environments | **pass** | app env **588 passed**; `venv_train` **585 passed, 3 skipped** |
| 9 | no test data, no teacher calls, no training | **pass** | validation rows only; GPU job was decode-only on `val.jsonl` |
| 10 | run-1 and earlier hybrid files unchanged | **pass** | adapter and checkpoints read only; all new output under `outputs/analysis/phase3a/fit` (ignored); earlier reports not edited |

**Estimate given before the one long job:** three-order decode ≈ 10 min (measured 617.5 s). Nothing
else in this round exceeded a few minutes.

---

## 8. What this changes, and what it does not

**Changes:**

1. The adapter-only reference row is now known to be **order-dependent**: 0.9263 macro-F1 in KB order,
   0.3324 / 0.3508 under two permutations. Every earlier report that quoted the KB-order row as if it
   were a property of the model needs that qualifier.
2. The deterministic fallback is **cheaper and more faithful in schema, worse in grounding** (32.1 %
   of rows) and **cannot write a differential or an item-specific narrative**. §19.2's rule was
   written expecting a contest between two viable prose sources; the measurement makes it a contest
   between two partly broken ones, and says why: the fallback quotes the KB's numbers, the adapter
   quotes nothing it cannot see but also writes no recommendation.
3. The **tag-kind validator guards a failure that has not been observed**. §9.1's 21 "mislabels" are
   withdrawn at their source, and the retry budget spent on them in §17.7 bought nothing.

**Does not change:** the G1 architecture (hybrid; classifier + menu rule diagnose, adapter writes
prose). Nothing in this round touches it, and the order result supports the half of it that is
order-invariant.

**Still open:** the §19.2 verdict itself, which needs the sheet scored. If adapter prose lands below
85 % correct or 90 % grounded, or wins fewer than 60 % of non-tie pairs, the report path becomes the
deterministic explanation — and §3 says that path currently fails grounding on a third of rows, so
that outcome would need the fallback's mechanism quoting fixed before it ships.

---

## 9. Reproduce

```
python scripts/tag_kind_audit.py            # 0/242, 0/58, 0/21 + the 10 near-misses
python scripts/fallback_baseline.py         # the 834-row baseline and the 10 examples
python scripts/make_ab_sheet.py             # the 100/100 A/B sheet and its key
.\venv_train\Scripts\python.exe scripts/menu_order_decode.py --orders kb,shuffled-1,shuffled-2
python -m pytest tests -q                   # 588 passed
.\venv_train\Scripts\python.exe -m pytest tests -q   # 585 passed, 3 skipped
```
