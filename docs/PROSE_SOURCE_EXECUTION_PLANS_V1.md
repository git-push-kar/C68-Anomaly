# Execution plans for report prose: adapter against fallback v3.1

Status: consolidation of the readings already recorded in `docs/PROTOCOL.md` §29 and the
follow-up specification. No new measurement is claimed here, nothing was trained, and no stored
generation, sheet, key or checkpoint was modified. Where this document and a dated protocol
section disagree, the protocol section wins and the discrepancy is listed in §7. Sections 9 to 11
extend it with the combination matrix and the alternatives that were never costed.

---

## 1. The invariant that holds in every plan

The **diagnosis, verdict, severity and confidence are code outputs** in all six plans. No plan
lets prose decide what the system says a fault is (§29.1). The plans differ only in *who writes
the sentences*, and — for two of them — whether the writer is retrained.

Consequence for reading the rest of this document: a preference score between two texts is a
statement about prose quality, never about diagnostic accuracy.

## 2. The six plans

`Runtime` is what runs when a report is produced. `Work` is what has to happen before that
runtime exists. Only E and F train.

| Plan | Runtime behaviour | Work to get there | Gate | Failure mode |
|---|---|---|---|---|
| **A** fallback v3.1 authors every report | deterministic assembler: fixed sentence templates keyed on the supplied diagnosis, store values slotted in. **No model, no inference, no reasoning step.** | none — this is the measured outcome of §29.9 | already passed: 0 flagged rows of 40, anchor 40/40, `correct` and `useful` 40/40 | prose is flatter on the menu-removed open-set rows, where the adapter is preferred 8 to 1; a quality ceiling, not a correctness risk |
| **B** validated adapter prose (3B-17) | adapter generates the whole report including numbers; four checks run; **one** corrective retry; still-flagged items ship as fallback text | one GPU session: stored generations reused, checks run, up to 21 retries, gates measured on delivered text | fallback rate ≤ 25 % of 40 items (≤ 10) plus the §29.5 gates | at a 52.5 % first-pass flag rate the retry must rescue more than half the rows just to reach the ceiling; the retry rewrites text and does not recompute a threshold |
| **C** hybrid: code writes numbers, adapter writes mechanism | code writes the numeric sentences **into the report itself**; the adapter's section is **number-free by construction**. It is not handed numbers to paraphrase. | a restricted narrative-only prompt variant, one 40-item generation pass, then the same gates re-measured on assembled text; the report becomes two sections with a defined seam | §29.5 gates on assembled text, plus a check that no number appears in the adapter section | assembly reads as two authors; the adapter can still misattribute a tag role (v5-32, v5-37) |
| **D** adapter raw | adapter text delivered unvalidated | none | none — excluded | 21 of 40 rows flagged, 9 of 40 anchors, 0.560 preference |
| **E** retrain the report adapter | adapter authors reports after retraining | teacher corpus whose numeric sentences are correct by construction (fallback's numeric sections) plus mechanism narrative; multiple seeds; **needs a dated §30** because it is outside the §29 order | §29.5 gates per seed, variance across seeds | hours of GPU per attempt; must beat a deterministic writer that scores 0 defects on 40 rows |
| **F** adapter for follow-ups only | reports from fallback; adapter serves the dialogue path (mechanism, comparison, refusal, clarification prose) | the run specified in `docs/TRAINING_SPEC_FOLLOWUP_V1.md`: 961 rows, 180 steps, hyperparameters unchanged | the follow-up gates fixed in that spec before launch | adapter quality on a surface with no numeric sentence to break — its measured strong stratum |

A and F are **not competing**: they serve different code paths. A never touches the model, F never
touches the report. E is the plan that competes with F for the same GPU budget.

## 3. The measured numbers, and what each one means

### 3.1 The sheet

40 distinct events, one row each, no event appearing twice:

| stratum | rows | what the writer was told |
|---|---|---|
| open-set, menu present | 10 | `UNCLASSIFIED_OPEN_SET`, menu present minus the true entry |
| open-set, menu removed | 10 | same label, no menu in the prompt |
| `NORMAL_TRANSIENT` | 20 | no fault, menu present |

The adapter sits on side A for 5 / 5 / 10 of these, so position carries no information.

No fault ID is supplied to either writer in these 40 rows, and none is the correct answer: the
gold label behind them is `UNCLASSIFIED_OPEN_SET` ×20 and `NORMAL_TRANSIENT` ×20. The 20 open-set
rows are `::openset` episodes — from classifiable faults whose episode is outside the KB library,
and from faults 16–20 whose KB names are "Unknown disturbance in …". The abstention is the target,
not a classifier failure.

### 3.2 Preference

One blind call per row: `A`, `B`, or `tie`, meaning better for the operator's next decision. Ties
leave the denominator (§29.2).

| stratum | adapter | fallback | tie | adapter share of non-ties |
|---|---|---|---|---|
| open-set, menu present | 4 | 5 | 1 | 0.444 |
| open-set, menu removed | 8 | 1 | 1 | 0.889 |
| `NORMAL_TRANSIENT` | 2 | 5 | 13 | 0.286 |
| **pooled** | **14** | **11** | **15** | **0.560** |

- Threshold: 60 % of pooled non-ties, fixed before any score was read. 15/25 would be 0.600, so
  **the decision is one row from flipping**.
- Tie rate 37.5 %, concentrated in `NORMAL_TRANSIENT` (13 of 20), where the two texts tell an
  operator nearly the same thing.
- Strictness: resolving the 11 adapter-flagged ties against the adapter gives 14/36 = **0.389**.
  The direction is robust to strictness; the 0.60 line is not.

### 3.3 The four cells

Judgments, 0 or 1 per side per row: 320 binary cells plus 40 preferences = **360 entries** per
rater.

| cell | adapter | fallback | definition |
|---|---|---|---|
| `correct` | 40 | 40 | no fault, verdict or mechanism the prompt does not support |
| `grounded` | 37 | 40 | every number and tag exists in the store, comparisons point the right way |
| `consistent` | 35 | 40 | no self-contradiction |
| `useful` | 40 | 40 | top sensor with direction, plus an operator next step |

`correct` and `useful` at 40/40 on both sides is why the decision could not be made on those
cells: on these strata both writers are right and actionable, and the difference is in how
carefully numbers are handled.

### 3.4 The machine instrument

The §28 checkers run per side (`audit_v5_adjudication_keyed.csv`). One row can carry several
defects.

| | rows flagged | defect instances |
|---|---|---|
| adapter | **21 / 40** | 23 — numeric 18, grounding 3, tag role 2 |
| fallback | **0 / 40** | 0 |
| clean on both | 19 | — |
| clean on neither | 0 | — |

Anchor — the top sensor named with its z to two decimals:

| source | exact | z omitted | sensor absent | contradicting value |
|---|---|---|---|---|
| adapter | **9 / 40** | 14 | 11 | 6 |
| fallback | **40 / 40** | 0 | 0 | 0 |

The 9 adapter-exact rows are the same 9 rows where both sides are exact: the adapter anchors no
row the fallback does not already anchor.

**Flags are not cells.** The checkers fire on any deviation from a stored value; a cell fires only
where the defect would mislead an operator. Eight adapter cells failed against 21 flagged rows,
and at least four flag families are instrument false positives: v5-05 (two true inequalities in
one sentence), v5-20 (`value_not_stored` on values that are stored), v5-08 (a tag inside a
hypothetical), v5-37 (a valve position the XMV rule accepts). The instrument is a searchlight, not
an oracle (§29.10).

## 4. What each plan costs in GPU time

Measured rates on record, both quoted because they differ by 74 %:

| rate | value |
|---|---|
| pilot step time | **12.5 s/step** (288 steps in 1.0 h) |
| run-1 step time | **21.71 s/step** (288 steps ≈ 1 h 44 m) |

| plan | trains? | GPU time | wall clock |
|---|---|---|---|
| A | no | **0** | 0 — CPU only |
| B | no | inference only; ≤ ~21 retry generations — **single-digit minutes, unmeasured** | ~15–20 min, one session |
| C | no | one 40-item generation pass — **single-digit minutes, unmeasured** | ~1 day, dominated by the code seam |
| D | no | 0 | 0 |
| **F** | **yes** | 961 rows ÷ 16 = 60 steps/epoch × 3 = **180 steps → ≈ 38 min at 12.5 s/step, ≈ 65 min at 21.71 s/step** | ~1.5 h including load and gates |
| E | **yes** | 288 steps → **1.0 h at 12.5 s/step, 1 h 44 m at 21.71 s/step**, per seed; 3 seeds ≈ **3.0–5.2 h** | days, including the teacher rebuild |

Two caveats. The step times are measured; **no per-item generation rate is recorded anywhere in
this repository**, so the minutes against B and C are an estimate from output length, not a
measurement — whichever session runs should record its own wall clock, as the follow-up spec
already requires of its own estimate. And the ±30 % budget rule of §24 applies because the two
recorded rates differ by so much.

## 5. Decision

**A at delivery, F for the training budget** — fallback v3.1 authors the three report strata on a
deterministic path with no model in it, and the adapter is trained and used for follow-ups.

Reasons, in order of weight:

1. **A is the only plan whose gates are passed rather than projected** — 0 flagged rows of 40,
   anchor 40/40, `correct` and `useful` 40/40 — and it adds no inference cost to the report path.
2. **The adapter's failure class is mechanical.** All 23 defect instances are numbers, tag roles or
   grounding; none is a reasoning error. A template that fills `"{tag} (z={z})"` from the store
   cannot emit a wrong inequality direction, because nothing in it composes a comparison.
3. **B cannot fix what it validates.** Validation retries text generation; it does not recompute a
   threshold. At the observed 52.5 % flag rate it must rescue more than half the rows to stay under
   the 25 % fallback ceiling, and if it cannot, the GPU pass is sunk and C is the destination.
4. **C is the only design matched to the failure.** It removes numeric sentences from the adapter's
   remit instead of asking the model to get them right, and it preserves the stratum where the
   adapter wins 8 to 1.
5. **F spends the budget where the adapter is measured strong** — mechanism and comparison prose
   with no anchor obligation — for 38–65 minutes, against E's 3.0–5.2 h for three seeds.

Not chosen: **D** (excluded by measurement), **E** now (most expensive, least justified while a
deterministic writer already scores 0 defects), **B** (a detour around a problem it cannot fix).

### 5.1 Order of execution

1. Human rescore of the 40 rows — 320 binary cells plus 40 preferences. It supersedes the model
   reading and is reported beside it, never merged. Nothing below starts before it; the
   `NORMAL_TRANSIENT` stratum is where it should start, because it holds 13 of the 15 ties.
2. Plan A is already live; no report-path work remains.
3. CPU track in parallel: folds, baselines, calibration refit from validation, then the single
   pre-registered test pass.
4. Plan F after explicit approval, on the training device, gates fixed before launch.
5. Write-ups: report and follow-up audits, and the three stale companion docs.
6. Plan C only on a **dated §30**, measured against the same §29.5 gates.

### 5.2 What a human rescore would change

A rescore that lifts the pooled share to ≥ 60 % flips the report-prose decision in §29.9. It does
not change the argument for C over B, which rests on the flag profile, not on the preference share.
Until a run moves the adapter's anchor figure off 9/40, no validation loop can accept its prose at
a rate that keeps the fallback share under 25 %.

## 6. Artefacts

| Path | What it holds |
|---|---|
| `docs/PROTOCOL.md` §29 | the pre-registered question, rule, branch tree, 3B-17 loop, gates, measured outcome |
| `docs/TRAINING_SPEC_FOLLOWUP_V1.md` | the Plan F run: data, hyperparameters, 180 steps, cost range, gates |
| `scripts/audit_v5_sheet.py` | structure audit, scoring packet, join, adjudication, reveal |
| `tests/test_v5_scoring_decision.py` | regression test locking the structural gate and the 14/11/15 reading |
| `outputs/analysis/phase3a/fit/audit_v5_join.json` | per-stratum and pooled tallies, cell counts, decision |
| `outputs/analysis/phase3a/fit/audit_v5_adjudication_keyed.csv` | per row, per side: flags and anchor status, sources named |
| `outputs/analysis/phase3a/fit/audit_v5_reveal.csv` | the model rater's preference with sources named |

## 7. Correction ledger

Two figures were misstated in conversation while this document was being written. The protocol
record was correct in both cases; the errors were in the verbal restatement.

| figure | misstated as | correct | source |
|---|---|---|---|
| adapter anchor, exact | 0 / 40 | **9 / 40** | §29.10: exact on both 9, fallback only 31, adapter only 0; per-source tally exact 9, z omitted 14, sensor absent 11, contradicting value 6 |
| fallback anchor, exact | 31 / 40 | **40 / 40** | same sentence read per source rather than per side |
| human rescore size | 200 judgments | **360 entries** (320 binary cells + 40 preferences) | §29.2 defines four cells per side plus one preference |

The first correction makes the case against Plan B slightly stronger, not weaker: the adapter never
anchors a row the fallback misses.

## 8. Prohibitions carried forward

- No fold adapters or extra seeds before the human rescore is in.
- No reopening of the prose question after the rescore; a reversal is a dated §30, not an edit to
  §29.
- No plan in this document may change the diagnosis, verdict, severity or confidence.
- Plan F runs only after explicit approval, and its gates are fixed before the first step.

---

## 9. Combinations that can be implemented

The six plans are not mutually exclusive alternatives on one axis. They split across three
independent axes, and a delivered system is a point on each:

| axis | options |
|---|---|
| **1. who writes report prose** | A (fallback, deterministic) · B (validated adapter) · C (code writes numbers, adapter writes mechanism) · D excluded |
| **2. who writes follow-up prose** | template lookup only · the existing adapter as-is · the adapter retrained on follow-ups (F) |
| **3. training spend** | none · F (180 steps) · E (retrain the report adapter) |

A clarification on the shorthand used in the earlier cost table: "A + B" and "A + C" are
redundant labels, because B and C already contain A as the fallback guarantee for reports. The only
combination that genuinely adds a surface is **A + F**.

| # | report | follow-ups | training | GPU | wall clock | protocol status | verdict |
|---|---|---|---|---|---|---|---|
| 1 | A | template only | none | 0 | 0 | legal | cheapest; the free-form follow-ups are lost — the three LLM intents and the valve/causal cases degrade to refusals |
| 2 | A | adapter **as-is** | none | inference only, 97-row eval | ~10 min | legal | a probe, not a destination: tests whether the pilot adapter transfers to dialogue at all |
| **3** | **A** | **F** | **F, 180 steps** | **38–65 min** | **~1.5 h** | **legal, spec written** | **the recommendation: every gate either passed or pre-registered** |
| 4 | A, then C | F | F, then C on §30 | 38–65 min + minutes | ~1.5 h, then +1 day of code | C needs a dated §30 | the staged path if adapter mechanism prose is wanted in reports later |
| 5 | B | F | F | ~15–20 min + 38–65 min | ~2 h | legal but conditional | validator and retry on the report path; most likely breaches the 25 % ceiling and falls through to 4 |
| 6 | C | F | F | 38–65 min + minutes | ~1.5 h + a day of code | needs a §30 | the strongest measured-quality combination, with the largest code change |
| 7 | B + post-hoc repair | F | F | minutes ×2 + 38–65 min | ~2 h + code | needs a §30 | more moving parts than 6 for the same destination |
| 8 | stratified routing: adapter only on menu-removed | F | F | 38–65 min + minutes | ~1.5 h + a routing rule | needs a §30 | wins a 10-row stratum; C delivers the same benefit without a per-item voice switch |
| 9 | E | F | E + F | 3.6–5.4 h | days | needs a §30 | only if the adapter must author reports as well |
| 10 | A | retired (no adapter anywhere) | none | 0 | 0 | legal | the free-form follow-up surface is given up entirely |

### 9.1 What would actually be run, in order

1. **#3** — the only combination whose report gates are already passed and whose training run is
   specified. 38–65 minutes, one session.
2. **#2** as a zero-cost probe before #3 is launched, if the transfer of the pilot adapter to
   dialogue is in doubt: it costs an evaluation, not a training run.
3. **#4** only on a dated §30, and only if adapter mechanism prose in reports is wanted.

Rejected: **#1** and **#10** (give up the follow-up surface for no saving), **#5** and **#7** (spend
GPU to arrive at #4), **#9** (3.6–5.4 h to duplicate what a deterministic writer already does at 0
defects), **#6** and **#8** without a §30.

## 10. Options that were never costed

Five alternatives are real and absent from §2. None has been run, so none has evidence behind it.

| option | what it is | GPU | why it is not in §2 |
|---|---|---|---|
| **post-hoc repair** | keep the adapter text; a deterministic fixer rewrites only the flagged phrases (inequality direction, stored value, tag role) instead of replacing the whole item with fallback | minutes | a genuine third way between B (retry) and C (seam) that was never costed; by the time the fixer handles every number, the code is writing them, so it converges on C with a generation step still attached |
| **best-of-n** | generate 3 candidates, ship the first that passes validation | minutes ×3 | a cheaper cousin of B's retry-once; unbudgeted |
| **stratified routing** | adapter authors only the strata where it is preferred (menu-removed, 0.889), fallback elsewhere | 0 | the most tempting omission and the flattest: it wins on a 10-row stratum and adds a per-item voice switch an operator would notice — a product cost the preference score does not measure |
| **constrained decoding / slot filling** | the adapter writes around placeholders and code substitutes every number | minutes | a softer C, at prompt level rather than at section level |
| **different base model or scale** | retrain at a larger model, or a quantised larger one, for numeric fidelity | hours and up | a variant of E with a different model, not a different plan family |

Of these, **post-hoc repair** is the one worth costing first: it targets the defect class directly
(18 of the 23 defect instances are numeric), needs no retraining, and unlike B does not discard the
item when the retry fails.

## 11. What §9 is not

§9 is a list of what has been measured, costed or pre-registered — **not a proof that these are the
only ways this can be implemented.** No finite table exhausts a design space.

What is genuinely closed is narrower:

- **Report prose.** §29.3 admits three outcomes only — fallback authors, validated adapter prose, or
  the hybrid rebuild. A fourth report path needs a dated §30.
- **Training runs.** F is specified; E requires a §30.
- **Everything else** in §10 is possible, uncosted, and would arrive the same way: a dated section,
  a fixed gate, and a measurement, in that order.
