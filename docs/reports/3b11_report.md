# 3B-11 — the v3 outcome, fallback v3.1, and the prose pilot, built but not trained

Repo `C:/Users/Admin/Desktop/anomaly`, branch `benchmark-upgrade`. §23 was committed as `8f70135`
before any number below was computed. No adapter training, no test component opened, no teacher
call, no GPU use. Run-1 files, the KB, the menu/dataset hashes and `train_mixed.jsonl` are untouched.

## Gate

| # | gate | result |
|---|---|---|
| 1 | §23 dated, committed first | **pass** — `8f70135`, with the scored 40-row sheet |
| 2 | v3.1 table, truncation reproduced then fixed | **pass** — 520 rows broken in v3, 0 in v3.1 |
| 3 | blind sheet + leak test | **pass, with a disclosed confound** |
| 4 | `train_prose_v1.jsonl`, counts, hashes, shuffle assertion | **pass** — 1,523 rows, 0 identity menu orders |
| 5 | validation exports | **pass** — 520 items x 3 orders + 97 follow-up turns |
| 6 | pilot plan | **pass** — `docs/PILOT_PLAN_PROSE_V1.md` |
| 7 | tests | **pass** — 598 app; 595 passed, 3 skipped in `venv_train` |
| 8 | nothing trained, nothing opened, no teacher, no GPU | **pass** |

## 1. The record

§23.1 carries the rater's per-stratum figures and was committed before this round's work: classified
faults, adapter 100 / 100 / 90 against v3 100 / 100 / 100, preference 4 of 10 non-ties;
abstention + NORMAL, adapter 100 / 100 / 100 against v3 100 / 100 / 0, preference 18 of 19. Under
§21.2 as written that splits the architecture — **v3.1 for classified faults, the adapter for
abstention** — which §23.2 then froze.

Those aggregates are the rater's and were **not recomputed** here: §23.1 records them as given, and
the scored CSV is committed beside them so a recomputation needs no chat table. What §23.1 does say
is decisive for reading them: the v3 abstention column measured a **malformed** text, so the 18-of-19
preference is not evidence for the adapter.

## 2. Fallback v3.1 — v3 versus v3.1 on 834 validation rows

| metric | v3 | v3.1 |
|---|---|---|
| schema valid | 834 | 834 |
| rows with a structural issue | **520** | **0** |
| — candidate opener with no candidates | 780 occurrences | 0 |
| — orphan conclusion | 1,036 | 0 |
| — unbalanced parentheses | 260 | 0 |
| narrative byte-identical to reasoning | **520** | **0** |
| direction conflicts (single-sensor clauses) | 0 | 0 |
| rows with a number flag against the prompt alone | 308 | 308 |
| rows with a flag against prompt + supplied values | 0 | 0 |
| actions naming a printed leading sensor | 520 / 520 | 520 / 520 |
| mean words, reasoning / narrative | 80.3 / 59.4 | 84.3 / 32.7 |

The 308 prompt-only flags are `0.00` in the differential, counted at display precision as a supplied
value — the class §20.6 separates, and zero once supplied values are included. v3.1 does not change
that; it is reported so the figure is not read as a regression.

**Actions.** 59 replaced (56 cited a sensor outside the printed leading deviations, 3 were emergency
wording on a low-signal event). 115 rows are low-signal, 89 of them abstentions, and there the action
reads check / verify. **Trend words.** 1,163 printed, 819 omitted because the page's trend did not
agree with the sign of its own deviation. **60 sentences** dropped whole by the structural checker
rather than printed as fragments.

**The truncation, reproduced.** v3 lists candidates as `FAULT_16 (initiating XMEAS_9)`; those tags
never deviate — that is why the fault is named as absent — so `terminal_guard` saw unprintable tags,
cut the clause and left `"…, of the listed faults, so neither is supported by this page."` behind an
unclosed parenthesis. v3.1 prints the candidate **ids only** and asserts closure on every sentence.
The 20 abstention/NORMAL rows of the scored sheet, before and after, are in
`outputs/analysis/phase3a/fit/sheet_abstention_before_after.json`.

## 3. Blind sheet — and a confound now on the record

`audit_ab_100_v3_blind.csv` holds the same 40 rows and the same texts, with no `source` column and no
scores. `tests/test_ab_sheet_blind.py` fails if any column other than `text_A` / `text_B` separates
the sides.

**It does not pass on strata.** The key is stratum-confounded: `removed` is 0 adapter / 5 fallback and
`normal` is 9 / 1, so `family`, `diagnosis` and `verdict` each separate the sides. Removing `source`
therefore did **not** make the sheet blind — a rater who knows the strata can guess. The test pins
this exactly (stratum columns plus `source` in the scored file) so any *new* leaking column fails,
but the honest reading is that the next sheet needs a **balanced key**, not a hidden column. That is
the second time this sheet's construction, not its scoring, was the problem.

## 4. `train_prose_v1.jsonl`

| | value |
|---|---|
| rows | **1,523** (562 report + 961 dialogue) |
| report candidates | 563 |
| dropped: §17 cross-fitted rule disagrees with the gold label | **1** |
| menu identity orders | **0** (seed 7; chance about 1/16 per row) |
| supervised tokens (512 cap) | **223,631** |
| max / mean prompt tokens | 1,382 / 950.2 |
| rows over the 4,096 cap | **0** |
| steps (batch 4 x accum 4, 3 epochs) | 96 / epoch, **288** total |
| estimate at run-1's 21.71 s/step | **6,252 s, about 1 h 44 m** |
| SHA-256 | `9d0ecdfb98edcc953a6ea1e93333b4c533c4791a67437eb7823ee770e6888e62` |

`fault_id`, `verdict`, `ranked_hypotheses` and the classifier's top-2 sit in the prompt; the answer
carries `physical_reasoning` and `narrative` only, so masking is a property of the file rather than a
trainer option. Re-running the §17 rule out of fold on the training events is what makes the one-row
drop checkable (`crossfit_train_oof.jsonl`, 1,919 events, 1,677 label agreement).

## 5. Validation exports

`outputs/benchmark_v2/eval/prose_pilot/`: 520 abstention / NORMAL / unclassified items in KB order
and two seeded shuffles (1, 2), plus the follow-up turns of those events; hashes in `manifest.json`.

**The 179 follow-up turns in the round brief do not reproduce.** The abstention-flagged dialogues
carry **97 turns across 42 dialogues**; the whole validation split carries 608. 97 is what the export
contains, and the pilot plan says so rather than restating 179.

## 6. What this round does not establish

* v3.1 is judged by **deterministic checks**, not by a reader or a rater. It is well-formed; whether
  it reads better than the adapter is the blind-sheet gate in §23.4, and that sheet does not exist.
* The 40-row sheet was single-rater and stratum-confounded. The only architecture decision resting on
  it is the split §23.2 froze, **with** that caveat written into §23.1.
* The 1 h 44 m is run-1's seconds-per-step applied to 288 steps. No step of this file has run.
