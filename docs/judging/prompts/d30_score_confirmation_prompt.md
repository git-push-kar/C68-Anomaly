# D30 - what to do with the 40-row scoring sheet

This is a confirmation pass, not a fresh scoring of 100 items. It exists because one column of
the sheet is uniform in a way that needs a named rule, not because the answers need re-reading
from scratch.

## The files

| file | role |
|---|---|
| `outputs/analysis/phase3a/fit/audit_ab_100_v5.csv` | the blind sheet you filled: `item_id, text_A, text_B`, eight score cells, `preference`, `notes` - no panel, no key |
| `docs/judging/confirm/evidence_2026-11-06/audit_ab_100_v5_scored_rater1.csv` | the same 40 rows plus the four panel fields they are scored against, and the cleaned scores |
| `docs/judging/confirm/evidence_2026-11-06/audit_v5_reveal.csv` | which side is the adapter and which the fallback, per row |
| `outputs/analysis/phase3a/fit/audit_ab_100_v5_key.csv` | the closed key: `item_id, family, diagnosis, verdict, A, B, seed` |
| `docs/judging/confirm/evidence_2026-11-06/v5_join_report.json` | the current join result |
| `outputs/analysis/phase3a/fit/tep_blind_answer_evaluation.xlsx` | the original workbook |

Open the reveal and the key only after your read is saved; both turn an independent read into a
confirmatory one.

## What the sheet asks of you

Forty rows, each with one panel and two answers to it. Per row you produce nine values:

1. `correct_A`, `grounded_A`, `consistent_A`, `useful_A` - 1 or 0 each;
2. the same four for `_B`;
3. `preference` - `A`, `B` or `TIE`.

The unit is the **answer**, not the pair: you score A on its own merits against the panel, then B
on its own merits, then say which of the two you would rather an operator read. A row where both
answers score 1,1,1,1 still needs a preference, and a row where both are poor still needs one.

## The four cells, and what each is measured against

| cell | measured against | the question |
|---|---|---|
| `correct` | `supplied_diagnosis`, `detectors` | does it assert anything the page does not support? |
| `grounded` | `detectors`, `top_sensors`, `residual_shares` | is every number and tag traceable to the panel, in the right direction? |
| `consistent` | the answer alone | does it contradict itself? (not: does it agree with the supplied diagnosis) |
| `useful` | the operator's next action | does it name the top sensor, its direction, and what to check? |

`correct` and `grounded` overlap but are not the same: a text can be internally careful and still
cite a number that is not on the page.

## Worked example: `v5-01`

Panel: `supplied_diagnosis = UNCLASSIFIED_OPEN_SET`; detectors `reconstruction error 229.2001
(thr 0.6870)`, `CVA Q-statistic 2585907.4 (thr 194.6)`; top sensors `XMEAS_16 z 31.75`, `XMEAS_29
z -24.09`, `XMEAS_31 z 23.48`, `XMEAS_7 z 22.54`, `XMEAS_13 z 20.72`; residual shares `XMV_44
0.359`, `XMEAS_1 0.351`, `XMEAS_23 0.285`.

Side A says the evidence shows a slow-drift anomaly, names XMEAS_16/7/13 as rising, XMEAS_29 as
falling, points at XMV_44 and XMEAS_23 from the residual shares, adds "slow-drift ratio 0.34",
and rules FAULT_01 out on the ground that XMEAS_23 rises first rather than falls.

Side B repeats the panel facts, states detector ratios as `333.60x` and `13286.16x`, cites the same
two top sensors, says of the listed faults that FAULT_01 and FAULT_02 would each start at a tag
that is not among these deviations, and ends with a next step on the stripper steam valve.

The sheet scores A 1,0,1,1 and B 1,1,1,1, preferring B.

Both texts cite numbers that are not panel values - `0.34` in A, `333.60x` and `13286.16x` in B.
That is the whole question: one of them was accepted and the other was not.

## The asymmetry to resolve

Checked field by field over all 40 rows:

| side | numbers cited | not present verbatim in the panel | rows with at least one |
|---|---|---|---|
| adapter | 540 | 243 | 40 of 40 |
| fallback | 566 | 227 | 40 of 40 |

Every fallback row cites at least one non-panel number and scored grounded 40 of 40. Every adapter
row does too and scored 0 of 40. So the rule that was applied cannot have been "every number
appears in the panel", and whatever it was, it was not the same on both sides.

What is being asked, in one sentence: **name the rule for derived numbers, and apply it to both
sides.** Two rules are defensible and they differ in what they permit:

- **strict** - a derived number is not panel content, so a text that reports one is not fully
  grounded. Then the fallback's grounded column is wrong too and falls to 0 of 40.
- **permissive** - derived arithmetic is allowed when its inputs are in the panel and the
  arithmetic is right. Then the adapter's grounded column has to be re-read: its deviation
  percentages and ratios may pass, row by row, and any that do not keep their 0.

What neither rule changes: `correct` adapter 19 against fallback 40, `useful` 12 against 40, and
`preference` fallback 40 of 40. Under both rules the pre-registered decision goes against the
adapter.

What does change is what may be claimed about the fallback. Under the strict rule nothing in the
sheet clears the 90 % grounded bar and the honest claim is "the adapter does not meet the bar",
not "the fallback passes it".

## Recording the outcome

- **Unchanged** - reply `confirm`. The entry names you, the channel and the date, and D30 freezes
  at adapter 0 of 40 with the grounded column recorded as stated.
- **Strict rule applied** - set `grounded_B` (the fallback column) to 0 wherever the text cites a
  derived number, save as `audit_ab_100_v5_scored_rater1_revised.csv`, then run:

      python scripts/import_blind_scores.py --workbook docs/judging/confirm/evidence_2026-11-06/audit_ab_100_v5_scored_rater1_revised.csv
      python scripts/audit_v5_sheet.py join --scored outputs/analysis/phase3a/fit/audit_ab_100_v5_scored_rater1.csv --out outputs/analysis/phase3a/fit/v5_join_report.json
      python scripts/audit_v5_sheet.py reveal --scored outputs/analysis/phase3a/fit/audit_ab_100_v5_scored_rater1.csv --out outputs/analysis/phase3a/fit/audit_v5_reveal.csv

  `import_blind_scores.py` validates ids, cell count and text identity and refuses an incomplete
  sheet with exit 2, so a run that returns 0 is a run whose cells are all present.
- **Permissive rule applied** - re-score the nine cells of the adapter rows only, save, and run
  the same three commands.

Neither route edits `tests/test_v5_scoring_decision.py`, which asserts the pre-registered 14 A /
11 B / 15 tie. A different result is recorded as a new dated section, which sections 16 and 18 of
`docs/HANDOVER_BENCHMARK_PLAN_V2.md` do.

## Report back

Which rule you named, which rows changed (if any), and - for the strict route - whether you want
the fallback column rewritten to 0 or left as recorded with the discrepancy noted.
