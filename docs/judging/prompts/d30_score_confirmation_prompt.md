# D30 - confirmation of the 40-row scoring sheet

This is a confirmation prompt, not a fresh scoring pass. It exists because the join result
is extreme enough that it should not be frozen without a second look.

## Why it is being asked

`audit_v5_sheet.py join` on your sheet returns adapter 0, fallback 40, tie 0 - share 0.0000,
decision `fallback_v31`, in every stratum (removed 0/10, open_set 0/10, normal 0/20). The
fallback side scored `1` on all 160 cells while the adapter scored `0/40` on grounded. The
preference column agrees with the higher-scored side on all 40 rows, so the sheet is
internally coherent - this is not a swapped-column or sign-error artefact.

The likely explanation is the rubric, not the answers: the v5 template writes derived
multiples such as `reconstruction error 333.60x`, which are not panel values, so a strict
"grounded" reading fails every adapter row while the fallback prose, which repeats panel
facts and adds none, passes every one. Confirm that is what you meant, or correct the cells.

## Files

- Sheet to check: `docs/judging/confirm/evidence_2026-11-06/audit_ab_100_v5_scored_rater1.csv`
  (40 rows, 320 score cells, preference 20 A / 20 B).
- Blind sheet, no key: `outputs/analysis/phase3a/fit/audit_ab_100_v5.csv`.
- Reveal, names each side: `docs/judging/confirm/evidence_2026-11-06/audit_v5_reveal.csv`.
- Join report: `docs/judging/confirm/evidence_2026-11-06/v5_join_report.json`.
- Key, only after your read: `outputs/analysis/phase3a/fit/audit_ab_100_v5_key.csv`.
- Original workbook: `outputs/analysis/phase3a/fit/tep_blind_answer_evaluation.xlsx`.

## The four criteria, as pre-registered

- `correct` - nothing is asserted that the evidence does not support.
- `grounded` - every number and tag appears in the panel, in the right direction.
- `consistent` - internally consistent only; not measured against the supplied diagnosis.
- `useful` - names the top sensor, its direction and the operator's next step.

## What to look at

Re-read the rows where the fallback scored 1,1,1,1 and the adapter scored 0 on grounded.
Ask one question per row: did I mean "this answer contains a number that is not in the
panel", or did I mean "this answer is worse"? The first is a rubric finding, the second is
a preference finding, and they are recorded differently.

## Recording the outcome

- If unchanged: reply "confirm". It is recorded as a dated attestation naming you and the
  channel, and D30 freezes at adapter 0/40.
- If cells change: save as `audit_ab_100_v5_scored_rater1_revised.csv`, then run

      python scripts/import_blind_scores.py --workbook <revised.csv>
      python scripts/audit_v5_sheet.py join --scored outputs/analysis/phase3a/fit/audit_ab_100_v5_scored_rater1.csv --out outputs/analysis/phase3a/fit/v5_join_report.json
      python scripts/audit_v5_sheet.py reveal --scored outputs/analysis/phase3a/fit/audit_ab_100_v5_scored_rater1.csv --out outputs/analysis/phase3a/fit/audit_v5_reveal.csv

## Note on the test

`tests/test_v5_scoring_decision.py` asserts the pre-registered 14 A / 11 B / 15 tie. It is
not edited to match new data; a different result is recorded as a new dated section, which
section 16 of `docs/HANDOVER_BENCHMARK_PLAN_V2.md` already does.
