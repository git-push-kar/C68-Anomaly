# D29 - third-read adjudication prompt

Hand this to a third person. It is the only remaining human step in D29.

## What this is

Two independent human raters each scored the same 180 blind pairs (conditions A, C and D
against each other; no pair states which condition it is). Their choices agree on 90 pairs
and disagree on 90. Cohen's kappa on all 180 is 0.2556 against a 0.60 floor, so the
preference instrument cannot be used as it stands. Your read settles the 90 disputed pairs
and the report states the adjudication count beside kappa.

## Your task

Fill the `choice` column of every row in the sheet: `left`, `right` or `tie`, lowercase.

## Files

- Work in: `docs/judging/confirm/d29_sheets/third_read_sheet.csv` - 90 rows, columns
  `pair_id, item_id, text_left, text_right, choice, notes`.
- Write only the `choice` and (optionally) `notes` columns. Do not edit the texts, do not
  add or delete rows.
- Do not open `docs/judging/confirm/d29_sheets/third_read_key.csv` or
  `outputs/analysis/phase3a/fit/d29/key.csv` until the sheet is saved. The first records
  what the two earlier raters chose, the second records which side is which condition.
  Either one turns your read into a confirmatory one.
- The two completed sheets, for context only - do not edit them:
  `docs/judging/confirm/d29_sheets/sheet_rater1_human.csv`,
  `docs/judging/confirm/d29_sheets/sheet_rater2.csv`.
- Background if you want it: `outputs/analysis/phase3a/fit/d29/d29_receipt.json`
  (parameters, hashes, selection rule) and `outputs/analysis/phase3a/fit/d29/texts.jsonl`
  (the corpus the pairs were drawn from).

## How to choose

- `left` or `right`: which of the two answers you would put in front of an operator.
- `tie`: only when neither is better. If the two texts are identical, `tie` is the only
  defensible answer.
- Judge each pair on its own. Do not try to infer which condition wrote which side, and do
  not discuss pairs with the earlier raters until your sheet is saved.
- Pace yourself: about 90 pairs, roughly 20-30 minutes. Fatigue shows up as a drift toward
  `tie`; the earlier raters show exactly that, one at 63 ties and the other at 22.

## What happens to your file

`python scripts/preference_agreement.py --rater1 <sheet_rater1_human.csv> --rater2 <rater2.csv>`
is re-run with your choices replacing the disputed ones, and the result is reported as kappa
plus adjudication count. Adjudication does not raise kappa; it records how much of the
instrument needed a third read.

## Report back

Rows filled, your left/right/tie counts, minutes taken, and any pair you could not decide.
