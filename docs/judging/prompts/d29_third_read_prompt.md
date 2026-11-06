# D29 - third-read adjudication prompt

Hand this to the third reader. It is the last read D29 needs.

## What this is

The same 180 blind pairs (conditions A, C and D against each other; no pair states which condition
it is) were scored twice and independently: first by Claude, then by ChatGPT. They agree on 90
pairs and disagree on 90; Cohen's kappa on all 180 is 0.2556 against a 0.60 floor. A human read of
the 90 disputed pairs is the missing piece: it is the only human reading in D29, and the report
puts it beside the model-versus-model kappa rather than replacing it.

## Your task

Fill the `choice` column of every row: `left`, `right` or `tie`, lowercase.

## Files

- Work in: `docs/judging/confirm/d29_sheets/third_read_sheet.csv` - 90 rows, columns
  `pair_id, item_id, text_left, text_right, choice, notes`.
- Write only the `choice` and (optionally) `notes` columns. Do not edit the texts, do not add or
  delete rows.
- Do not open `docs/judging/confirm/d29_sheets/third_read_key.csv` or
  `outputs/analysis/phase3a/fit/d29/key.csv` until your sheet is saved. The first records what
  Claude and ChatGPT chose, the second records which side is which condition.
- The two model sheets, for context only - do not edit them:
  `docs/judging/confirm/d29_sheets/sheet_rater1_human.csv` (Claude; the `_human` in the filename is
  a historical error, see `provenance.json`) and
  `docs/judging/confirm/d29_sheets/sheet_rater2.csv` (ChatGPT).
- Background if you want it: `outputs/analysis/phase3a/fit/d29/d29_receipt.json` (parameters,
  hashes, selection rule) and `outputs/analysis/phase3a/fit/d29/texts.jsonl` (the source corpus).

## How to choose

- `left` or `right`: which of the two answers you would put in front of an operator.
- `tie`: only when neither is better. If the two texts are identical, `tie` is the only defensible
  answer.
- Judge each pair yourself, without looking at either model's choice and without trying to infer
  which condition wrote which side.
- Pace yourself: about 90 pairs, roughly 20-30 minutes. Fatigue shows up as drift toward `tie`;
  the two model readers show exactly that, Claude at 63 ties and ChatGPT at 22.

## What happens to your file

`python scripts/preference_agreement.py --rater1 docs/judging/confirm/d29_sheets/sheet_rater1_human.csv
--rater2 docs/judging/confirm/d29_sheets/sheet_rater2.csv` already gives the model-versus-model
kappa. Your read is reported alongside it: how often you agree with each model, how often all three
agree, and on how many pairs no majority exists. It does not raise kappa; it records how much of
the instrument needed a third reader.

## Report back

Rows filled, your left/right/tie counts, minutes taken, and any pair you could not decide.
