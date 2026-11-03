# Pilot v5 A/B sheet (PROTOCOL 26)

This directory holds the rebuilt judging package. The v4 sheet was found invalid by the owner's
review — its two texts described different events, its key did not match text style, it carried
seven classified-fault rows inside the NORMAL family, and only 13 `NORMAL_TRANSIENT` rows instead
of 20. It is kept at `outputs/analysis/phase3a/fit/audit_ab_100_v4.csv` and marked SUPERSEDED. No
number from it is used anywhere.

## Files

| File | What it is |
|---|---|
| `audit_ab_100_v5.csv` | 40 rows, scoring blank, no family/diagnosis/verdict/source column |
| `audit_ab_100_v5_key.csv` | `item_id, family, diagnosis, verdict, A, B, seed`; 5/5, 5/5, 10/10 |
| `ab_v4_diagnosis.csv` | per-row diagnosis of every v4 row: item each text was generated for, same-event check, style check, real stratum |
| `ab_v4_diagnosis_summary.json` | counts, root-cause lines and the evaluation join check |

## What is in the sheet

- 10 open-set items presented on a menu, 10 menu-removed, 20 `NORMAL_TRANSIENT`
  (`verdict no_fault`). No classified-fault row anywhere.
- Both texts on a row come from the **same item**: the pilot adapter's stored answer for that item
  and the v3.1 deterministic explainer answering the same prompt. The build asserts this.
- Both sides use one identical layout — `Diagnosis:`, `Reasoning:`, `Narrative:` — with the
  fallback's own labels (`Differential:`, `Recommended:`, stock openers) normalised away, so the
  layout cannot identify the source.
- Every scoring column is blank. The preference gate is unmeasured until a rater who has not seen
  the key scores the sheet.

## Reading the key

`A` and `B` name the generator on each side. `family` is `open_set`, `removed` or `normal`;
`diagnosis` is the gold label the row is judged against.
