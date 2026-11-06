# D29 sheets, archived 2026-11-06

Committed copies, because `outputs/` is gitignored and this project has already lost a scored
sheet that lived only there. The pipeline still reads and writes the copies under
`outputs/analysis/phase3a/fit/d29/`; these are the record, not the working set.

| file | what it is |
|---|---|
| `sheet_rater1_human.csv` | rater 1's 180 choices: 59 left, 58 right, 63 tie |
| `sheet_rater2.csv` | rater 2's 180 choices: 88 left, 70 right, 22 tie |
| `third_read_sheet.csv` | the 90 disputed pairs, `choice` column blank, for a third reader |
| `third_read_key.csv` | what the two raters chose on those 90 pairs - open only after the third read is saved |

Side order is identical across both rater sheets on all 180 pairs, so choices pair row by row.
Seventeen of the 180 pairs have byte-identical text on both sides; both raters called all
seventeen `tie`, and those pairs are excluded from the 90-pair adjudication sheet because there
is nothing to adjudicate.
