# D29 sheets

Committed copies, because `outputs/` is gitignored and this project has already lost a scored
sheet that lived only there. The pipeline still reads and writes the copies under
`outputs/analysis/phase3a/fit/d29/`; these are the record, not the working set.

**Reader identities are in `provenance.json`, which is the authority.** The `_human` suffix in the
rater-1 filename is a historical error: that sheet was read by Claude.

| file | reader | choices |
|---|---|---|
| `sheet_rater1_human.csv` | Claude (Anthropic) | 59 left, 58 right, 63 tie |
| `sheet_rater2.csv` | ChatGPT (OpenAI) | 88 left, 70 right, 22 tie |
| `third_read_sheet.csv` | human | 32 left, 38 right, 20 tie, over the 90 disputed pairs |
| `third_read_key.csv` | closed key | what the two model readers chose on those 90 |

Side order is identical across both model sheets on all 180 pairs, so choices pair row by row.
Seventeen of the 180 pairs have byte-identical text on both sides; both model readers called all
seventeen `tie`, and those pairs are excluded from the 90-pair adjudication sheet because there is
nothing to adjudicate.

Because neither original rater is human, D29 has no human-versus-human kappa. The 0.2556 figure
is agreement between two models; the one human read is the third sheet.
