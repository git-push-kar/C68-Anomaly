# Pilot evaluation: item_id join check and gate recomputation (PROTOCOL 27)

## 1. Join

| set | rows | joined by item_id | not joined |
|---|---|---|---|
| kb (520) | 520 | 520 | 0 |
| shuffled-1 (520) | 520 | 520 | 0 |
| shuffled-2 (520) | 520 | 520 | 0 |
| follow-up turns (179) | 179 | 179 | 0 |

Rows with no prompt for their item_id: 0.
Rows whose written order field disagrees with its file: 0.
Rows where a positional join would pick the same prompt: 1739 of 1739 (the files are in build order, so position agrees today; the join is nevertheless by item_id and a test pins it).

## 2. Anchor decomposition (520 kb rows)

| class | rows | share |
|---|---|---|
| exact | 187 | 35.96% |
| sensor_absent | 68 | 13.08% |
| z_format_differs | 11 | 2.12% |
| z_true_mismatch | 254 | 48.85% |

Mismatch sub-classes: contradicting_value 106, z_omitted 148.

Own top sensor and z: 187 (35.96%); next row's: 6 (1.15%).

## 3. Gates, joined by item_id: stored vs recomputed

| gate | threshold | order | stored | recomputed | same |
|---|---|---|---|---|---|
| schema_valid | 98.00% | kb | 520 / 520 = 100.00% pass | 520 / 520 = 100.00% pass | yes |
| schema_valid | 98.00% | shuffled-1 | 520 / 520 = 100.00% pass | 520 / 520 = 100.00% pass | yes |
| schema_valid | 98.00% | shuffled-2 | 520 / 520 = 100.00% pass | 520 / 520 = 100.00% pass | yes |
| complete | 95.00% | kb | 520 / 520 = 100.00% pass | 520 / 520 = 100.00% pass | yes |
| complete | 95.00% | shuffled-1 | 520 / 520 = 100.00% pass | 520 / 520 = 100.00% pass | yes |
| complete | 95.00% | shuffled-2 | 520 / 520 = 100.00% pass | 520 / 520 = 100.00% pass | yes |
| empty_narrative | 0.00% | kb | 0 / 520 = 0.00% pass | 0 / 520 = 0.00% pass | yes |
| empty_narrative | 0.00% | shuffled-1 | 0 / 520 = 0.00% pass | 0 / 520 = 0.00% pass | yes |
| empty_narrative | 0.00% | shuffled-2 | 0 / 520 = 0.00% pass | 0 / 520 = 0.00% pass | yes |
| grounding | 2.00% | kb | 33 / 520 = 6.35% FAIL | 33 / 520 = 6.35% FAIL | yes |
| grounding | 2.00% | shuffled-1 | 36 / 520 = 6.92% FAIL | 36 / 520 = 6.92% FAIL | yes |
| grounding | 2.00% | shuffled-2 | 36 / 520 = 6.92% FAIL | 36 / 520 = 6.92% FAIL | yes |
| numeric | 2.00% | kb | 69 / 520 = 13.27% FAIL | 69 / 520 = 13.27% FAIL | yes |
| numeric | 2.00% | shuffled-1 | 58 / 520 = 11.15% FAIL | 58 / 520 = 11.15% FAIL | yes |
| numeric | 2.00% | shuffled-2 | 54 / 520 = 10.38% FAIL | 54 / 520 = 10.38% FAIL | yes |
| tag_role | 2.00% | kb | 396 / 520 = 76.15% FAIL | 396 / 520 = 76.15% FAIL | yes |
| tag_role | 2.00% | shuffled-1 | 396 / 520 = 76.15% FAIL | 396 / 520 = 76.15% FAIL | yes |
| tag_role | 2.00% | shuffled-2 | 398 / 520 = 76.54% FAIL | 398 / 520 = 76.54% FAIL | yes |
| contradiction | 0.00% | kb | 462 / 520 = 88.85% FAIL | 462 / 520 = 88.85% FAIL | yes |
| contradiction | 0.00% | shuffled-1 | 464 / 520 = 89.23% FAIL | 464 / 520 = 89.23% FAIL | yes |
| contradiction | 0.00% | shuffled-2 | 464 / 520 = 89.23% FAIL | 464 / 520 = 89.23% FAIL | yes |
| over_refusal | 10.00% | kb | n/a / 40 = 0.00% pass | n/a / 40 = 0.00% pass | yes |
| over_refusal | 10.00% | shuffled-1 | n/a / 40 = 0.00% pass | n/a / 40 = 0.00% pass | yes |
| over_refusal | 10.00% | shuffled-2 | n/a / 40 = 0.00% pass | n/a / 40 = 0.00% pass | yes |
| refusal | 90.00% | kb | n/a / 28 = 100.00% pass | n/a / 28 = 100.00% pass | yes |
| refusal | 90.00% | shuffled-1 | n/a / 28 = 100.00% pass | n/a / 28 = 100.00% pass | yes |
| refusal | 90.00% | shuffled-2 | n/a / 28 = 100.00% pass | n/a / 28 = 100.00% pass | yes |

Cells reproduced: 27 of 27. Follow-up readings: refusal 100.00% on 28, over-refusal 0.00% on 40.

## 4. Per-order differences

| reading | stored | recomputed |
|---|---|---|
| identical text kb vs shuffled-1 | 2 | 2 |
| complete spread | 0.0000 | 0.0000 |
| contradiction spread | 0.0038 | 0.0038 |
| empty_narrative spread | 0.0000 | 0.0000 |
| grounding spread | 0.0057 | 0.0057 |
| numeric spread | 0.0289 | 0.0289 |
| schema_valid spread | 0.0000 | 0.0000 |
| tag_role spread | 0.0039 | 0.0039 |

## 5. Verdict

The pilot gate table stands: all 27 gated cells and both follow-up readings reproduce exactly when every row is joined to the prompt of its own item_id, with 0 of 1739 generations lacking a matching prompt.

