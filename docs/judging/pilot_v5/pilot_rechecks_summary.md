# Pilot artefact checks, re-specified (PROTOCOL 28)

Rows: 520. Strata: classified 19, menu 125, normal 47, removed 329. CPU only, stored generations only.

## 1. Synthetic calibration (40 hand-written cases)

| check | cases | tp | fp | fn | precision | recall |
|---|---|---|---|---|---|---|
| contradiction | 14 | 7 | 0 | 0 | 1.00 | 1.00 |
| tag_role | 12 | 6 | 0 | 0 | 1.00 | 1.00 |
| anchor | 8 | 4 | 0 | 0 | 1.00 | 1.00 |
| numeric | 6 | 3 | 0 | 0 | 1.00 | 1.00 |

## 2. Old vs new

| gate | stored | re-specified | threshold | note |
|---|---|---|---|---|
| contradiction | 462 / 520 = 88.85% | 0 / 520 = 0.00% | 2% | re-specified 28.2; stored value not a prose defect |
| tag_role | 396 / 520 = 76.15% | 57 / 520 = 10.96% | 2% | re-specified 28.3; stored value not a prose defect |
| numeric | 69 / 520 = 13.27% | 62 / 520 = 11.92% | 2% | same checker, decomposed by cause |
| grounding | 33 / 520 = 6.35% | 33 / 520 = 6.35% | 2% | same checker, split into numbers and tags |
| anchor | 187 / 520 = 35.96% | 170 / 520 = 51.67% | 90% | higher is better; rows with top |z| >= 1 |

## 3. Flagged rows by stratum (kb order)

| check | flagged | open-set on menu | menu-removed | NORMAL_TRANSIENT |
|---|---|---|---|---|
| contradiction | 0 | 0 | 0 | 0 |
| tag_role | 57 | 12 | 40 | 0 |
| numeric | 62 | 1 | 23 | 36 |
| grounding | 33 | 3 | 25 | 3 |

Per order: kb grounding 33, numeric 62, tag_role 57; shuffled-1 grounding 36, numeric 49, tag_role 39; shuffled-2 grounding 36, numeric 46, tag_role 38.

## 4. Anchor

| rows | exact | rate |
|---|---|---|
| top |z| >= 5 | 115 of 206 | 55.83% |
| 1 <= top |z| < 5 | 55 of 123 | 44.72% |
| top |z| < 1 | 18 of 191 | 9.42% |

Anchor gate (top |z| >= 1): 170 of 329 = 51.67% against a 90% threshold: FAIL.
Statuses (kb): contradicting_value 34, exact 188, sensor_absent 68, z_omitted 230.

## 5. Numeric and grounding breakdown

Numeric by cause (flags): threshold_misquoted 1, value_not_stored 34, wrong_inequality_direction 47.
Numeric rows by first cause: threshold_misquoted 1, value_not_stored 34, wrong_inequality_direction 27.
Grounding rows: numbers 28, tags 5.
Grounding codes: number_not_in_store 45, tag_not_in_evidence 5.

## 6. Side by side with fallback v3.1

| reading | adapter (kb) | fallback v3.1 |
|---|---|---|
| contradiction rows | 0 | 0 |
| tag_role rows | 57 | 0 |
| numeric rows | 62 | 19 |
| grounding rows | 33 | 0 |
| anchor exact, top |z| >= 1 | 170 of 329 = 51.67% | 329 of 329 = 100.00% |

## 7. Twenty most frequent offending phrases (all three orders)

| phrase | count |
|---|---|
| `number_not_in_store:<N>` | 145 |
| `The CVA Q-statistic (<N>) is below threshold (<N>), and the LSTM autoencoder reconstruction error (<` | 88 |
| `The CVA Q-statistic alarm (<N> > <N>) indicates a process deviation, but the top deviating sensors (` | 28 |
| `The CVA Q-statistic alarm (<N> > <N>) indicates a significant process deviation, but the top deviati` | 18 |
| `The CVA Q-statistic alarm (<N> > <N>) indicates a process deviation, but the top deviating sensors s` | 18 |
| `The CVA Q-statistic alarm (<N> vs threshold <N>) indicates a minor process deviation, but the LSTM r` | 15 |
| `but the temperature measurement (<TAG>)` | 13 |
| `tag_not_in_evidence:<TAG>` | 11 |
| `but the valve position (<TAG>)` | 9 |
| `let conditions and flow control valve position (<TAG>)` | 8 |
| `but here the flow valve position (<TAG>)` | 6 |
| `water inlet temperature and flow control valve (<TAG>)` | 6 |
| `The CVA Q-statistic (<N>) is just below threshold (<N>), and the LSTM autoencoder reconstruction err` | 6 |
| `The CVA Q-statistic (<N>) remains below threshold (<N>), and the LSTM autoencoder reconstruction err` | 6 |
| `viation in the cooling water inlet temperature (<TAG>)` | 5 |
| `is ruled out because the valve position (<TAG>)` | 4 |
| `d the valve position is a manipulated variable (<TAG>)` | 4 |
| `C feed flow (<TAG>)` | 4 |
| `iation in the cooling water outlet temperature (<TAG>)` | 4 |
| `but the flow valve position (<TAG>)` | 4 |
