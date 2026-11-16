# The prose instrument, v5 against v5b, 2026-11-06

Two dated readings of the same 520 stored generations coexist. The v5 reading is untouched and stays
where it was recorded; v5b is the reader in force. This note is the diff the P0 acceptance asks for.

## Which reader is which

| label | reader | status |
|---|---|---|
| v5 | `scripts/prose_pilot_prompts.contradiction` and its companions | **retired**; its counts remain in `docs/judging/pilot_v5/pilot_rechecks.json` |
| v5b | `scripts/pilot_rechecks.contradiction_flags` and the clause-bound checks | **in force**; the gate reads this reader only |

`outputs/overnight_2026-11-05/n1_parity.json` compares them on every stored row and order: **1,560
comparisons, 170 agree, 1,390 disagree**. The disagreement is not noise between two estimators of the
same quantity — the two readers were answering different questions, which is why one was retired
rather than reconciled.

## The re-specifications, and their effect

From `pilot_rechecks.json:old_vs_new`, over 520 rows in the `kb` order:

| check | v5 count | v5 rate | v5b count | v5b rate | threshold | note |
|---|---|---|---|---|---|---|
| contradiction | 462 | 0.8885 | **0** | 0.0000 | ≤ 0.02 | re-specified 28.2: a stored value is not a prose defect |
| tag_role | 396 | 0.7615 | **57** | 0.1096 | ≤ 0.02 | re-specified 28.3: a canonical name is not a role claim |
| numeric | 69 | 0.1327 | 62 | 0.1192 | ≤ 0.02 | same checker, decomposed by cause |
| grounding | 33 | 0.0635 | 33 | 0.0635 | ≤ 0.02 | same checker, split into numbers and tags |
| anchor | 187 | 0.3596 (of 520) | 170 | 0.5167 (of 329) | ≥ 0.90 | higher is better; denominator is rows with top \|z\| ≥ 1 |

The two re-specifications are the whole story. The v5 contradiction reader fired on **89 % of rows**,
so a gate at 0.02 would have eliminated every cell but A on a defect that was in the checker, not in
the prose. The clause binding cut tag_role from 396 to 57. Numeric and grounding are unchanged in
count: the defect is real and survives the re-specification.

The anchor row is not comparable across readings — v5 scored all 520 rows, v5b scores the 329 rows
whose top \|z\| ≥ 1 — so the two rates sit in different denominators and only the v5b one is a
reading of the instrument as it now stands.

## Calibration

The overnight stage records **46/46 at precision 1.00 and recall 1.00** on the frozen set
(`n1_calibration.json`, `stages.json`). `pilot_rechecks.json` carries the per-check split over the 40
cases that existed when it was written: contradiction 14 (7/7), tag_role 12 (6/6), anchor 8 (4/4),
numeric 6 (3/3), all at 1.00/1.00. The six later cases are the mixed-direction, stored-value,
hypothetical, `XMV`-as-valve and two contradiction-rejection shapes; the frozen record is
`docs/judging/confirm/cases46_freeze.json`.

## Three orders, v5b in force

| order | tag_role | grounding | numeric |
|---|---|---|---|
| kb | 57 | 33 | 62 |
| shuffled-1 | 39 | 36 | 49 |
| shuffled-2 | 38 | 36 | 46 |

## What this changes and what it does not

The direction of every gate is unchanged: numeric and grounding still fail the 2 % threshold, tag_role
still fails it, contradiction now passes, and the anchor still misses 0.90. What changed is that
**the failures are now attributable to the prose**: numeric 62 rows with 34 value-not-stored and 47
wrong-inequality-direction flags; grounding 33 rows, 28 by number and 5 by tag.

One acceptance item is not met literally: P0.4 asked for the two prose readers to produce *identical
counts*, and they do not — 1,390 of 1,560 comparisons still disagree because one reader was retired
instead of being made to agree. The deviation is on record in `n1_parity.json` with the reader in
force named, and no run reads the retired reader. Stating it as "parity achieved" would be wrong;
stating it as "one reader in force, the other retired, with the diff measured" is what happened.
