# 3B-5 CORRECTIONS — four statements the stored artefacts do not support

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade`
**Date:** 2026-11-01 · **Scope:** corrections to [hybrid_hardening_report.md](hybrid_hardening_report.md) (§17). The old report is **not edited** — this file states what its text should have said, and where a sentence cannot be supported it is **withdrawn** rather than restated. No new model was run to produce any figure here; every number is re-derived from artefacts that already existed.

Sources: `outputs/analysis/phase3a/hybrid/hardening_checks.json`, `.../crossfit.json`, `.../surrogate_wide.json`, `.../fold_holdout.json`, `.../crossfit_val_rows.jsonl`.

---

## a. The unobservable rows: the two halves were reported as one behaviour, and they are not

Old text (§5.1): *"Both halves answer NORMAL_TRANSIENT on 12/12; the menu-removed gold agrees with that on 6/12, and the menu-present gold (`FAULT_03`, `FAULT_09`) never does."*

**The menu-removed gold is `UNCLASSIFIED_OPEN_SET`** (six rows), and the **adapter never answers `NORMAL_TRANSIENT` on those six** — it answers `UNCLASSIFIED_OPEN_SET`. The old sentence fused the classifier's behaviour with the adapter's, and then read a coincidence into the removed half that was actually a correct answer.

| variant | n | gold | classifier | rule output (diagnosis) | adapter @ step 500 |
|---|---|---|---|---|---|
| menu | 6 | `FAULT_03` ×2, `FAULT_09` ×4 | `NORMAL_TRANSIENT` 6/6 → **0/6 agree** | `NORMAL_TRANSIENT` 6/6 → **0/6 agree** | `NORMAL_TRANSIENT` 6/6 → **0/6 agree** |
| removed | 6 | `UNCLASSIFIED_OPEN_SET` 6/6 | `NORMAL_TRANSIENT` 6/6 → **0/6 agree** | `NORMAL_TRANSIENT` 6/6 → **0/6 agree** | `UNCLASSIFIED_OPEN_SET` 6/6 → **6/6 agree** |
| **all 12** | 12 | — | **0/12** | **0/12** | **6/12** |

```bash
python - <<'PY'
import json, pathlib
from collections import Counter
rows = json.loads(pathlib.Path("outputs/analysis/phase3a/hybrid/hardening_checks.json")
                  .read_text(encoding="utf-8"))["unobservable"]["rows"]
for split in ("menu", "removed"):
    part = [r for r in rows if r["variant"] == split]
    print(split, len(part),
          sum(1 for r in part if r["classifier_label"] == r["gold"]),
          sum(1 for r in part if r["diagnosis"] == r["gold"]),
          sum(1 for r in part if r["adapter_step500"] == r["gold"]),
          Counter(r["adapter_step500"] for r in part))
PY
```

What the numbers mean, stated plainly:

* The classifier and the menu rule earn **nothing** here: `NORMAL_TRANSIENT` on all twelve, which is wrong on both halves for different reasons. The rule's reason code is `no_fault_predicted` on every row.
* The adapter earns **6 of 12**, and all six are on the removed half — because it reads the menu, sees no entry matching, and abstains. It is not scoring a fortunate match between `NORMAL_TRANSIENT` and the gold; the old sentence's "agrees with that" described agreement between a *classifier's* answer and the gold and attributed it to the adapter.
* **The 12 rows are 6 distinct events** (`f03-r302`, `f03-r317`, `f09-r301`, `f09-r302`, `f09-r317`, `f09-r319`), each contributing a menu-present and a menu-removed twin with **different golds**. Any count over these rows is a count over variants, not events.
* Under §18.2's second scoring ("acceptable non-diagnosis": `NORMAL_TRANSIENT` or `UNCLASSIFIED_OPEN_SET` counts on a 3/9 row), the classifier and the rule score **6/12** rather than 0/12, and the adapter scores **12/12**. The strict column in the old §2 stratum table (`unobservable (3, 9) … accuracy 0.0000`, `macro-F1 0.0000`) is correct as a strict number and is not withdrawn; it is simply not the only defensible reading, which is why §18.2 reports both and neither enters a headline.

---

## b. "44 of 50 surrogate errors" — withdrawn, with the definition that produced it

Old text (§5.3, and its restatement in §11): *"44 of the 50 surrogate errors and all 8 natural errors are absorbed by `FAULT_08`, `FAULT_11` or the open-set class."*

**Withdrawn.** Both halves of that fraction belong to a single fault, not to the surrogate:

* **numerator 44** = the items of the **removed fault `FAULT_15`** that came back as `NORMAL_TRANSIENT`: 22 of its 25 menu-present items + 22 of its 25 menu-removed items.
* **denominator 50** = `FAULT_15`'s **own** held-out items (25 + 25).
* The surrogate's own named-error total is **136 of 325 per variant** (272 item-level across both variants) — see the table below, which sums to 136 for the menu variant.

Per-lookalike mislabel table for the **menu variant** (`surrogate_wide.json → per_fault[*].named_as.menu`), summing to 136:

| removed (held-out) fault | answered as | items of 25 |
|---|---|---|
| `FAULT_01` | `FAULT_08` | 25 |
| `FAULT_02` | `FAULT_08` | 11 |
| `FAULT_05` | `FAULT_12` | 25 |
| `FAULT_06` | `FAULT_08` | 9 |
| `FAULT_07` | `FAULT_08` | 13 |
| `FAULT_08` | `FAULT_01` ×3, `FAULT_07` ×1 | 4 |
| `FAULT_11` | `FAULT_04` | 2 |
| `FAULT_14` | `FAULT_11` | 25 |
| `FAULT_15` | `NORMAL_TRANSIENT` | 22 |
| **total named** | | **136** |

Same total on the removed variant (136). By absorbing class: `FAULT_08` 58 per variant (116 across both), `FAULT_12` 25 (50), `FAULT_11` 25 (50), `NORMAL_TRANSIENT` 22 (44 — the number that was mistaken for a denominator), `FAULT_01` 3 (6), `FAULT_04` 2 (4), `FAULT_07` 1 (2).

**Correct statement:** across the 13-fault surrogate, 136 of 325 menu-present held-out items (41.8 %) are answered with a *named* wrong fault or `NORMAL_TRANSIENT`; 116 of those 136 (85.3 %) go to `FAULT_08` or `FAULT_12`, and 22 go to `NORMAL_TRANSIENT`, all of them `FAULT_15`'s.

---

## c. The natural wrong labels: three different populations were added together, and one conclusion was inverted

Old text (§5.3): *"8 wrong, 254 abstained … **44 of the 50 surrogate errors and all 8 natural errors are absorbed by `FAULT_08`, `FAULT_11` or the open-set class**."*

The two claims fail differently. The first is a definition problem; the second is simply wrong about the predicted class.

**(i) The three populations, separated.** Of 1,919 out-of-fold train events:

| population | rows | variants | what it is |
|---|---|---|---|
| argmax wrong | **8** | menu-present only | the classifier named some class other than the gold |
| abstained on a menu-present seen fault | **64** | menu-present | **false abstention**: the class was in the menu and in training |
| abstained on a menu-removed item | **190** | menu-removed | **correct abstention**: gold is `UNCLASSIFIED_OPEN_SET` |
| wrong **or** abstained (distinct events) | **258** | — | 13.4 % of 1,919 |

8 + 254 = 262 rows against 258 distinct events, because **4 rows are counted in both** the wrong and the abstained sets: their gold is `UNCLASSIFIED_OPEN_SET`, their argmax is `FAULT_10`, `FAULT_10` is not on an open-set item's menu, and so the §17 rule withholds — the row is an argmax error *and* an abstention. Nothing double-counted that should be; the union is reported.

**(ii) "All 8 absorbed by FAULT_08, FAULT_11 or open-set" — wrong.** By predicted class (`crossfit.json → natural_wrong_labels.by_predicted_fault`):

| predicted as | count | which |
|---|---|---|
| `FAULT_10` | **4** | gold `UNCLASSIFIED_OPEN_SET` |
| `FAULT_08` | 2 | gold `FAULT_01`, `FAULT_02` |
| `UNCLASSIFIED_OPEN_SET` | 2 | gold `FAULT_10`, `FAULT_13` |
| `FAULT_11` | **0** | — |

`FAULT_11` absorbs nothing here; four of the eight go to `FAULT_10`, a class the old sentence never mentions.

**(iii) The hybrid's own error pool is smaller, and this is the useful correction.** The four `gold = OPEN_SET → argmax FAULT_10` rows are **not errors of the hybrid at all**: `FAULT_10` is off the menu, so the rule abstains and the diagnosis becomes `UNCLASSIFIED_OPEN_SET`, which is the gold. Applying the rule to the out-of-fold predictions gives:

| hybrid diagnosis error | events |
|---|---|
| named a wrong fault (`01→08`, `02→08`) | 2 |
| answered `UNCLASSIFIED_OPEN_SET` where the gold was a seen fault (`10`, `13`) | 2 |
| false abstention on a menu-present seen fault | 64 |
| **deployed error pool** | **68 distinct events (3.5 % of 1,919)** |

So the argmax pool (8 wrong + 254 abstentions = 258) is **not** the pool any "learn from natural wrong labels" dataset would be drawn from; that pool is 68 events, of which 64 are abstentions rather than misnames, and 66 of the 68 are "declined something it should have named". The catalogue remains dominated by the §3.3 lookalike pairs, but it is four times smaller than §5.3 implies and it contains no `FAULT_11` absorption at all.

---

## d. What the 323 features are parsed from, and what never reaches them

Both lines, as asked, from `scripts/evidence_classifier.py::prompt_features`:

* **Parsed from:** the five bullet sensor lines of *TOP DEVIATING SENSORS* (`z`, `deviation %`, `trend` word, and a `present` flag per tag — 52 tags × 4 = 208 columns, plus `n_listed`); the numbered lines of *ONSET ORDER* (one `onset` minute per tag, 52 columns); the *residual share* lines of *DYNAMIC RESIDUAL ATTRIBUTION* (52 columns); and the detector header (*overall anomaly score*, LSTM reconstruction error + its ratio, CVA Q-statistic + its ratio, dominant lag, slow-drift ratio, and three fused-trigger flags).
* **Never in the features:** the candidate menu (the classifier cannot see which faults were offered, which is why the hybrid needs the menu rule to turn a label into an answer); the question/instruction text and the schema line; the fault names and KB prose; the sensors' display names and their tag *kind* (valve position vs measurement — the regex captures the kind and discards it); and any event, run, item, split or ordering identifier, which `identifier_audit` asserts rather than assumes.

**One consequence worth stating**, because it bounds what the classifier can be credited with: only **5 sensors are printed** per prompt (measured: exactly 5 in every one of the 3,714 measured rows) and **5 residual variables**, while the feature vector has a column for all 52 tags. A tag that is not listed contributes `present = 0`, which is information the adapter also has — by absence. Neither side sees the full 52-tag matrix.

---

## Summary of what changed

| # | old statement | status |
|---|---|---|
| a | "both halves answer NORMAL on 12/12; removed gold agrees on 6/12" | **corrected** — classifier 0/12, rule 0/12, adapter 6/12 (all on the removed half), 6 distinct events |
| b | "44 of the 50 surrogate errors" | **withdrawn** — 44 is `FAULT_15`'s absorbed count out of its own 50 items; the surrogate's named total is 136 of 325 per variant |
| c | "all 8 natural errors absorbed by FAULT_08, FAULT_11 or open-set" | **corrected** — `FAULT_10` takes 4, `FAULT_08` 2, open-set 2, `FAULT_11` 0; the deployed-rule error pool is 68 events, not 258 |
| d | (unstated) | **confirmed** — feature sources and non-sources listed above |
