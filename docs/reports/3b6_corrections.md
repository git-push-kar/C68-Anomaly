# 3B-6 CORRECTIONS — seven statements, six of which the rows do not support

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade`
**Date:** 2026-11-01 · **Scope:** corrections to [g1_fit_signal_report.md](g1_fit_signal_report.md) and to §a of [3b5_corrections.md](3b5_corrections.md). Neither old file is edited. Every figure here is counted from rows, not read from an aggregate: `scripts/corrections_3b6.py` rebuilds the out-of-fold predictions exactly as `scripts/hybrid_crossfit.py` builds them and reproduces §17's threshold **exactly** (0.9792, 82 abstained of 1,729), so the rows it lists are the rows §17 scored.

Artefacts: `outputs/analysis/phase3a/fit/corrections_3b6.json` (all numbers), appendix A below (the 65 events).

---

## a. Faults 3/9 under "acceptable non-diagnosis": the reading is saturated and tells us nothing

§19.3 fixed the rule before this ran: a scoring that gives every system the same perfect number has measured the scoring, not the systems.

| system | strict (gold = dataset target) | acceptable non-diagnosis (`NORMAL`/`UNCLASSIFIED` counts) |
|---|---|---|
| classifier | 0 / 12 | **12 / 12** |
| rule output (diagnosis) | 0 / 12 | **12 / 12** |
| adapter @ step 500 | **6 / 12** | **12 / 12** |

Per variant: menu 6 rows (gold `FAULT_03` ×2, `FAULT_09` ×4) — classifier `NORMAL` 6/6, rule `NORMAL` 6/6, adapter `NORMAL` 6/6; removed 6 rows (gold `UNCLASSIFIED_OPEN_SET` 6/6) — classifier `NORMAL` 6/6, rule `NORMAL` 6/6, adapter `UNCLASSIFIED` 6/6.

**New statement replacing the "6/12" reading:** under the acceptable-non-diagnosis reading the classifier scores 12/12, the rule 12/12 and the adapter 12/12 — because `NORMAL_TRANSIENT` is acceptable on a 3/9 row, and the classifier answers `NORMAL_TRANSIENT` on all twelve. The reading is therefore **uninformative** (§19.3) and is reported as such, beside the strict column, entering no headline. The strict column is unchanged and is the one that discriminates: **0 / 0 / 6**, and the adapter's six are all on the menu-removed half where abstaining is the intended behaviour.

---

## b. The natural error pool: 65 events, and the two named in the question are inside the 64

Two different golds are in play, and confusing them is what produced 68:

* an event's **true fault** (`hc.true_label`), which is what a classifier's label is scored against;
* the **dataset's target** (`target.fault_id`), which is what a diagnosis is scored against — and for the train split's 190 `::openset` rows the target is `UNCLASSIFIED_OPEN_SET` while the event's true fault is a seen one.

Measured on the rebuilt rows:

| population | rows | events |
|---|---|---|
| classifier label ≠ true fault (the §17 census) | 8 | 8 |
| of those, **rescued** by the menu rule (argmax wrong, diagnosis = target) | 3 | 3 |
| of those, still wrong as a diagnosis | 5 | 5 |
| **deployed diagnosis ≠ dataset target — named a wrong fault** | **2** | 2 |
| **deployed diagnosis ≠ dataset target — abstained on a seen-fault row** | **63** | 63 |
| **deployed error pool** | **65** | **65** |
| correct abstentions (target `UNCLASSIFIED`, diagnosis `UNCLASSIFIED`) | 451 | 451 |

**The pool is 65, not 66 and not 68.** The two events in the question — `EVT-train-f10-r091::menu` and `EVT-train-f13-r086::menu`, gold `FAULT_10` / `FAULT_13`, argmax `UNCLASSIFIED_OPEN_SET` at `below_threshold` — **do sit inside the 63 false abstentions**; they are not additional, which is why the pool is not 67. The earlier 68 came from two mistakes, both now visible:

1. it counted **64** false abstentions where there are **63** — one of §17's 64 menu abstentions sits on an open-set-gold row (`gold = UNCLASSIFIED_OPEN_SET`), and abstaining there is **correct**, so it belongs in the 451 and not in the pool;
2. it assumed **4** of the 8 label errors were rescued where the rule rescues **3** — `EVT-train-f16-r040::menu` answers `FAULT_10` on a fault-16 episode, and `FAULT_10` **is** on that episode's menu (only `FAULT_16` is removed), so `on_menu_above_threshold` names it and the rule cannot withhold.

The 2 named errors, in full:

| item | gold | argmax | diagnosis | reason |
|---|---|---|---|---|
| `EVT-train-f01-r086::menu` | `FAULT_01` | `FAULT_08` | `FAULT_08` | `on_menu_above_threshold` |
| `EVT-train-f16-r040::menu` | `UNCLASSIFIED_OPEN_SET` | `FAULT_10` | `FAULT_10` | `on_menu_above_threshold` |

The 63 false abstentions by gold fault: `FAULT_08` 24, `FAULT_10` 12, `FAULT_12` 9, `FAULT_13` 7, `FAULT_07` 5, `FAULT_15` 2, and one each of `FAULT_02`, `04`, `11`, `14`. Appendix A lists all 65 events.

**Consequence for the 3B-5 correction.** Its statement "the deployed error pool is 68 events (3.5 % of 1,919)" is **withdrawn and replaced by 65 (3.4 %)**, and its claim that 4 of the 8 label errors are correct diagnoses is replaced by **3**.

---

## c. The two threshold populations, reconciled exactly

Both are populations of the **same** 1,919 out-of-fold train events; they differ by a filter, and the filter is visible in the split's own composition:

| variant | gold kind | rows |
|---|---|---|
| menu | seen fault (`FAULT_01`…`FAULT_15`) | **1,356** |
| menu | open-set gold (`UNCLASSIFIED_OPEN_SET`, faults 16-20) | **262** |
| menu | `NORMAL_TRANSIENT` | **111** |
| menu, total | | **1,729** |
| removed | open-set gold (190 `::openset` rows) | **190** |
| total | | **1,919** |

* **§17.2's threshold population: 1,729 rows, 82 abstained (4.74 %).** The code's filter is `variant == "menu" and true_label in training_classes`, and every row's label is a training class, so the filter reduces to **all menu-present rows** — which is why it includes the 262 open-set-gold rows and the 111 `NORMAL_TRANSIENT` rows.
* **§18.3's novelty population: 1,356 rows, 67 flagged (4.94 %).** The filter is `variant == "menu" and label startswith "FAULT"`, i.e. **seen-fault menu rows only**.
* **Difference: 373 = 262 open-set-gold + 111 normal.** Not an error in either place, but they are not the same population and their rates are not comparable as if they were: §17 abstains on a set that contains 262 rows whose right answer is an abstention, and §18.3 flags a set that contains none. The novelty threshold is therefore *stricter* than it looks — 5 % of 1,356 seen-fault rows rather than 5 % of 1,729 mixed rows.

**Which is right for what:** §17.2's rule is "false abstention ≤ 5 % on seen-fault items (menu present)", and its own code's extra filters (`true_label in labels`) do nothing, so its population is too broad by 373 rows and its 4.74 % is a rate over a set that includes rows where abstaining is correct. §18.3's narrower population is the one the rule names. No number is recomputed here — §17's artefacts stay as they are — but the §17 rate should be read as "over all menu rows", not "over seen-fault rows".

---

## d. The named-stratum false-abstention denominators: 331 was used, 325 was claimed

| figure in the 3B-6 report | numerator / denominator | rate |
|---|---|---|
| "false abstention on the named-fault stratum (325) … 0.0363" | 12 / **331** | 0.0363 |
| "… → 0.0846" | 28 / **331** | 0.0846 |
| **on the 325-row stratum (the label in the table)** | 12 / **325** | **0.0369** |
| **after the channel, on 325** | 28 / **325** | **0.0862** |

The channel's before/after lists were passed to the rate function unfiltered, so the six menu-present faults-3/9 rows stayed in the denominator and made it 331. **0.0369 is the correct figure on the stratum the table names**, and it is exactly §17's number, which is the check that the 331 was the anomaly. The change the conclusion depends on is unaffected: false abstention rises from 12 to 28 abstentions of the same rows, i.e. **+4.98 percentage points on 325** (+4.83 on 331).

---

## e. Fold table: "new declines" was the total, and fold_2 is 11 → 13

Relabelled, with the arithmetic shown:

| fold | held-out menu items | §17 declines | novelty flags | of the previously **named** | **total declines after** | kept-fault accuracy |
|---|---|---|---|---|---|---|
| `fold_1` | 75 | 50 | 27 | 0 (none were named) | **50** (= 50, no change) | 0.9580 → 0.9213 |
| `fold_2` | 50 | 11 | 5 | **2** of 23 | **13** (= 11 + 2) | 0.9532 → 0.9187 |

The column is the **total** decline count after the channel is OR-ed in, not a count of newly declined items. For `fold_1` the total is unchanged at 50 even though 27 items are flagged, because those 27 were **already declined** — which is itself the finding: on that fold the channel only re-flags what the rule already withheld. `fold_2` moves 11 → 13, a gain of 2 items out of the 23 that were confidently named wrong.

---

## f. Fit score: 0.476 without the structurally decided pairs, and 15-vs-11 cannot be swapped

| aggregation | value |
|---|---|
| mean paired AUROC, all 9 pairs | 0.6508 |
| pairs excluded as structurally decided (`02`, `06`, `07` vs `08`) | 3 |
| **mean paired AUROC over the 6 informative pairs** | **0.4762** |
| `FAULT_15` vs `FAULT_11` | 0.6441 |
| `FAULT_15` vs `NORMAL_TRANSIENT` | **not computable** |

* **0.4762** is the honest summary of the score's discrimination: below chance. The 0.6508 in the 3B-6 report is carried by three pairs that are 1.000 because `FAULT_08`'s required elements are never printed, so *any* event with a positive fit beats it — a property of the KB's phrasing, not of discrimination. The 3B-6 report said this; it did not compute what remains, and what remains is worse.
* **`FAULT_15` vs `NORMAL_TRANSIENT` cannot be computed**, and not for want of data: `NORMAL_TRANSIENT` has **no KB signature** — no initiating variable, no direction, no onset — so there is no second candidate to fit. The pair that would matter most for `FAULT_15` (22 of its 25 held-out items are answered `NORMAL_TRANSIENT`) is outside the score's reach by construction.
* **The §18.4 verdict stands unchanged: not feasible**, against the 0.80 bar fixed before measuring. 0.4762 makes it firmer, and neither number is a substitute for the other — both are reported.

---

## g. The novelty rejection, restated with both bases and with the order of events

The 3B-6 report rejected the channel on one basis and did not disclose that the criterion had been implemented twice. Corrected in full:

**Basis 1 — criterion (a) on `fold_1`.** At τ = 18.7678 (4.94 % out-of-fold false-novel) the channel costs `fold_1` **3.67 points** of kept-fault accuracy (0.9580 → 0.9213) while converting **none** of its held-out declines (50 → 50). §18.6(a) rejects exactly this: a >2-point accuracy cost with no decline gained **on the same instrument**.

**Basis 2 — the validation result, which does not depend on any criterion.** On the 822 observable rows the channel changes 16 decisions and **all 16 go from correct to abstention** (0 improved, 16 worsened): accuracy 0.9781 → 0.9586, macro-F1 0.9708 → 0.9457, false abstention on the named stratum 12 → 28 of 325. A channel that is wrong on every decision it touches is rejected on measurement, not on a threshold.

**Why the second basis matters here.** On `fold_1` the conversion was **impossible by construction**: the 27 flagged items were already declined, so no flag could ever increase the decline count on that instrument — the criterion is satisfied for a structural reason rather than for a discriminating one. That makes basis 1 a weak rejection on its own, and basis 2 the substantive one. Both are now stated, and the conclusion is the same.

**Disclosure — the criterion was implemented on validation first, and the wider reading was adopted after seeing that result.** My first implementation computed §18.6(a) from the validation rows only (drop 1.95 points, under the 2-point bar), which would have returned "not rejected". Reading the criterion as written — "on an instrument" — puts the folds inside the test, and `fold_1` triggers it. **The wider reading was adopted after the narrow one had already produced its number**, and that sequence is the kind of post-hoc widening the protocol exists to prevent. It is recorded here rather than smoothed over. What it does not change: basis 2 was measured before either reading, and the channel is rejected on it either way.

**Two further disclosures, same category.** The surrogate's per-fault gain is an upper bound (`min(named, flagged)`) and was excluded from the verdict for that reason. And the `fold_1`/`fold_2` per-item decisions were **reproduced** in this round rather than stored in §17; the reproduction matches §17's aggregates exactly (`declined` 50 / 11, `named` 0 / 23), which is what makes the per-item overlap usable.

---

## Appendix A — the 65 events, one per line

| # | item | gold | argmax | diagnosis | rule reason | class |
|---|---|---|---|---|---|---|
| 1 | `EVT-train-f01-r086::menu` | `FAULT_01` | `FAULT_08` | `FAULT_08` | `on_menu_above_threshold` | named wrong |
| 2 | `EVT-train-f16-r040::menu` | `UNCLASSIFIED_OPEN_SET` | `FAULT_10` | `FAULT_10` | `on_menu_above_threshold` | named wrong |
| 3 | `EVT-train-f02-r086::menu` | `FAULT_02` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 4 | `EVT-train-f04-r120::menu` | `FAULT_04` | `FAULT_04` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 5 | `EVT-train-f07-r004::menu` | `FAULT_07` | `FAULT_07` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 6 | `EVT-train-f07-r032::menu` | `FAULT_07` | `FAULT_07` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 7 | `EVT-train-f07-r038::menu` | `FAULT_07` | `FAULT_07` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 8 | `EVT-train-f07-r066::menu` | `FAULT_07` | `FAULT_07` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 9 | `EVT-train-f07-r086::menu` | `FAULT_07` | `FAULT_07` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 10 | `EVT-train-f08-r003::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 11 | `EVT-train-f08-r010::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 12 | `EVT-train-f08-r012::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 13 | `EVT-train-f08-r013::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 14 | `EVT-train-f08-r034::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 15 | `EVT-train-f08-r039::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 16 | `EVT-train-f08-r040::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 17 | `EVT-train-f08-r041::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 18 | `EVT-train-f08-r047::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 19 | `EVT-train-f08-r053::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 20 | `EVT-train-f08-r056::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 21 | `EVT-train-f08-r061::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 22 | `EVT-train-f08-r071::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 23 | `EVT-train-f08-r073::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 24 | `EVT-train-f08-r075::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 25 | `EVT-train-f08-r083::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 26 | `EVT-train-f08-r086::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 27 | `EVT-train-f08-r091::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 28 | `EVT-train-f08-r097::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 29 | `EVT-train-f08-r098::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 30 | `EVT-train-f08-r099::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 31 | `EVT-train-f08-r100::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 32 | `EVT-train-f08-r112::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 33 | `EVT-train-f08-r120::menu` | `FAULT_08` | `FAULT_08` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 34 | `EVT-train-f10-r002::menu` | `FAULT_10` | `FAULT_10` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 35 | `EVT-train-f10-r044::menu` | `FAULT_10` | `FAULT_10` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 36 | `EVT-train-f10-r046::menu` | `FAULT_10` | `FAULT_10` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 37 | `EVT-train-f10-r050::menu` | `FAULT_10` | `FAULT_10` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 38 | `EVT-train-f10-r070::menu` | `FAULT_10` | `FAULT_10` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 39 | `EVT-train-f10-r073::menu` | `FAULT_10` | `FAULT_10` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 40 | `EVT-train-f10-r087::menu` | `FAULT_10` | `FAULT_10` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 41 | `EVT-train-f10-r091::menu` | `FAULT_10` | `UNCLASSIFIED_OPEN_SET` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 42 | `EVT-train-f10-r092::menu` | `FAULT_10` | `FAULT_10` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 43 | `EVT-train-f10-r096::menu` | `FAULT_10` | `FAULT_10` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 44 | `EVT-train-f10-r098::menu` | `FAULT_10` | `FAULT_10` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 45 | `EVT-train-f10-r108::menu` | `FAULT_10` | `FAULT_10` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 46 | `EVT-train-f11-r063::menu` | `FAULT_11` | `FAULT_11` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 47 | `EVT-train-f12-r004::menu` | `FAULT_12` | `FAULT_12` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 48 | `EVT-train-f12-r039::menu` | `FAULT_12` | `FAULT_12` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 49 | `EVT-train-f12-r043::menu` | `FAULT_12` | `FAULT_12` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 50 | `EVT-train-f12-r053::menu` | `FAULT_12` | `FAULT_12` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 51 | `EVT-train-f12-r066::menu` | `FAULT_12` | `FAULT_12` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 52 | `EVT-train-f12-r069::menu` | `FAULT_12` | `FAULT_12` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 53 | `EVT-train-f12-r088::menu` | `FAULT_12` | `FAULT_12` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 54 | `EVT-train-f12-r107::menu` | `FAULT_12` | `FAULT_12` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 55 | `EVT-train-f12-r108::menu` | `FAULT_12` | `FAULT_12` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 56 | `EVT-train-f13-r012::menu` | `FAULT_13` | `FAULT_13` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 57 | `EVT-train-f13-r037::menu` | `FAULT_13` | `FAULT_13` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 58 | `EVT-train-f13-r063::menu` | `FAULT_13` | `FAULT_13` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 59 | `EVT-train-f13-r066::menu` | `FAULT_13` | `FAULT_13` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 60 | `EVT-train-f13-r086::menu` | `FAULT_13` | `UNCLASSIFIED_OPEN_SET` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 61 | `EVT-train-f13-r101::menu` | `FAULT_13` | `FAULT_13` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 62 | `EVT-train-f13-r113::menu` | `FAULT_13` | `FAULT_13` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 63 | `EVT-train-f14-r077::menu` | `FAULT_14` | `FAULT_14` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 64 | `EVT-train-f15-r024::menu` | `FAULT_15` | `FAULT_15` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |
| 65 | `EVT-train-f15-r108::menu` | `FAULT_15` | `FAULT_15` | `UNCLASSIFIED_OPEN_SET` | `below_threshold` | false abstention |

Total: **65** events (2 named wrong + 63 false abstentions).
