# 3B-8 CORRECTIONS — the shuffled-order table, the masking arithmetic, and the §9.2 label errors

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade` · **Date:** 2026-11-01
**Scope:** corrections to [3b7_prose_source_report.md](3b7_prose_source_report.md) §5 and to
[RESPONSE_SPEC.md](../RESPONSE_SPEC.md) §9.2. **No old report is edited.** Every figure here is
recomputed from rows by `scripts/reconcile_3b8.py`; artefacts in
`outputs/analysis/phase3a/fit/reconcile_3b8.json`.

Definitions were fixed **before** these numbers were computed, in PROTOCOL **§20.10**: the 325-row
observable gold-fault stratum, two labelled "named" denominators, per-table false-abstention
denominators, and the dataset-target basis for label errors.

---

## a. The shuffled-order table, recomputed on the population each row names

| order | 822 observable: accuracy | macro-F1 | abstention P / R | **325 gold-fault: accuracy** | named | correct \| named | false abstention |
|---|---|---|---|---|---|---|---|
| **KB** | 0.9416 | 0.9263 | 0.9852 / 0.9376 | **0.9508** | **318** | 0.9717 | **7 / 325 = 0.0215** |
| **shuffled-1** | 0.6594 | 0.3116 | 0.7579 / 0.9698 | **0.2123** | **171** | 0.4035 | **154 / 325 = 0.4738** |
| **shuffled-2** | 0.6642 | 0.3096 | 0.7726 / 0.9638 | **0.2338** | **184** | 0.4130 | **141 / 325 = 0.4338** |

Support of the 822: **450** abstention-class + **325** gold-fault + **47** `NORMAL_TRANSIENT`. The
classifier-plus-menu-rule row is order-invariant by construction and by test
(`tests/test_menu_order.py`), so it is not re-run here.

**The two "named" denominators, and why both exist.** The 3B-7 table mixed them in adjacent
sentences, which made 171 and 186 look like a contradiction:

| order | named on the **331** gold-fault rows | named **anywhere in the 822** | difference | correct \| named (822 basis) | correct \| named (gold-fault basis) |
|---|---|---|---|---|---|
| KB | **318** | **349** | **31** | 0.8854 (309 / 349) | 0.9717 (309 / 318) |
| shuffled-1 | **171** | **186** | **15** | 0.3710 (69 / 186) | 0.4035 (69 / 171) |
| shuffled-2 | **184** | **202** | **18** | 0.3762 (76 / 202) | 0.4130 (76 / 184) |

* *named on the gold-fault rows* answers **"how often does it commit on a fault episode"**;
* *named anywhere in the 822* answers **"of the faults it asserts, how many are right"**, and its
  denominator must include the faults asserted on rows where abstaining is correct — that is the
  difference (31 / 15 / 18 rows), and it is a precision denominator, not a coverage one.
* Both are reported from now on with their population in the row.

**Why 331 and 325 both appear.** 331 = every gold-fault row in the decode output. Six of those are
the **menu-present faults-3/9 rows**, whose evidence does not separate the fault from
`NORMAL_TRANSIENT`; they are not in the 822 support and §18.2 scores them separately. On the 325 the
3B-7 "named" counts are unchanged (318 / 171 / 184) — all six unobservable rows name a fault in every
order — so the two strata differ in denominator, not in numerator.

**False abstention: the 3B-7 column was a different population.** That table printed
0.0108 / 0.4328 / 0.3978, which the decode job's own `false_abstention_rate` supplies; its
denominator is not the 325-row stratum. Recomputed on the population the row names, the rates are
**0.0215 / 0.4738 / 0.4338**. The direction and size of the finding are unchanged — permuting the
menu turns a ~2 % false-abstention rate into ~44–47 % — and the corrected numbers are strictly
worse for the adapter, not better.

**macro-F1.** Recomputed macro-F1 over the labels present differs slightly from the decode job's
(0.9263 KB is identical; 0.3116 / 0.3096 against 0.3324 / 0.3508 shuffled) because the two
aggregations treat labels with empty support differently. Accuracy, abstention precision/recall and
every named count reproduce exactly.

---

## b. The masking counts: 34,431 − 22,917 = 11,514, and where those tokens are

Measured directly on `train_mixed.jsonl`, per row, with the model's own tokenizer
(`transformers` 4.57.6, the `venv_train` interpreter), by tokenising the answer with and without the
masked prefix.

| quantity | measure | value |
|---|---|---|
| report rows carrying the prefix | rows | **1,919** |
| dialogue rows | rows, **no** prefix | **961** (skipped: their answers carry no `fault_id`) |
| **measured removal** | tokens | **32,512** |
| §9.2's claimed removal (644,128 − 609,697) | tokens | **34,431** |
| difference | tokens | **1,919 = exactly one per report row** |

**The 1,919 is a double-counted terminator.** §9.2's two field counts are
`fault_id` **13,774** and `verdict` **9,143**; measured on their own, the *value* spans are
**11,855** and **7,224**, and 13,774 − 11,855 = 9,143 − 7,224 = **1,919**. Each field was counted
with the trailing `",` token, so the comma between the two fields entered the total twice:

```
22,917 claimed = 11,855 (fault_id value) + 7,224 (verdict value) + 2 × 1,919 (terminator)
```

**Where the remaining 11,514 goes**, from the component measurement (all on the 1,919 report rows):

| token group | tokens | note |
|---|---|---|
| opening `{` | **1,919** | one per row |
| `"fault_id"` key, with its quotes | **7,676** | 4.0 per row |
| `"verdict"` key, with its quotes | **7,676** | 4.0 per row |
| separators (`: `, `, `) | **9,595** | |
| the second terminator, already counted above | **−1,919** | |
| **BPE boundary merges** (`": "` + value, `", ` + next key) | **−13,433** | pieces exist only because the prefix and the answer are adjacent |
| **= 11,514** | | 1,919 + 7,676 + 7,676 + 9,595 − 1,919 − 13,433 |

The negative term is the point: a component tokenised alone carries its own pieces, so the
components cannot sum to the removal. The merge term is *larger* than the key term, which is why
"22,917 removed" and "34,431 removed" are both true statements about different spans rather than an
arithmetic error in either.

**3B-3's 19,531 and 7,224 are on inconsistent bases**, and that is the root of the confusion:

| basis | fault_id | verdict | sum |
|---|---|---|---|
| 3B-3 as reported | **19,531** = key + value | **7,224** = value only | 26,755 |
| §9.2 as reported | **13,774** = value + terminator | **9,143** = value + terminator | 22,917 |
| measured, value only | 11,855 | 7,224 | 19,079 |

So 19,531 is a **key-inclusive** count and 7,224 is a **value-only** count; adding them (26,755) or
subtracting them from 34,431 mixes three bases. On one basis — value plus terminator — the removal
is **22,917**, and the measured removal including keys, quotes, brace and separators, net of merges,
is **32,512**. §9.2's 34,431 is 1,919 higher than the direct measurement, by exactly the
double-counted terminator.

---

## c. The §9.2 label errors, restated on the dataset-target basis

Basis (§20.10): an error is a deployed diagnosis that differs from `target.fault_id`; an abstention
on a row whose target is a named fault is a **false abstention**; an abstention on a row whose
target is `UNCLASSIFIED_OPEN_SET` is a **correct abstention**.

| quantity | value | rate |
|---|---|---|
| out-of-fold train events | **1,919** | |
| **named a wrong fault** | **2** | 0.10 % |
| **false abstention** (abstained, target a named fault) | **63** | 3.28 % |
| **deployed error pool** | **65** | **3.39 %** |
| **correct abstentions** (target `UNCLASSIFIED_OPEN_SET`) | **451** | |

**"254 / 13.2 %" is withdrawn.** No stated definition reproduces it. The nearest population is
`63 + 190 = 253`, where the 190 are the **menu-removed** rows: their target is
`UNCLASSIFIED_OPEN_SET`, so abstaining on them is *correct* and counting them as false abstentions
is precisely the mistake the earlier correction removed. One row is unaccounted for by that
arithmetic and no fourth population of size 1 was identified; rather than invent one, the figure is
withdrawn and the basis is stated. The like-for-like number is **63 / 1,919 = 3.28 %**, or
**63 / 1,729 = 3.64 %** over the menu-present rows alone — both are reported, neither is called
"the" rate.

**Consequence for the prose-only contract.** §9.2's "13.2 % is the number to worry about" was an
argument against supplying the cross-fitted prediction. On the dataset-target basis the number is
**3.3 %**, and the argument is much weaker than it was written: supplying the cross-fitted label
would give the model an abstention where the truth is a named fault on **63 of 1,919 events**, not
254. That does not make the supply question easy — it makes it a decision that needs its own
numbers, and `docs/TRAINING_SPEC_PROSE_V1.md` lists both options without recommending one.

---

## Reproduce

```
./venv_train/Scripts/python.exe scripts/reconcile_3b8.py
```

CPU only (the tokenizer runs on CPU). No model, no generation, no test component, no teacher calls,
and no earlier report edited.
