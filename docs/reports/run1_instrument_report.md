# RUN 1 INSTRUMENT CHECK — which measurement is the instrument

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade` · **Environment:** `venv_train` (GPU), app env (CPU)
**Date:** 2026-11-01 · **Scope:** compare four readings of the same answer field on the same validation subsample, then freeze one. No training, no test component opened, run-1 adapter and checkpoints untouched.

---

## 0. Gate table

| # | Gate item | Status | Evidence |
|---|---|---|---|
| 1 | Prefix-sharing diagnosis on the disagreement items | **PASS** | §1 — first-token family mass, P(number\|FAULT), per-candidate joint |
| 2 | Four instruments × 6 checkpoints, same 300-item subsample | **PASS** | §2 — table, and the checkpoint each would select |
| 3 | Full validation, step 300 and final, instruments (a) and (c), per class, menu-present vs menu-removed | **PASS** | §3 |
| 4 | Adapter-(c) vs GBDT/MLP on the 497 intersection with McNemar | **PASS** | §4 |
| 5 | GBDT error count corrected | **PASS** | §4.3 — 3 errors, not 2 |
| 6 | Top-25 features, no identifier-derived feature | **PASS** | §4.4 — audit clean |
| 7 | Follow-up corpus decline rates + 10 examples + comparison | **PASS** | §5 |
| 8 | Frozen amendment, dated, with the rule and all instruments' numbers | **PASS** | PROTOCOL §14 |
| 9 | Tests in both environments | **PASS** | 515 + 515 (§6) |
| 10 | No test data touched | **PASS** | only `val` and follow-up `val` read |

**Headline.** Three of the four candidates are not really four: the chain-rule
decomposition (d) collapses onto (b), and both are dominated by the fact that 15 fault
labels share one first token. The two instruments that carry information are the
mean-per-token score and **greedy decoding of the field** — and decoding wins on the
same subsample (macro-F1 **0.9353** against **0.8720**) while costing **0.22 s/item
against 0.82–0.92**. Decoding also selects a different checkpoint (500, not 300).
The freeze in §6 adopts decoding, on the rule that was written before these numbers
were read.

**And the number that matters more than any of them.** On the 497 menu-variant items
where both can be asked, a 400-tree gradient-boosted model reading the same evidence
reaches **0.9940 accuracy / 0.9926 macro-F1**, against the adapter's **0.9034 /
0.9199** (McNemar: 46 discordant in the classifier's favour, 1 in the adapter's,
p ≈ 0). The instrument choice changes a couple of points; the gap between the adapter
and a tabular baseline on the same numbers is 9 points of accuracy and 46-to-1 paired.
That is a finding for the pre-test close-out, not something a decode tweak fixes (§4).

---

## 1. Prefix-sharing diagnosis

The answer field is one JSON string after a fixed prefix. Three candidates share the
first token `FAULT`, one starts `UNCLASSIFIED`, one starts `NORMAL`. That means a
scorer that compares whole strings is not comparing a 17-way decision: it is comparing
15 numbers that share a prefix against two strings that do not.

### 1.1 What the disagreement items actually look like

Ten items are in scope — the six where generation and the scorer disagreed at step 400
and the four at step 300 (they overlap; the union is ten). Numbers below are the
sweep's batch-16 predictions, with the first-token probability of the winning label
from a batch-1 probe:

| Item | Gold | mean argmax | P(first token of argmax) | Family mass FAULT / NORMAL / UNCL | P(number \| FAULT), top 3 |
|---|---|---|---|---|---|
| `EVT-val-f11-r317::menu` | FAULT_11 | UNCLASSIFIED_OPEN_SET | 0.268 | 0.020 / 0.980 / 0.000 | 11: 0.984 · 01: 0.007 · 04: 0.006 |
| `EVT-val-f12-r311::menu` | FAULT_12 | UNCLASSIFIED_OPEN_SET | 0.437 | 0.043 / 0.955 / 0.002 | 12: 0.97 · 15: 0.01 · 10: 0.01 |
| `EVT-val-f14-r310::menu` | FAULT_14 | UNCLASSIFIED_OPEN_SET | 0.107 | 0.054 / 0.944 / 0.001 | 14: 0.98 · 11: 0.01 · 09: 0.00 |
| `EVT-val-f14-r324::menu` | FAULT_14 | UNCLASSIFIED_OPEN_SET | 0.182 | 0.030 / 0.969 / 0.001 | 14: 0.99 · 15: 0.00 · 10: 0.00 |
| `EVT-val-f15-r308::menu` | FAULT_15 | NORMAL_TRANSIENT | 0.075 | 0.005 / 0.995 / 0.000 | 15: 0.98 · 12: 0.01 · 11: 0.00 |
| `EVT-val-f15-r323::openset` | UNCLASSIFIED | UNCLASSIFIED_OPEN_SET | 0.998 | 0.000 / 0.987 / 0.012 | — |
| `EVT-val-f16-r317::menu` | UNCLASSIFIED | UNCLASSIFIED_OPEN_SET | 0.779 | 0.004 / 0.996 / 0.000 | 10: 0.72 · 09: 0.11 · 16: 0.06 |
| `EVT-val-f19-r302::menu` | UNCLASSIFIED | UNCLASSIFIED_OPEN_SET | 0.983 | 0.011 / 0.980 / 0.009 | 10: 0.85 · 12: 0.08 · 04: 0.03 |
| `EVT-val-f19-r315::menu` | UNCLASSIFIED | UNCLASSIFIED_OPEN_SET | 0.962 | 0.003 / 0.995 / 0.002 | 10: 0.62 · 12: 0.10 · 15: 0.09 |

Two readings, and they disagree about what the model is doing:

* the **first token** votes `NORMAL` with 0.90–0.99 mass on eight of these nine items,
  even when the truth is a fault. The model's first decision is "probably nothing
  wrong", and the probe's argmax only matches that family on **2 of 20** item/step
  pairs (10 items × 2 checkpoints);
* the **per-token mean** still lands on `UNCLASSIFIED_OPEN_SET` for five of them,
  because it divides each path by its length and `UNCLASSIFIED_OPEN_SET` has one token
  more to average over than `FAULT_nn`.

So no, the scorer's argmax does **not** follow a unique-first-token candidate: in 18
of 20 pairs it picks a label whose first token the model had assigned 0.001–0.44
probability. The conditional numbers are unambiguous though — given `FAULT`, the model
names the right number with 0.96–0.99 probability on every fault item above. The
information is there; the per-token average is what loses it, by letting a 15-way
family pay for its own spread while a 1-way family collects a single term.

---

## 2. Four instruments, six checkpoints, one subsample

Same 300-item stratified validation subsample as the earlier selection
(`val_subsample_ids.json`), batch 16 for scoring, batch 8 for decoding, seen faults,
both menu variants. Instrument definitions:

| | Instrument | Score |
|---|---|---|
| **a** | `mean` | mean log-probability per token over the candidate's suffix |
| **b** | `joint` | summed log-probability (the string's own probability) |
| **c** | `decode` | greedy-decode the field, ≤16 tokens, stop at the closing quote; prediction is the decoded string, confidence the chain-rule sum along the path |
| **d** | `hierarchical` | family chosen by first-token mass, then the label by its within-family conditional |

### 2.1 Table

| Step | (a) mean acc / F1 | (b) joint acc / F1 | (d) hierarchical acc / F1 | (c) decode acc / F1 |
|---|---|---|---|---|
| 100 | 0.4833 / 0.2515 | 0.0567 / 0.0072 | 0.0633 / 0.0278 | 0.2367 / 0.3424 |
| 200 | 0.7633 / 0.5623 | 0.1000 / 0.1351 | 0.1267 / 0.1871 | 0.7600 / 0.6903 |
| 300 | **0.8833 / 0.8720** | 0.3833 / 0.4988 | 0.3733 / 0.4839 | 0.8733 / 0.8706 |
| 400 | 0.8533 / 0.7325 | 0.4300 / 0.4521 | 0.4300 / 0.4521 | 0.9267 / 0.8853 |
| 500 | 0.9033 / 0.8540 | 0.4100 / 0.5345 | 0.4067 / 0.5318 | **0.9367 / 0.9353** |
| 540 | 0.9033 / 0.8545 | 0.4200 / 0.5308 | 0.4200 / 0.5253 | 0.9300 / 0.9230 |

Per-class recall that separates them:

| Step | FAULT_15 recall (a / b / d / c) | NORMAL recall (a / b / d / c) |
|---|---|---|
| 100 | 0.222 / 0.000 / 0.000 / 0.556 | 0.000 / 1.000 / 1.000 / 0.000 |
| 200 | 0.000 / 0.000 / 0.000 / 0.556 | 0.941 / 1.000 / 1.000 / 0.059 |
| 300 | 0.556 / 0.000 / 0.000 / **0.889** | 1.000 / 1.000 / 1.000 / 0.706 |
| 400 | 0.000 / 0.000 / 0.000 / 0.222 | 1.000 / 1.000 / 1.000 / 0.941 |
| 500 | 0.000 / 0.000 / 0.000 / **0.889** | 1.000 / 1.000 / 1.000 / 0.941 |
| 540 | 0.000 / 0.000 / 0.000 / **0.889** | 1.000 / 1.000 / 1.000 / 0.882 |

Abstention precision / recall at the same steps (a / c):

| Step | (a) precision / recall | (c) precision / recall |
|---|---|---|
| 300 | 0.986 / 0.861 | 1.000 / 0.831 |
| 400 | 0.878 / 0.958 | 0.926 / 0.976 |
| 500 | 0.968 / 0.922 | 0.987 / 0.916 |
| 540 | 0.968 / 0.922 | 0.987 / 0.916 |

Cost: one scoring pass serves (a), (b) and (d) at **0.82–0.92 s/item**; (c) costs
**0.216–0.226 s/item** at decode batch 8, i.e. roughly a quarter, because a label is
≤6 tokens and batching amortises the prefill. Invalid decodes: **0 of 300 at every
checkpoint** — the field always parsed into a legal label.

### 2.2 Which checkpoint each instrument selects

Applying the pre-registered rule (macro-F1, accuracy as tie-break, later step second):

| Instrument | Selected step | Sub-sample macro-F1 | Accuracy |
|---|---|---|---|
| (a) mean | **300** | 0.8720 | 0.8833 |
| (b) joint | 500 | 0.5345 | 0.4100 |
| (d) hierarchical | 500 | 0.5318 | 0.4067 |
| (c) decode | **500** | 0.9353 | 0.9367 |

The instrument decides the checkpoint, which is the reason this check exists: (a)
selects 300, (c) selects 500, and between them they disagree on 4 points of sub-sample
macro-F1.

### 2.3 Why (b) and (d) are not independent instruments

The chain rule says P(family) · P(label | family) = P(label), so multiplying the two
gives (b) back. The two rows differ only because (d) *decides* in two stages (argmax
family first, then argmax within it), which is a different decision rule on the same
quantity — and even that barely moves it (0.5345 vs 0.5318, and 0.4100 vs 0.4067
accuracy). Both are broken by the same fact: 15 FAULT labels split one family's mass,
so a single-label family wins whenever the number is uncertain. That is visible in the
table — joint peaks at 0.53 macro-F1 while decode, which commits to a number, reaches
0.94.

---

## 3. Full validation, step 300, step 500 and final, instruments (a) and (c)

All 834 Protocol A validation rows: **822 observable** (14 seen faults and
`NORMAL_TRANSIENT`, 47 normal items) plus **12 unobservable** (faults 3 and 9, which
have no oracle and can only be answered by abstaining). The 300-item subsample is
1,464 rows of a different population and its numbers are not comparable with these.

### 3.1 Headline, 822 observable rows, batch 16

| Step | Instrument | Acc | mF1 | Abstain P | Abstain R |
|---|---|---|---|---|---|
| 300 | (a) mean | 0.8869 | 0.8689 | 0.9707 | 0.8822 |
| 300 | (c) decode | 0.8796 | 0.8741 | 0.9974 | 0.8489 |
| **500** | (a) mean | 0.9173 | 0.8619 | 0.9727 | 0.9489 |
| **500** | **(c) decode** | **0.9416** | **0.9263** | **0.9907** | 0.9422 |
| 540 | (a) mean | 0.9161 | 0.8651 | 0.9704 | 0.9467 |
| 540 | (c) decode | 0.9380 | 0.9204 | 0.9907 | 0.9422 |

The ordering of the subsample holds on the full split: decoding at step 500 is the
highest macro-F1 anywhere in the table (0.9263) **and** clears the abstention-precision
floor (0.9907), while the mean instrument's best full-validation macro-F1 is 0.8689.
Step 540 is 6 points of accuracy behind step 500 under the same instrument, which is
the noisiest part of this run: the last 40 steps move FAULT_15 recall from 0.68 to
0.68 but drop `NORMAL_TRANSIENT` recall from 0.872 to 0.851.

### 3.2 Per class at the two steps that matter (decode, step 500 / step 540)

| Class | n | decode@500 P / R / F1 | decode@540 P / R / F1 | mean@500 F1 |
|---|---|---|---|---|
| FAULT_01 | 25 | 0.962 / 1.000 / 0.980 | 0.962 / 1.000 / 0.980 | 0.980 |
| FAULT_02 | 25 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 |
| FAULT_04 | 25 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 |
| FAULT_05 | 25 | 0.962 / 1.000 / 0.980 | 0.926 / 1.000 / 0.962 | 0.962 |
| FAULT_06 | 25 | 1.000 / 0.960 / 0.980 | 1.000 / 0.960 / 0.980 | 0.958 |
| FAULT_07 | 25 | 0.962 / 1.000 / 0.980 | 1.000 / 1.000 / 1.000 | 1.000 |
| FAULT_08 | 25 | 0.958 / 0.920 / 0.939 | 0.958 / 0.920 / 0.939 | 0.920 |
| **FAULT_10** | 25 | **0.511 / 0.920 / 0.657** | 0.524 / 0.880 / 0.657 | 0.691 |
| FAULT_11 | 25 | 0.926 / 1.000 / 0.962 | 0.926 / 1.000 / 0.962 | 0.941 |
| FAULT_12 | 25 | 1.000 / 0.960 / 0.980 | 1.000 / 0.920 / 0.958 | 0.864 |
| FAULT_13 | 25 | 0.958 / 0.920 / 0.939 | 0.958 / 0.920 / 0.939 | 0.939 |
| FAULT_14 | 25 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 |
| **FAULT_15** | 25 | **0.607 / 0.680 / 0.641** | 0.515 / 0.680 / 0.586 | **0.000** |
| NORMAL_TRANSIENT | 47 | 0.911 / 0.872 / 0.891 | 0.909 / 0.851 / 0.879 | 0.713 |
| UNCLASSIFIED_OPEN_SET | 450 | 0.991 / 0.942 / 0.966 | 0.991 / 0.942 / 0.966 | 0.961 |

Two classes carry essentially all of the error and both are confusions with the
open set: FAULT_10 (0.511 precision — half of its predictions are wrong, mostly
abstentions it should not have made) and FAULT_15 (0.607 precision, 0.680 recall —
the class that the mean instrument **never predicts at all**, 0.000 precision and
recall at step 500 and 540). The step-300 prediction that the mean instrument
"prefers the checkpoint that keeps every class alive" was wrong about FAULT_15: at
step 300 it fires 5 times out of 25, at 500 and 540 it fires zero times.

### 3.3 Menu-present versus menu-removed

`menu` items list the true fault in the menu; `removed` items are the twin where it
was taken out, so the only correct answer is `UNCLASSIFIED_OPEN_SET`.

| Step | Instrument | menu-present n=497 acc / mF1 / abstain P | menu-removed n=325 acc / mF1 |
|---|---|---|---|
| 300 | (a) mean | 0.8330 / 0.8635 / 0.8723 | 0.9692 / 0.9844 |
| 300 | (c) decode | 0.8330 / 0.8798 / 0.9865 | 0.9508 / 0.9748 |
| **500** | (a) mean | 0.8632 / 0.8548 / 0.8947 | **1.0000 / 1.0000** |
| **500** | **(c) decode** | **0.9034 / 0.9199 / 0.9612** | **1.0000 / 1.0000** |
| 540 | (a) mean | 0.8612 / 0.8575 / 0.8860 | 1.0000 / 1.0000 |
| 540 | (c) decode | 0.8974 / 0.9140 / 0.9612 | 1.0000 / 1.0000 |

At step 500 both instruments get **every one of the 325 removed-menu twins right**,
which means the remaining 48 decode errors are one kind only: the model reads the
menu, the right fault is on it, and it still names a different label or abstains. The
sub-task the prompt advertises ("answer `UNCLASSIFIED_OPEN_SET` when the cause is not
on the list") is solved; the sub-task it does not ("pick the right one of 14") is not.

### 3.4 The 12 unobservable rows

| Step | Instrument | Acc | Abstain P / R |
|---|---|---|---|
| 300 | mean | 0.000 | 0.000 / 0.000 |
| 300 | decode | 0.333 | 1.000 / 0.667 |
| 500 | mean | 0.417 | 1.000 / 0.833 |
| 500 | decode | **0.500** | **1.000 / 1.000** |
| 540 | mean | 0.417 | 1.000 / 0.833 |
| 540 | decode | 0.500 | 1.000 / 1.000 |

Six fault-3 and six fault-9 rows. Decode at step 500 abstains on all twelve and
names the gold label on the six whose gold is `UNCLASSIFIED_OPEN_SET`; the other six
gold labels (FAULT_03, FAULT_09) are not in the candidate set by construction and
cannot be named, so 0.500 is the ceiling for this instrument and this table is a
statement about abstention, not about accuracy.

---

## 4. Adapter-(c) versus the tabular reference, on the same 497 items

The classifier from §4 of the previous report is re-compared here against the **frozen
instrument** rather than against whatever prediction file happened to be on disk. Its
`preds_{gbdt,mlp}_val503.jsonl` cover 497 of the adapter's 822 observable rows (it
never sees the 325 open-set twins, by design: they are questions about a menu it
cannot read), so the comparison is on that intersection, both sides scored by
`evaluate_v2.report`.

### 4.1 Table

| Model | n | Errors | Acc | mF1 | Accuracy CI (run-clustered) |
|---|---|---|---|---|---|
| Adapter, decode @ step 500 | 497 | 48 | 0.9034 | 0.9199 | 0.8753 – 0.9276 |
| GBDT (400 trees, gain) | 497 | **3** | **0.9940** | **0.9926** | 0.9859 – 1.0000 |
| MLP (128-64, class-weighted) | 497 | 15 | 0.9698 | 0.9697 | 0.9537 – 0.9839 |

### 4.2 McNemar, paired on item id

| Pair | Discordant | p | Method |
|---|---|---|---|
| adapter vs GBDT | b=1 (adapter only), c=46 (GBDT only) | 0.0000 | χ² continuity, n=497 |
| adapter vs MLP | b=6, c=39 | 0.000002 | χ² continuity, n=497 |

The direction is not ambiguous. **On the items where the adapter is asked a question
whose answer is in its context, a 400-tree gradient-boosted model reading the same
numbers is right 46 times where the adapter is wrong, and the adapter is right once
where the GBDT is wrong.** This is the single most important number in the run-1
close-out: the closed-set task, as currently rendered, does not require a language
model, and no amount of prompt or decode tuning closes a 46-to-1 discordance.

### 4.3 Per-class, adapter minus classifier (F1)

| Class | n | adapter | GBDT | Δ | MLP |
|---|---|---|---|---|---|
| FAULT_01 | 25 | 0.980 | 0.980 | 0.000 | 1.000 |
| FAULT_02 | 25 | 1.000 | 1.000 | 0.000 | 1.000 |
| FAULT_04 | 25 | 1.000 | 0.980 | +0.020 | 0.943 |
| FAULT_05 | 25 | 0.980 | 1.000 | −0.020 | 1.000 |
| FAULT_06 | 25 | 0.980 | 1.000 | −0.020 | 1.000 |
| FAULT_07 | 25 | 0.980 | 1.000 | −0.020 | 1.000 |
| FAULT_08 | 25 | 0.939 | 0.980 | −0.041 | 0.980 |
| **FAULT_10** | 25 | **0.657** | 1.000 | **−0.343** | 0.923 |
| FAULT_11 | 25 | 0.962 | 0.980 | −0.018 | 0.958 |
| FAULT_12 | 25 | 0.980 | 1.000 | −0.020 | 1.000 |
| FAULT_13 | 25 | 0.939 | 1.000 | −0.061 | 0.980 |
| FAULT_14 | 25 | 1.000 | 1.000 | 0.000 | 0.980 |
| **FAULT_15** | 25 | **0.641** | 0.980 | **−0.338** | 0.868 |
| NORMAL_TRANSIENT | 47 | 0.891 | 0.990 | −0.098 | 0.933 |
| UNCLASSIFIED_OPEN_SET | 125 | 0.868 | 1.000 | −0.132 | 0.980 |

The adapter matches or beats the GBDT on the easy classes and loses heavily on exactly
the three that a production system would care about: the two fault families that are
confused with the open set (10, 15), normal-vs-fault, and abstention itself.

### 4.4 The GBDT error count, corrected

`0.9940 × 497 = 494.02`, so the GBDT makes **3 errors** on the 497-item intersection,
not 2 as the run-1 report stated — `0.9940` is a rounded rate and the count is
`497 − round(0.9940 × 497)`. Over all **503** classifier items it makes 9 errors
(0.9821); the six extra are the unobservable fault-3/9 rows, which the classifier
answers from evidence and the adapter cannot be asked at all.

### 4.5 Feature audit

`feature_audit` reports **323 features, 0 offending**: no feature name contains
`item_id`, `run`, `event`, `evt`, `index`, `order`, `seed`, `split`, `hash` or
`simulation`. The 25 highest-gain features are all evidence fields — the overall
anomaly score, the two detector scores and their ratios, the slow-drift ratio, and
per-tag `z` / `deviation` / `trend` / `residual share`:

```
overall_score            8775.1    XMEAS_10:z               2289.4
XMEAS_17:residual        4364.8    cva_ratio                2153.7
cva_q                    4165.2    XMEAS_21:residual        2135.2
XMEAS_9:residual         3787.5    XMEAS_18:residual        1794.9
XMEAS_1:z                3742.7    XMEAS_38:deviation       1737.2
XMEAS_19:residual        3505.4    XMEAS_24:residual        1719.6
lstm_error               3500.8    XMEAS_1:residual         1603.5
slow_drift               3409.1    XMEAS_16:trend           1588.0
lstm_ratio               2795.9    XMEAS_20:residual        1574.8
XMEAS_22:residual        2733.9    XMEAS_11:residual        1495.8
XMEAS_26:trend           2674.8    XMEAS_38:z               1388.4
XMEAS_13:residual        2323.4    XMEAS_21:trend           1154.7
                                  XMEAS_31:deviation       1069.8
```

Nothing here identifies an item: the strongest single feature is the anomaly score,
and the rest are the variables the prompt prints. The classifier cannot be winning on
memorised identity.

---

## 5. Follow-up: the corpus already declines these questions

The 179-turn sample refused 53 % of re-asks, 73 % of `causal_why` and 40 % of
`valve_actuator`. Measured against the accepted **training** prose for the same case
types, using the project's own refusal test plus a second pattern for the
"the context does not mention …" form that the refusal filter deliberately tolerates:

| Case | Train turns | Declines (either test) | Refusal vocabulary | "Context does not mention" | Answered | Validation refusal (n) |
|---|---|---|---|---|---|---|
| causal_why | 368 | **0.486** | 0.481 | 0.410 | 0.514 | 0.733 (15) |
| mechanism | 558 | **0.622** | 0.618 | 0.539 | 0.378 | 0.467 (15) |
| valve_actuator | 1043 | **0.496** | 0.489 | 0.471 | 0.504 | 0.400 (15) |
| repeated_question | 41 | **0.220** | 0.195 | 0.171 | 0.780 | 0.529 (17) |
| hypothesis_comparison | 108 | **0.018** | 0.009 | 0.018 | 0.982 | 0.000 (15) |

The model is not inventing the refusal behaviour: it was trained on a corpus in which
half to two-thirds of these turns decline. Where it diverges is the re-ask — 22 % in
the corpus, 53 % from the adapter — and `causal_why`, where it refuses more than it was
taught (73 % vs 49 %).

Ten examples from the accepted training prose (the question in each case asks about a
valve or a mechanism, and the event context does not contain it):

1. **[valve_actuator]** "The context does not mention a purge valve or its behavior. The incident involves the Stripper_Steam_Valve [XMV_50] and related measurements, but no purge valve is referenced in the provided data."
2. **[valve_actuator]** "The context does not mention the recycle compressor valve or any indication of valve hunting. The leading diagnosis focuses on feed system issues…"
3. **[valve_actuator]** "The condenser cooling water valve is not mentioned in the incident context. The detected issue involves A_Feed_Stream1, Stripper_Temperature, Stripper_Steam_Flow…"
4. **[causal_why]** "The incident context does not mention coolant restriction or its propagation into the separator. The leading diagnosis focuses on feed system issues, specifically FAULT_01."
5. **[valve_actuator]** "The condenser valve is not mentioned in the incident context. The context describes changes in A_Feed_Stream1, Stripper_Temperature, Stripper_Steam_Flow, A_Feed_Flow, and Stripper_Steam_Valve, but…"
6. **[valve_actuator]** "The context does not mention the recycle compressor valve or any indication of valve hunting. The variables showing significant changes are A_Feed_Stream1, Stripper_Pressure…"
7. **[mechanism]** "The context does not provide information about a purge mechanism or how it affects composition. The incident involves changes in A_Feed_Stream1, Stripper_Pressure…"
8. **[valve_actuator]** "The purge valve is not mentioned in the incident context. The context describes the stripper steam valve and other variables, but there is no reference to a purge valve."
9. **[valve_actuator]** "The context does not mention the recycle compressor valve or any indication of valve hunting. The observed variables show A_Feed_Stream1 increasing, A_Feed_Flow valve position increasing…"
10. **[mechanism]** "The context does not provide information about a purge mechanism or its effect on composition. The context describes feed stream, stripper temperature, and steam flow, but no purge-related…"

The conclusion the previous check should have drawn: the refusal rate on "in-scope"
cases is mostly a property of the question generator, not of the adapter. The fix is
in the templates (ask about a valve the event actually stores) and in the corpus, not
in a prompt or a filter.

---

## 6. Freeze (PROTOCOL §14)

Recorded in [PROTOCOL.md](../PROTOCOL.md) §14 with the rule, the numbers and the date.
The rule as written before these numbers were read: *highest full-validation
macro-F1 with abstention precision ≥ 0.95; ties go to the instrument that matches
deployed generation.* Applied:

| Instrument | Best full-val mF1 | At step | Abstain P there | Eligible (P ≥ 0.95) |
|---|---|---|---|---|
| (a) mean | 0.8689 | 300 | 0.9707 | yes |
| (c) decode | **0.9263** | **500** | **0.9907** | yes |
| (b) joint | — | (not run on full val) | — | retired: collapses onto (d) |
| (d) hierarchical | — | (not run on full val) | — | retired: same reason |

*(b) and (d) were not carried to the full split because the subsample already showed
them selecting a checkpoint with 0.42 accuracy, and the freeze decision cannot turn on
them; saying so now, before the freeze, is the point of writing it down.)*

Frozen from here, no further changes without a dated amendment:

* **instrument** — greedy decoding of the first answer field (`fault_id`), ≤16 tokens,
  stop at the closing quote; probability = chain-rule sum along the decoded path;
* **candidate set** — the 17 labels of §13; `NORMAL_TRANSIENT` included;
* **reported checkpoint** — step 500 of run 1 (seeded 42, Protocol A), all metrics in
  this section and the next are step 500 unless a row says otherwise;
* (a) mean stays as a diagnostic column because four earlier tables were computed
  with it, and retiring it would make those tables unreadable.

---

## 7. Tests and artifacts

* `python -m pytest -q` — **515 passed** in the app environment; `venv_train` — **515 passed**.
* `scripts/instrument_check.py` (measurement), `scripts/instrument_report.py`
  (tables + McNemar), `scripts/classifier_vs_adapter.py` (the §4 comparison against the
  frozen instrument), `scripts/followup_decline_audit.py` (corpus decline audit).
* Artifacts under `outputs/analysis/phase3a/instrument_check/`:
  `checkpoint_table_instruments.json`, `preds_{mean,joint,hierarchical,decode}_step*.jsonl`,
  `preds_{mean,decode}_step*_full.jsonl`, `fullval_{300,500,540}.json`,
  `prefix_diagnosis.json`, `classifier_vs_adapter.json`; the summary the report quotes is
  `outputs/analysis/phase3a/instrument_report.json`.
* The classifier comparison reuses `preds_{gbdt,mlp}_val503.jsonl` from
  `outputs/analysis/phase3a/evidence_classifier/`; `classifier_vs_adapter.py` filters them
  to the 497 items the adapter can also be asked and calls `evaluate_v2.mcnemar` directly,
  so the paired test is the project's own implementation, not a re-derivation.
