# P2 — the merged classifier row, 2026-11-16

**Generation: pre-vocabulary-fix** (adapter 0.8330 = 414/497). The retrained table is
`outputs/analysis/phase3a/hybrid/vocabfix/classifier_contrasts.json`: menu rule 1.0000, hybrid 0.9960,
LightGBM 0.9980, MLP 0.9799, **adapter 0.7646 (0.7253–0.7998)**, hybrid vs adapter 2 vs 117.

Three stored systems, compared on the **497 items all three scored**:

| system | where it comes from |
|---|---|
| menu rule | `classifier_label` in `hybrid/preds_protocolA_val.jsonl` — the menu classifier's label before any rule overrode it |
| hybrid | `pred` in the same file — after the threshold, menu and no-fault rules |
| adapter | `pred` in `evidence_classifier/preds_adapter_val503.jsonl` — the 2B model's own decoded label |

Abstention is the label `UNCLASSIFIED_OPEN_SET` on either side. A system abstains when it predicts
it; an abstention is correct when the gold label is the same. Reproduce with:

    python scripts/classifier_contrasts.py

## Result

| system | root-cause accuracy (95 % CI) | abstains / gold abstains | precision | recall | false-abstention | 5 % gate |
|---|---|---|---|---|---|---|
| menu rule | 0.9980 (0.9887–0.9996) | 125 / 125 | 1.0000 | 1.0000 | 0.0000 | pass |
| hybrid | 0.9940 (0.9824–0.9979) | 128 / 125 | 0.9766 | 1.0000 | 0.0081 | pass |
| adapter (pre-vocabulary-fix) | 0.8330 (0.7977–0.8632) | 94 / 125 | 0.8723 | 0.6560 | 0.0323 | pass |

Paired exact McNemar on correctness, same 497 items:

| contrast | first wrong, second right | first right, second wrong | exact p |
|---|---|---|---|
| menu rule vs hybrid | 0 | 2 | 0.5 |
| menu rule vs adapter | 0 | 82 | < 1e-06 |
| hybrid vs adapter | 1 | 81 | < 1e-06 |

## What the row says

The menu rule and the hybrid are statistically indistinguishable on these items — two discordant
pairs, p = 0.5 — so the rules buy nothing over the menu label on the subset where the menu is
present, and the hybrid's slight deficit is inside noise. The adapter is decisively worse: 82 items
it gets wrong that the menu rule gets right, against none the other way, and its abstention recall is
0.656 against 1.000 — it names a fault on 31 of the 125 items whose gold label is "not on the menu".

Every system passes the 5 % false-abstention budget, including the adapter at 3.23 %. That budget
alone therefore does not separate them; the separation is in accuracy and abstention recall, which is
why reporting only the budget would flatter the adapter.

## The merged row, and what is genuinely missing

The plan asks for root cause, subsystem and severity as one merged row. Only root cause is stored.
Subsystem and severity are knowledge-base functions of the predicted fault id in this checkout — no
system emits them independently — so a separate accuracy for either would restate the root-cause
number rather than measure a new quantity. The merged row is therefore:

> **classifier = menu rule** (accuracy 0.9980, abstention precision 1.0000 / recall 1.0000, false
> abstention 0.0000) · **subsystem = not measured** · **severity = not measured**

If a subsystem or severity claim is wanted, a system that emits those outputs has to be built and
scored; it cannot be extracted from these three prediction files.

## Limits

1. The 497-item intersection is the menu-present subset. The hybrid scored 834 items; the adapter
   scored 497. Open-set-only variants are outside this comparison, so no claim here covers them.
2. The abstention precision and recall thresholds are not signed. Only the 5 % false-abstention
   budget has a declared value, and the "gate" column reflects that budget alone.
3. Cross-check against the stored evaluations: adapter 0.833 over 497 items matches
   `evidence_classifier/summary.json` (`adapter` and `adapter_common`), and hybrid 0.994 matches
   `hybrid/evaluation.json:menu_present`. The pairing of labels is therefore the same one those
   artefacts used.
