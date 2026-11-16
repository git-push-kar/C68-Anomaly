# D33: the confirmatory pass on the test split

**Date:** 2026-11-16
**Pass:** 3,168 test rows, kb order, batch 16, adapter seed 42 — 3.14 h, 3.57 s/item, no missing rows
(`outputs/benchmark_v2/eval/prose_test/run_kb16/raw_testkb_kb.jsonl`).
**Contrast:** B (InternVL3-2B adapter) against A (deterministic fallback v3.1), same rows, same instrument.

---

## 1. Instrument revision, declared in full

The first scoring of this pass reported A at 41.48 % numeric defects against 3.65 % on validation and was
correctly rejected as an artefact. The cause is now fixed in `scripts/pilot_rechecks.numeric_flags`:

> a number was attributed **only** to detectors named in its sentence. A sentence naming LSTM and CVA while
> also quoting a sensor z-score (17.99), a lag (4) or a slow-drift ratio (0.52) had those *stored* values
> counted as invented. On validation the sentences are short and rarely mix detectors with sensors, so the
> effect was small; on the test prompts (3,270 tokens against 1,030) the writer's first sentence covers both,
> and the effect became 41 %.

The fix allows values from the non-detector evidence sections (sensor z and deviation, lag and ratio, onset
minutes, residual shares) to satisfy the "stored" test, while detector numbers are still checked against the
detector that is named, and wrong-inequality-direction checks are untouched.

**The requested check could not be preserved, and that is itself a finding.** A narrower fix that kept
validation at 19/520 and 62/520 is not possible, because validation was affected by the same defect, just
less: with the corrected instrument the validation numbers become **A 13/520 (2.50 %) and B 50/520
(9.62 %)**, against 19/520 (3.65 %) and 62/520 (11.92 %) as stored. Six A items and twelve B items were
false positives — sentences that quoted real sensor values. A "check" that reproduces the old numbers would
have reproduced the bug.

What does **not** change: both conditions still exceed the 2 % numeric gate on validation, so the rung-1 veto
direction is unchanged; anchor, grounding, contradiction and tag-role checks are untouched (their counts
reproduce exactly); the selection receipt's `decided_by_rung 5` stands.

## 2. The contrast on 3,168 held-out test items

| check | A (fallback v3.1) | B (adapter) | gate | reading |
|---|---|---|---|---|
| **numeric defect** | **19.54 % (619)** | **3.50 % (111)** | ≤ 2 % | **B better by 16.0 pp**; both fail the gate |
| grounding violation | 7.89 % (250) | 10.26 % (325) | ≤ 2 % | A better; both fail |
| negate-aware contradiction | 0.00 % (0) | 0.16 % (5) | ≤ 2 % | both pass |
| tag-role claim | 0.00 % (0) | 14.46 % (458) | ≤ 2 % | A better; B fails |
| **anchor exact** | **100 % (3,168)** | **36.93 % (1,170)** | ≥ 90 % | A passes, B fails |

McNemar on the primary, numeric defect: **B-only 97, A-only 605, p ≈ 0** (two-sided exact, 97 + 605
discordant). Test items where A invents a number and B does not outnumber the reverse 6.2 : 1.

B's anchor breakdown: exact 1,170 (36.93 %), z-score omitted 1,459 (46.06 %), contradicting value 203
(6.41 %), sensor absent 336 (10.61 %).

## 3. What the contrast supports

1. **The adapter is more numerically faithful than the deterministic writer on held-out items** — a reversal
   of the validation ordering, where A was the more faithful of the two (2.50 % vs 9.62 %). Any claim must
   name the split; the paper reports both.
2. **The reversal does not change the selection.** A still fails fewer *other* gates (grounding 7.9 % vs
   10.3 %, tag-role 0 % vs 14.5 %, anchor 100 % vs 36.9 %), and both fail the numeric gate. The shipped
   condition remains A, decided at rung 5 on cost.
3. **Anchor exactness is the decisive quality gap**: 100 % against 36.9 % on 3,168 items with the adapter
   dropping the top sensor or its z-score in 46 % of texts.
4. **No claim is made that either writer passes the numeric gate.** On test B is 3.50 % and A is 19.54 %
   against a 2 % bar.

## 4. Reproduce

```
python scripts/crossfit_test_rows.py --check-val     # 834/834 stored validation probability vectors
python scripts/score_test_pass.py                    # §2 tables, writes d33/test_contrast.json
```

Artefacts: `outputs/analysis/phase3a/d33/test_contrast.json`, `run_kb16/raw_testkb_kb.jsonl`,
`hybrid/crossfit_test_rows.jsonl`.
