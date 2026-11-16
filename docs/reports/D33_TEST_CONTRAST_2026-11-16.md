# D33: the confirmatory pass on the test split

**Date:** 2026-11-16 (revised after the identifier fix — supersedes the earlier versions)
**Pass:** 3,168 test rows, kb order, batch 16, adapter seed 42 — 3.14 h, 3.57 s/item, no missing rows
(`outputs/benchmark_v2/eval/prose_test/run_kb16/raw_testkb_kb.jsonl`).
**Contrast:** B (InternVL3-2B adapter) against A (deterministic fallback v3.1), same rows, same instrument.

---

## 1. Two instrument corrections, in order

**Correction 1 — role attribution.** The checker attributed a number only to the detectors named in its
sentence, so a sentence naming LSTM and CVA while quoting a stored sensor z-score counted that value as
invented. Values from the non-detector sections now count as stored, and a number quoted in a role must match
that role (`z=` against a sensor z-score, `deviation` against a deviation, `lag` against the lag).

**Correction 2 — identifiers are not quantities.** The number reader still tokenised the digits inside names:
`FAULT_08` produced `08`, `Component_D_Stream9` produced `9`, `XMEAS_17` produced `17`. Every token that
starts with a letter and contains a digit is now masked before numbers are read, while numbers with a unit
suffix (`24.12x`, `221.80%`) survive.

**Effect.** Validation moves from A 19/520 and B 62/520 (original) → A 13/520 and B 50/520 (correction 1) →
**A 0/520 and B 30/520 (correction 2)**. On test, A goes 619 → **0** and B 111 → **40**. The deterministic
writer's entire numeric-defect count was name artefacts: it copies numbers from the page, so a count of zero
is what it should have been all along.

## 2. The contrast on 3,168 held-out test items

| check | A (fallback v3.1) | B (adapter) | gate | reading |
|---|---|---|---|---|
| **numeric defect** | **0.00 % (0)** | **1.26 % (40)** | ≤ 2 % | **both pass**; B is the only one that produces any |
| grounding violation | 7.89 % (250) | 10.26 % (325) | ≤ 2 % | both fail |
| negate-aware contradiction | 0.00 % (0) | 0.16 % (5) | ≤ 2 % | both pass |
| tag-role claim | 0.00 % (0) | 14.46 % (458) | ≤ 2 % | A passes, B fails |
| **anchor exact** | **100 % (3,168)** | **36.93 % (1,170)** | ≥ 90 % | A passes, B fails |

McNemar on the primary check: **B-only 40, A-only 0, exact p ≈ 2 × 10⁻¹²**. There is no discordance in A's
favour, so the test is one-sided by construction and the ratio is undefined rather than "6 : 1".

B's anchor breakdown: exact 1,170 (36.93 %), z-score omitted 1,459 (46.06 %), contradicting value 203
(6.41 %), sensor absent 336 (10.61 %).

## 3. What the contrast supports

1. **On validation A is numerically flawless and B is not** (0.00 % against 5.77 %, 30 discordant pairs, p ≈
   1.9 × 10⁻⁹). On test **B is inside the 2 % numeric gate at 1.26 % while producing 40 defects, and A
   produces none.**
2. **The claim is no longer "the adapter is more faithful on test".** Under the corrected instrument the
   adapter never beats A on numeric fidelity: it is clean on validation only relative to its own earlier,
   artefactual count. What survives is narrower and stronger: **the adapter meets the numeric gate on
   held-out data (1.26 %) but fails grounding (10.26 %), tag-role (14.46 %) and anchor exactness (36.93 %)**,
   where A fails none of those except grounding.
3. **Anchor exactness remains the decisive quality gap**: 100 % against 36.9 %, with the z-score omitted in
   46 % of adapter texts.
4. The shipped condition is unchanged: A, chosen at rung 5 on cost.

## 4. Reproduce

```
python scripts/crossfit_test_rows.py --check-val     # 834/834 stored validation probability vectors
python scripts/score_test_pass.py                    # §2 tables, writes d33/test_contrast.json
```

Artefacts: `outputs/analysis/phase3a/d33/test_contrast.json`, `run_kb16/raw_testkb_kb.jsonl`,
`hybrid/crossfit_test_rows.jsonl`.
