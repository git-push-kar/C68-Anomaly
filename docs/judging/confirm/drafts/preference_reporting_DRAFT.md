# DRAFT — reporting the preference gate that was missed. Unsigned; no claim changes until the owner chooses.

## 1. Where the 0.60 floor came from

The floor is not a statistician's convention applied after the fact; it was a signed decision taken
before any of these 360 pairs existed.

- `docs/judging/confirm/signed.json` → `decisions.D8`:
  `"value": "120 items x 3 pairings, two raters, adjudicator, kappa >= 0.60"`, `"status": "approved"`,
  `"blocked_by": "human labour, 360 entries per rater per condition pair; cannot be automated"`.
- `signed_by`: **"project owner, session approval"**, `signed_at_local`: **2026-11-05T01:05:00+05:30**.
- The plan proposed that value (`docs/HANDOVER_BENCHMARK_PLAN_V2.md` §11, D8) and the owner signed it.
  0.60 is the conventional lower bound of "substantial agreement" (Landis & Koch), chosen for that reason
  and fixed on 2026-11-05, eleven days before the data.

So the record is unambiguous: **the owner set the threshold, in writing, before the study.** That is the
fact worth putting in the paper, because it is what makes the miss meaningful rather than embarrassing.

## 2. What was measured

`docs/judging/confirm/preference_2026-11-16_result.json`, `…_kappa.json`:

| quantity | value |
|---|---|
| pairs | 360 (120 each of C–A, C–B, M–B) |
| observed agreement | 0.7778 (280/360) |
| expected agreement | 0.4739 |
| Cohen's κ | **0.5776** |
| floor | 0.60 → **not met, by 0.0224** |
| marginals | rater 1: 189 / 161 / 10 · rater 2: 183 / 166 / 11 |

Per pairing, exploratory: A beats C for both readers (98–20 and 70–48); M beats B almost unanimously
(114–0 and 112–2); **B against C is where they part** — 108–10 for one reader, 64–53 for the other.

Standard error of κ at n = 360 is ≈ 0.03, so the point estimate sits about one SE below the floor and a
95 % interval would straddle 0.60. That is a fair sentence to include. It is not a licence to call the
floor met: the signed criterion was on the estimate, and 0.5776 < 0.60.

## 3. Three ways to report it

### Option 1 — recommended: report the miss, demote the claim

> Agreement between the two readers was κ = 0.578 (observed 0.778, expected 0.474, n = 360), below the
> pre-registered floor of 0.60 set on 2026-11-05. Because the reliability criterion was not met, the
> preference outcomes are reported as exploratory and no confirmatory preference claim is made. For
> transparency we report them: A was preferred to C by both readers (0.831 and 0.593 of decided pairs),
> M to B almost unanimously (0.981 and 0.982), while the readers disagreed on B against C (0.085 and
> 0.453). The disagreement on that contrast is what the κ measures.

This is honest, complete, and costs only the confirmatory sentence. It is also exactly how the plan
already treats the 40-pair pilot (0.560): exploratory, explicitly not confirmatory. The paper then states
Version B's claims and reports the preference tier as an instrument that did not reach reliability.

### Option 2 — deviation with disclosure, claim demoted not promoted

Legitimate only in this shape: report the floor, report the miss, give the reason (one SE, high observed
agreement, skewed marginals), and then explicitly say the preference analysis is **secondary**. The
sentence "we adopted it because it was closest" may appear only as an explanation of why the exploratory
numbers are shown at all — never as the basis for a confirmatory claim.

> The pre-registered floor was not met (κ = 0.578 vs 0.60; the gap is within one standard error at
> n = 360). We therefore report the preference comparison as a secondary, descriptive analysis and do
> not claim a preference effect. [Then the rates, then the disagreement.]

### Option 3 — a declared replication

Pre-register a new study before collecting data: new items drawn (the 120 already read are excluded),
new readers, the same or a justified floor, and a power statement for how many pairs are needed to
distinguish 0.60 from 0.55. Report **both** studies and both κ values, in that order, whatever the second
one says. A replication declared in advance is science; repetition until the floor clears is not.

## 4. Not permitted

- Lowering the floor to 0.55 (or switching to PABAK/Gwet's AC1) *after* seeing 0.5776 and calling the
  gate met. Both can be shown as labelled robustness figures only, beside the primary κ.
- Dropping the B-against-C row because the readers disagreed, and reporting the two contrasts they agree
  on. That is selective reporting of the one comparison the paper actually needs.
- Writing "κ ≈ 0.6, meeting the threshold". 0.5776 is not 0.60, and a reader who checks will find the
  sentence, not the number, was rounded.

## 5. What this does to the paper

It does not weaken the measured claims: numeric defects (62 vs 19), grounding (33 vs 0), tag-role
(57 vs 0), order invariance (0.708), the seam certification and the cost column are untouched. It removes
one sentence — "readers preferred X" — and replaces it with a reported instrument failure, which is a
contribution of its own kind: a pre-registered reliability gate that the data did not clear, disclosed
with the numbers that failed it.
