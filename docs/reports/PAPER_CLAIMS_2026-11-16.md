# Paper claims — measured, bounded, and sourced

**Date:** 2026-11-16
**Rule for this document:** a claim appears here only if an artefact in this repository produces it. Every
claim names the split it holds on. Anything not measured is listed in §3 and is not claimed in the paper.

---

## 1. Claims the paper can make

**C1 — The deterministic writer carries no numeric-defect liability; the 2B adapter carries a small one
that meets the held-out budget.** On the 520-item validation cell table A and C score **0/520** numeric
defects while B and M score 30/520 (5.77 %); on 3,168 held-out test items A scores **0/3,168** and B
**40/3,168 (1.26 %)**, inside the 2 % gate (McNemar 40 vs 0, p ≈ 2 × 10⁻¹²). The adapter's failures are
structural rather than numeric: grounding 10.26 %, tag-role 14.46 %, anchor exactness 36.93 %. Sources:
`n4_cells.json`, `d33/test_contrast.json`.

**C2 — Anchor exactness, not defect counts, is where the language model fails structurally.**
The adapter reproduces the top anomalous sensor's z-score in 188/520 validation texts (36.2 %) and
1,170/3,168 test texts (36.93 %), against 100 % for the deterministic writer, and omits the z-score entirely
in 46.06 % of test texts. The 90 % gate is failed on both splits. Sources: `pilot_rechecks.json`
`anchor_gate`, `d33/test_contrast.json` `anchor_status`.

**C3 — The calibrated hybrid classifier is statistically indistinguishable from a plain gradient-boosted
model on the same features.** 0.9940 against 0.9940 on 497 shared items, 2 vs 2 discordant, p = 1. The
knowledge-base rules earn their place in *explanation*, not in accuracy. Adapter alone 0.8330, MLP 0.9698,
menu rule 0.9980. Source: `confirm/p2_classifier_contrasts_2026-11-16.json`.

**C4 — Pairwise preference cannot carry a claim.** 360 pairs, two readers, κ = 0.5776 (observed 0.7778,
expected 0.4739), below the declared 0.60 floor by 0.0224. All three contrast orders are reported
descriptively: A over C (98/20 and 70/48), B over C (10/108 and 53/64), M over B (0/114 and 2/112). Signed
deviation: `confirm/d8_deviation_2026-11-16.json`.

**C5 — The selection is decided on cost, not quality.** The adapter is vetoed at rung 1 (numeric gate);
rungs 2–4 are unavailable or uninformative (preference below its floor; no eligible passer carries a
mechanism claim); rung 5 compares seconds per 520-row pass on a signed basis and condition A ships. Receipt:
`confirm/selection_receipt_2026-11-16_final.json`, `decided_by_rung 5`, `measured_rungs_with_input [5]`.
No dollar figure is claimed.

**C6 — The fallback wrapper stays inside its budget, and the paper's D wording awaits re-signoff.**
Condition D substitutes the fallback report when the adapter is flagged; on the corrected instrument it falls
back on **114/520 rows (21.9 %)**, inside the pre-registered 25 % ceiling. The signed
`d_ceiling_decision_2026-11-16.json` records a 27.3 % breach measured before two checker corrections, so D is
**not** claimed as a pass here: the owner must re-sign the decision, and the conservative "A with extra
steps" framing remains available because D's own contribution is zero on the rows it hands back. Source:
`n4_cells.json` seam certification.

**C7 — Generation is not batch-invariant, so batch size is a measurement parameter.**
Same prompts at batch 16 and batch 8: 1 of 12 texts identical; batch 1 against padding-free grouped batch 8:
3 of 8 identical. All reported numbers come from one batch size per experiment. Evidence:
`outputs/analysis/phase3a/fit/eq_B1`, `eq_B8`, `prose_pilot/pb16`, `pb8`.

**C8 — The system's diagnosis channel is deterministic by construction.** Diagnosis, verdict, severity and
confidence are computed by code and calibrated classifiers; the model writes prose only. The paper's
architectural claim rests on that separation, not on model accuracy.

## 2. Numbers the paper may quote (single source of truth)

| quantity | value | source |
|---|---|---|
| cell-table numeric defects (identifier-masked) | A 0/520, B 30/520, C 0/520, M 30/520, D 0/520 | `n4_cells.json` summary |
| validation numeric defects (final instrument) | A 0/520 (0.00 %), B 30/520 (5.77 %) | `n4_cells.json` |
| test numeric defects | A 0/3,168 (0.00 %), B 40/3,168 (1.26 %) | `d33/test_contrast.json` |
| test grounding / contradiction / tag-role | A 7.89 / 0 / 0 %, B 10.26 / 0.16 / 14.46 % | same |
| anchors exact | validation A 520/520, B 188/520; test A 3,168/3,168, B 1,170/3,168 | same |
| primary validation contrast | B vs A McNemar 30/0, exact p ≈ 1.9 × 10⁻⁹; order invariance 0.7077 | `n4_cells`, overnight receipt |
| classifier contrast | hybrid 0.9940 = LightGBM 0.9940, 2/2, p = 1 | `p2_classifier_contrasts` |
| preference | κ 0.5776 over 360 pairs | `preference_2026-11-16_result.json` |
| cost basis | seconds per 520-row pass, tie-break signed | `cost_basis_seconds_2026-11-16.json` |

## 3. Not claimed (and why)

1. **Subsystem and severity accuracy** — not measured; would duplicate the root-cause number.
2. **A numerical advantage for the adapter** — on test it meets the gate (1.26 %) but the deterministic
   writer has no numeric defects at all (0/3,168); the adapter's claim is budget compliance, not superiority.
3. **A preference ordering** — instrument below its reliability floor (C4).
4. **Mechanism quality (rung 3)** — no eligible passer carries a mechanism claim among passers; the
   feasible comparison (C 12/19 vs B 14/26) overlaps.
5. **Order invariance on the test split** — only the kb order was run (shuffled-1/2 unmeasured, ≈6 h each).
6. **E (seed-robustness ablation)** — adapters trained (`seed43`, `seed44`, 288 steps, 3 epochs) but
   unscored; the 2026-11-05 seed-43/42 quarantine incident is not yet cleared.
7. **Detector-frontier table (table 7)** — the P1 arm table was never built.
8. **Follow-up condition T** — deferred; U vs F only.
9. **Parity between scoring paths** — 1,390 disagreements unresolved; every number above is reported as
   produced by the path named beside it.
10. **Line-level latency for any component except the generator** — no timing artefact exists.

## 4. Honest framing for the abstract

> We evaluate a 2B parameter language model as a report writer for an industrial anomaly-detection system,
> with diagnosis kept deterministic. On 3,168 held-out events the model meets the numeric-fidelity budget
> (1.3 % defective reports against 2 %) while the deterministic writer produces none at all, so no numerical
> advantage is claimed for the model. Its failures are elsewhere: it omits or misstates the top sensor's
> z-score in 63 % of reports, cites an unsupported tag role in 14 %, and violates grounding in 10 %.
> Pairwise preference between writers fails its reliability floor (κ 0.58), and the shipped configuration is
> chosen on cost, not quality. A gradient-boosted classifier matches our calibrated hybrid exactly (0.9940 vs
> 0.9940, p = 1), so the knowledge-base contribution is explanatory rather than predictive.

## 5. Remaining work, and what each item can change in the claims

Nothing below blocks submission of the claims as written, but each one bounds how strongly a claim can be
phrased. "Impact" names the claim that would move if the task came back badly.

| task | what it settles | impact if it comes back badly | cost |
|---|---|---|---|
| **Test order runs (shuffled-1/2)** | whether B's 1.26 % numeric rate is order-stable, or an artefact of the one menu order measured | **C1**: today C1's test half rests on the kb order alone. If B's rate swings with order, the 1.26 % figure must be quoted as "on the kb order" | ≈6.3 h GPU |
| **E scoring (seeds 43/44 on 520 rows)** | whether the adapter's gate compliance is seed-stable | **C1, C3**: a seed that behaves like seed 42 keeps C1; a seed above the 2 % gate turns C1 from "meets the budget" into "meets it at seed 42" | ≈0.9 h GPU per seed plus scoring |
| **Seed-43/44 integrity check** | whether those adapters are genuinely distinct runs after the 2026-11-05 checkpoint incident | **E claims only**: until cleared, E cannot be reported at all. Clearing it changes no existing claim | minutes |
| **Detector-frontier table (7)** | how the prose layer compares with detector-only and classifier-only front-ends | **none of C1–C8**: it is a new table. Its absence costs a comparison, not a claim | a build, plus P1 arm table |
| **Follow-up condition T** | whether store-backed answers beat router-only answers on follow-ups | **no claim**: dialogue is not claimed in the paper. T would only fill the report's §5 | ≈2 h |
| **Parity (1,390 disagreements)** | whether every number was produced by one instrument | **C1–C6 provenance**: unresolved, each figure must keep the scoring path named beside it. Resolving it either confirms the numbers or replaces them with one path's | instrument work, unknown |
| **Release packaging (D10)** | whether a reader can rebuild the artefacts | **no numeric claim**: affects reproducibility, not values | ≈1 h |
| **Version A/B signature** | which framing the paper uses (with or without the preference and mechanism sections) | **C4, C5**: Version A reports them descriptively, Version B omits them. Neither changes C1–C3 | owner decision |

**Identifier over-count, now fixed.** An earlier version of this document said every numeric rate was an
upper bound because names such as `FAULT_08` yielded digit tokens. That is fixed: identifiers are masked
before numbers are read, the deterministic writers score zero, and the adapter's rates are 5.77 % on
validation and 1.26 % on test. These numbers are final for this instrument.
