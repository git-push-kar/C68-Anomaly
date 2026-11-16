# Figure audit: `BENCHMARK_ACHIEVED_METRICS_REPORT.md` against the stored artefacts

**Date:** 2026-11-16 (second pass — supersedes the first pass of the same day)
**Method:** every quantitative claim was searched for in the form printed, then in **decimal** (0.9416 for
94.16 %), **ratio** (424/450 = 0.9422) and **count** form, across `outputs/**/*.json` and
`docs/judging/**/*.csv`, with the JSON path recorded so a hit can be judged in context rather than trusted.

## 1. Correction to the first pass

The first audit reported twelve figures as "withdrawn" on the strength of a literal search for the printed
strings. That was wrong for most of them. The numbers were stored as decimals inside nested JSON, so
`grep 94.16` found nothing while `0.9416` sat at
`outputs/analysis/phase3a/hybrid/comparison.json: strata/observable/adapter/accuracy`. Nine of the twelve
are **supported** and have been restored to the report with their sources. This is exactly the failure mode
the re-check was intended to catch.

## 2. Supported (restored, now cited)

| Claim | Stored source (JSON path) |
|---|---|
| Adapter accuracy 94.16 %, macro F1 0.9263 | `hybrid/comparison.json` → `strata/observable/adapter/{accuracy,macro_f1}` |
| Hybrid accuracy 99.64 %, macro F1 0.9963 | `hybrid/evaluation.json` → `observable/{accuracy,macro_f1}` (also `hybrid/comparison.json` → `classifier_menu_rule`) |
| Abstention recall 94.22 % (424/450) | `instrument_report.json` → `fullval_step540/decode:observable/abstention/abstention_recall` = 0.9422 |
| Abstention precision 99.07 % / 99.34 % | `instrument_report.json` = 0.9907; `hybrid/evaluation.json` = 0.9934 |
| False abstention 1.08 % / 0.81 % | `instrument_report.json` = 0.0108; `hybrid/evaluation.json` = 0.0081 |
| Missed abstention 5.78 % / 0.00 % | `instrument_report.json` = 0.0578; `hybrid/evaluation.json` = 0.0 |
| **ECE 0.0009, Brier 0.0017** | `hybrid/evaluation.json` → `observable.ece`, `observable.brier` |
| **AUROC 0.8808 / 0.8828 / 0.8815 / 0.8456** | `hybrid/surrogate_wide.json` → `signal_choice.mean_auroc` (the first pass looked in `surrogate.json`, which has no AUROC) |
| **Anchor exactness A 329/329, B 170/329 (51.67 %)** | `pilot_rechecks.json` → `fallback_v31.anchor_gate` {rows 329, exact 329}; `anchor_gate` {rows 329, exact 170, rate 0.5167} |
| Mean output 287.3 tokens (1,086 chars) | `pilot_eval_summary.json` → `orders/kb/mean_tokens`, `mean_prose_chars` |
| Numeric defect 3.65 % / 11.92 %; grounding 0 / 6.35 % | `n4_cells.summary.json` |
| FAR 2.0 % (10/500) | `evaluation/detector_v2/v2_point/A_val.json` → `far.fused.far_run = 0.02` |
| Follow-up agreement 73.18 % vs 73.74 % | `followup_rescore.json`, `farm_2026-11-06/*` |
| Over-refusal 77.78 % before retraining | `followup_rescore.json` → `comparison/causal_why/over_refusal_rate_router` |

Two scope notes, not errors: the anchor denominator **329** is the anchor-bearing stratum (top |z| ≥ 1),
which is why `n4_cells` reports 520 anchor rows with 188 exact — a wider population measured by a later
scorer. Both are artefact-backed; the paper must say which population it quotes. Likewise numeric defects are
62/520 in `n4_cells` and 69/520 in the older `pilot_eval_summary`; the report quotes the former.

## 3. Withdrawn — still no artefact in any form

| Claim | Why it goes |
|---|---|
| **χ² = 41.19, p < 0.0001** | No artefact stores this statistic. The only near-matches are sensor readings (`current_value` 41.19) in event files. The primary contrast is a McNemar test, not a Pearson χ². |
| Routing **545/550** | No artefact; the nearest values are `pr_auc` and `deviation_percent` in detector outputs. |
| Routing **146/150** | Same — coincidence with a `deviation_percent`. |
| Over-refusal after retraining **55.56 %** | `followup_rescore.json` gives `adapter_refusal_rate = 0.7333` for `causal_why`, not 0.5556. |
| Adapter latency **0.65 s/report** | Nothing times the adapter at that rate. Measured: 2.47 s/item at batch 16 (`raw_pilot_run.json`), 29–38 s at batch 1. |

## 4. Contradicted — the artefact says something else

| Claim | The artefact |
|---|---|
| D30 correctness 40/40 vs 22/40; usefulness 40/40 vs 13/40; preference 30/8/2 (93.75 %) | `evidence_2026-11-06/audit_ab_100_v5_scored_rater2_strict_grounded.csv` (40 rows): correct **31 vs 31**, useful **29 vs 24**, preference **A 18 / B 14 / tie 8**. The second sheet `audit_v5_reveal_strict.csv` gives the same preference tally. |
| Rung 2 decided for A | A preference instrument at κ 0.5776 < 0.60 carries no confirmatory weight (signed D8 deviation); all three contrast orders are reported descriptively. |
| Rung 6 "default winner" | The signed receipt records `decided_by_rung 5` with the cost tie-break and `measured_rungs_with_input [5]`; rung 6 was never reached. |
| Adapter 0.65 s/report | Contradicted by the two measured rates above. |

## 5. Limits of this audit

The scan covers JSON and CSV artefacts under `outputs/` and `docs/judging/`. It does not re-derive the
detector table (FDR, delay per fault) beyond the fused FAR, and it does not treat a value that appears only
inside an unrelated per-item record as support — §3 lists the cases where a match was coincidental. Where an
artefact and a later scorer disagree (§2 scope notes), the report states both rather than choosing silently.
