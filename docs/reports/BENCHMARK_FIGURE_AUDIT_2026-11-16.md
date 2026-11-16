# Figure audit: `BENCHMARK_ACHIEVED_METRICS_REPORT.md` against the stored artefacts

**Date:** 2026-11-16
**Scope:** every quantitative claim in the older report, checked against a file in this checkout.
**Method:** each number was searched for in `outputs/` and `docs/judging/` (JSON, JSONL, CSV); where a
number was claimed as a rate, the numerator and denominator were recomputed from the row-level artefact.
A figure is **supported** only if an artefact reproduces it, **withdrawn** if nothing does, and
**contradicted** if an artefact says something else.

---

## 1. Supported — kept unchanged

| Claim | Stored source | Check |
|---|---|---|
| Numeric defect rate A = 3.65 % (19/520) | `outputs/analysis/phase3a/n4_cells.summary.json` | 19/520 = 0.03654 ✓ |
| Numeric defect rate B = 11.92 % (62/520) | same | 62/520 = 0.11923 ✓ |
| Grounding violations B = 6.35 % (33/520) | same | 33/520 = 0.06346 ✓ |
| Grounding violations A = 0 | same | ✓ |
| Negate-aware contradictions = 0 both | same | ✓ |
| Run-level FAR 2.0 % (10/500) | `outputs/evaluation/detector_v2/v2_point/A_val.json` → `far.fused.far_run = 0.02`, `n_runs = 500` | ✓ |
| Follow-up turn agreement 73.18 % vs baseline 73.74 % | `outputs/analysis/phase3a/followup/*` | ✓ |

## 2. Withdrawn — no artefact in this checkout produces these

Each was searched as a literal across `outputs/` and `docs/judging/`; the only hits are substrings inside
per-row prediction files, never an aggregate.

| Claim | Status |
|---|---|
| Adapter class accuracy 94.16 % (CI 92.34–95.59) | withdrawn |
| Hybrid class accuracy 99.64 % (CI 99.15–100.00) | withdrawn |
| Macro F1 0.9263 / 0.9963 | withdrawn |
| Abstention recall 94.22 % (424/450); precision 99.07 % / 99.34 % | withdrawn |
| False abstention 1.08 % / 0.81 %; missed abstention 5.78 % / 0.00 % | withdrawn |
| ECE 0.0009; Brier 0.0017 | withdrawn |
| χ² = 41.19, p < 0.0001 | withdrawn — no Pearson χ² exists; the primary contrast is McNemar 0/82, p < 1e-06 |
| Surrogate AUROC 0.8808 / 0.8828 / 0.8815 / 0.8456 | withdrawn — `hybrid/surrogate.json` holds per-fault LOFO counts only, no AUROC |
| D30 correctness 40/40 vs 22/40; usefulness 40/40 vs 13/40 | withdrawn |
| Mean output length 287.3 tokens | withdrawn — not recomputed from stored texts |
| Dialogue routing 545/550 and 146/150; over-refusal 77.78 → 55.56 | withdrawn |
| Latency table (§6) in full | withdrawn — no timing artefact backs any cell |

The measured replacement for the classifier block exists and is committed:
`docs/judging/confirm/p2_classifier_contrasts_2026-11-16.json` — adapter 83.30 %, hybrid 99.40 %,
LightGBM 99.40 %, MLP 96.98 %, menu rule 99.80 % on the 497 shared items.

## 3. Contradicted — an artefact says otherwise

| Claim | Artefact says |
|---|---|
| Anchor exactness A 100 % (329/329), B 51.67 % (170/329) | `n4_cells.summary.json`: anchor rows **520** for both conditions, exact **520** (A) and **188** (B) → 100 % and 36.2 %, not 51.67 %. The 329 denominator belongs to an older item subset. |
| D30 preference "A 30 wins, 8 ties, 2 adapter wins" (93.75 % of non-ties) | `docs/judging/confirm/evidence_2026-11-06/audit_v5_reveal_strict.csv` (40 rows): **A 18, B 14, tie 8** → 56.25 % of non-ties. No stored sheet yields 30/8/2. |
| Rung 2 decides for A | The signed D8 deviation records the preference instrument as failing its κ floor (0.5776 < 0.60), so rung 2 carries no confirmatory weight and all three contrast orders must be reported. |
| Rung 6 "default winner" | The final receipt records `decided_by_rung 5` with the signed `tie_break` (seconds per 520-row pass) and `measured_rungs_with_input [5]`; rung 6 was never reached. |
| Adapter inference 0.65 s per report on GPU | Measured on this machine: **2.47 s/item at batch 16** (`raw_pilot_run.json`) and **29–38 s/item at batch 1**. The claim is off by 4× at best and 58× against a single-item decode. |

## 4. Consequence

`BENCHMARK_ACHIEVED_METRICS_REPORT.md` now carries a banner and the withdrawn figures have been removed
from its tables; the measured replacements are cited inline. The authoritative report for the paper is
`docs/reports/BENCHMARK_FINAL_REPORT_2026-11-16.md`.
