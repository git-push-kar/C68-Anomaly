"""Rewrite the withdrawn figures in BENCHMARK_ACHIEVED_METRICS_REPORT.md.

Every replacement is asserted: the script refuses to run if a stored figure it expects is not present,
so a partially applied edit cannot pass silently. See docs/reports/BENCHMARK_FIGURE_AUDIT_2026-11-16.md.
"""
import pathlib

PATH = pathlib.Path("docs/reports/BENCHMARK_ACHIEVED_METRICS_REPORT.md")

BANNER_OLD = """**Active Backbones:** `InternVL3-2B` (Adapter `internvl3_2b_pristine_rca`) · LightGBM Multiclass Classifier · Fallback v3.1 Engine · Fused LSTM-AE + CVA $Q$  

---
"""

BANNER_NEW = """**Active Backbones:** `InternVL3-2B` (Adapter `internvl3_2b_pristine_rca`) · LightGBM Multiclass Classifier · Fallback v3.1 Engine · Fused LSTM-AE + CVA $Q$  

> [!WARNING]
> **Figure audit, 2026-11-16.** Numbers previously printed in this report were not produced by any stored
> artefact and have been **withdrawn** or **replaced with measured values**; three are contradicted
> outright (anchor denominators, the D30 preference tally, adapter latency). The audit is
> [`BENCHMARK_FIGURE_AUDIT_2026-11-16.md`](BENCHMARK_FIGURE_AUDIT_2026-11-16.md). The authoritative report
> for the paper is [`BENCHMARK_FINAL_REPORT_2026-11-16.md`](BENCHMARK_FINAL_REPORT_2026-11-16.md); where the
> two differ, that one governs.

---
"""

CLASS_OLD = r"""| Metric | InternVL3-2B Adapter Alone | Hybrid (LightGBM + Menu Rule) | Statistical Significance |
|---|---|---|---|
| **Exact Diagnosis Accuracy** | 94.16% (95% CI: 92.34%–95.59%) | **99.64% (95% CI: 99.15%–100.00%)** | **+5.48 pp** ($\chi^2 = 41.19, p < 0.0001$) |
| **Macro F1-Score** | 0.9263 | **0.9963** | **+0.0700** |
| **Open-Set Abstention Recall** | 94.22% (424 / 450) | **100.0% (450 / 450)** | Captures all open-set events |
| **Abstention Precision** | 99.07% | **99.34%** | Near-zero false abstentions |
| **False Abstention Rate** | 1.08% | **0.81%** | Well within the strict 5.0% budget |
| **Missed Abstention Rate** | 5.78% | **0.00%** | Zero unhandled open-set events |
| **Expected Calibration Error (ECE)**| Uncalibrated | **0.0009** | Isotonic cross-fitted calibration |
| **Brier Score** | Uncalibrated | **0.0017** | Highly confident, well-calibrated |"""

CLASS_NEW = r"""Measured on the 497 items the two systems share
([`confirm/p2_classifier_contrasts_2026-11-16.json`](../judging/confirm/p2_classifier_contrasts_2026-11-16.json)):

| Metric | InternVL3-2B Adapter Alone | Hybrid (LightGBM + Menu Rule) | Statistical Significance |
|---|---|---|---|
| **Exact Diagnosis Accuracy** | 83.30% (95% CI: 79.77–86.32) | **99.40% (95% CI: 98.24–99.79)** | **+16.10 pp**; McNemar 0 / 82, $p < 10^{-6}$ |
| **Open-Set Abstention Recall** | 65.60% (82 / 125) | **100.0% (125 / 125)** | Captures all open-set events |
| **Abstention Precision** | 87.23% | **97.66%** | Against the hybrid's calibrated threshold |
| **False Abstention Rate** | 3.23% | **0.81%** | Both inside the 5.0% budget |
| **Missed Abstention Rate** | 34.40% | **0.00%** | Zero unhandled open-set events |

*Withdrawn 2026-11-16:* macro F1 0.9263 / 0.9963, abstention recall 424/450, precision 99.07 / 99.34,
false abstention 1.08 %, missed abstention 5.78 %, ECE 0.0009, Brier 0.0017 and chi-square 41.19. None of
these is produced by any artefact on this checkout, and an ECE of 0.0009 is not credible for a 497-item
holdout. The full five-system table (LightGBM 99.40 %, MLP 96.98 %, menu rule 99.80 %) is in the final
report, section 6."""

AUROC_OLD = r"""| Signal | Mathematical Definition | Mean AUROC | Selection Rule Outcome |
|---|---|---|---|
| **`margin`** | Top-1 probability minus Top-2 probability | **0.8808** | **Selected Signal** (Won on simplicity tie-break) |
| **`entropy`** | Shannon entropy of calibrated vector | **0.8828** | Tied within 0.005 tolerance |
| **`max_prob`** | Maximum calibrated class probability | **0.8815** | Tied within 0.005 tolerance |
| **`novelty`** | Euclidean distance to 10 nearest train events | **0.8456** | **Rejected** (degraded known-fault accuracy by 3.67 pp) |"""

AUROC_NEW = r"""**Withdrawn 2026-11-16.** The AUROC column (0.8808 margin, 0.8828 entropy, 0.8815 max_prob, 0.8456
novelty) is not produced by any artefact: `outputs/analysis/phase3a/hybrid/surrogate.json` stores per-fault
leave-one-fault-out counts, abstention rates and thresholds, but no AUROC. What the stored file does support
is the qualitative picture: on an unseen fault the dominant failure is mislabelling rather than abstention
(`FAULT_05` -> 25/25 labelled `FAULT_12`, 0 abstentions), while a fault whose signal is distinctive does
abstain (`FAULT_12` -> 23/25). A per-signal AUROC table would have to be recomputed from the prediction
files before it can appear again."""

ANCHOR_OLD = r"""| **Anchor Exactness ($\|z\| \ge 1$)** | $\ge 90.0\%$ | **100.0% (329 / 329)** | **51.67% (170 / 329)** | Adapter drops top sensor or $z$-score |"""

ANCHOR_NEW = r"""| **Anchor Exactness ($\|z\| \ge 1$)** | $\ge 90.0\%$ | **100.0% (520 / 520)** | **36.2% (188 / 520)** | Adapter drops top sensor or $z$-score |"""

D30_OLD = """1. **D30 Audit (40 Paired Items, Independent Second Rater under Strict Grounded Rules)**:
   * **Correctness**: Fallback 40/40 (100%) vs. Adapter 22/40 (55.0%)
   * **Usefulness**: Fallback 40/40 (100%) vs. Adapter 13/40 (32.5%)
   * **Preference**: **Fallback 30 wins, 8 ties, 2 adapter wins** (Fallback won **93.75% of non-ties**)."""

D30_NEW = """1. **D30 Audit (40 Paired Items, Independent Second Rater under Strict Grounded Rules)**:
   The only stored 40-row preference artefact, `evidence_2026-11-06/audit_v5_reveal_strict.csv`, tallies
   **Fallback 18, Adapter 14, 8 ties** — 56.25 % of non-ties, not the 93.75 % printed previously. The
   earlier correctness (40/40 vs 22/40) and usefulness (40/40 vs 13/40) counts are withdrawn as well: no
   scored sheet reproduces them, and the rater-agreement record says the grounded column differs from the
   first sheet in 42 of 50 score cells, i.e. a rule difference rather than reader noise.
   **This 40-item audit cannot decide the comparison in any case** — see the D29 line below. The
   confirmatory preference measurement is 360 pairs over two readers, kappa 0.5776, reported in the final
   report section 9."""

LADDER_OLD = """```
[ Rung 1: Numeric Defect Gate ] ──► Fallback: 3.65% vs Adapter: 11.92% ──► VETO ON ADAPTER
                                                                                  │
[ Rung 2: Blind Preference ]    ──► Fallback: 93.75% non-tie share    ──► FALLBACK WINS
                                                                                  │
[ Rung 5: Compute & Cost ]      ──► Fallback: < 2 ms (CPU, $0)        ──► FALLBACK WINS
                                                                                  │
[ Rung 6: Default Winner ]      ──► Condition A (Fallback v3.1) ships as the report writer!
```"""

LADDER_NEW = """```
[ Rung 1: Numeric Defect Gate ] ──► Fallback: 3.65% vs Adapter: 11.92% ──► ADAPTER VETOED
                                                                                  │
[ Rung 2: Blind Preference ]    ──► kappa 0.5776 < 0.60 floor ──► NO CONFIRMATORY WEIGHT
                                    (all three contrast orders reported descriptively)
                                                                                  │
[ Rungs 3-4 ]                    ──► not available: no eligible passer shows a tag-role claim
                                                                                  │
[ Rung 5: Compute & Cost ]      ──► seconds per 520-row pass (signed basis) ──► DECIDES
                                                                                  │
[ Rung 6: Default Winner ]      ──► never reached
```

The signed receipt records `decided_by_rung 5` with the cost tie-break, condition A, and
`measured_rungs_with_input [5]` — see `docs/judging/confirm/selection_receipt_2026-11-16_final.json`.
Condition A ships because it is the cheaper of two conditions once the adapter is vetoed, **not** because a
preference test or a default rule chose it."""

LAT_OLD = r"""| **InternVL3-2B LoRA Adapter Inference** | 4.4 GB FP16 weights (~4.8 GB VRAM) | $5.0\text{ – }10.0\text{ s}$ / report | **$0.65\text{ s}$ / report** |"""

LAT_NEW = """| **InternVL3-2B LoRA Adapter Inference** | 4.4 GB FP16 weights (~4.8 GB VRAM) | 29–38 s / report (batch 1) | **2.47 s / item (batch 16)** |"""

REPLAY_OLD = r"""| **Full 48-Hour Stream Replay (960 Samples)**| Continuous sliding window (Stride 5) | **4.8 seconds** | **1.5 seconds** |"""

REPLAY_NEW = """| **Full 48-Hour Stream Replay (960 Samples)**| Continuous sliding window (Stride 5) | unverified | unverified |

> [!NOTE]
> Only the adapter row is measured on this checkout (`raw_pilot_run.json`, 520 items, batch 16: 2.47 s per
> item; single-item decode 29–38 s). Every other latency cell in this table is withdrawn: no timing
> artefact backs it, and the 0.65 s/report figure previously printed here understated the same adapter's
> measured cost by 4x at best."""

SUBS = [
    (BANNER_OLD, BANNER_NEW),
    (CLASS_OLD, CLASS_NEW),
    (AUROC_OLD, AUROC_NEW),
    (ANCHOR_OLD, ANCHOR_NEW),
    (D30_OLD, D30_NEW),
    (LADDER_OLD, LADDER_NEW),
    (LAT_OLD, LAT_NEW),
    (REPLAY_OLD, REPLAY_NEW),
]


def main() -> int:
    text = PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    for index, (old, new) in enumerate(SUBS, 1):
        if old not in text:
            print("PATTERN MISS on edit %d: %s" % (index, old.splitlines()[0][:70]))
            return 2
        text = text.replace(old, new, 1)
    PATH.write_text(text, encoding="utf-8")
    print("applied %d edits; %d lines" % (len(SUBS), len(text.splitlines())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
