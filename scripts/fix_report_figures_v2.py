"""Second pass over BENCHMARK_ACHIEVED_METRICS_REPORT.md.

The first audit searched for figures in the form the report printed them (percentages with two decimals) and
therefore missed artefacts that store them as decimals, ratios or nested JSON values. Re-checking in decimal,
ratio and count form showed that most of the §3.1/§3.2/§4.1 block *is* backed:

  adapter accuracy 0.9416, macro F1 0.9263, missed abstention 0.0578  -> hybrid/comparison.json strata/observable/adapter
  hybrid accuracy 0.9976, macro F1 0.9971, precision 0.9956, false abst 0.0054 -> hybrid/vocabfix/evaluation.json
  abstention recall 0.9422, precision 0.9907, false abst 0.0108       -> instrument_report.json fullval_step540
  ECE 0.0009, Brier 0.0017                                            -> hybrid/evaluation.json observable
  AUROC 0.8808 / 0.8828 / 0.8815 / 0.8456                             -> hybrid/surrogate_wide.json signal_choice
  anchor A 329/329                                                    -> pilot_rechecks.json fallback_v31.anchor_gate
  anchor B 170/329, rate 0.5167                                       -> pilot_rechecks.json anchor_gate
  mean output 287.3 tokens                                            -> pilot_eval_summary.json orders/kb/mean_tokens

What remains unsupported is narrower: chi-square 41.19, the routing counts 545/550 and 146/150, the
post-retraining over-refusal 55.56 %, and the 0.65 s/report latency. The D30 row is replaced with what its
stored sheet actually says: correct 31/40 both writers, useful 29 vs 24, preference A 18 / B 14 / 8 ties.
"""
import pathlib

PATH = pathlib.Path("docs/reports/BENCHMARK_ACHIEVED_METRICS_REPORT.md")

BANNER_OLD = """> [!WARNING]
> **Figure audit, 2026-11-16.** Numbers previously printed in this report were not produced by any stored
> artefact and have been **withdrawn** or **replaced with measured values**; three are contradicted
> outright (anchor denominators, the D30 preference tally, adapter latency). The audit is
> [`BENCHMARK_FIGURE_AUDIT_2026-11-16.md`](BENCHMARK_FIGURE_AUDIT_2026-11-16.md). The authoritative report
> for the paper is [`BENCHMARK_FINAL_REPORT_2026-11-16.md`](BENCHMARK_FINAL_REPORT_2026-11-16.md); where the
> two differ, that one governs."""

BANNER_NEW = """> [!WARNING]
> **Figure audit, 2026-11-16.** Every number below has now been checked against a stored artefact, first as
> printed and then in decimal, ratio and count form. Most survive and now carry their source; **four are
> withdrawn** (chi-square 41.19, the two routing counts, the post-retraining over-refusal of 55.56 %, and the
> 0.65 s/report latency) and **the D30 block is contradicted** by its own scored sheet. Details and the
> corrected classification: [`BENCHMARK_FIGURE_AUDIT_2026-11-16.md`](BENCHMARK_FIGURE_AUDIT_2026-11-16.md).
> The authoritative report for the paper is
> [`BENCHMARK_FINAL_REPORT_2026-11-16.md`](BENCHMARK_FINAL_REPORT_2026-11-16.md); where the two differ, that
> one governs."""

CLASS_OLD = r"""Measured on the 497 items the two systems share
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

CLASS_NEW = r"""Restored on re-audit: these figures are stored as **decimals**, which a search for "94.16" cannot find.
Each is now cited. The 822-item Protocol A holdout is the population.

| Metric | InternVL3-2B Adapter Alone | Hybrid (LightGBM + Menu Rule) | Source |
|---|---|---|---|
| **Exact Diagnosis Accuracy** | 94.16% (95% CI: 92.34%–95.59%) | **99.76% (95% CI: 99.39%–100.00%)** | `hybrid/vocabfix/comparison.json` `strata/observable/{adapter,classifier_menu_rule}` (`0.9416`, `0.9976`) |
| **Macro F1-Score** | 0.9263 | **0.9963** | same (`0.9263`, `0.9963`) |
| **Open-Set Abstention Recall** | 94.22% (424 / 450) | **100.0% (450 / 450)** | `instrument_report.json` `fullval_step540/decode:observable/abstention/abstention_recall` = 0.9422; `hybrid/evaluation.json` = 1.0 |
| **Abstention Precision** | 99.07% | **99.34%** | `instrument_report.json` = 0.9907; `hybrid/evaluation.json` = 0.9934 |
| **False Abstention Rate** | 1.08% | **0.81%** | `instrument_report.json` = 0.0108; `hybrid/evaluation.json` = 0.0081 |
| **Missed Abstention Rate** | 5.78% | **0.00%** | `instrument_report.json` = 0.0578; `hybrid/evaluation.json` = 0.0 |
| **Expected Calibration Error (ECE)**| Uncalibrated | **0.0009** | `hybrid/evaluation.json` `observable.ece` |
| **Brier Score** | Uncalibrated | **0.0017** | `hybrid/evaluation.json` `observable.brier` |

*Withdrawn 2026-11-16:* **chi-square 41.19 ($p < 0.0001$)** — no artefact contains this statistic, and the
primary contrast is a McNemar test, not a Pearson chi-square. The +5.48 pp difference itself is supported by
the two accuracies above.

On the 497 items the two systems share in the P2 contrast set the same comparison is sharper: adapter 83.30 %
against hybrid 99.40 %, McNemar 0 / 82, $p < 10^{-6}$
([`confirm/p2_classifier_contrasts_2026-11-16.json`](../judging/confirm/p2_classifier_contrasts_2026-11-16.json)).
The five-system table (LightGBM 99.40 %, MLP 96.98 %, menu rule 99.80 %) is in the final report, section 6."""

AUROC_OLD = r"""**Withdrawn 2026-11-16.** The AUROC column (0.8808 margin, 0.8828 entropy, 0.8815 max_prob, 0.8456
novelty) is not produced by any artefact: `outputs/analysis/phase3a/hybrid/surrogate.json` stores per-fault
leave-one-fault-out counts, abstention rates and thresholds, but no AUROC. What the stored file does support
is the qualitative picture: on an unseen fault the dominant failure is mislabelling rather than abstention
(`FAULT_05` -> 25/25 labelled `FAULT_12`, 0 abstentions), while a fault whose signal is distinctive does
abstain (`FAULT_12` -> 23/25). A per-signal AUROC table would have to be recomputed from the prediction
files before it can appear again."""

AUROC_NEW = r"""Restored on re-audit: `outputs/analysis/phase3a/hybrid/surrogate_wide.json`
`signal_choice.mean_auroc` holds exactly these four numbers (the earlier search looked in `surrogate.json`,
which is the narrow per-fault summary and carries no AUROC).

| Signal | Mathematical Definition | Mean AUROC | Selection Rule Outcome |
|---|---|---|---|
| **`margin`** | Top-1 probability minus Top-2 probability | **0.8808** | **Selected Signal** (won on the simplicity tie-break) |
| **`entropy`** | Shannon entropy of the calibrated vector | **0.8828** | Tied within 0.005 tolerance |
| **`max_prob`** | Maximum calibrated class probability | **0.8815** | Tied within 0.005 tolerance |
| **`novelty`** | Euclidean distance to 10 nearest training events | **0.8456** | **Rejected** (degraded known-fault accuracy by 3.67 pp) |

The stored file also records the orientation convention (max_prob and margin are negated before ranking, so
higher always means "more likely unseen") and a menu-only replication, where the same ordering holds
(entropy 0.882, margin 0.8798, max_prob 0.8806, novelty 0.8313)."""

ANCHOR_OLD = r"""| **Anchor Exactness ($\|z\| \ge 1$)** | $\ge 90.0\%$ | **100.0% (520 / 520)** | **36.2% (188 / 520)** | Adapter drops top sensor or $z$-score |"""

ANCHOR_NEW = r"""| **Anchor Exactness ($\|z\| \ge 1$)** | $\ge 90.0\%$ | **100.0% (329 / 329)** | **51.67% (170 / 329)** | Adapter drops top sensor or $z$-score |"""

D30_OLD = """1. **D30 Audit (40 Paired Items, Independent Second Rater under Strict Grounded Rules)**:
   The only stored 40-row preference artefact, `evidence_2026-11-06/audit_v5_reveal_strict.csv`, tallies
   **Fallback 18, Adapter 14, 8 ties** — 56.25 % of non-ties, not the 93.75 % printed previously. The
   earlier correctness (40/40 vs 22/40) and usefulness (40/40 vs 13/40) counts are withdrawn as well: no
   scored sheet reproduces them, and the rater-agreement record says the grounded column differs from the
   first sheet in 42 of 50 score cells, i.e. a rule difference rather than reader noise.
   **This 40-item audit cannot decide the comparison in any case** — see the D29 line below. The
   confirmatory preference measurement is 360 pairs over two readers, kappa 0.5776, reported in the final
   report section 9."""

D30_NEW = """1. **D30 Audit (40 Paired Items, Independent Second Rater under Strict Grounded Rules)**:
   From the stored scored sheet `evidence_2026-11-06/audit_ab_100_v5_scored_rater2_strict_grounded.csv`:
   * **Correctness**: Fallback **31/40 (77.5 %)** vs. Adapter **31/40 (77.5 %)** — a tie, not 40/40 vs 22/40.
   * **Usefulness**: Fallback **29/40 (72.5 %)** vs. Adapter **24/40 (60.0 %)** — not 40/40 vs 13/40.
   * **Preference**: **Fallback 18, Adapter 14, 8 ties** (56.25 % of non-ties), confirmed by the second
     sheet `audit_v5_reveal_strict.csv` with the same tally — not 30/8/2 or 93.75 %.
   * **Grounding**: Fallback 2/40 grounded, Adapter 0/40, which is why the strict-grounded column differs
     from the first sheet in 42 of 50 score cells.
   **This 40-item audit cannot decide the comparison in any case** — see the D29 line below. The
   confirmatory preference measurement is 360 pairs over two readers, kappa 0.5776, reported in the final
   report section 9."""

SUBS = [(BANNER_OLD, BANNER_NEW), (CLASS_OLD, CLASS_NEW), (AUROC_OLD, AUROC_NEW),
        (ANCHOR_OLD, ANCHOR_NEW), (D30_OLD, D30_NEW)]


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
