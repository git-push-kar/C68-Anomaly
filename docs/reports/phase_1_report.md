# Phase 1 report — detector evaluation and baselines

**Repo:** `C:\Users\Admin\Desktop\anomaly` · branch `benchmark-upgrade` · CPU only,
no LLM training, no GPU job. Phase 0 patch approved as the precondition.

---

## 1. Summary

1. The frozen detector (LSTM AE + CVA/DPCA, OR-fused) **reproduces its published
   per-fault numbers**: per-fault `mean_max` within **2.0e-4** (relative 6e-6),
   event rate within 1–4 runs of 500, delay within 0.13 windows, over faults
   1/3/4/9/15 × 500 runs.
2. Protocol A is evaluated end to end: **2,000 runs per component** (20 faults ×
   100), **254 s per component on CPU**, with per-fault FDR, delay, AUROC, PR-AUC
   for LSTM / CVA / DPCA / fused and PCA / Isolation Forest baselines, plus a
   **2,000-row per-run alarm table** per component.
3. Validation and test agree tightly: max fused-FDR difference **0.55 points**
   across the 20 faults, mean 0.13 points. Test was not used for any decision.
4. **The observable set at the frozen thresholds is the 18 faults other than
   {3, 9}** (fused FDR 2.4% / 2.5% on validation; 1.8% / 2.0% on test).
5. Two findings that need an owner decision before Phase 2A generates events:
   * the fused detector raises an event on **30.6% of fault-free runs**;
   * the dynamic detectors' thresholds were fitted per *sample*, so at window
     level they are near-inert and the OR-gate is effectively the LSTM alone
     (fault 15: 7.7% → 97.2% fused FDR once every detector sits at a common 1%
     window FAR).
6. Baselines are strong: mean fused FDR over the 18 observable faults is 92.9%,
   but CVA Q alone reaches 92.5% and PCA SPE 72.3%; the fused gain is largest on
   faults 4/5/10/16/19/20, where PCA T² collapses to 0.1–24%.

---

## 2. Changes (commit hashes)

| Commit | Change |
|---|---|
| `7161d5b` | `scripts/evaluate_detector_v2.py` (streamed, split-aware, per-run table) and `configs/detector_thresholds_v2.json` |
| `af15201` | `tests/test_detector_metrics_v2.py` (12 tests; caught a real disjointness bug) |
| `65c5676` | `scripts/verify_detector_reproduction.py` and `configs/seeds/` (three seed configs + README; nothing trained) |
| _this commit_ | Protocol A results, PROTOCOL.md §8, this report |

No existing module was modified: `scripts/evaluate_anomaly_detector.py`,
`utils.py`, `preprocessing/tep_loader.py` and `main.py` are untouched.

---

## 3. Inspection findings (tasks a–c)

### a. How an alarm is decided

| Element | Value | Where |
|---|---|---|
| LSTM score | mean per-sensor MSE over a 60-sample window, scaled | `anomaly_detection/inference.py:118` |
| LSTM threshold | 0.687 frozen | `scripts/evaluate_all_faults_full.py` docstring; artifact deleted (see §6) |
| CVA | `n_past=5`, `n_future=5`, `r=67`, statistics T²/Q/Tr, primary **Q** | `outputs/anomaly_detector/dynamic/fit_metadata.json` |
| DPCA | `n_lags=3`, T²/SPE, primary **SPE** | same |
| Dynamic thresholds | CVA Q 239.75, T² 127.54; DPCA SPE 40.91, T² 124.25 | `dynamic/dynamic_thresholds.json` (== `fit_metadata`) |
| Smoothing | **none at calibration**; `configs/config.yaml` asks EWMA λ=0.1, `configs/config_a5000.yaml` asks none | `dynamic_thresholds.json:config`, runner default `ewma` at `dynamic/runner.py:45` |
| Fusion | OR of LSTM and dynamic primary; `combined = max(norm_lstm, norm_dyn) * lstm_thr` | `dynamic/runner.py:235-260` |
| Event aggregation | opens after 3 consecutive alarmed windows, closes after 20 consecutive quiet windows or 200 alarmed windows | `main.py:178-180, 270-310` |

The EWMA discrepancy is real: the runner defaults to EWMA for streaming, but the
stored thresholds were fitted on unsmoothed statistics. The evaluation therefore
uses the calibration-consistent setting (no smoothing) and records the config
value as a mismatch to fix separately.

### b. Calibration runs vs free runs

* LSTM weights + scaler + threshold: `TEP_FaultFree_Training` (window build in
  `scripts/prepare_tep.py`; run partition in
  `data/processed/manifests/detector_split.json`, 350/75/75).
* CVA/DPCA fit: `TEP_FaultFree_Training` runs **1–400**; thresholds from runs
  **401–500** (`fit_metadata.json`, `fit_n_runs=400`, `holdout_n_runs=100`).
* **FAR set: `TEP_FaultFree_Testing` runs 1–500** — a different file, no run
  reuse. Verified by the script (`far_disjoint_from_calibration.ok = true`,
  `overlapping_files = []`, `overlapping_runs = []`).
* The window-calibrated operating point in §5 uses Training 401–500 only, i.e.
  threshold-fitting data, never the FAR set.

Note: `fit_metadata.json` stores only the **first ten** runs of each partition
plus the counts; reconstructing the calibration set from the stored list alone
undercounts it (20 instead of 500). The loader expands from
`fit_n_runs`/`holdout_n_runs`.

### c. Existing results (reproduction target)

* `outputs/evaluation/all_faults_detector_summary.json` — LSTM only, 20 faults ×
  500 Training runs, threshold 0.687. **This is the usable target.**
* `data/processed/anomaly_detector_eval.json` — LSTM + CVA + DPCA + fused, but
  only faults 1/3/9/15, 5 runs each, and its `n_samples` (3,980 / 4,000 for 5
  runs) corresponds to 800-sample runs with older thresholds (CVA Q 204.53 vs
  the current 239.75). **Not reproducible from the current artifacts**; it is a
  stale smoke run, listed as a mismatch rather than treated as a baseline.

---

## 4. Reproduction (gate item 1)

`python scripts/verify_detector_reproduction.py --faults 1 3 4 9 15`
(5 × 500 = 2,500 Training runs, 66 s):

| Fault | runs | mean_max | published | Δ | event rate | published | delay (win) | published |
|---|---|---|---|---|---|---|---|---|
| 1 | 500 | 34.9632 | 34.9634 | −2.0e-4 | 1.000 | 1.000 | 0.00 | 0.00 |
| 3 | 500 | 0.6779 | 0.6779 | +5.3e-7 | 0.120 | 0.118 | 41.22 | 41.26 |
| 4 | 500 | 1.5871 | 1.5871 | +6.5e-6 | 1.000 | 1.000 | 0.00 | 0.00 |
| 9 | 500 | 0.6788 | 0.6788 | −6.5e-8 | 0.132 | 0.126 | 40.32 | 40.28 |
| 15 | 500 | 0.6845 | 0.6845 | +1.8e-7 | 0.204 | 0.196 | 39.29 | 39.16 |

**PASS.** Mismatches, all boundary effects on `max_consec >= 3` (1–4 runs out of
500) and on the mean of the first-alarm window: faults 3/9/15 event rate
+0.002…+0.008 and delay ±0.13 windows. No score-level mismatch.

---

## 5. Protocol A results

Commands (each ~254 s, CPU, single process):

```bash
python scripts/evaluate_detector_v2.py --protocol A --component val
python scripts/evaluate_detector_v2.py --protocol A --component test
```

Outputs: `outputs/evaluation/detector_v2/A_{val,test}.json`, `.md`,
`_per_run.csv` (2,000 rows each).

### 5.1 Per-fault (validation; test in the JSON/CSV)

| Fault | LSTM FDR | CVA Q | DPCA SPE | Fused | Fused det. | Delay (samples) | AUROC | Events/run |
|---|---|---|---|---|---|---|---|---|
| 1 | 99.5% | 99.4% | 98.4% | 99.5% | 100% | 8.4 | 0.999 | 1.00 |
| 2 | 98.3% | 97.0% | 95.4% | 98.3% | 100% | 17.6 | 0.994 | 1.00 |
| 3 | 2.4% | 0.0% | 0.0% | **2.4%** | 39% | 370.4 | 0.541 | 0.49 |
| 4 | 100.0% | 99.4% | 99.4% | 100.0% | 100% | 4.4 | 1.000 | 1.00 |
| 5 | 100.0% | 100.0% | 7.6% | 100.0% | 100% | 4.0 | 1.000 | 1.00 |
| 6 | 100.0% | 100.0% | 100.0% | 100.0% | 100% | 4.0 | 1.000 | 1.00 |
| 7 | 100.0% | 100.0% | 99.4% | 100.0% | 100% | 4.0 | 1.000 | 1.00 |
| 8 | 97.5% | 97.0% | 92.9% | 97.5% | 100% | 24.2 | 0.989 | 1.00 |
| 9 | 2.5% | 0.0% | 0.0% | **2.5%** | 40% | 339.1 | 0.547 | 0.49 |
| 10 | 75.6% | 96.2% | 17.6% | 96.3% | 100% | 33.1 | 0.983 | 1.02 |
| 11 | 98.8% | 98.3% | 98.2% | 98.8% | 100% | 13.3 | 0.996 | 1.00 |
| 12 | 99.0% | 99.2% | 96.8% | 99.2% | 100% | 10.2 | 0.997 | 1.00 |
| 13 | 94.4% | 94.2% | 92.5% | 94.6% | 100% | 47.6 | 0.975 | 1.03 |
| 14 | 100.0% | 99.0% | 98.1% | 100.0% | 100% | 4.0 | 1.000 | 1.00 |
| 15 | 3.3% | 5.0% | 0.0% | **7.7%** | 84% | 315.2 | 0.726 | 1.28 |
| 16 | 75.3% | 97.2% | 14.0% | 97.3% | 100% | 25.5 | 0.988 | 1.00 |
| 17 | 95.7% | 95.7% | 95.3% | 95.8% | 100% | 37.4 | 0.979 | 1.02 |
| 18 | 93.1% | 93.3% | 92.2% | 93.5% | 100% | 55.3 | 0.970 | 1.01 |
| 19 | 97.5% | 99.0% | 0.0% | 99.1% | 100% | 11.6 | 0.997 | 1.00 |
| 20 | 93.9% | 94.8% | 93.0% | 94.9% | 100% | 44.6 | 0.976 | 1.02 |

Mean delay over the 18 observable faults: **36.9 samples (≈1.8 h at 3-min
sampling)**, 1.02 events per run.

### 5.2 Baseline comparison (mean over the 18 observable faults)

| Detector | FDR val | FDR test | AUROC val | AUROC test |
|---|---|---|---|---|
| Fused (OR) | **92.9%** | 92.9% | 0.976 | 0.976 |
| CVA Q | 92.5% | 92.5% | **0.993** | 0.992 |
| LSTM AE | 90.1% | 90.2% | 0.964 | 0.964 |
| PCA SPE | 72.3% | 72.6% | 0.972 | 0.970 |
| DPCA SPE | 71.7% | 71.8% | 0.969 | 0.968 |
| PCA T² | 62.0% | 62.0% | 0.951 | 0.958 |
| Isolation Forest | 39.1% | 39.2% | 0.914 | 0.937 |

Baselines are fitted on 100,000 normal *Training* samples only; PCA keeps 31
components (90% variance); thresholds at the 1% quantile of training scores.
The fused detector's advantage over CVA Q alone is small on average and
concentrated in faults where PCA T² fails (4: 19.1%, 5: 24.1%, 10: 10.2%, 16:
0.5%, 19: 0.1%, 20: 10.6%).

### 5.3 FAR (fault-free Testing runs 1–500, no calibration overlap)

| Detector | window FAR | sample FAR | run FAR (event) |
|---|---|---|---|
| LSTM AE | 1.66% | – | **30.6%** |
| CVA Q | 0.00% | 0.94% | 0.0% |
| DPCA SPE | 0.00% | 0.96% | 0.0% |
| PCA T² | 0.00% | 1.07% | 0.0% |
| PCA SPE | 0.00% | 1.01% | 0.0% |
| Isolation Forest | 0.04% | 1.13% | 0.8% |
| **Fused** | 1.66% | – | **30.6%** |

---

## 6. Observable set (gate item 3) — PROPOSED

Written to `docs/PROTOCOL.md` §8.3. Applying the owner rule (fused FDR ≥ 5% on
validation) to the frozen artifacts gives:

* **Observable: 18 faults. Excluded: {3, 9}** (2.4%, 2.5%). Test agrees exactly
  (1.8%, 2.0%, same two faults). Fault 15 is just above the line (7.7% / 8.0%).
* At a common **1% window** operating point (thresholds from the dynamic holdout
  runs, not the FAR set) the set becomes 19 faults with only {3} excluded: fault
  9 reaches 6.6% and fault 15 reaches 97.2%.

Recommendation: keep {3, 9} excluded, but fix the threshold chain before event
generation — the dynamic thresholds are per-sample p99 applied to a window mean,
which is why their window FAR is 0% and fault 15 is nearly invisible.

---

## 7. Seeds (task 4) — written, not trained

`configs/seeds/config_a5000_seed{42,123,999}.yaml` (identical except `seed`),
plus `configs/seeds/README.md` with the commands. Estimated **20–35 min per
seed on CPU**, **1–1.75 h for three seeds**, anchored on the measured 0.20
ms/window inference cost and ~31k windows per epoch. Unmeasured: the GPU factor
and the early-stopping point.

---

## 8. Tests

```
python -m pytest tests/ -q   ->  88 passed in 3.50s
python -m pytest tests/test_detector_metrics_v2.py -q  ->  12 passed
```

New tests cover window labelling, the post-onset detection rule, event
segmentation, degenerate AUROC/PR-AUC, mean vs max aggregation with a lagged
index, and the calibration/FAR disjointness check. The disjointness test found
a real defect — the check compared a basename against full-path keys, so a
same-file overlap reported "no overlap" — which is fixed and now covered.

---

## 9. Gate table

| Gate item | Status | Evidence |
|---|---|---|
| Reproduces existing numbers within tolerance | **PASS** (scores; mismatches listed) | §4, `reproduction.json` |
| Per-run alarm table exists | **PASS** | `A_{val,test}_per_run.csv`, 2,000 rows each, 17 columns |
| No calibration run in the FAR set, script-verified | **PASS** | `far_disjoint_from_calibration.ok = true`, 500 FAR runs, 500 calibration runs, no overlap |
| Per-fault FDR/delay/AUROC/PR-AUC for 4 detectors + baselines | **PASS** | §5, JSON |
| Faults with fused FDR < 5% flagged, set proposed in PROTOCOL.md | **PASS** (proposal) | §6, PROTOCOL.md §8.3 |
| Seed configs + commands, nothing trained | **PASS** | `configs/seeds/` |
| Metric unit tests | **PASS** | §8 |

---

## 10. Deviations from the brief

1. **Window labelling changed.** A window is positive if it *contains* a
   post-onset sample, not if it starts after the onset. The legacy rule counts 12
   windows per run as negatives while they already hold fault data, and books
   their alarms as pre-onset false alarms (the published `false_before` of 21 for
   every strong fault is this artefact). Without the fix, "false alarm before
   onset" read 100% for faults 1/2/4–8/11/12/14/19 — a measurement bug, not a
   detector property. Consequence: delay is measured to the completing sample of
   the alarm window, so it is larger than the legacy number by up to 59 samples.
2. **Window aggregation is the mean, not the max.** A max over 60 samples turns
   a 1%-per-sample threshold into ≈45% window FAR; the mean keeps every detector
   at the level its threshold was set for and matches the LSTM's own window
   definition. Sample-level FAR is reported separately.
3. **The LSTM threshold is passed in explicitly (0.687).** The artifact
   `outputs/anomaly_detector/threshold.json` was deleted in `49b006b`, so
   `AnomalyDetector.from_artifacts` cannot load this model at all; the script
   rebuilds it from `model.pt` + the config's LSTM block and reads the frozen
   threshold from `configs/detector_thresholds_v2.json`, with provenance. The
   alternative value in history is 1.047 (a 69-score smoke calibration).
4. **A second operating point was added** (all detectors at 1% window FAR,
   thresholds from Training 401–500). It is reported alongside, never used for
   the headline numbers, and never fitted on the FAR set.
5. **`data/processed/anomaly_detector_eval.json` is not reproducible** and is
   reported as stale rather than matched (§3c).
6. **The first test-component run used the buggy labelling** (same 254 s) and was
   overwritten by the corrected run; only the corrected files remain.

---

## 11. Open questions (owner)

1. **Observable set:** approve `{3, 9}` excluded (frozen thresholds) or `{3}`
   (window-calibrated)? The answer changes the RQ1 headline denominator.
2. **Run-level FAR of 30.6%**: fix before generating events, by requiring longer
   persistence, by moving the LSTM threshold to a 0.1% window target, or by
   accepting normal-run events and labelling them as such? As it stands, one in
   three normal runs would hand the adapter an event to explain.
3. **Refit the dynamic thresholds at window level** (recommended, changes the
   published detector), or keep the per-sample chain and accept that the OR-gate
   is the LSTM alone?
4. **EWMA**: `configs/config.yaml` asks for EWMA λ=0.1 but the artifacts were
   calibrated unsmoothed. Stream with EWMA (thresholds then need refitting) or
   set the config to `none` for consistency?
5. **Baseline fairness**: baselines are at 1% *sample* FAR, so their window FAR is
   ~0 like CVA/DPCA. Recalibrating them at 1% window FAR would raise their FDR
   and is a fairer comparison — do it before citing the baseline table as a
   lower bound?

---

## 12. Cost (measured where possible)

**Phase 1 — done.** 254 s per protocol component on CPU (2,000 runs, one 3.6 GB
CSV pass, 500 FAR runs, 100 holdout calibration runs, baselines fitted in 2.5 s).
Reproduction check 66 s. No GPU used. A Protocol B run is the same cost per
component (folds have 17/18 faults, so 1,700/1,800 runs).

**Phase 2A — unchanged estimate, still CPU-only.** Evaluator rewrite + item
generator + routing harness: 1–2 days. Routing/format evaluation on 200 items:
well under 1 GPU-hour once an adapter exists. Event generation is blocked on
open question 2 (or it will emit ~0.35 normal events per normal run).

**LLM training (not this phase, for reference).** 3 configurations × 3 seeds;
the only measured anchor is 8,877 s per 2,376 examples, i.e. ≈3.7 s/example, so
2,000–3,000 examples ≈ 2.1–3.1 h per seed plus evaluation. Marked as an
extrapolation; inference speed still to be measured.
