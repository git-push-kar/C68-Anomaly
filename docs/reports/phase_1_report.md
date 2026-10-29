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
---

# Phase 1 / 2A patch — operating point frozen, response layer corrected

**Repo:** `C:\Users\Admin\Desktop\anomaly` · branch `benchmark-upgrade` · CPU only.
No LLM training, no GPU. Phase 0 patch approved as the precondition.

## 1. Summary

1. **A0 is broken, and it was broken before this patch.** `outputs/anomaly_detector/threshold.json`
   was deleted in `49b006b`; `AnomalyDetector.from_artifacts` calls
   `load_threshold(model_dir)`, which raises `FileNotFoundError`. The live app
   constructs exactly that way (`main.py:141`), so `TEPApp()` cannot load the
   detector at all. Reproduced with a real construction, not inferred. The fix
   is proposed separately (§10) and **not** applied, per the brief.
2. **The 0.687 threshold now has a provenance.** It is `p99.0` of the LSTM
   window scores over the 75 normal validation runs: **0.68704**, with mean
   0.61923, std 0.02832, max 0.72148 over **6,675 windows** — reproducing the
   historical note's mean/std/max/n exactly. It is not a fitted value; it is
   that percentile.
3. **A window-level operating point is frozen** (1% window FAR, persistence 12,
   no smoothing) from calibration data only, with SHA-256 of the file and of
   every artifact. Fused run-level FAR falls **30.6% -> 2.0%**, mean FDR over the
   18 observable faults rises **92.9% -> 98.2%**, and fault 15 goes
   **7.7% -> 97.2%** with delay **315 -> 25 samples**.
4. **The headline observable set is unchanged**: the deployed point still
   excludes `{3, 9}`. The recalibrated set (19 faults, `{3}` excluded) is
   recorded as a clearly labelled secondary row and is not substituted anywhere.
5. **Event-level behaviour is now measured, not assumed**: 0.02 events per
   fault-free run, 1,824 events over 2,000 faulty runs with **zero** pre-onset,
   and fault 15 post-onset in **100/100** runs.
6. **Severity had two answers.** The production rule is now the only one quoted;
   the response-layer rule is a cross-check. On all 489 stored events it agrees
   **16.2%** of the time and is lower on 288, so adopting it would have been a
   corpus-wide downgrade, not a judgement call.
7. **The number check can finally fail a wrong count.** `LABEL_INTEGERS` is gone;
   a bare integer needs a stored count or a fault-id reading. Gate flipped from
   FAIL to PASS.

## 2. Changes (commit hashes)

| Commit | Change |
|---|---|
| `543d8b2` | `scripts/derive_lstm_threshold.py` — A1 provenance re-derivation |
| `0f71119` | `scripts/fit_operating_point_v2.py`, `configs/detector_v2_operating_point.json` + `.sha256.json`, `configs/config_a5000_no_smoothing.yaml`, sweep results — A2/A3 |
| `8adb48b` | `scripts/evaluate_detector_v2.py` operating-point input, `outputs/evaluation/detector_v2/v2_point/` — A4 |
| `207ba84` | `scripts/event_level_stats_v2.py` + results — A5 |
| `23027c3` | `llm/response/validators.py` number rule + 9 tests — B1 |
| `4fad4fe` | severity authority, cross-check, `scripts/severity_agreement_audit.py` + results — B2 |
| `7d8dd8f` | `docs/PROTOCOL.md` §8.3-8.8, `docs/RESPONSE_SPEC.md` §8, `scripts/render_examples.py` — A6/B3 |

`llm/inference.py` and `llm/dataset.py` are **unmodified** (`git status` clean,
no diff). The live dispatcher, its 11 intents and the SYSTEM_PROMPT rubric are
untouched.

## 3. A0 — the live detector does not load (VERIFIED, broken)

```
$ python -c "AnomalyDetector.from_artifacts(model_dir='outputs/anomaly_detector',
    scaler_dir='outputs/preprocessing', threshold_dir='outputs/anomaly_detector')"
FAILED: FileNotFoundError Threshold file not found: outputs\anomaly_detector\threshold.json
$ git show --name-status 49b006b | grep threshold
D   outputs/anomaly_detector/threshold.json
```

`anomaly_detection/threshold.py:96` raises rather than defaulting, which is the
right behaviour for a threshold; the artifact is what went missing. The value in
history (`49b006b^`) is `1.0472214221954346`, method `percentile`, `n_normal_scores`
69 — a smoke-sized calibration, 150x smaller than the real validation set.

## 4. A1 — threshold provenance (VERIFIED)

`python scripts/derive_lstm_threshold.py` (2.8 s):

| quantity | value | historical claim |
|---|---|---|
| windows | 6,675 (75 runs x 89) | `n 6675` |
| mean | 0.61923 | 0.619 |
| std | 0.02832 | 0.028 |
| max | 0.72148 | 0.721 |
| **p99.0** | **0.68704** | frozen **0.687** (delta +4e-5) |
| p99.5 | 0.69337 | — |
| p99.9 | 0.70729 | — |

The historical note lived in a different checkout (`docs/THRESHOLD_COMPARISON.md`
refers to `C:\Users\Admin\Desktop\anom\C68-Anomaly`), so this is a re-derivation,
not a citation. It reproduces to 4e-5.

## 5. A2/A3 — operating point (calibration data only)

Command: `python scripts/fit_operating_point_v2.py --rebuild-cache`
(**139 s** to score 2,675 runs from CSV, **3.4 s** to sweep 12 combinations).

Pareto table (full version in `outputs/evaluation/detector_v2/operating_point_sweep.md`):

| target | persistence | run FAR sel (1-250) | run FAR rep (251-500) | mean FDR (18) | mean delay | eligible |
|---|---|---|---|---|---|---|
| 0.1% | 3 | 8.4% | 9.2% | 98.0% | 19.6 | no |
| 0.1% | 5 | 2.8% | 4.0% | 98.0% | 19.6 | yes |
| 0.1% | 8 | 0.8% | 0.8% | 98.0% | 19.6 | yes |
| 0.1% | 12 | 0.0% | 0.0% | 98.0% | 19.8 | yes |
| 0.5% | 3 | 27.6% | 26.8% | 98.1% | 18.8 | no |
| 0.5% | 5 | 15.6% | 14.0% | 98.1% | 18.8 | no |
| 0.5% | 8 | 4.4% | 6.0% | 98.1% | 18.9 | yes |
| 0.5% | 12 | 0.4% | 1.2% | 98.1% | 18.9 | yes |
| **1.0%** | **12** | **2.4%** | **1.6%** | **98.2%** | **18.3** | **selected** |
| 1.0% | 3 | 37.6% | 44.0% | 98.2% | 18.1 | no |
| 1.0% | 5 | 24.0% | 24.8% | 98.2% | 18.3 | no |
| 1.0% | 8 | 11.2% | 10.0% | 98.2% | 18.3 | no |

Frozen: `configs/detector_v2_operating_point.json`, sha256
`e7fe73eb0b75b4d5703136e6c997e223d2a878c7e36764ed8d6ccf00e559624b`; LSTM 0.68704,
CVA Q 194.632, DPCA SPE 24.043, persistence 12, smoothing none. Artifact hashes
(`model.pt`, `cva_model.npz`, `dpca_model.npz`, `dynamic_thresholds.json`,
`detector_split.json`, `splits_v2.json`, sweep JSON) are inside the file.

Two honest notes:

* At 1% target the LSTM lands on 0.68704 — the arithmetic identity with the
  deployed threshold is a coincidence worth stating: the deployed 0.687 was
  already a p99.0 window percentile, so the LSTM part of v2 changes nothing. All
  of the gain is the dynamic head finally being calibrated at window level.
* The instruction "select the combination with the smallest delay" necessarily
  reads a delay, and the only faulty data permitted is Protocol A validation.
  Delay is therefore the selection criterion; **FDR never entered selection**.
  This is the one place where the brief's "report the trade-off, do not tune"
  and "smallest delay" pull against each other, and delay won.

## 6. A4 — Protocol A at both points

Commands (each **251 s**): `evaluate_detector_v2.py --protocol A --component {val,test}
--operating-point configs/detector_v2_operating_point.json --out-dir .../v2_point`.

| | v1 (deployed) | v2 (frozen) | v2 test |
|---|---|---|---|
| mean fused FDR, 18 observable | 92.9% | **98.2%** | 98.2% |
| mean delay, 18 observable | 36.9 | **18.3** | — |
| fused run FAR (500 fault-free) | 30.6% | **2.0%** | 2.0% |
| fused window FAR | 1.66% | 2.33% | 2.33% |
| CVA Q window FAR | 0.00% | 1.02% | 1.02% |
| fault 15 FDR / delay | 7.7% / 315.2 | **97.2% / 24.9** | 96.8% / 27.5 |
| fault 3 FDR | 2.4% | 4.8% | 4.8% |
| fault 9 FDR | 2.5% | 7.0% | 6.3% |

Baselines, recalibrated to the same 1% window FAR (previously 1% *sample*, which
gave them ~0% window FAR and made the old table a soft target):

| Detector | window FAR | run FAR | note |
|---|---|---|---|
| PCA T2 | 1.21% | 6.4% | was 0.00% / 0.0% |
| PCA SPE | 1.24% | 1.8% | was 0.00% / 0.0% |
| Isolation Forest | 1.02% | 4.8% | was 0.04% / 0.8% |

Protocol A test was run **once**, after the freeze, and agrees with validation
within 0.5 points on every fault. No v1 number was recomputed, so the Phase 1
tables remain valid as published.

## 7. A5 — event-level statistics (v2)

| Population | events/run | runs with an event |
|---|---|---|
| fault-free Testing 1-500 | 0.020 | 10/500 (2.0%) |
| faulty, Protocol A val | 0.91 | 1,724/2,000 |
| fault 15 | 1.00 | 100/100 post-onset |

Events starting before onset: **0 of 1,824 (0.0%)**. With persistence 12 and 21
pre-onset windows per run, twelve consecutive pre-onset alarms is effectively
impossible; the v1 pre-onset counts came from the 3-window confirmation plus the
legacy labelling bug. Event-count histogram on fault-free runs: 490x0, 10x1 —
never more than one.

## 8. A6 — observable set, both points

| Point | Set | Excluded |
|---|---|---|
| **v1, deployed (HEADLINE)** | **18** | **{3, 9}** |
| v2, recalibrated (SECONDARY row) | 19 | {3} |

Recorded in `docs/PROTOCOL.md` §8.3. 3-seed training restated, not run
(`configs/seeds/`, ~20-35 min/seed, 1-1.75 h for three).

## 9. B1/B2/B3 — response layer

**B1.** `LABEL_INTEGERS` replaced by `STORE_COUNT_KEYS` + `fault_ids_in`. Nine
tests: stored count passes; `17 windows` on a 2-window event fails; bare `41`
fails but `XMEAS_12` passes; `FAULT_15` and "Faults 4 and 11" pass; "Fault 15
produced 17 alarmed windows" fails on 17 only; list markers are not claims.
Two real false positives surfaced and were fixed in the rule, not the tests:
`\bfaults?\b` never matched `FAULT_15` (underscore is a word character) and
numbered reasoning steps were read as measurements.

**B2.** `severity.py` now returns
`evidence.process_limits.compute_deterministic_severity` and exposes
`components['severity_agreement' | 'severity_gap' | 'proposed_level']`.
`scripts/severity_agreement_audit.py` over **all 489 stored events**:

| | count |
|---|---|
| agree | 79 (16.2%) |
| cross-check 1 level above | 29 |
| cross-check 1 level below | 93 |
| cross-check 2 levels below | 218 |
| cross-check 3 levels below | 70 |
| authoritative critical -> proposed medium | 205 |
| stored column equals authoritative | 413/489 (76 differ) |

The 76 stored-vs-authoritative differences are disclosed in answers ("the event
store carries LOW ...") and never quoted as the level. Two rules that disagree
on 84% of a corpus were not a tie to break: keeping the production rule avoids
rewriting 205 criticals to medium.

**B3.** `docs/RESPONSE_SPEC.md` §8 records: schema-shaped answers and
code-rendered tables; the three proposed intents behind an off flag; the 0.15
margin as an explicit placeholder pending adapter measurements; severity
authority; and the tightened integer rule. `docs/PROTOCOL.md` §8.3-8.8 records
the point, the sweep, both observable sets and the event statistics.

## 10. A0 fix — PROPOSED, not applied

Two options, both one line of code plus an artifact:

1. **Regenerate the artifact** (preferred): `scripts/derive_lstm_threshold.py`
   already produces the exact fields; write them with
   `anomaly_detection.threshold.save_threshold` to
   `outputs/anomaly_detector/threshold.json` and keep that path out of the
   ignore list, or generate it as part of `prepare_tep.py`. Reproducible, and it
   keeps `load_threshold` strict.
2. **Fall back explicitly** in `AnomalyDetector.from_artifacts` to
   `configs/detector_thresholds_v2.json` (already committed, with provenance)
   and log a warning. This is the smaller change but creates a second source of
   truth for one number.

Option 1 keeps one authority. Either way the deployed detector is broken today,
so this outranks any further evaluation work.

## 11. Tests

```
python -m pytest tests/ -q   ->  167 passed in 5.14s
```

155 before this patch, 12 added (9 number-rule, 3 severity/agreement). The three
pre-existing severity tests were rewritten because their subject moved from
"the proposed rule decides the level" to "the cross-check's level is reported
separately" — that is a changed specification, not a loosened assertion.

## 12. Gate table

| Gate item | Status | Evidence |
|---|---|---|
| A0 resolved (determined and recorded) | **PASS (broken, fix proposed)** | §3, reproduction command, `49b006b` |
| Frozen operating-point file with hashes | **PASS** | `configs/detector_v2_operating_point.json` + `.sha256.json` |
| Pareto table | **PASS** | §5, `operating_point_sweep.md` (12 rows) |
| v1 vs v2 validation tables | **PASS** | §6, `v2_point/A_val.json`, PROTOCOL §8.6 |
| Test run once after freeze | **PASS** | `v2_point/A_test.json`, no retuning after |
| Event-level stats | **PASS** | §7, `event_level_stats_v2.json` |
| Observable set under both points | **PASS** | §8, PROTOCOL §8.3 |
| B1 tests pass, gate flipped | **PASS** | 9 tests; FAIL -> PASS |
| B2 agreement report | **PASS** | §9, `severity_agreement.md`, 489 events |
| Full suite passes | **PASS** | 167 passed |
| No LLM training / no GPU | **PASS** | no training job; all timings are CPU |
| `llm/inference.py` unchanged | **PASS** | not in any diff |

## 13. Deviations and judgement calls

1. **Delay was read to satisfy "smallest delay".** FDR and Protocol A test were
   not. If that reading is wrong, the alternative selection — 0.1%/12, run FAR
   0.0%/0.0%, delay 19.8 — is one row away in the frozen file's provenance.
2. **The secondary `cva_T2`/`cva_Tr`/`dpca_T2` thresholds were not recalibrated.**
   They never enter the OR-gate or a headline table; recalibrating them would
   mean re-running a 3-minute scoring pass to produce numbers nothing cites.
   Recorded explicitly in the frozen file instead of silently left.
3. **The old uniform-1%-window experiment is skipped when a frozen point is
   supplied.** Two differently-derived "1% window" columns in one file is how
   the wrong number gets quoted later. The v1 file still carries it.
4. **`per_fault_window_calibrated` for v2 is `{}`**, for the same reason.
5. **Sample-level FAR is reported for v2 but is not comparable** (CVA Q reads
   26.1% against a window-calibrated threshold). Stated in PROTOCOL §8.6 rather
   than quietly dropped.
6. **Severity wording changed in the renderer and the lookup answer** (no score,
   no module path). The module path in prose tripped the tag-shaped-token check,
   which is a real defect the validator caught.
7. **Intermediate commits were not individually test-run.** Commits are grouped
   by task; the suite was run on the final tree (167 passed). Commit `23027c3`
   is self-consistent by construction (it carries the old severity assertions
   that its dependencies still satisfy), but "self-consistent by construction"
   is weaker than a measured result.

## 14. Open questions (owner)

1. **Fix A0 before anything else?** Both options are in §10; the deployed app
   cannot load the detector today.
2. **Which observable set becomes the RQ1 denominator** — 18 at v1 (headline,
   unchanged) or 19 at v2? The patch keeps 18 as headline and reports 19 as a
   secondary row, so this needs an explicit decision before Phase 4.
3. **Adopt the v2 point for event generation?** It turns the OR-gate into a real
   fusion (fault 15 7.7% -> 97.2%) and cuts run FAR 30.6% -> 2.0%. Cost: every
   published detector number would need restating at v2.
4. **Persistence 12** is the selected operating point but it is also a long
   confirmation window (~12 windows = 60 samples at stride 5). If the operator
   needs earlier alarms, the 0.1%/8 row (run FAR 0.8%, delay 19.6) is the
   cheaper trade.
5. **Severity:** the production rule ignores `fault_label` entirely (evidence
   only). Is that intended? It means severity cannot leak a fault identity — good
   for benchmark integrity — but a critical-family fault is not escalated by its
   family.
6. **The 76 stored-vs-authoritative differences**: regenerate the `severity`
   column, or keep disclosing it in answers? Disclosing is honest but noisy.

## 15. Cost

**This patch — measured, CPU only.** Cache build 139 s (2,675 runs), sweep 3.4 s,
each Protocol A component 251 s (2,000 faulty + 500 fault-free runs + baseline
fit + baseline recalibration), event stats <5 s, agreement audit <10 s, threshold
derivation 2.8 s. Total wall clock under 15 minutes of compute. No job exceeded
30 minutes, so no pre-announcement was required; the longest single command was
251 s.

**Phase 2A (response layer) — done, unchanged estimate for the rest.** Evaluator
rewrite + item generator + routing harness: 1-2 days. Routing/format evaluation
on 200 items: well under 1 GPU-hour once an adapter exists.

**LLM training (not this phase, for reference).** 3 configurations (Protocol A,
fold_1, fold_2) x 3 seeds. Anchor: 8,877 s per 2,376 examples = 3.74 s/example,
so a 2,000-3,000 example training set is 2.1-3.1 h per seed, 6-9 h per
configuration, **18-28 h for nine runs**, plus evaluation. Inference speed is
still an assumption and must be measured before any per-query cost claim.
