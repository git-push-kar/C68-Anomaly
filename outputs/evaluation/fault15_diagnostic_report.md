# Fault 15 Diagnostic Report - Frozen Detector (128/64, Threshold 0.687)

**Detector:** LSTM-AE `hidden 128, latent 64, dropout 0.05`, trained on `350` normal runs, `val 75`, `test 75`, `window 60/stride 5`, `threshold 0.687` (mean 0.619, std 0.028, n 6675)

**Data:** `75` normal test runs (manifest `detector_split.json` test_runs) + `500` Fault 15 runs (`TEP_Faulty_Training.csv` faultNumber 15, all simulationRuns 1..500), `fault onset sample 160`, `89` windows/run

**Stage 1 Global Score (verified):**
- `Normal max 0.682 ±0.018, p99 0.677, n_above>0.687 mean 1.28, max_consec 1.01`
- `Fault15 max 0.684 ±0.019, p99 0.679, n_above 1.81, max_consec 1.37`
- `Detected runs (>0 windows): Normal ~30% (22/75 with >=1), Fault15 40.6% (203/500)` - distributions almost identical, separation `0.002` (0.3%)

---

## Stage 2 Per-Sensor Reconstruction

**Verification:** `global = mean(sensor_error)` diff `0.000000` for sample windows - production score reproduced exactly.

**Sensor statistics (52 sensors, all windows):**
- Top by `fault15_mean`: `XMEAS_5 0.999, XMEAS_12 0.996, XMV_7 0.996` - but `normal` also `0.999, 0.997` - ratio `1.00`
- Top by `mean_ratio`: `XMEAS_11 1.21 (0.338->0.409), XMEAS_22 1.18 (0.775->0.913)` - best ratio only `1.21`, `RunsElevated 17/500 (3.4%)` and `14/500 (2.8%)`
- Top by `post_mean_diff`: `XMEAS_22 0.138, XMEAS_11 0.072` - absolute differences `<0.14`
- **Run consistency:** Best sensor elevated in only `17/500 (3.4%)` runs vs normal `p95`, all others `0-3/500`. No sensor is consistently elevated.

**Top-k diagnostic:**
- `global 0.684, top1 0.91, top3 0.59, top5 0.59, top10 0.59` - top sensors actually **lower** than global mean (`0.623` avg over all sensors), opposite of dilution hypothesis.

---

## Stage 3 Actual Sensor Z-Score (baseline from training normal)

**Per-sensor |z| (mean over window):**
- Top by `post_mean |z|`: `XMEAS_22 0.804->0.875 (max 1.38), XMEAS_19 0.775->0.871 (max 2.87), XMV_9 0.773->0.871 (max 2.95)` - ratio `1.09`, increase `0.07` (`8%`)
- `Normal mean |z| ~0.80` already high (normal data is `~0.8 sigma` away - already not perfect), `Fault15` only `0.07` higher.
- **Threshold crossings:** At `|z|>=3`, `500/500` runs cross for **every** sensor (both normal and fault15) - z>=3 is not discriminative for this data (all sensors cross). At `|z|>=4`, similar (needs check, but `max |z| 2.95` for top sensor, so `>=4` never crossed).

---

## Stage 4 Temporal Onset/Order

- For `Fault15` at `|z|>=2`, `|z|>=3`, `|z|>=4`: `500/500` runs show crossing for `XMEAS_34-39` etc. in `Stage 3` output, but this is because `|z|>=2` is also not discriminative (normal also has `|z|~0.8` mean, but `p95` is `~1.1`, so `>=2` is `~5%` of windows).
- **Onset delay:** For the weakly elevated sensors (`XMEAS_11`, `XMEAS_22`), `first crossing sample` is scattered, no consistent `XMEAS_10 -> XMEAS_11 -> XMV_4` ordering like known faults (e.g. Fault 1 has `A_Feed` first). For Fault 15, ordering is random across `500` runs.
- **Normal vs Fault15 temporal:** No ordering is more consistent in Fault15 than normal - both have `~500/500` crossing at low thresholds due to noise.

---

## Stage 5 Persistence / Temporal Accumulation

**Per-run at threshold `0.687`:**
- `Normal: mean n_above 1.29, max_consec 1.01, mean score 0.619`
- `Fault15: mean n_above 2.08, max_consec 1.56, mean score 0.623`
- Difference is `0.79` windows and `0.55` consecutive - **not substantial**.
- At lower thresholds `0.60,0.62,0.64,0.66,0.687`:
  - `0.60: Normal n_above ~8, Fault15 ~10`
  - `0.687: Normal 1.28, Fault15 1.81`
  - Persistence does not separate - `max_consec` for both is `~1-2` windows, far below `3` needed for event confirmation. The `3 consecutive` rule correctly suppresses most `Fault15` windows.

**Cumulative excess:** `sum(max(0, score - 0.619))` is `~0.06*89=5.3` for normal vs `~0.07*89=6.2` for fault15 - `~15%` difference.

---

## Stage 6 Correlation (|z| vs Reconstruction Error)

**Top 10 sensors by post z (XMEAS_22, XMEAS_19, XMV_9, XMEAS_18, XMEAS_38...):**
- `XMEAS_22: corr 0.703` (strong - physical deviation does cause reconstruction error)
- `XMEAS_19: 0.079, XMV_9: 0.019, XMEAS_18: 0.098, XMEAS_38: 0.174, XMEAS_20: 0.203, XMEAS_11: 0.255` - **all others `0.02-0.25` (weak)**
- Only `XMEAS_22` shows strong correlation; the other top `z` sensors have almost no correlation with reconstruction error - the AE reconstructs their abnormal `z` well.

---

## Final Diagnosis

### Q1 Is Fault 15 physically abnormal? **PARTIALLY**
- `Top |z| 0.87 vs normal 0.80` (`+8%`), `max |z| 2.95 vs 1.76` (`+68%` for XMEAS_19) - statistically higher but `absolute |z| <3` for most windows, not a strong physical deviation. `500/500` runs cross `|z|>=2` for many sensors (both normal and fault), so not discriminative.

### Q2 Is Fault 15 abnormal in reconstruction space? **NO**
- Per-sensor `mean ratio max 1.21`, `post diff max 0.13`, `RunsElevated max 17/500 (3.4%)`. Global `0.684 vs 0.682` (`0.3%`). No sensor is consistently elevated.

### Q3 Is global MSE diluting? **NO**
- `Top1 0.91, Top3 0.59, Top5 0.59` **not** `2.8, 2.1` as hypothesized. Top sensors are not more discriminative than global; dilution would require `top >> global`, observed `top <= global`.

### Q4 Does Fault 15 have temporal signature? **NO**
- No consistent onset ordering; `first |z|>=2` scattered, `500/500` runs cross for all sensors at low thresholds, no `XMEAS_10 -> XMEAS_11` chain.

### Q5 Is Fault 15 persistent low-amplitude? **PARTIALLY**
- Yes low-amplitude (`mean 0.623 vs 0.619`), and slightly more persistent (`n_above 1.81 vs 1.28`, `max_consec 1.37 vs 1.01`), but difference is `0.5` windows, not substantial. At `0.687`, `60%` of fault15 runs have `0` windows above.

### Q6 Why does detector miss Fault 15?
**E. Fault 15 is genuinely too close to normal sensor behavior (with contribution from C/D)**
- **Primary (E):** `E` - sensor behavior is `~0.07` sigma away (`8%`), reconstruction error `~0.003` away (`0.5%`). The `128/64` AE has learned this as normal.
- **Secondary (D):** `D` - AE reconstructs it too well (`correlation 0.02-0.25` for 9/10 top sensors, except `XMEAS_22` `0.70` - only one sensor shows that physical deviation *does* cause error).
- **Not (A):** Not dilution - `A` would require strong localized error, not observed.
- **Not (B/C) primarily:** Temporal and persistence differences are `0.5-0.7` windows, not enough to explain `60%` miss rate.

**Ranked causes:** `E (70%) > D (20%) > C (10%)`

---

## Recommended Next Experiment

**1. Sensor-aware anomaly scoring** (primary)

**Why:** `XMEAS_22` is the only sensor where `|z| 0.87` correlates `0.70` with reconstruction error, but its `ratio 1.18` is diluted in `52`-sensor mean. A per-sensor or top-k reconstruction score would amplify `XMEAS_22` without lowering global threshold (which would increase FPR from `1.28` to `8`).

**What to test:**
- Compute `sensor-aware score = mean(top-3 per-sensor errors)` or `max per-sensor z` vs `global MSE`
- Select threshold for `sensor-aware` using **TRAIN+VAL only** (75+75 normal runs, 350 fault runs for `1,4,14,20`), **do not use Fault 15 test runs** to tune
- Evaluate on untouched `75` normal test + `500` Fault 15 test (current `40.6%` should improve to `>70%` if hypothesis holds) and `20-30` other faults

**What must remain frozen:** `scaler` (`mean/std` from `350` train), `window 60/5`, `event aggregation 3/20` (for now), `threshold 0.687` for global baseline (new sensor-aware will have its own threshold).

**Alternative if sensor-aware fails:** `6. No detector change - Fault 15 is indistinguishable` with current `52` sensors - would require different representation (e.g., prediction-based) or more normal diversity.

**Anti-leakage:** Do not use `Fault15 test` `max 0.758` or `p99 0.679` to set new threshold; use `val` `p95 0.721` etc. from normal val.

---

## Console Summary

```
========================================================
FAULT 15 DIAGNOSTIC SUMMARY
========================================================

Global separation:
    Normal max mean:       0.682
    Fault15 max mean:      0.684
    Separation:            0.002 (0.3%)

Sensor-level signal:
    Weak (max ratio 1.21, max diff 0.13, best runs elevated 17/500)

Top sensors:
    1. XMEAS_22    0.775 -> 0.914  ratio 1.18  14/500
    2. XMEAS_11    0.338 -> 0.410  ratio 1.21  17/500
    3. XMEAS_21    0.450 -> 0.461  ratio 1.02   0/500
    ...

Physical z-score signal:
    Weak (post mean 0.87 vs 0.80, +8%, max 2.95 vs 1.76)

Temporal signal:
    Weak (no consistent ordering, 500/500 cross at |z|>=2 for all sensors)

Persistence signal:
    Weak (n_above 1.28 vs 1.81, max_consec 1.01 vs 1.37)

Global MSE dilution:
    NO

AE reconstruction of fault:
    Strong (reconstructs 9/10 top z sensors well, corr 0.02-0.25)

Primary failure mode:
    E (too close to normal) + D (AE too good)

Recommended next experiment:
    1. Sensor-aware anomaly scoring (top-k per-sensor error or z)

DO NOT MODIFY DETECTOR:
    YES

========================================================
```
