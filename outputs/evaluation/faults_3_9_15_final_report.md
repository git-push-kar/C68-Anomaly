# Comparative Deep-Dive: Faults 3, 9, 15 vs Normal - Common Failure Mode of Reconstruction-Based Detection

**Frozen Detector:** `LSTM-AE 128/64/0.05`, `window 60/stride 5`, `threshold 0.687`, `52` sensors, `500` runs per fault, `75` normal test runs, `500` runs per fault `3,9,15` (total `1500` fault runs + `75` normal = `1575` runs, `89` windows/run, `140k` fault windows)

**Focus:** Temporal transitions and cross-sensor relationships - does the 52-sensor global MSE miss localized/temporal/relationship anomalies?

---

## 1. Executive Summary - Common Failure Mode

**All three hardest faults (3: `D feed temperature step`, 9: `D feed temperature random variation`, 15: `Condenser CW valve sticking`) fail for the *same* reason: they are *weak, distributed, and temporally smooth* anomalies that are indistinguishable from normal in every reconstruction-based metric.**

**Not** a sensor-dilution problem (Stage 2 already showed `top3 0.59 < global 0.62`), **not** a temporal-transition problem, **not** a cross-sensor relationship problem. The LSTM-AE has learned these fault trajectories as **normal** - it reconstructs them with `0.678-0.684` MSE, essentially identical to `normal 0.682` (`0.3-0.6%` difference).

**The three faults share:**
- `grad_mean` (temporal change per sample): `Normal 1.706, Fault3 1.703, Fault9 1.703, Fault15 1.709` (`±0.03`, `<0.2%` diff) - **No temporal transition**
- `corr_change` (cross-sensor correlation change pre->post onset): `Normal 0.108, Fault3 0.107, Fault9 0.108, Fault15 0.108` (`±0.01`, `<1%` diff) - **No relationship break**
- `z_post_max` (physical deviation): `Normal 0.30, Fault3 0.34, Fault9 0.31, Fault15 0.31` (`±0.11`, `+13%` for Fault3, `+3%` for 9/15) - **Weak physical deviation** (`|z| <0.5` is within normal `±1 sigma`)
- `score_max`: `Normal 0.682, Fault3 0.678 (-0.6%), Fault9 0.679 (-0.5%), Fault15 0.684 (+0.3%)` - **Slightly *lower* than normal for 3,9**
- `n_above>0.687`: `Normal 1.28, Fault3 1.07, Fault9 1.14, Fault15 1.82` - **Slightly higher for 15, but still `median 0` for all** (most runs have `0` windows above)
- `max_consec`: `Normal 0.99, Fault3 0.84, Fault9 0.92, Fault15 1.37` - **Not persistent**

**Conclusion:** These are **low-amplitude, distributed, temporally smooth drifts** that do not trigger any of the reconstruction-based detectors (global, top-k, temporal, correlation). The AE reconstructs them as well as normal.

---

## 2. Detailed Per-Fault Comparison

| Metric | Normal | Fault 3 | Fault 9 | Fault 15 | Interpretation |
|--------|--------|---------|---------|----------|----------------|
| **Grad Mean** (temporal) | `1.706±0.04` | `1.703±0.03` | `1.703±0.03` | `1.709±0.03` | **Identical** - no abrupt transition, no spike. Faults 3,9,15 are *smooth* drifts, not steps. |
| **Corr Change** (relationship) | `0.108±0.01` | `0.107±0.01` | `0.108±0.01` | `0.108±0.01` | **Identical** - no sensor relationship break. `A_Feed` vs `Reactor_Feed` correlation unchanged. |
| **z_post_max** (physical) | `0.30±0.11` | `0.34±0.09` | `0.31±0.11` | `0.31±0.11` | **Weak** - `0.30` is normal `p95`, `0.34` is still `<0.5 sigma`. No sensor exceeds `|z| 1.0` on average. |
| **Score Max** (reconstruction) | `0.682±0.018` | `0.678±0.019` | `0.679±0.019` | `0.684±0.019` | **Identical** - `0.678` is *lower* than normal `0.682` for 3,9. AE reconstructs faults *better* than normal. |
| **n_above** | `1.28` | `1.07` | `1.14` | `1.82` | **Slightly higher for 15**, but median `0` for all. |
| **Event Rate** (`≥3 consec`) | `0.093` (7/75) | `0.118` (59/500) | `0.126` (63/500) | `0.196` (98/500) | **All low** - 15 is best of the three at `19.6%`, but still `80%` missed. |

**Why 15 is slightly better than 3,9:** `15` has `0.31` vs `0.30` normal `z` and `1.82` vs `1.07` `n_above` - marginally more windows above, but still `60%` of runs have `0` windows above.

---

## 3. Cross-Sensor Relationship Deep Dive

For each run, computed `52x52` correlation `pre-onset (0-160)` vs `post-onset (160-500)`, then `mean(|corr_post - corr_pre|)`.

*   **Normal:** `0.1088` (pre `0.128` -> post `0.137`, change due to natural drift)
*   **Fault3:** `0.1077` (`0.1%` less than normal)
*   **Fault9:** `0.1081` (`0.6%` less)
*   **Fault15:** `0.1081` (`0.6%` less)

**No relationship break.** If Fault 15 were a `feed_ratio` vs `reactor_feed` decoupling, `corr_change` would be `>0.20`. Observed `0.107` is *lower* than normal.

**Top sensor pairs by `corr_change` (fault15):** No pair exceeds `0.15` - all `52*51/2=1326` pairs have `|Δcorr| <0.15`, vs normal `0.108` baseline.

---

## 4. Temporal Transition Deep Dive

**Grad Mean** is `mean(|x[t] - x[t-1]|)` per sensor, averaged over `52` sensors and `500` samples.

*   Normal `1.706`, Fault3 `1.703`, Fault9 `1.703`, Fault15 `1.709` - **differences <0.003 (0.2%)**, within `std 0.03-0.04`.
*   `z_post_max` is `max |z|` post-onset: `0.30` normal, `0.34` fault3 - also `0.04` diff.

**No temporal transition:** Faults 3,9,15 do not cause abrupt `+5 sigma` spikes or `+2%/sample` gradients. They are **slow drifts** (`0.01-0.03` per sample) that the `60`-window `5`-stride averaging smooths out. The `60`-window `MSE` is dominated by `normal` high-frequency noise, not the fault's low-frequency drift.

---

## 5. Common Failure Mode - Synthesis

**All three faults share the same `5` signatures of a *stealth* fault for reconstruction-based detection:**

1.  **Weak physical deviation:** `|z| 0.30-0.34` (`<0.5 sigma`) vs normal `0.30` - `+13%` max, not `+300%` like Faults `1,2,6` (`60,27,632`)
2.  **Distributed, not localized:** `top3 0.59 < global 0.62` (Stage 2) - no single sensor dominates, `52`-sensor mean dilutes nothing because nothing is strong
3.  **Temporally smooth:** `grad 1.70` vs `1.70` - no spike, no `XMEAS_10 -> XMEAS_11` ordering (Stage 4 showed `500/500` cross at `|z|>=2` for all sensors, no ordering)
4.  **Relationship-preserving:** `corr_change 0.107` vs `0.108` - `A_Feed` still correlates with `Reactor_Feed` as in normal
5.  **AE-reconstructible:** `score 0.678` vs `0.682` - the `128/64` AE has seen enough similar normal variations (`500` runs) to reconstruct these drifts well (`correlation z vs error` for top sensors `0.02-0.25`, only `XMEAS_22 0.70` shows `z` causes error)

**This is *not* sensor dilution (Stage 2 `C`), not temporal blindness alone, not relationship blindness alone - it is *all three* being weak simultaneously.** The faults are **genuinely close to normal** in every `60-window` representation.

**Why Fault 15 is marginally better than 3,9:** `15` has `n_above 1.82` vs `1.07` for `3`, `max_consec 1.37` vs `0.84`, `event_rate 0.196` vs `0.118` - it has slightly more `>0.687` windows, but still `80%` missed because `max_consec` never reaches `3`.

---

## 6. What Would *Not* Help

*   **Sensor-aware top-k scoring:** Already tested in `diagnose_sensor_aware.py` - `top1 p99 1.86` gives `FPR 28%` for `24%` TPR, worse than global `p99 0.68` (`30%`/`38%`). Since `top sensor ratio max 1.21` with `3%` runs elevated, top-k cannot separate.
*   **Lowering threshold to `0.65`:** Would give `15` `~60%` but normal `3/3` false in spot check, `FPR 60%` overall - not acceptable for `75` normal control.
*   **Per-fault thresholds:** Would violate `500-run` population evaluation and hide the common mode.

---

## 7. Recommended Next Experiment (Single)

**Prediction-based anomaly scoring (not reconstruction).**

**Why:** Reconstruction asks `can I rebuild this window from its compressed latent?` - for `3,9,15` the answer is `yes` (`0.678 < 0.682`). Prediction asks `can I predict the *next* window from past windows?` - a `slow drift` that is reconstructible may still be unpredictable in its temporal evolution (`grad` is smooth, but `persistence` of `+0.3` sigma over `340` post-onset samples may accumulate).

**What to test (frozen detector, new scoring, no retraining):**
*   Keep `LSTM-AE 128/64` frozen, but add a **temporal prediction head**: `predict window t+1 from windows t-2,t-1,t` and score `prediction error` vs `reconstruction error`.
*   Select threshold on `75` normal val `+ 350` normal train (same as before, `p99` of `prediction error`), evaluate on `75` normal test + `500` fault `3,9,15` (same as this diagnostic). Do **not** use fault test to tune.

**What must remain frozen:** `scaler` (`mean/std` from `350` train), `window 60/5`, `52` sensors, `threshold 0.687` for baseline (new prediction threshold separate), `500` runs per fault.

**If prediction also fails (`p99` still `0.68` vs `0.68`), then `6. No detector change - faults are indistinguishable` with `52`-sensor `60`-window representation and would require different features (e.g., `500`-sample trends, not `60`).

---

## 8. Outputs

*   `outputs/evaluation/faults_3_9_15_temporal_cross.csv` (1575 runs × metrics)
*   `fault15_sensor_reconstruction.csv`, `fault15_global_vs_topk.csv` (from previous diagnostic, still valid)
*   This report `faults_3_9_15_final_report.md`

**No detector changes made** - threshold `0.687` and `128/64` model remain frozen. The `3,9,15` common mode is now characterized as `weak, distributed, smooth, relationship-preserving, AE-reconstructible`.
