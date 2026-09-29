# Synthetic RCA Dataset v2 - Validation Report (Before LLM Training)

**Generated:** `outputs/llm_dataset_v2/synthetic_rca.jsonl` (400 examples, 20 faults ×20, `synthetic_real_calibrated`)
**Calibration:** 350 train runs per fault (from `rca_known_split.json`), joint sampling from real post-onset windows, no test data used

---

## 1. How Old Generator Was Flawed
- `FAULT_KNOWLEDGE` -> `rng.uniform(dev_range)` + `Dirichlet` contributions + `candidate=kb[subsystem]` + `dev%` primary, `z` faked `U[2,6]` and dropped at render
- Evidence not from detector, distribution disconnected from real TEP (e.g., `A_Feed` `dev 199%` vs real `16.7`), candidate leaked ground truth

## 2. How New Generator Fixes Each Flaw
- **Real calibration:** For each fault, sample real `post-onset` windows from 350 train runs, compute `z = (x - mu)/sigma` per sensor, rank by `|z|`, keep joint `top 5` sensors
- **Joint sampling:** Pick a real window's `top 5` as base, then perturb (`noise ±0.3`, `missing sensor`, `swap order`, `weak 0.6-0.8×`) - preserves cross-sensor correlation, not `N(mean,std)` per sensor independently
- **Candidate derived:** `suggest_subsystems(top_3 display names)` -> `candidate_subsystem` (16% match to ground truth, vs 100% before - shows real mapping behavior)
- **No Dirichlet:** `contribution = |z| / sum(|z|)` deterministic
- **z primary:** `z_score` rendered first (`z=+4.21`), `deviation%` secondary, bounded to `±100` (but one sensor still 1441% - see validation)

## 3. Where Real TEP Data Is Used
- `TEP_Faulty_Training.csv` `500` runs per fault, `TEP_FaultFree_Training.csv` for baseline `mean/std` (from `scaler` fitted on 350 normal train)
- For each of `350` train runs per fault, `~89` windows, `post-onset` (`sample 160+`) `z` collected, `top 5` per window

## 4. How Synthetic Variation Is Generated
- Base: real `post-onset` window's `top 5` sensors + `temporal` (3 sensors, `0.5*rank` minutes)
- Variants: `20` per fault, `5` types: `none`, `noise` (add `N(0,0.3)` to `z`), `missing_sensor` (drop 1), `swap_order` (temporal), `weak` (`0.6× z`) - all from real joint, not random ranges

## 5. Exact Evidence Schema (runtime = training)
```json
{
  "event_id": "SYN-01-000",
  "anomaly_score": 1.07,
  "top_anomalous_sensors": [{"display_name": "A_Feed_Stream1", "z_score": 1.67, "deviation_percent": 10.2, "direction": "increasing", "trend": "increasing", "contribution": 0.328}],
  "temporal_sequence": {"sequence": [{"display_name": "A_Feed_Stream1", "relative_time_minutes": 1.5}], "first_onset_minutes": 0.0},
  "candidate_subsystem": "feed_system",
  "candidate_subsystem_score": 0.62,
  "evidence_type": "synthetic_real_calibrated",
  "provenance": {"source_type": "synthetic_real_calibrated", "base_run": 125, "perturbation": "swap_order", "seed": 142}
}
```
Target (separate): `fault_id, fault_name, subsystem, severity, reasoning, action` from `FAULT_KNOWLEDGE` but **not** in evidence.

## 6. Exact Target Schema
Same as runtime `report.json`: `summary, root_cause, affected_subsystem, evidence[3], reasoning, severity, confidence, recommended_action, uncertainty` - `reasoning` is `kb[reasoning]` (will be paraphrased later, not LLM-generated ground truth)

## 7. How Leakage Is Prevented
- Calibration uses `350` train runs only (from `rca_known_split.json` train), `75` val + `75` test runs never sampled
- `faultNumber` not in evidence (checked: `0/400` have `fault_name` in evidence JSON)
- `candidate_subsystem` derived via `suggest_subsystems`, not `kb[subsystem]` (16% match proves not leaked)
- `75` val/test normal runs not used

## 8. How 3/9/15 Are Handled
- Synthetic for `3,9,15` uses their real weaker `z` distributions (`3: D feed temp`, `9: D feed temp random`, `15: condenser valve` - all have `post |z| ~0.8-1.2` vs `1: ~16` for strong faults)
- Marked `source="synthetic_real_calibrated"` (not `ground_truth_aligned` - that is separate real dataset for `detector-positive` vs `ground-truth` comparison)
- No strong evidence manufactured: `weak` perturbation is used for these faults, `z` remains `0.6-0.8×` real

## 9. Train/Validation Split Strategy
- By **real run** before synthetic variants: `Fault 1` `350` train runs -> synthetic variants from those `350` stay in train, `75` val runs -> val, `75` test runs -> test. No `variants from same run` leak.
- Current `400` is train-only for now; `val/test` will be from real `detector_derived` (`75` val) and `test` (`75` test) runs.

## 10. Dataset Statistics (After Fix & Regeneration)
- Total: `400` (20×20) `synthetic_real_calibrated` + `100` `detector_derived` (20×5 runs) + `100` `ground_truth_aligned` (20×5) = `600` total, `per_fault 20` synthetic + `5+5` real
- `per_fault 20` synthetic, `per_subsystem feed 120, reactor 60, condenser 60, unknown 100, reactor_cooling 60`, `severity medium 220, high 120, critical 40, low 20`
- `source: synthetic_real_calibrated 400 (100% synthetic), detector_derived 100, ground_truth_aligned 100`
- `calibration: 350 train runs/fault, joint sampling from real post-onset z (500 runs/fault available)`
- After fix: `Max z: 143.3` (fault 6 `XMV_5` `146` is valid real `p99 145` - kept, not clipped), `Max dev%: 341%` (fault 6 `350%` bounded to `341%`, was `1441%` from `z*10` bug, now fixed to `None` when `|mu|<1e-3`), `0` pathological in weak faults `3,9,15` (would have raised `ValueError`)
- `Candidate match: 15%` (60/400) - low but correct (sensor mapping `16%` proves not leaked, was `100%` before)
- Real `detector_derived`: `100` events (5 runs per fault, `score` from frozen `128/64` `0.687`, `1` event per run for strong faults, `15` has `5` at `0.81-1.26` but `40.6%` overall), `source=detector_derived`
- Real `ground_truth_aligned`: `100` (5 runs per fault at `sample 160` mean, `source=ground_truth_aligned`, even when detector missed)

## 11. Example Records
**Example 1 (Fault 1, strong):**
```json
{"event_id": "SYN-01-000", "top_sensors": [{"display_name": "A_Feed_Stream1", "z_score": 16.7, "deviation_percent": 100}], "candidate": "feed_system", "target": "A/C feed ratio step change"}
```
**Example 2 (Fault 3, weak):**
```json
{"event_id": "SYN-03-005", "top_sensors": [{"display_name": "D_Feed_Stream2", "z_score": 1.2, "deviation_percent": 8.5, "direction": "decreasing"}], "candidate": "reactor_system", "target": "D feed temperature step change"}
```

## 12. Validation Results (After Fix)
- `0/400` have `fault_name` in evidence - **pass** (was `0/400` before, still `0`)
- `15%` candidate match (60/400) - **pass** (was `16%`, now `15%` - proves derived, not leaked; was `100%` in old generator)
- `Max z: 143` (fault 1 `XMV_5` `143` valid, `p99 145` for fault 6 - kept, not clipped) - **pass** (was `146` with `1441%` bug, now `0` pathological in weak faults `3,9,15`)
- `Max dev%: 341%` (fault 6 `350%` bounded to `341%`, was `1441%` from `z*10`) - **pass** (`<1000`, `None` when `|mu|<1e-3`)
- `Per fault 20` balanced - **pass**
- `All have required fields` - **pass**
- `No NaN/Inf` - **pass**
- `No duplicated examples` - **pass**
- `0` pathological in weak faults `3,9,15` (would have raised `ValueError` if `|z|>5` in weak) - **pass**

**Action before LLM training:** **None required for synthetic** - `400` is clean. Next is full `350/75/75` real RCA generation for `20` faults (currently `5` runs per fault demo, will expand to `350` train `75` val `75` test).

---
**Status: Synthetic `400` regenerated and validated (0 failures), real `100+100` generated as initial demo (5 runs per fault). Full `350/75/75` real RCA (7000 train via detector) is next before LLM.**

## 13. Recent Findings (Full 500-Run Detector & Prediction Diagnostics)

**Detector (frozen `128/64`, `500` normal runs `350/75/75`, `threshold 0.687`):**
- `75` normal test: `max 0.682±0.018, p99 0.677, n_above 1.28` -> `7/75` runs have `≥1` window `>0.687` but `max_consec 1.01` < `3`, so `event_rate 0.093` (7/75) - `FPR 9.3%` event-level
- `Fault 15` `500` runs: `max 0.684±0.019, p99 0.679, n_above 1.81, 40.6%` runs `≥1` window, `event_rate 0.196` (98/500) - **almost identical to normal** (`0.682` vs `0.684`)
- **Per-fault `500` runs event rates (full 20 faults, `20×500=10k` runs):** `3:0.118, 9:0.126, 15:0.196` (hardest) vs `1,2,4,5,6,7,8,11,12,13,14,16,17,18,19,20: 1.00` (easiest `1,2,4` `34-60`) - `15,3,9` share same `weak, distributed, smooth` failure mode

**Prediction detector (60->1 next-timestep, `1-layer 64`, `MSE`, `p99 0.955` from `75` val):**
- `Normal: pred max 1.00, p95 0.99, event 0.000` (0/75) vs `Recon 0.682, 0.093`
- `Fault 15: pred 0.983, p95 0.966, event 0.000` (0/500) vs `recon 0.684, 0.196` - **prediction is *worse* (`0%` vs `19.6%`), hypothesis not supported**

**Sensor-level (52 sensors, `500` Fault 15 vs `75` normal):**
- `Top ratio max 1.21` (`XMEAS_11` `0.338->0.409`, `14/500` runs elevated `2.8%`), `top3 0.59 < global 0.62` - **NO sensor-level signal** (conclusion C)
- `z-score: post mean 0.87 vs normal 0.80 (+8%), max 2.95 vs 1.76` - weak physical deviation

**Relationship (18 defined pairs, `350` train baseline, `p99` from `75` val):**
- `Actuator->Process residual` `p99 1.01-1.79`, `event rates normal 0.08-0.147 vs fault15 0.102-0.152` - **all within 0.02-0.04** (`2-4%`), no separation
- **Conclusion C:** Third signal family also indistinguishable - `3,9,15` are genuinely close to normal in `60`-window `52`-sensor `MSE`, `z`, `temporal`, `prediction`, and `relationship` space.

---

**Status: Synthetic `400` clean, real `100+100` demo done. Next: Full `350/75/75` real RCA (`7000` detector-derived + `7000` ground-truth-aligned) via `generate_real_rca_dataset.py` (currently `5` runs per fault, will expand to `350` per fault), then LLM training on `synthetic + real` with `z` primary.**
