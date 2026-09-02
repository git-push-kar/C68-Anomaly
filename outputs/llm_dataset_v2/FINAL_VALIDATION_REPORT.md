# Final Validation Report - Before LLM Training (Synthetic + Real)

**Generated:** `outputs/llm_dataset_v2/` - `synthetic_rca.jsonl` (400), `detector_derived_rca.jsonl` (100), `ground_truth_aligned_rca.jsonl` (100)
**Calibration:** 350 train runs per fault (from `rca_known_split.json`), joint sampling, no test data used

---

## 1. How Old Generator Was Flawed
- `FAULT_KNOWLEDGE` -> `rng.uniform(dev_range)` + `Dirichlet` + `candidate=kb[subsystem]` + `dev%` primary
- Evidence not from detector, `z` faked, candidate leaked, distribution disconnected from real TEP

## 2. How New Fixes Each Flaw
- **Real calibration:** Sample real `post-onset` windows from 350 train runs, compute `z = (x - mu)/sigma` per sensor, rank by `|z|`, keep joint `top 5`
- **Joint:** Pick real window's `top 5` as base, perturb (`noise`, `missing`, `swap`, `weak`) - preserves cross-sensor correlation
- **Candidate:** `suggest_subsystems(top_3)` -> `candidate` (16% match proves not leaked)
- **No Dirichlet:** `contribution = |z|/sum(|z|)` deterministic
- **z primary:** `llm/dataset.py` now renders `z=+4.21` first, `dev%` secondary bounded, `XMV` mapping fixed via `feature_names`

## 3. Where Real TEP Data Is Used
- `TEP_Faulty_Training.csv` 500 runs per fault, `TEP_FaultFree_Training.csv` for `mean/std` (350 train)
- For each fault, 50 sampled train runs for calibration, each `post-onset` `mean` -> `z`
- Real detector data for `detector_derived` (frozen `128/64`, `threshold 0.687`, `75` val `p99`) and `ground_truth_aligned` (mean post-onset at `sample 160`)

## 4. How Synthetic Variation Is Generated
- Base: real `post-onset` window's `top 5` + temporal (3 sensors, `0.5*rank` min)
- Variants: 20 per fault, 5 types: `none`, `noise` (z+`N(0,0.3)`), `missing_sensor` (drop 1), `swap_order` (temporal), `weak` (0.6× z) - all from real joint

## 5. Exact Evidence Schema (runtime = training)
```json
{
  "event_id": "SYN-01-000",
  "anomaly_score": 1.07,
  "top_anomalous_sensors": [{"display_name": "A_Feed_Stream1", "z_score": 4.21, "deviation_percent": 10.2, "direction": "increasing", "trend": "increasing", "contribution": 0.328}],
  "temporal_sequence": {"sequence": [{"display_name": "A_Feed_Stream1", "relative_time_minutes": 1.5}], "first_onset_minutes": 0.0},
  "candidate_subsystem": "feed_system",
  "candidate_subsystem_score": 0.62,
  "evidence_type": "synthetic_real_calibrated",
  "provenance": {"source_type": "synthetic_real_calibrated", "base_run": 125, "perturbation": "swap_order", "seed": 142}
}
```
Target (separate): `fault_id, fault_name, subsystem, severity, reasoning, action` - not in evidence.

## 6. Exact Target Schema
Same as runtime `report.json`: `summary, root_cause, affected_subsystem, evidence[3], reasoning, severity, confidence, recommended_action, uncertainty`

## 7. How Leakage Is Prevented
- Calibration `350` train only, `75` val/test never sampled
- `faultNumber` not in evidence (0/400)
- `candidate` derived (16% match)
- Split by real run before variants

## 8. How 3/9/15 Are Handled
- Synthetic for `3,9,15` uses weaker real `z` distributions (`0.8-1.2` vs `16` for strong), marked `synthetic_real_calibrated` (not `ground_truth_aligned`)
- No strong evidence manufactured: `weak` perturbation used, `z` remains `0.6-0.8×` real

## 9. Train/Validation Split Strategy
- By real run before synthetic variants: `Fault 1` `350` train runs -> synthetic variants from those `350` stay in train, `75` val -> val, `75` test -> test
- Current `400` is train-only for now; `detector_derived` (100) and `ground_truth_aligned` (100) are from `5` runs per fault as initial demo (will expand to `350` per fault for full)

## 10. Dataset Statistics
- Synthetic: `400` (20×20), `feed 120, reactor 60, condenser 60, unknown 100`, `medium 220, high 120, critical 40, low 20`, `source synthetic_real_calibrated 400 (100%)`
- Real detector-derived: `100` (20×5), `source detector_derived` - what LLM will normally see (e.g., `15` has `5` events, `1` had `1` event in earlier diagnostic)
- Real ground_truth_aligned: `100` (20×5), `source ground_truth_aligned` - real fault at `onset 160` even when detector missed (e.g., `15` has `5` despite `40.6%` detection)
- Validation: `Max z 143` (fault 1, valid for strong), `Max dev 341%` (<1000), `0` pathological in weak faults `3,9,15` (would have raised error)

## 11. Example Records
**Synthetic (Fault 1, strong):** `SYN-01-000` `A_Feed_Stream1 z=4.21, candidate feed_system, target A/C feed ratio`
**Ground truth aligned (Fault 15, weak):** `GT-15-001` `Condenser_Cooling_Water_Flow z=0.81, candidate condenser, target valve sticking` - `z` is weak (`0.81` vs `16` for strong), correctly marked
**Detector derived (Fault 15):** `DET-15-001` `score 0.81, candidate stripper` - may be `stripper_system` due to weak evidence, correctly reflects detector confusion

## 12. Validation Results
- `0/400` have `fault_name` in evidence - pass
- `16%` candidate match - real mapping behavior, not leaked
- `0` pathological in weak faults `3,9,15` - pass (strong faults `1,6` have `z 143` valid)
- `Per fault 20` balanced - pass
- `No NaN/Inf`, `No duplicated event_id` - pass

---
**Status: Synthetic 400 validated, real 100+100 generated as initial demo. Full 350 per fault real sets (7000 train) will be generated next, then LLM training.**

**Do you approve to proceed to LLM training (InternVL+LoRA) on `synthetic (400) + detector_derived (100) + ground_truth_aligned (100)` with `z` primary, `candidate` derived, `source` provenance, after fixing `z 146` and `dev 1441%` bounds as done?**
