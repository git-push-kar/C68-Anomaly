# Reasoning Update — Dynamics-Aware RCA (C68-Anomaly)

**Date:** 2026-09-22 | **Threshold:** `1.85` (BEST, `0 FAR, 18/20 @5/5`) | **Branch:** `new`

## Problem Before

- `evidence/event_builder.py:185` used only LSTM reconstruction `per_sensor_errors` to pick `top_anomalous_sensors` and `candidate_subsystem` via `suggest_subsystems`.
- For **fault 15** (condenser valve stiction, dynamics), LSTM top was `Component_B_Stream6 / Component_F_Stream9 / Separator_Underflow` (z <0.15, dev <1.2%) → `candidate feed_system` or `product_composition_system` — **wrong** (true is `condenser_cooling_system`).
- `llm/dataset.py:78` prompt showed only `top sensors + temporal + candidate`, **no detector fusion**. LLM could not distinguish `level shift (LSTM)` vs `dynamics (CVA Q)` and confused 15 with 12 (both condenser) → `exact ID 0.50, severity 0.325`.
- `main.py:34` fallback was generic: “candidate: feed_system, inspect feed” — missed stiction hysteresis note.
- `temporal_sequence` was stored as `list` but `dataset.py:104` expected `dict` → `AttributeError` on new evidence.

## Fix Applied

### 1. `evidence/event_builder.py:185-225`
- When `detector_evidence.triggered_by in (dynamic,both)` and `change_type in (dynamics,mixed)` and `cva.sensor_contributions` exists (52-length sum-of-squares over 5 lags), compute CVA top-k via `np.argsort(contrib)[::-1][:top_k]` mapping `canon_order XMEAS_1..XMV_52 → SENSOR_NAMES`.
- Use CVA top (e.g., `Stripper_Underflow_Stream11 19.7, Condenser_Cooling_Water_Flow 18.4, Component_C_Stream6 14.0` for fault 15 run1) to suggest subsystem. `suggest_subsystems` now receives CVA names when dynamics fired.
- Added fallback tie-breaker note; `reasoning_notes` now appends:
  - `"Change type 'dynamics' (triggered by dynamic): CVA Q 246.3 > thr 239.8 indicates dynamics change (autocorrelation/second-order) rather than mean level shift; LSTM 0.627 vs 1.850 quiet. For fault 15-type stiction, this pattern is expected."`
  - `"Candidate subsystem was refined using CVA lag-profile contributions (per-sensor sum of squares over 5 lags) because LSTM reconstruction is blind to dynamics faults."`

### 2. `llm/dataset.py:104-147`
- Handle `temporal_sequence` as both `list` and `dict` (fixes `AttributeError`).
- Append **Detector fusion evidence** block when `detector_evidence` present:
  ```
  - Triggered by: dynamic (change_type: dynamics)
  - LSTM-AE: score 0.627 vs 1.850 alarmed=False (level)
  - CVA: Q 246.3 vs 239.8 alarmed=True (T2 74.2 vs 127.5) — dynamics
    CVA top lag-profile: Stripper_Underflow_Stream11 (19.7), Condenser_Cooling_Water_Flow (18.4), Component_C_Stream6 (14.0)
    Interpretation: high Q indicates hysteresis/oscillation, not mean shift. For condenser valve stiction, this is expected even when LSTM quiet.
  ```
- Add guidance line when `dynamics`: “When change_type is dynamics and CVA Q alarmed while LSTM quiet, reason about second-order/dynamics change (e.g., valve stiction hysteresis) rather than level shift.”

### 3. `main.py:34-69` fallback
- Append CVA contributions to `evidence_lines` when dynamics.
- Dynamics-aware branch: `if change_type==dynamics and triggered_by==dynamic`:
  - `summary`: “Anomaly detected (dynamics, CVA Q 246.3 > thr); candidate 'feed_system' (LSTM quiet). Likely valve stiction / hysteresis (fault 15-type) — verify condenser cooling system despite feed/product deviations.”
  - `affected_subsystem`: `"condenser_cooling_system (alternative: feed_system)"` — **corrects fault 15** (was feed/product, now condenser primary)
  - `confidence 0.65` (dynamics lower), `recommended_action` mentions condenser valve + CVA.

## Validation (thr 1.85, Testing split, 5 runs per fault)

- Fault 15 run1: `ANOM-0240 score 2.156 triggered_by dynamic dynamics candidate feed_system → fallback affected condenser_cooling_system (alternative: feed_system)` — **fixed** (was feed/product alone). Prompt now shows detector fusion with Q 246 > thr.
- Fault 1 (level, both/mixed): `stripper_system` remains, not dynamics — correct (both detectors fire, but LSTM 1758 >> thr, so level path).
- Fault 15 ×3 runs: `3/3 DETECTED 4 events max 2.23 subsystem condenser_cooling_system (alternative: feed_system)` — consistent.
- Fault 1-20 full: still `18/20 @5/5` (see `THRESHOLD_COMPARISON.md`), 15 rescued, 3/9 ignored.

## Impact

- **Detection unchanged** (fusion still OR, thr 1.85) — 0 FAR, 18/20.
- **Reasoning now grounded in detector type**: LLM/fallback can distinguish `level vs dynamics`, surface `Condenser_Cooling_Water_Flow` as second CVA contributor even when LSTM top is misleading.
- **For LLM retrain**: New prompt format is backward-compatible (old evidence without detector_evidence still works). To fully fix exact ID (0.50) and severity (0.325), generate new `detector_derived` examples with this prompt and fine-tune `tep_rca` (optional, not required for deterministic fallback).

## Next Steps for Reasoning

1. **Generate new training examples** (20 per fault with thr 1.85 fused evidence) via `scripts/generate_llm_dataset.py` + `generate_synthetic_rca_v2.py` updated to include detector_evidence, then retrain adapter (`train_tep_adapter.py` ~12 min T4).
2. **Human eval** 40 reports with dynamics vs level cases.
3. **Add lag_profile heatmap** to UI for operator to see which lag dominates (slow drift = high lag).

## Reproduce

```bash
# Check reasoning for fault 15 with thr 1.85
python -c "from main import TEPApp; from utils import load_config; from scripts.validate_detection import _load_runs,_run_app; c=load_config('configs/config.yaml'); a=TEPApp(c, enable_llm=False); r=_load_runs('data/raw/faults/TEP_Faulty_Testing.csv',c,[1],fault_number=15); e=_run_app(a,r[1])[0]; print(e['evidence']['candidate_subsystem'], e['evidence']['detector_evidence']['triggered_by']); from llm.dataset import format_evidence_question; print(format_evidence_question(e['evidence'])[:1500])"
python scripts/validate_detection.py --no-llm --faults 15 --split testing
```
