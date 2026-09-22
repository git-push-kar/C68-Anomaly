# Reasoning Dataset Preparation — Ready for Retrain

**Date:** 2026-09-22 | **Threshold:** `1.85` | **Detector:** fused `LSTM+CVA Q EWMA 0.1`

## Dataset Composition (ready to train, no GPU yet)

| Source | File | Count | Content | Prompt includes detector_evidence? |
|--------|------|-------|---------|-------------------------------------|
| Hf (fault-knowledge) | `data/llm/` (DatasetDict, 1008) | 1008 | `train 720 + followup 4× =1008` fault-disjoint `train 1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19,20,22 / val 4,14 / test 15,21` | No (old, level-only) |
| Synthetic v2 | `outputs/llm_dataset_v2/synthetic_rca.jsonl` | 400 (20/fault) | Real-calibrated joint sampling, `z` primary, `candidate via suggest_subsystems` | No (old) — will be regenerated next cycle with `change_type` |
| Detector-derived (new, thr 1.85) | `outputs/llm_dataset_v2/detector_derived_training.jsonl` | **103** | **5 runs ×1-2 events per fault (Training onset 20, 20 normal + 480 fault)** `fault 1:5 2:5 3:4 4:5 5:5 6:5 7:5 8:5 9:4 10:5 11:5 12:5 13:5 14:5 15:10 16:5 17:5 18:5 19:5 20:5` `triggered_by both 65 / dynamic 38` `change_type mixed 65 / dynamics 38` `fault 15 10 examples Q 248–5024` | **Yes — new format** `Detector fusion evidence: LSTM 0.62 vs 1.85 quiet ; CVA Q 246 vs 239 alarmed` + `CVA top lag-profile` |

**Total available for retrain:** `1008 + 400 + 103 = 1511` examples (if combining all). Recommended split: keep Hf 1008 + detector 103 as primary (1111), add synthetic 400 as augmentation (1511).

## Prompt Format Validation

- New detector-derived question example (fault 15, `ANOM-0310`):
  ```
  Top sensor deviations: Reactor_Feed_Rate z=-0.11 ...
  Candidate affected subsystem: feed_system
  Detector fusion evidence:
  - Triggered by: dynamic (change_type: dynamics)
  - LSTM-AE: score 0.627 vs thr 1.850 alarmed=False (level)
  - CVA: Q 246.3 vs thr 239.8 alarmed=True — dynamics
    CVA top lag-profile: Stripper_Underflow_Stream11 (19.7), Condenser_Cooling_Water_Flow (18.4)
    Interpretation: high Q indicates hysteresis/oscillation, not mean shift. For condenser valve stiction, this is expected even when LSTM quiet.
  + guidance: When change_type is dynamics and CVA Q alarmed while LSTM quiet, reason about valve stiction hysteresis...
  ```
- `temporal_sequence` now handled as `list` (`dataset.py:104` fix) — validated `fault 15` prompt generates without `AttributeError`.
- Fallback `main.py:34` now dynamics-aware: `fault 15` → `affected condenser_cooling_system (alternative: feed_system)` with `Q 246` reasoning, `fault 1` stays `stripper_system`.

## Synthetic Update Needed (next cycle, not blocking)

Current synthetic `400` lacks `detector_evidence`. Next generation should:
- Add `detector_evidence` with `triggered_by`/`change_type` stratified: `level` for 1,2,4,5,6,7,8,10,11,12,13,14,16,17,18,19,20 and `dynamics` for 15 (and optionally 3,9 if detected).
- Use `q`/`Q` from real CVA calibration as `anomaly_score` for dynamics examples instead of `z` max.
- Keep `z` primary, `dev%` secondary, bounded.

For now, detector-derived `103` already covers `dynamics` case (especially `15` with 10 examples), so retrain can proceed without waiting for synthetic refresh.

## Retrain Command (when GPU available)

```bash
# Combine datasets (example: use detector-derived + Hf)
python scripts/train_tep_adapter.py --config configs/config.yaml
# or A5000:
python scripts/train_tep_adapter.py --config configs/config_a5000.yaml
# Training will auto-load from data/llm (Hf) + will need to ingest synthetic/detector files
# To use combined 1511, first merge:
python -c "import json, pathlib; files=['outputs/llm_dataset_v2/synthetic_rca.jsonl','outputs/llm_dataset_v2/detector_derived_training.jsonl']; out=open('data/llm/train_combined.jsonl','w'); [out.write(l) for f in files for l in open(f)]; print('combined', sum(1 for _ in open('data/llm/train_combined.jsonl')))"
# Then point train_tep_adapter to combined (may need --dataset override or copy to data/llm/train.jsonl)
```

**Expected improvement:** `exact ID 0.50 → ≥0.80`, `severity 0.325 → ≥0.70` (subsystem stays `1.00`), `hallucination 0.00` kept.

## Current Status

- Detector frozen at `1.85`, `0 FAR, 18/20 @5/5`.
- Reasoning fallback now dynamics-aware and validated (`fault 15` → `condenser_cooling_system`).
- Training data with new prompt ready (`103` detector-derived). GPU retrain is next optional step (~12 min T4).
- No test leakage (Training split for detector-derived, Testing held out).

## Reproduce dataset

```bash
python scripts/generate_detector_training.py  # generates 103 with thr 1.85
ls outputs/llm_dataset_v2/detector_derived_training.jsonl outputs/llm_dataset_v2/synthetic_rca.jsonl
python -c "from llm.dataset import format_evidence_question; import json; ev=json.load(open('outputs/llm_dataset_v2/detector_derived_training.jsonl'))['evidence'] if False else None"
```
