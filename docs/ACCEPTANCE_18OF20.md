# Acceptance — 18/20 Final (Faults 3,9 Documented Limitation)

**Date:** 2026-09-29 | **Branch:** `C68-Anomaly\new` | **Threshold:** `1.85` frozen | **Detector:** `LSTM 128/64 + CVA r=67 Q EWMA 0.1 OR fused`

## Accepted scope

`18/20 faults @5/5 runs (Testing, onset 160)`, normal `0/10 PASS`. Faults 3 (D feed temp step) and 9 (D feed temp random) at `0/5` — accepted as benchmark limitation, not a bug.

Evidence:
- `data/processed/c68_full_20_validation_1.85.json` — 18/20 @5/5, 15 @5/5 (8 ev, Q 246>239)
- `data/processed/c68_full_20_validation.json` (0.687) — same 18, 3/9 @1/5 blips, 2/10 FAR
- `outputs/llm_dataset_v2/llm_20x1_metrics.json` — LLM 20x1: detected 18/20, subsystem 0.722, exact 0.667, sev 0.722 (on detected); 0.65/0.60/0.65 on 20-denominator
- `docs/THRESHOLD_COMPARISON.md`, `docs/FINAL_REPORT.md`

## Why 3,9 excluded

Control-loop compensation: mean shift <0.15σ, variance ratio ≈1.018, `grad/corr/z` identical to normal, scores below normal (`0.678 vs 0.683`). Literature (Russell/Chiang/Braatz 2000, Yin 2012) reports 2–4% for unsupervised methods on standard 52 vars. Lowering threshold to catch them explodes FAR (9.3%→60%). Documented, not pursued.

## Reasoning status

- Fallback stabilized 2026-09-29: CVA contributions averaged over event windows (`main.py` aggregation) + physics-weighted condenser override (`event_builder.py`) + lag-energy trend (`dominant_lag`, `slow_drift_ratio`). Fault 15 Testing runs 1-3 now `3/3 condenser_cooling_system` (was feed/purge flip).
- LLM retrained `1111 ex, A5000 BF16 3ep, 62.9MB`: fault 15 → `Condenser cooling water valve sticking / condenser_cooling_system / critical 0.74` with hysteresis reasoning (was `D feed / reactor` pre-retrain). Fault 1 → `feed_system` (corrects fallback `stripper`).
- Known residual misses (20x1): 6→unknown, 7 fine-grained (C header→A/C, same subsystem), 8→unknown, 13→unknown, 16 over-specific, 18 over-specific. Subsystem 0.722 reflects these; exact 0.667 below 0.80 target — accepted for demo, next cycle would add strong-fault/unknown discrimination samples.

## Production hardening (this release)

- `api/server.py`: `/api/health` (threshold, CVA Q thr, LLM flag, accepted scope, latency mean/max), `/api/model-info` (detector + dynamic thresholds + fusion + training summary + metrics), per-sample `latency_ms` tracking.
- `ui/app.py`: sidebar shows `thr 1.85 + CVA Q thr + fusion`, accepted-scope caption; events tab shows `triggered_by/change_type`, `LSTM vs CVA Q`, `lag-energy`, `slow-drift`, last 3 reasoning notes.
- Fallback remains primary when LLM offline; LLM primary when loaded. Both validated end-to-end (`test_end_to_end.py` fault 1 + 15).

## Reproduce final claim

```bash
python scripts/validate_detection.py --no-llm --faults 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --fault-runs 1 2 3 4 5 --normal-runs 1 2 3 4 5 6 7 8 9 10 --split testing
python -u scripts/score_llm_20.py --runs 1 --faults 1 2 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 --out outputs/llm_dataset_v2/llm_20x1_metrics.json
python scripts/test_end_to_end.py --config configs/config_a5000.yaml --inject-at 200 --fault-number 15 --fault-run 1
```
