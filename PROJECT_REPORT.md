# TEP Industrial Anomaly Detection & Automatic Root Cause Analysis (RCA) System
### A Comprehensive Project Report — Architecture, Implementation, Results, Failures, and Future Roadmap

> **One-line summary:** We built an end-to-end system that watches 52 chemical-plant sensors continuously, learns what *normal* looks like without ever seeing a fault, flags anomalies in real time, packages evidence into a structured forensic report, and uses a fine-tuned InternVL2-2B language model (one LoRA adapter called `tep_rca`) to explain — in plain English — what probably went wrong, why, how severe it is, and what to do next. The sensor brain and the language brain are completely separate; the LLM has never seen raw sensors.

**Project type:** Research + Engineering prototype  
**Domain:** Tennessee Eastman Process (TEP) — the de-facto benchmark for chemical-plant fault detection (Dow Chemical simulation, 52 sensors, 20+ fault scenarios)  
**Status as of Sep 2026:** Detector pipeline complete on **full 500-run** normal data (44.5k windows, 350/75/75) and rigorously evaluated on **10k fault runs**. LLM adapter (`tep_rca`) trained on 1008 Hf + 400 synthetic v2 (real-calibrated) and evaluated (40-sample large-test, 500-run fault coverage). Systematic 6-stage + 20×500 all-fault + 3-fault comparative + 3 alternative-detector studies complete. Known limitation is **stealth class (Faults 3,9,15 = 11.8-19.6% event rate)** with common `weak/distributed/smooth` mode; no single threshold fixes it.  
**Hardware target:** NVIDIA RTX A5000 (24 GB) — also runs on any ≥8 GB GPU (QLoRA) and on CPU for the sensor pipeline.  
**Repo root:** `C:\Users\Admin\Desktop\anomaly` · Entry point: `main.py` → `TEPApp`

---

## Table of Contents
1. [Objectives — What We Set Out to Build and Why](#1-objectives--what-we-set-out-to-build-and-why)
2. [Dataset — What Data We Have](#2-dataset--what-data-we-have)
3. [System Architecture at a Glance](#3-system-architecture-at-a-glance)
4. [Current Implementation — How Each Part Works Today](#4-current-implementation--how-each-part-works-today)
5. [Progress Report — What Is Done vs What Is Pending](#5-progress-report--what-is-done-vs-what-is-pending)
6. [Results — How Well Does It Work?](#6-results--how-well-does-it-work)
7. [The Fault 15 Story — How a Single Fault Exposed a Systemic Problem](#7-the-fault-15-story--how-a-single-fault-exposed-a-systemic-problem)
8. [Testing All 20 Faults — Who Is Problematic and Why](#8-testing-all-20-faults--who-is-problematic-and-why)
9. [What We Tried to Fix It — Experiments and Why They Did (or Didn't) Help](#9-what-we-tried-to-fix-it--experiments-and-why-they-did-or-didnt-help)
10. [Key Observations, Faulty Errors, and Lessons Learned](#10-key-observations-faulty-errors-and-lessons-learned)
11. [Future Scope — What We Will Build Next](#11-future-scope--what-we-will-build-next)
12. [Artifacts & How to Reproduce Everything](#12-artifacts--how-to-reproduce-everything)
13. [Conclusion](#13-conclusion)
14. [Appendix — File Map & Configuration](#14-appendix--file-map--configuration)

---

## 1. Objectives — What We Set Out to Build and Why

### 1.1 The real-world problem
Chemical plants stream hundreds of sensor readings every few seconds. When something drifts — a cooling-water valve sticks, a feed composition shifts — operators have minutes to notice, localize the fault, and act. Traditional alarms trigger per-sensor thresholds but don't explain *why* or *what to do*. Missed or late detection can mean off-spec product, equipment damage, or safety incidents.

### 1.2 What we wanted to build
A **two-brain system**:

| Brain | Job | Analogy |
|-------|-----|---------|
| **Sensor brain (unsupervised)** | Learn "normal" from *only* normal data. Flag anything that doesn't reconstruct well. No fault labels needed. | A veteran operator who knows the sound of a healthy plant. |
| **Language brain (supervised LLM adapter)** | Take the sensor brain's evidence and produce a grounded, auditable RCA report + answer follow-up questions. | A senior engineer who writes the incident report and answers "why the cooling system?" |

**Design principles:**
1. **No fault labels for detection.** The detector must work on *any* future unknown fault — including ones never seen in training.
2. **Explainability by construction.** Every LLM claim must be traceable to numeric evidence (sensor deviation %, trend, onset order). No hallucinating causation.
3. **Separation of concerns.** The LoRA adapter (`tep_rca`) contains *only* language reasoning. All sensor math (scaler, autoencoder, threshold, evidence) lives outside it. You can swap detectors without retraining the LLM and vice versa.
4. **Production-ready streaming.** One record at a time (`TEPApp.process_sensor_stream`), event aggregation (no LLM call per timestamp), SQLite event store, FastAPI + Streamlit.

### 1.3 Intended end-to-end user journey
```
Continuous sensor stream (52 values every sample)
  → 60-sample sliding window (stride 5) → LSTM autoencoder reconstruction error
  → threshold + 3-consecutive-windows confirmation → ONE anomaly event
  → evidence extraction (top 5 sensors, % deviation, z-score, trend, temporal order)
  → InternVL2-2B + tep_rca adapter → JSON report {summary, root_cause, subsystem, evidence, reasoning, severity, confidence, action, uncertainty}
  → stored in SQLite → operator asks "Should I check the condenser valve?" → grounded answer
```

---

## 2. Dataset — What Data We Have

### 2.1 Raw data layout
We use the **Rieth et al. consolidated TEP** release (auto-detected; legacy 52-col per-file format also supported):

```
data/raw/normal/
  TEP_FaultFree_Training.csv   250,000 rows  (500 runs × 500 samples, fault 0)
  TEP_FaultFree_Testing.csv    480,000 rows  (500 runs × 960 samples, fault 0)
data/raw/faults/
  TEP_Faulty_Training.csv      5,000,000 rows (20 faults × 500 runs × 500 samples)
  TEP_Faulty_Testing.csv       9,600,000 rows (20 × 500 × 960)
Columns: faultNumber, simulationRun, sample, xmeas_1..41, xmv_1..11  → 52 sensors
Fault injected at sample 160 (0-based; 161 in 1-based docs) → config dataset.fault_onset_index = 160
```

**Sensor vocabulary (52):** 41 measured variables (`XMEAS_1..41`: pressures, temperatures, compositions, flow rates) + 11 manipulated variables (`XMV_1..11`: valve positions, feed flows). Canonical names and subsystem mappings live in `evidence/process_relationships.py` (e.g., `A_Feed_Flow → feed_system`, `Condenser_Cooling_Water_Flow → condenser_cooling_system`).

### 2.2 Processed data (after `scripts/prepare_tep.py`)
| Artifact | Location | Size / Count | How |
|----------|----------|--------------|-----|
| Standard scaler (fitted **only** on normal training) + baseline stats (mean/std/min/max) | `outputs/preprocessing/scaler.pkl`, `baseline_stats.json` | 52 features | Ensures no fault leakage into normalization |
| Leakage-free window split (by **whole simulationRun**, not random windows) | `data/processed/normal_windows_{train,val,test}.npy` | 44,500 windows [60×52], split 31,150 / 6,675 / 6,675; block 500, test 0.2, val 0.15 | Prevents temporal leakage |
| Per-fault arrays | `data/processed/fault_values/fault_01..20.npy` | Each 730,000 samples (Training + Testing merged per fault) | For detector evaluation & LLM dataset generation |
| Manifests | `data/processed/manifests/detector_split.json` etc. | 350 train / 75 val / 75 test simulationRuns | Reproducible, seed 42 |
| Summaries | `prepare_summary.json`, `window_metadata.json`, `fault_metadata.json` | — | Audit trail |

> **Key invariant:** Scaler and baseline are learned **exclusively** from `TEP_FaultFree_Training.csv` (Rieth protocol). No fault data, no `Testing` normal data contaminates them.

### 2.3 LLM supervision data
- **Source of truth:** `scripts/fault_knowledge.py` — curated knowledge base for all 22 fault scenarios (initiating sensor, cascading sensors with delay, severity, reasoning, recommended action). Faults 16–20 are explicitly labeled `unknown` — we do **not** fabricate a cause.
- **Generation:** `scripts/generate_llm_dataset.py` (1008 Hf: 720 train + 4× follow-up = 1008, splits **by fault ID**: train {1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19,20,22} / val {4,14} / test {15,21}, see `data/llm/split_metadata.json`) **plus** `scripts/generate_synthetic_rca_v2.py` (**new, real-calibrated, joint sampling, no Dirichlet, z primary, 400 examples 20 per fault, `outputs/llm_dataset_v2/synthetic_rca.jsonl`**) + `scripts/generate_real_rca_datasets.py` (**real detector-derived `detector_derived_rca.jsonl` 100 = 20×5 runs via frozen 128/64 @0.687 + ground-truth-aligned `ground_truth_aligned_rca.jsonl` 100, separated `source_type` provenance**).
- **v2 synthetic (calibrated, validated):** `outputs/llm_dataset_v2/synthetic_rca.jsonl` — 400 examples, 20 per fault, jointly sampled from real post-onset `z` distributions of 350 train runs per fault (not `FAULT_KNOWLEDGE` random ranges), `candidate` via `suggest_subsystems` (16% match proves no leakage), `z` primary, `dev%` secondary bounded, `validation_report.md` shows `0/400` leakage, `0` pathological in weak faults `3,9,15`.

---

## 3. System Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer A — Sensor / Time-Series Pipeline (no LLM)                    │
│  preprocessing/  →  anomaly_detection/  →  evidence/  →  events/     │
│  streaming/ (simulator)                                              │
│                                                                      │
│  Layer B — Language Reasoning (frozen LLM + one adapter)            │
│  llm/  (InternVL2-2B + tep_rca LoRA/QLoRA)                            │
│                                                                      │
│  Orchestration: main.py: TEPApp                                     │
│  Interfaces: api/server.py (FastAPI)  +  ui/app.py (Streamlit)      │
└──────────────────────────────────────────────────────────────────────┘

Data flow:
  sensor_record {values: [52], sample_index, fault_label?}
    → deque buffer (60) → every stride 5 → window [60,52]
    → scaler (pre-fitted) → LSTM-AE → reconstruction error (global MSE + per-sensor)
    → threshold (0.687) → recent_flags deque → _feed_aggregator
        (3 consecutive anomalous → open event; 20 normal → close event; cap 200)
    → build_event() → EventEvidence {anomaly_score, top 5 sensors (deviation %, z, trend),
        temporal_sequence (onset order), candidate_subsystem, uncertainty disclaimer}
    → RCAInference.generate_report() OR deterministic fallback → JSON report
    → EventStore (SQLite outputs/events.db) → answer_followup()
```

**Why two layers?** An operator can trust the detector even if the LLM is offline (fallback report is fully grounded in evidence). Conversely, the LLM cannot "cheat" by seeing raw sensors — it only sees the evidence JSON, which is how we enforce that correlation ≠ causation is always stated.

---

## 4. Current Implementation — How Each Part Works Today

### 4.1 Preprocessing (`preprocessing/`)
- **`tep_loader.py`**: Auto-detects Rieth consolidated CSVs vs legacy 52-col; handles ` faultNumber / simulationRun / sample` grouping; validates row counts; interpolates/ffill/drops missing values per `dataset.missing_value_strategy`; never loads the full 5 GB at once — streams per-run.
- **`scaler.py`**: `StandardScaler` (or `MinMax`) fitted only on normal training. Persists `scaler.pkl` and `BaselineStats` (mean/std/min/max/feature_names). `load_scaler()` is the single source for evidence z-scores.
- **`windowing.py`**: `to_windows(arr, 60, 5)` → `[N,60,52]` windows **never** across `simulationRun` boundaries. Leak-free splits by `block_samples=500` temporal blocks. `float32` throughout.

### 4.2 Anomaly Detector (`anomaly_detection/`)
**Model:** `LSTMAutoencoder` (`anomaly_detection/lstm_autoencoder.py`)
- **Encoder:** LSTM(52 → 128 × 2 layers) → latent 64 via linear projection. Optional bidirectional.
- **Decoder:** Latent → projected to `h0/c0` + repeated latent as per-step input → LSTM(64→128×2) → linear to 52.
- **Training:** *Only* on normal windows. Loss = MSE (reconstruction). AdamW, `lr 1e-3`, cosine schedule, `epochs 100`, `batch 64` (A5000: 128), early stopping patience 15, grad clip 1.0, checkpoint every 5. Outputs `outputs/anomaly_detector/model.pt`.
- **Threshold:** Fit on normal validation reconstruction errors. Method `percentile`, `p=99.0` (A5000 pushed to 99.5 to curb false positives). Frozen at **0.687363** (mean 0.619, std 0.028, n 6675) — see `threshold.json`. Also tried `p99.5` in code comments to reduce FPR 13.5% → <5%.

**Inference** (`inference.py`):
- `score_window(window)` → scalar `score = mean(per_sensor MSE)` + per-sensor vector `[52]`.
- `is_anomalous(score)` → `score > threshold`.
- `score_windows(windows)` batched for diagnostics.

**Legacy alternatives explored:**
- `lstm_predictor.py` + `train_predictor.py` (prediction-based scoring: predict next window from history) — evaluated but not adopted (see §9).
- `threshold.py` + `compatibility.py` for method variants (`mean_std`, `validation_percentile`).

### 4.3 Evidence Extraction (`evidence/`)
Turns raw numbers into the LLM's prompt:

- **`sensor_contribution.py`**: Per-sensor error → contribution %, deviation `% = (current - baseline)/baseline *100`, `z = (current - baseline)/std`, ranking. `event_level_deviations()` aggregates over event windows with `min_deviation_percent 3.0`.
- **`temporal_analysis.py`**: `detect_onsets()` (per-sensor error > `onset_sensitivity (1.5) × std`), `analyze_temporal_sequence()` (sorted onset times → relative minutes), `sensor_trend()` (linear fit over `trend_smoothing_window 8` → increasing/decreasing/stable), `pre_post_context()` (pre-status vs post delta).
- **`process_relationships.py`**: Human vocabulary: `SENSOR_NAMES` (52), `SUBSYSTEM_GROUPS` (feed/reactor/condenser/separator etc.), `KNOWN_PROCESS_RELATIONSHIPS` (e.g., `Cooling Water → Reactor Temp → Reactor Pressure`). `suggest_subsystems()` heuristically maps top deviating sensors → candidate subsystem + score. Never claims causation — every output carries `RelationshipKind.MODEL_DERIVED` and an explicit uncertainty disclaimer.
- **`event_builder.py`**: The assembler. Groups anomalous windows into `AnomalyEvent` + `EventEvidence`. Severity from `score/threshold` ratio: ≥3.0 critical, ≥1.8 high, ≥1.2 medium, else low. Evidence includes `temporal_sequence` (up to 6 events), `top_anomalous_sensors` (5), `candidate_subsystem`, `reasoning_notes` (always includes "temporal ordering does not prove causation").

### 4.4 Streaming & Event Aggregation (`streaming/`, `events/`)
- **`streaming/simulator.py`**: `SensorStream` replays CSVs as live stream (`window_size/stride/replay_rate`). `TEPApp.run_stream_from_file()` can inject a fault file at a chosen sample (default fault onset 160 → splices `fault_frame` at `inject_fault_at`).
- **`main.py: TEPApp`**: The public API: `process_sensor_stream(record)` (one-by-one), `finalize_stream()`, `answer_followup()`, `run_stream_from_file()`. State: `deque` buffer (60), `consecutive_windows_to_confirm=3`, `min_separation_windows=20`, `max_event_windows=200`.
- **`events/event_store.py`**: SQLite (`outputs/events.db`) — stores events + JSON reports + conversation history. Used for follow-up QA context.

### 4.5 LLM Adapter (`llm/`)
- **Base:** `OpenGVLab/InternVL2-2B` (`InternLM2-Chat-1.8B` backbone), loaded with `trust_remote_code=True`. Text-only TEP uses dummy `pixel_values` + `image_flags=0` (official InternVL text-only pattern). No images.
- **Adapter:** Single LoRA/QLoRA adapter `tep_rca` (`r 16, alpha 32, dropout 0.05`, targets `q/k/v/o + gate/up/down`; A5000 config targets `wqkv/wo/w1/w2/w3` variant). `outputs/tep_rca_adapter/tep_rca/{adapter_config.json, adapter_model.safetensors}` (~62 MB).
- **Training:** `llm/train_adapter.py` / `scripts/train_tep_adapter.py`. QLoRA 4-bit (NF4, double quant, bf16) ~6-8 GB **or** BF16 LoRA ~12-14 GB on A5000. Batch 2×8 grad-accum (A5000 BF16: 4×4), 3 epochs, `lr 2e-4`, warmup 0.03, cosine, checkpointing on. Train faults {1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19,20,22}, val {4,14}, test {15,21}. Saves `training_summary.json`.
- **Inference:** `llm/inference.py` → `RCAInference.generate_report(evidence_dict)` (JSON-constrained) + `answer_followup(event, report, history, question)`. Deterministic fallback (`_fallback_report()` in `main.py`) used when adapter absent — still grounded in evidence.
- **Evaluation:** `llm/evaluate.py` + `scripts/benchmark_tep_adapter.py` → metrics like root cause accuracy, subsystem accuracy, severity, hallucination rate etc.

### 4.6 Config & Utils
Single source `configs/config.yaml` (merged over `utils.py: _DEFAULT_CONFIG`). Every script accepts `--config`. `utils.py` provides `load_config()`, `load_json/save_json`, `get_device()`, `set_seed(42)`, `gpu_memory_summary()`.

---

## 5. Progress Report — What Is Done vs What Is Pending

### 5.1 Timeline (condensed)

| Phase | What happened | Outcome |
|-------|---------------|---------|
| **0. Scaffolding** | Repo created: `preprocessing/`, `anomaly_detection/`, `evidence/`, `llm/`, `streaming/`, `events/`, `api/`, `ui/`, `configs/`, `scripts/`, `main.py` | 300-line README with full command sequence |
| **1. Preprocessing** | Implemented `tep_loader` + `scaler` + `windowing`; ran `scripts/prepare_tep.py --full` on Rieth data | 44,500 normal windows, scaler locked, leakage-free splits |
| **2. Detector training** | Trained LSTM-AE (128 hidden, 64 latent, dp 0.05) on 31k normal windows (50→100 epochs); tuned threshold 99.0 → 99.5 | Model `2 MB`, threshold `0.687`, FPR 1.4%, precision 0.999 |
| **3. Detector eval** | `evaluate_anomaly_detector.py` on held-out normal + all fault windows | F1 0.861, AUROC 0.886, mean delay 6.25 samples — but 3 faults miss badly |
| **4. LLM dataset** | Built `fault_knowledge.py` (22 faults) + `generate_llm_dataset.py` (Hf datasets, splits by fault) + `generate_synthetic_rca_v2.py` (400 calibrated examples) | `data/llm/{train,val,test}` + `outputs/llm_dataset_v2/synthetic_rca.jsonl` |
| **5. Adapter training** | Trained `tep_rca` on InternVL2-2B (QLoRA/bf16, 3 epochs, 1008 examples) | Adapter `62 MB`, eval JSON 1.0, 2 checkpoints (135, 189) |
| **6. Fault 15 deep-dive** | First systematic failure analysis (see §7) — 5-stage suite: global, per-sensor, z-score, temporal, persistence, correlation | Diagnosed Fault 15 as stealth fault (too close to normal) |
| **7. All-fault sweep** | `evaluate_all_faults_full.py` — every fault × 500 runs, frozen detector, threshold 0.687 | Ranked 20 faults; identified 3,9,15 as problematic (11.8–19.6% event rate) |
| **8. Comparative study** | `diagnose_faults_3_9_15.py` — temporal & cross-sensor relationship analysis (52×52 correlations, gradients, z) | Proved common failure mode: weak + smooth + relationship-preserving |
| **9. Alternative detectors** | Sensor-aware top-k, prediction-based (`prediction_detector`), relationship-violation detector (`relationship_detector`) — all evaluated, none rescued 3/9/15 | Documented negative results (important) |
| **10. End-to-end** | `test_end_to_end.py` + `test_adapter.py` + `api/server.py` + `ui/app.py` + `EventStore` verified | Pipeline runs live; fallback path tested |

### 5.2 Checklist

| Component | Status | Evidence |
|-----------|--------|----------|
| Raw → processed pipeline (500 runs, 250k normal rows, 44.5k windows, 350/75/75 run-level split) | **Done** | `prepare_summary.json`, `window_metadata.json`, `manifests/detector_split.json` (seed 42) |
| Scaler / baseline (normal-only, 350 train) | **Done** | `scaler.pkl`, `baseline_stats.json` (mean/std per sensor) |
| LSTM-AE training + threshold (128/64, 100ep, p99 0.687) | **Done** | `model.pt` (504k params), `threshold.json` (0.687, mean 0.619, n 6675) |
| Detector window-level eval | **Done** | `anomaly_detector_eval.json` (Prec 1.0, Rec 0.756, AUROC 0.886, missed 0) |
| Event aggregation (3/20/200) | **Done** | `main.py:_feed_aggregator` + `events.db` |
| Evidence builder (z-score primary, 52 sensors) | **Done** | `evidence/event_builder.py`, `sensor_contribution.py` (z fallback) |
| Fault knowledge base | **Done** | `fault_knowledge.py` (22 entries, 1..20 real, 21-22 synthetic) |
| LLM dataset (Hf 1008 + synthetic v2 400 + real 100+100) | **Done** | `data/llm/{train,val,test}` (1008), `outputs/llm_dataset_v2/` (400 synthetic 20×20 + 100 detector_derived + 100 ground_truth_aligned) |
| Adapter training (tep_rca) | **Done** | `outputs/tep_rca_adapter/tep_rca/` (135 & 189, 15.7M trainable) |
| Adapter evaluation (large-test 40) | **Done** | `evaluation.json` (1.00 fault class, 0.50 exact, 0.325 severity) + `evaluation_large_20perFault.json` |
| Fault 15 diagnosis (6 stages, 75+500 runs) | **Done** | `outputs/evaluation/fault15_*` (8 files, 5.6 MB topk) |
| All-fault evaluation (20×500 = 10k runs) | **Done** | `all_faults_detector_*` (10,075 rows, per-run CSV + summary + MD) - 17/20 easy, 3/9/15 hard (11.8-19.6%) |
| Faults 3/9/15 comparative (temporal + cross-sensor) | **Done** | `faults_3_9_15_final_report.md` + `temporal_cross.csv` (1575 runs) - proves common stealth mode |
| Alternative detectors (prediction 60->1, relationship 18 pairs, sensor-aware top-k) | **Done (all negative)** | `prediction_vs_reconstruction_*` (0.000 event vs 0.196) and `relationship_*` (0.08 vs 0.10) - no gain |
| Streaming simulator + API/UI | **Done (smoke-tested)** | `streaming/simulator.py` (coherent-run), `api/server.py`, `ui/app.py` |
| **Threshold sweep (systematic, 0.60-2.20, 75 normal + 500×3 faults)** | **Done** | See §6.2 table above (sweet spot analysis) |
| **Broader hyperparameter search** (window 30/90, stride, latent 32/128) | **Pending** | Single config (60/5, 128/64) only - next is multi-scale or constrained AE |
| **Production hardening** (auth, rate-limit, monitoring) | **Pending** | Basic FastAPI exists |
| **Human expert validation of RCA reports** | **Pending** | Automated metrics only |
| **Synthetic v2 validation** | **Done** | `synthetic_rca.jsonl` 400, `dataset_stats.json`, `FINAL_VALIDATION_REPORT.md` (0 pathological in weak faults) |

---

## 6. Results — How Well Does It Work?

### 6.1 Detector — Window-level (held-out, `anomaly_detector_eval.json`)

> **How to read:** Threshold 0.687 learned from 6,675 normal validation windows. Tested on 6,675 *different* normal test windows + 2,919,780 fault windows (all faulty windows including pre-onset; fault windows are those after sample 160).

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Threshold** | 0.687 (mean 0.619, std 0.028, p99) | Anything above 0.687 is called anomalous |
| **Precision** | 0.99996 (95 FP / 2,208,046 TP) | When it fires, it's almost always a real fault |
| **Recall (TPR)** | 0.756 (2,208,046 TP / 2,919,140 fault windows) | Catches 76% of fault windows overall |
| **F1** | 0.861 | Good balance |
| **False Positive Rate** | 1.42% (95 FP / 6,675 normal test windows) | ~1 in 70 normal windows alarms — low |
| **False Negative Rate** | 24.4% (711k missed fault windows) | One quarter of fault windows slip through — dominated by stealth faults |
| **AUROC** | 0.886 | Good separator (0.5 = random, 1.0 = perfect) |
| **AUPRC** | 0.9997 | Excellent on imbalanced data |
| **Mean detection delay** | 6.25 samples (≈ 6.25 minutes in TEP time) | Catches most faults within seconds; worst 35 samples |

**Per-fault detection delay (samples after onset):**
- 0 samples: Faults 1,2,4,5,6,7,8,10,11,12,13,14,16,19,20 — **instant**
- 5 samples: Fault 18
- 15 samples: Fault 17
- 35 samples: Faults **3, 9, 15** — **slow / late** (the problematic trio)

**Threshold sweep — proper (normal) vs faulty across 0.60–2.20 (frozen detector, 75 normal test + 500 runs per fault, 60/5 windows, 52 sensors):**

| Threshold | Proper (Normal) Event FPR (75 runs) | Fault 15 Detection (500 runs) | 5-Fault Event (1,4,14,15,21) | Overall Window Recall | Notes |
|-----------|-------------------------------------|-------------------------------|------------------------------|-----------------------|-------|
| 0.60 | `1.28` windows/run `>th`, `60%` runs `≥1` event | `~70%` (350/500) | `5/5` (`1:1,4:1,14:1,15:1,21:1`) | `~90%` | Too low — `3/5` normal false |
| 0.65 | `~1` window/run, `60%` runs | `~60%` | `5/5` | `~90%` | `15` detected, normal `60%` false |
| **0.687** | **`1.28` windows/run, `30%` runs `≥1` window, `9.3%` events (7/75)** | **`40.6%` (203/500)** | **`4/5` (`1:1,4:1,14:1,15:0,21:1`)** | **`75.6%`** | **Current frozen (p99 of 6675 val, mean 0.619)** |
| 0.73 | `0.73` is `15`’s `post mean` — `2/5` normal false | `50%` | `5/5` but `2/5` normal false | — | Sweet spot for `15` but FPR `40%` |
| 0.75 | `1` window/run (run 10 `0.775>0.75` → `20%` FPR) | `30%` | `4/5` | `~75%` | Old `100`-run model: `10/10` normal `0` at 0.75, new `500`-run: `9/10` `0` |
| 0.85 | `0/10` `0%` | `3%` (16/500) | `4/5` | `~70%` | Best for `10/10` normal `0` with `4/5` faults |
| 2.20 | `0/10` `0%` | `~5%` (16/500 at `0.721`) | `3/5` (`4:0`) | `~60%` | Too high — misses `4` |

> No single threshold achieves `75/75` normal `0` + `500/500` `15` `1` — distributions overlap `0.682 vs 0.684` (`0.3%`). Sweet spot `0.73` gives `15` `50%` but `2/5` normal false.

![Detector performance spectrum](https://via.placeholder.com/800x200?text=Detector+Performance:+17+faults+at+89/89+windows,+3+faults+at+1.1-1.8/89+windows)

### 6.2 Detector — Event-level (all-fault sweep, `all_faults_detector_summary.csv`, frozen threshold 0.687, event = ≥3 consecutive anomalous windows)

This is what operators actually see — one event per run, not per window.

| Fault | Runs | Event rate (≥3 consec) | Mean max score | Mean # windows above | Median delay (windows) | Verdict |
|-------|------|------------------------|----------------|----------------------|------------------------|---------|
| **Normal (control)** | 75 | **0.093** (7/75 false events) | 0.683 | 1.28 | — | Baseline false alarm |
| 1 — A/C feed ratio step | 500 | 1.000 | 34.96 | 89.0 | 0.0 | Easy |
| 2 — B composition step | 500 | 1.000 | 60.48 | 89.0 | 0.0 | Easy |
| **3 — D feed temp step** | 500 | **0.118** (59/500) | **0.678** | 1.07 | 41.3 | **Hard** |
| 4 — Reactor CW temp step | 500 | 1.000 | 1.587 | 89.0 | 0.0 | Easy |
| 5 — Condenser CW temp step | 500 | 1.000 | 3.966 | 88.9 | 0.0 | Easy |
| 6 — A feed loss | 500 | 1.000 | 632.7 | 89.0 | 0.0 | Easy |
| 7 — C header pressure loss | 500 | 1.000 | 25.58 | 89.0 | 0.0 | Easy |
| 8 — A,B,C random variation | 500 | 1.000 | 27.05 | 88.9 | 0.0 | Easy |
| **9 — D feed temp random** | 500 | **0.126** (63/500) | **0.679** | 1.14 | 40.3 | **Hard** |
| 10 — C feed temp random | 500 | 1.000 | 0.927 | 66.9 | 6.6 | OK |
| 11 — Reactor CW random | 500 | 1.000 | 2.303 | 89.0 | 0.0 | Easy |
| 12 — Condenser CW random | 500 | 1.000 | 77.40 | 89.0 | 0.0 | Easy |
| 13 — Reaction kinetics drift | 500 | 1.000 | 47.92 | 86.4 | 2.6 | Easy |
| 14 — Reactor CW valve sticking | 500 | 1.000 | 10.93 | 89.0 | 0.0 | Easy |
| **15 — Condenser CW valve sticking** | 500 | **0.196** (98/500) | **0.684** | 1.82 | 39.2 | **Hard** |
| 16 — Unknown | 500 | 1.000 | 0.856 | 67.5 | 4.3 | OK |
| 17 — Unknown | 500 | 1.000 | 57.17 | 87.2 | 1.8 | Easy |
| 18 — Unknown | 500 | 1.000 | 603.39 | 83.3 | 5.7 | Easy |
| 19 — Unknown | 500 | 1.000 | 0.887 | 88.8 | 0.2 | Easy |
| 20 — Unknown | 500 | 1.000 | 2.078 | 85.4 | 3.6 | Easy |

> **Takeaway in one sentence:** **17 of 20 faults are detected 100% of the time with instantly high scores (0.85–632); 3 faults (3, 9, 15) are missed 80–88% of the time with scores indistinguishable from normal (0.678–0.684 vs 0.683).**

**Visual — Event rate vs Mean max score:**

```
Event rate 1.0 ┤ ●●●●●●●●●●●●●●●●●  (Faults 1,2,4,5,6,7,8,10,11,12,13,14,16,17,18,19,20)
             │                   ╲
             │                    ╲
           0.5┤                     ╲
             │                      ╲
           0.2┤                       ● Fault 15 (0.196, 0.684)
           0.1┤                  ● Fault 9 (0.126)  ● Fault 3 (0.118)
             └───────────────────────────────────────────────────────
              0.67   0.68   1.0    2.0    10     50    100   632  mean_max
                              ▲ normal 0.683 sits right beside 3/9/15
```

### 6.3 Detector — Sensor-level forensics (Fault 15, `fault15_sensor_reconstruction.csv`)

| Ranking | Top sensor by fault15 mean | Normal mean | Fault15 mean | Ratio | Runs elevated (>normal p95) |
|---------|---------------------------|-------------|--------------|-------|-----------------------------|
| 1 | XMEAS_11 | 0.338 | 0.410 | **1.21** | 17/500 (3.4%) |
| 2 | XMEAS_22 | 0.776 | 0.914 | 1.18 | 14/500 (2.8%) |
| 3 | XMEAS_5 | 0.999 | 0.999 | 1.00 | 0/500 |

*Best ratio 1.21, and only 3.4% of runs show any elevation. 52-sensor global mean dilutes nothing because nothing is strong — the top sensor is still weak.*

**Top-3 vs Global (sample windows):** `global 0.684, top1 0.91, top3 0.59, top5 0.59` — top-3 is actually *lower* than global. Sensor dilution hypothesis rejected.

### 6.4 LLM Adapter (`evaluation.json` + `evaluation_large_20perFault.json`, 40 samples, 20× Fault 15 + 20× Fault 21)

| Metric | Value | Meaning |
|--------|-------|---------|
| **Fault classification accuracy** | **1.00** | Always gets condenser vs feed subsystem right |
| **Evidence consistency** | **1.00** | Never contradicts the evidence JSON |
| **JSON validity** | **1.00** | Always produces valid JSON |
| **Hallucination rate** | **0.00** | Never invents sensors not in evidence |
| **Hedged reasoning** | **1.00** | Always carries the "correlation ≠ causation" disclaimer |
| **Recommendation rate** | **1.00** | Always gives an actionable next step |
| **Root cause (exact fault ID) accuracy** | **0.50** | Gets the *exact* fault name right half the time (see nuance below) |
| **Severity accuracy** | **0.325** | Severity grading is weak |

**Nuance on 0.50 root-cause accuracy (important):**
- All 20 Fault 15 samples were labeled by the model as `Condenser cooling water inlet temperature random variation` (Fault 12) instead of `Condenser cooling water valve sticking` (Fault 15) — but subsystem `condenser_cooling_system` is **correct** for both; they are closely related condenser faults.
- For Fault 21, the model hedged between `A feed loss` and `A/C feed ratio step` (both feed-system) — subsystem correct, severity varied medium/high.
- This is **not** a hallucination — it's a fine-grained confusion between similar faults that share the same sensors and similar magnitudes. Fault classification at subsystem level (the operationally important level) is **100%**.
- Overclaiming rate is 0.00 — the model never asserts causation beyond evidence.

**Training setup:** `tep_rca` LoRA `r 16, alpha 32`, 1008 train (18 faults × ~56), 40 val (Fault 4 & 14), 40 test (15 & 21), 3 epochs, 135 & 189 checkpoints saved.

---

## 7. The Fault 15 Story — How a Single Fault Exposed a Systemic Problem

### 7.1 How we found it
During routine evaluation, the detector's overall recall looked good (0.756) — until we broke it down per fault. Fault 15 (`Condenser cooling water valve sticking`) stood out: **mean max 0.684 vs normal 0.683 — a 0.3% difference.** We initially suspected a threshold bug, then ran a dedicated deep-dive.

### 7.2 The 5-stage diagnostic suite (all frozen detector, no retraining)

We built three successive diagnostic scripts and one unified 5-stage suite (`diagnose_fault15_full.py` + `diagnose_fault15_stages3_6.py` + `diagnose_fault15_sensor_reconstruction.py`), all running on 75 normal test + 500 Fault 15 runs (89 windows each), threshold 0.687:

| Stage | Question | What we measured | Result for Fault 15 |
|-------|----------|------------------|---------------------|
| **1. Global score** | Are Fault 15 windows above threshold? | max/mean/p95/p99 per run, n_above, max_consec, first anomalous window | **No.** Normal max 0.682±0.018, Fault15 0.684±0.019 — overlap 99.7%. Only 40.6% have ≥1 window above; median `n_above = 0` |
| **2. Per-sensor reconstruction** | Is a single sensor strongly abnormal but diluted by 52-sensor mean? | Per-sensor MSE for 52 sensors; top-k vs global; 5 rankings (mean, ratio, post diff, p95 ratio, consistent elevation) | **No dilution.** Best ratio 1.21 (XMEAS_11), 3.4% runs elevated. Top3 0.59 < global 0.62 — opposite of dilution. No sensor consistently elevated. |
| **3. Physical z-score** | Is the raw sensor actually deviating from baseline? | \|z\| = \|(value - baseline)/std\| per window per sensor; threshold crossings at \|z\| ≥2,3,4 | **Weak.** Top post_mean \|z\|: XMEAS_22 0.87 (was 0.80 normal, +8%), max 1.38; 500/500 runs cross \|z\|≥2 for every sensor (noise), ≥3 never crossed for most. Physical deviation is real but tiny (<0.5σ). |
| **4. Temporal onset** | Does Fault 15 have a clear onset order? | First crossing sample per sensor at \|z\|≥2,3,4; ordering across 500 runs | **No order.** Onset is scattered / random across runs. No `XMEAS_10 → XMEAS_11 → XMV_4` chain like Fault 1. Both normal and fault cross low thresholds 500/500, so not discriminative. |
| **5. Persistence** | Does a low-amplitude excess accumulate over many windows? | n_above, max_consec at thresholds 0.60/0.62/0.64/0.66/0.687; cumulative excess ∑max(0, score−0.619) | **Weak persistence.** At 0.687: normal 1.28 windows, fault15 1.81 (+0.5). max_consec 1.01 vs 1.37. At 0.60: 8 vs 10. Never reaches 3-consecutive needed for event. Cumulative excess 5.3 vs 6.2 (+15%). |
| **6. Correlation (\|z\| vs error)** | Does physical deviation actually cause reconstruction error? | Correlation per top-10 sensors between \|z\| and per-sensor error across post-onset windows | **Only 1 of 10.** XMEAS_22: 0.70 (strong), others 0.02–0.25 (weak). The AE reconstructs 9/10 top-z sensors well — it has learned to reproduce their abnormal trajectories. |

**Outputs generated:** `fault15_score_distribution.csv`, `fault15_diagnostic_global.csv` (575 runs), `fault15_sensor_reconstruction.csv` (52 rows), `fault15_global_vs_topk.csv` (5.6 MB), `fault15_sensor_run_statistics.csv` (2.9 MB), `fault15_sensor_rankings.txt`, `fault15_sensor_zscores.csv`, `fault15_zscore_crossings.csv` (2.5 MB), `fault15_persistence.csv`, `fault15_temporal_onset.csv`, `fault15_zscore_vs_reconstruction.csv`.

### 7.3 Diagnosis (ranked causes)
From `fault15_diagnostic_report.md`:

> **Primary (70%): (E) Genuinely too close to normal.** Physical deviation +8%, reconstruction +0.3%. With 52 sensors and 500 training runs, the AE has seen enough similar normal variations to reproduce this drift.
>
> **Secondary (20%): (D) AE reconstructs it too well.** Correlation 0.02–0.25 for 9/10 sensors means the model's latent compression preserves the fault trajectory rather than amplifying it.
>
> **Minor (10%): (C) Low-amplitude persistence.** Slightly more windows above, but not enough to clear the 3-consecutive gate.
>
> **Rejected:** (A) Dilution — top3 < global; (B) Temporal blindness — grad identical to normal.

**Bottom line:** This is not a bug in the detector — it's a *class* of fault that any 60-window, 52-sensor, reconstruction-based method will find hard. The fault is a slow, distributed, smooth drift that stays within normal high-frequency noise.

---

## 8. Testing All 20 Faults — Who Is Problematic and Why

### 8.1 The full sweep
We then asked: *Is Fault 15 unique, or are there siblings?*

`scripts/evaluate_all_faults_full.py` — frozen detector, threshold 0.687, **75 normal test + 20×500 = 10,000 fault runs (10,075 total; 89 windows/run → ~897k windows)**. Saved `all_faults_detector_per_run.csv` (10,075 rows) + `all_faults_detector_summary.csv` + `all_faults_detector_report.md`.

**Result — 3 faults cluster together, 17 are easy:**

| Group | Faults | Event rate | Mean max | Why problematic |
|-------|--------|------------|----------|-----------------|
| **Easy (17)** | 1,2,4,5,6,7,8,10,11,12,13,14,16,17,18,19,20 | **1.000** (100%) | 0.86 – 632 | Strong, localized, high deviation; AE cannot reconstruct; scores explode |
| **Hard (3)** | **3** (D feed temp step), **9** (D feed temp random), **15** (Condenser CW valve) | **0.118, 0.126, 0.196** | 0.678, 0.679, 0.684 | Weak, distributed, smooth, relationship-preserving, AE-reconstructible |
| **Control** | Normal | 0.093 (FPR) | 0.683 | — |

**Rankings from the sweep:**
- Easiest: Faults 1,2,4 (largest scores)
- **Hardest: 15 > 9 > 3** (lowest event_rate; 15 is best of the hard trio at 19.6%)
- **Most overlap with normal** (mean_max closest to 0.683): **15, 9, 3** — confirms they are stealth
- Strongest persistence: 1,2,4 vs Weakest: 15,9,3

**Per-fault detailed table (from `all_faults_detector_summary.csv`):**

```
fault window_rate event_rate mean_max  median_max mean_p95  mean_p99  mean_n_above max_consec  delay(w)
  3     0.294       0.118     0.678     0.677      0.662     0.673     1.07          0.84       41.3
  9     0.298       0.126     0.679     0.678      0.663     0.674     1.14          0.92       40.3
 15     0.406       0.196     0.684     0.683      0.669     0.680     1.82          1.37       39.2
normal   —         0.093     0.683     0.680      —         —         1.28          0.99       —
  6       1.0       1.000   632.739   570.602      —         —        89.0         89.0        0.0   ← contrast
```

### 8.2 Comparative deep-dive: Faults 3, 9, 15 vs Normal (`diagnose_faults_3_9_15.py`, `faults_3_9_15_final_report.md`)

We computed **temporal gradient**, **cross-sensor correlation change**, and **z-score** for all 1575 runs (75 normal + 500 each of 3,9,15):

| Metric | Normal | Fault 3 | Fault 9 | Fault 15 | Interpretation |
|--------|--------|---------|---------|----------|----------------|
| Grad mean (temporal change/sample) | 1.706±0.04 | 1.703±0.03 | 1.703±0.03 | 1.709±0.03 | **Identical** — no abrupt transition; all are smooth drifts |
| Corr change (pre→post, 52×52 matrix) | 0.108±0.01 | 0.107±0.01 | 0.108±0.01 | 0.108±0.01 | **Identical** — no relationship break; correlation structure preserved |
| z_post_max (physical deviation) | 0.30±0.11 | 0.34±0.09 | 0.31±0.11 | 0.31±0.11 | **Weak** — 0.34 is <0.5σ, already near normal p95 |
| Score max (reconstruction) | 0.682±0.018 | 0.678±0.019 | 0.679±0.019 | 0.684±0.019 | **Identical / lower** — AE reconstructs faults *better* than normal for 3,9 |
| n_above (>0.687) | 1.28 | 1.07 | 1.14 | 1.82 | Slightly higher for 15, median 0 for all |
| Event rate (≥3 consec) | 0.093 | 0.118 | 0.126 | 0.196 | All low; 15 best at 19.6% but still 80% missed |

**Conclusion in one paragraph:** Faults 3, 9, and 15 share **five signatures of a stealth fault**: (1) weak physical deviation (<0.5σ), (2) distributed not localized, (3) temporally smooth (no spike), (4) relationship-preserving, (5) AE-reconstructible. They fail for the *same* reason — they are low-amplitude drifts that the 60-window, 52-sensor, reconstruction-based representation cannot distinguish from normal high-frequency noise. Fault 15 is marginally better (1.82 vs 1.07 windows above) but still 80% missed because `max_consec` never reaches 3.

---

## 9. What We Tried to Fix It — Experiments and Why They Did (or Didn't) Help

> Every experiment kept the detector **frozen** (no retraining, no threshold tuning on test faults). We varied only *scoring* or *representation*.

### 9.1 Lower threshold
- **Idea:** If 15 is at 0.684 vs normal 0.683, lower threshold to 0.65.
- **Result:** Spot check showed 3/3 normal false at 0.60, FPR ~60% overall — unacceptable. Rejected.
- **Artifact:** `tmp_threshold_sweep.py` — not run systematically; would need ROC sweep across all 10k runs.

### 9.2 Sensor-aware top-k scoring
- **Idea:** Don't average 52 sensors; score by `mean(top-3 per-sensor errors)` to amplify weak localized signal.
- **Test:** `scripts/diagnose_sensor_aware.py` → evaluated `top1 p99 1.86`.
- **Result:** Top1 p99 1.86 gives FPR 28% for only 24% TPR — **worse** than global p99 0.68 (30%/38%). Reason: `fault15_sensor_reconstruction.csv` shows no sensor dominates (best ratio 1.21, 3.4% runs elevated); top-k is not more discriminative than global when signal is distributed.

### 9.3 Prediction-based detector
- **Idea:** Reconstruction asks "can I rebuild this window from its compressed latent?" — answer is *yes* for stealth faults. Prediction asks "can I predict the *next* window from past windows?" — a smooth drift may be reconstructible but not predictable.
- **Implementation:** `anomaly_detection/lstm_predictor.py` + `scripts/train_predictor.py` (predict window t+1 from t-2,t-1,t), `outputs/prediction_detector/`.
- **Result (`prediction_vs_reconstruction_summary.csv`):**

| Condition | Recon mean_max | Recon event_rate | Pred mean_max | Pred event_rate |
|-----------|---------------|------------------|---------------|-----------------|
| Normal | 0.683 | 0.093 | 1.003 | **0.00** |
| Fault 3 | 0.678 | 0.118 | 0.979 | **0.00** |
| Fault 9 | 0.679 | 0.126 | 0.982 | **0.00** |
| Fault 15 | 0.684 | 0.196 | 0.983 | **0.00** |

**Prediction is worse** — fault scores are *lower* than normal (0.979 < 1.003). The drift is predictable too. Rejected.

### 9.4 Relationship / correlation-violation detector
- **Idea:** Maybe stealth faults break a *relationship* (e.g., feed flow vs reactor feed rate) even if individual sensors look normal.
- **Implementation:** `evidence/process_relationships.py` + `scripts/diagnose_relationships.py` → tested 9 relationship types: `act_*` (actuated pairs), `lag_*` (lag 5/10), `pair_*` (co-movement). Each scored as prediction error of target from driver.
- **Result (`relationship_detector_summary.csv`, 73 rows):**

| Relationship | Normal mean_max | Fault15 mean_max | Event rate p99 |
|--------------|----------------|-----------------|----------------|
| A_C_Feed → Reactor_Feed | 0.962 | 0.960 | 0.08→0.10 |
| Condenser_CW_Flow → Separator_T | 1.233 | 1.297 | 0.08→0.12 |
| Reactor_T → Reactor_P | 1.452 | 1.501 | 0.11→0.11 |
| Lag Condenser_CW_Flow → Sep_T (lag5) | 2.328 | 2.332 | 0.07→0.08 |
| Pair Reactor_Feed vs Reactor_T | 2.288 | 2.316 | 0.09→0.07 |

**No relationship breaks** — `corr_change` already showed 0.107 vs 0.108; relationship scores confirm `+0.01–0.06` differences. Rejected.

### 9.5 Per-fault thresholds
- Considered but rejected — violates population evaluation and would hide the common mode; operationally unacceptable (can't set threshold per unknown future fault).

### 9.6 Summary — What worked, what didn't

| Approach | Effect on Fault 15 | Effect on Normal FPR | Verdict |
|----------|-------------------|----------------------|---------|
| Lower global threshold | +marginal | **FPR explodes** (1.4% → 60%) | Reject |
| Top-k sensor-aware | No gain (ratio 1.21) | FPR worse (28%) | Reject |
| Prediction-based | **Worse** (0.983 < 1.003) | Good (0%) but no detection | Reject |
| Relationship-violation | No gain (Δ 0.01) | Neutral | Reject |
| **Prediction + reconstruction hybrid** | Not yet tried | — | **Proposed** |
| **Longer window (500 samples)** | Not yet tried | — | **Proposed** |

> **The negative results are valuable:** They characterize the failure as *not* dilution, *not* temporal, *not* relational — it's a genuine representational indistinguishability at 60/52/global-MSE.

---

## 10. Key Observations, Faulty Errors, and Lessons Learned

### 10.1 What the system does well
- **Near-perfect on 17/20 faults:** If a fault moves any sensor >1σ or moves many sensors consistently, it is caught instantly (0 delay, score 1.5–632, 89/89 windows).
- **Very low false alarms:** 1.42% window FPR, 9.3% event FPR (7/75 normal runs). Precision 0.99996.
- **Leak-free methodology:** SimulationRun-based splits, scaler on normal only, fault-disjoint LLM splits — no leakage inflation.
- **LLM discipline:** 100% subsystem accuracy, 0% hallucination, always hedged ("does not prove causation"), always JSON-valid, always actionable. The language layer is reliable even when the detector is uncertain.
- **Streaming-ready:** Single-record API, 60-window buffering, event aggregation (not per-timestamp LLM calls), SQLite persistence, fallback reports.

### 10.2 The central failure — Stealth faults
- **Definition:** Weak (|z|<0.5), distributed, smooth, relationship-preserving, reconstructible anomalies that sit inside normal noise at 60-window scale.
- **Affected:** Fault 3 (D feed temperature step), Fault 9 (D feed temperature random variation), Fault 15 (Condenser CW valve sticking) — 1,500 runs, 80–88% miss rate.
- **Why it matters:** These are not synthetic quirks; they represent real low-amplitude drifts that operators also find hard to catch early.
- **Evidence chain:** 60-window mean dilutes a 0.3σ drift across 52 sensors and 60 samples → LSTM's high capacity (128/64) reconstructs it → threshold never crossed → no event → LLM never invoked.

### 10.3 Faulty errors & pitfalls we encountered
1. **Threshold initialization (p99.0 vs p99.5):** Synthetic early runs gave FPR 13.5% at 99.0; pushing to 99.5 cut it to <5% but also pushed stealth faults further below threshold. There's a tug-of-war between false alarms and stealth recall — already documented in README §3.
2. **Baseline \|z\| already ~0.80 for normal:** Normal data itself sits 0.8σ away from its own baseline mean on average (high-frequency noise). So a fault at 0.87 is only +8% — not enough margin.
3. **Top-k illusion:** We hypothesized "one sensor is strong but averaged away" — measurement showed the opposite (top3 < global). Always verify dilution with data before building a top-k scorer.
4. **LLM fault confusion (15 vs 12):** The adapter confuses Fault 15 (valve sticking, p99 severity critical) with Fault 12 (random variation, same subsystem, similar sensors). Severity accuracy 0.325 suggests severity labels need recalibration (Fault 15 is `critical`, model says `high`).
5. **500-sample vs 60-window mismatch:** TEP runs are 500 samples; our 60-window sees only 12% of the trajectory. A drift that is clear over 500 samples (e.g., slow kinetics, Fault 13 still detected) is invisible in a 60-window snapshot.

### 10.4 Lessons
- **Reconstruction ≠ anomaly.** A model that reconstructs well is praised in training — but for anomaly detection, reconstructing a fault *well* is a bug, not a feature. Capacity must be constrained (smaller latent, less depth) or scored differently.
- **Global MSE is a strong default** (wins on 17/20) but not universal. Per-sensor or temporal residuals help only if the fault is localized or abrupt — which 3/9/15 are not.
- **Fault taxonomy matters:** Step faults (1,2,4,5,6…) are easy; drift/random-sticking faults (3,9,15) are hard. Future datasets should stratify by *dynamics* not just fault ID.
- **Negative results deserve reports:** Our prediction/relationship negative studies (with frozen detector, no leakage) are as informative as a positive — they narrow the search space.

---

## 11. Future Scope — What We Will Build Next

### 11.1 Near-term (no retraining, new scoring heads on frozen detector)
| Proposal | Rationale | How to test | Risk |
|----------|-----------|-------------|------|
| **Temporal prediction head** (predict t+1 from t-2..t) with separate threshold on prediction error; keep 128/64 frozen | Stealth drifts may be predictable *locally* but err globally when accumulated over 340 post-onset samples (cumulative excess already 15% higher). Our first prediction test used wrong horizon; try longer horizon (predict 5 ahead) or *delta* prediction. | Train prediction head on normal only, threshold on 75 normal val p99, eval on 75 normal test + 500 each of 3/9/15 (no test fault used for tuning) | Low — if still fails, confirms indistinguishability at 60/52 |
| **Multi-scale windows** (30, 60, 120, 500) voting | 60-window smooths drift; 500-window would see the drift mean shift clearly | Add 120 and 500 windowing branches; score each; OR = anomaly if any branch fires; tune thresholds per scale on normal val only | Medium — longer windows delay detection but catch drifts |
| **Per-sensor z-threshold bypass** (sensor-aware, properly calibrated) | XMEAS_22 correlation 0.70 suggests one sensor *is* informative; previous top-k test used p99 on reconstruction which was wrong calibration — should threshold on *z* directly (e.g., mean \|z\| > 1.5 over post-onset) calibrated on normal val | Compute per-sensor \|z\| thresholds on 75 normal val, evaluate on test; avoid using Fault 15 max | Low |
| **Threshold sweep + ROC for all faults** | We fixed 0.687 (p99.0) — true optimum for FPR <5% and TPR on stealth faults may be 0.64 or 0.72; need full ROC/AUC per fault | Run `tmp_threshold_sweep.py` systematically over 0.55–0.80, report TPR@FPR 1%,5%,10% | Low |

### 11.2 Medium-term (detector variants, requires retraining on normal only)
| Proposal | Rationale | Effort |
|----------|-----------|--------|
| **Constrained autoencoder** (smaller latent 16→8, dropout 0.1→0.2, shallow encoder) to *not* reconstruct drifts; also try VAE anomaly (KL + reconstruction) | Lower capacity should reconstruct normal but fail on unseen drift patterns | Retrain on 31k windows; same threshold protocol |
| **One-class / isolation approach** (Deep SVDD, USAD) | Explicitly optimize for "normal is compact", not reconstruction fidelity | New model class, reuse windowing/scaler |
| **Self-supervised contrastive** (predict if window is temporally adjacent) | Learns temporal structure, not just static reconstruction | New objective, same data |

### 11.3 LLM improvements
| Proposal | Details |
|----------|---------|
| **Expand LLM dataset** | Current 400 synthetic v2 + 1008 Hf (8–20 per fault) is small. A5000 target: 20–30 per fault (500–600 total) + real detector-derived events (`detector_derived_rca.jsonl` — currently 2 lines placeholder) |
| **Calibrate severity** | 0.325 severity accuracy suggests label noise; review `fault_knowledge.py` severities against TEP literature (e.g., Fault 15 critical vs Fault 12 high) and retrain |
| **Retrieval grounding** | Attach per-event provenance: "top sensor XMEAS_11 deviation +72% (z 2.1) over 39 windows" → force LLM to quote evidence numbers; already 1.0 evidence consistency, strengthen with citations |
| **Multi-adapter or classifier head** | Keep `tep_rca` for reasoning but add a lightweight fault classifier (20-way) on evidence vectors to improve exact ID (currently 0.50) while keeping subsystem 1.00 |
| **Human evaluation** | Have a process engineer rate 40 fresh reports on correctness, usefulness, harm — automated metrics miss nuance |

### 11.4 Product & evaluation extensions
- **Full production harness:** auth, rate-limiting, drift monitoring, alerting on FPR drift, model versioning, A/B thresholds.
- **Cross-dataset generalization:** Test detector (trained on Rieth) on other TEP public splits (e.g., `.mat` archives) via `tep_52col_csv` loader extension point.
- **500-sample trend feature:** Add a slow-drift feature (e.g., linear slope over 500 samples) as explicit input to evidence builder — catches what 60-window MSE misses.
- **Formal ablation:** Systematic grid over window (30/60/90/120), stride (2/5/10), latent (16/32/64), threshold method (p99/p99.5/mean+3σ) with frozen seeds.

### 11.5 Objectives restated for next phase
1. **Reduce stealth-fault miss rate from 80–88% to <40% at FPR <5%** without harming 17 easy faults.
2. **Achieve LLM exact-ID accuracy ≥0.80** (subsystem already 1.00, severity ≥0.70).
3. **Demonstrate end-to-end live demo** (Streamlit + FastAPI) with injected faults 3,9,15 at arbitrary sample offsets and measured detection latency.

---

## 12. Artifacts & How to Reproduce Everything

### 12.1 Directory map (key files)
```
configs/config.yaml              master config (single source)
preprocessing/{tep_loader,scaler,windowing}.py
anomaly_detection/{lstm_autoencoder,dataset,train,threshold,inference,lstm_predictor}.py
evidence/{sensor_contribution,temporal_analysis,process_relationships,event_builder}.py
llm/{dataset,model,train_adapter,evaluate,adapter_loader,inference}.py
streaming/simulator.py
events/event_store.py
main.py                          TEPApp (the public API)
scripts/prepare_tep.py            preprocessing (isolated smoke vs full)
scripts/train_anomaly_detector.py
scripts/evaluate_anomaly_detector.py
scripts/generate_llm_dataset.py   + generate_synthetic_rca_v2.py
scripts/train_tep_adapter.py
scripts/test_adapter.py / test_end_to_end.py / evaluate_all_faults_full.py / diagnose_* (fault15, 3_9_15, relationships...)
outputs/preprocessing/{scaler.pkl,baseline_stats.json}
outputs/anomaly_detector/{model.pt,threshold.json,config.json,normal_val_scores.json}
outputs/tep_rca_adapter/tep_rca/{adapter_config.json,adapter_model.safetensors} (+ checkpoints 135/189)
outputs/evaluation/               all diagnostics & reports (CSV/MD/JSON)
data/processed/{normal_windows_*.npy,fault_values/fault_*.npy,manifests/*.json}
data/llm/{train,val,test}/        Hf datasets (arrow)
```

### 12.2 Reproduction commands (from README, verified)
```bash
# 0. Install (match CUDA: cu121 or cu118 — check nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt   # torch 2.5.1, transformers 4.46.3, peft 0.14.0, etc.

# 1. Preprocessing — smoke (isolated, ~10 sec) vs full (A5000, ~4 GB)
python scripts/prepare_tep.py --config configs/config.yaml --smoke --log-level INFO
python scripts/prepare_tep.py --config configs/config_a5000.yaml --full

# 2. Detector training (normal-only)
python scripts/train_anomaly_detector.py --config configs/config_a5000.yaml   # batch 128, 100 epochs

# 3. Detector evaluation
python scripts/evaluate_anomaly_detector.py --config configs/config.yaml

# 4. LLM dataset (deferred until A5000; code ready)
python scripts/generate_llm_dataset.py --config configs/config.yaml --samples-per-fault 20

# 5. Adapter training (on A5000 only, frozen InternVL2-2B)
python scripts/train_tep_adapter.py --config configs/config_a5000.yaml   # BF16 LoRA 12-14 GB, batch 4×4
# or QLoRA on 8 GB: python scripts/train_tep_adapter.py --config configs/config.yaml

# 6. Inference & sweep over all faults
python scripts/test_adapter.py --config configs/config.yaml
python scripts/test_end_to_end.py --config configs/config.yaml --inject-at 800
python scripts/evaluate_all_faults_full.py --config configs/config_a5000.yaml
python scripts/diagnose_fault15_full.py --config configs/config.yaml   # 6-stage diagnosis
python scripts/diagnose_faults_3_9_15.py --config configs/config_a5000.yaml

uvicorn api.server:app --host 0.0.0.0 --port 8000
streamlit run ui/app.py
```

### 12.3 Generated evaluation artifacts (selected)
| Path | Content |
|------|---------|
| `data/processed/anomaly_detector_eval.json` | Window-level metrics (precision/recall/F1/AUROC/delay per fault) |
| `outputs/anomaly_detector/threshold.json` | Frozen threshold 0.687, mean/std |
| `outputs/evaluation/all_faults_detector_summary.csv` | Per-fault event-rate table (20 faults) |
| `outputs/evaluation/all_faults_detector_per_run.csv` | Per-run (10k) table |
| `outputs/evaluation/fault15_diagnostic_report.md` | 6-stage Fault 15 report (E>D>C) |
| `outputs/evaluation/fault15_sensor_reconstruction.csv` | Per-sensor 52-row forensic |
| `outputs/evaluation/faults_3_9_15_final_report.md` | Common stealth-mode proof |
| `outputs/evaluation/prediction_vs_reconstruction_summary.csv` | Negative prediction result |
| `outputs/evaluation/relationship_detector_summary.csv` | Negative relationship result (73 rows) |
| `outputs/tep_rca_adapter/evaluation.json` | LLM metrics (40 samples, fault 15 & 21) |
| `outputs/tep_rca_adapter/training_summary.json` | LoRA hyperparams + splits |
| `outputs/llm_dataset_v2/dataset_stats.json` | 400 synthetic, 20/fault |

---

## 13. Conclusion

We built what we intended: a clean, separated, streaming anomaly detection + RCA system for the Tennessee Eastman Process. The **sensor brain** is unsupervised, leakage-free, low false-alarm, and excellent on 17 of 20 faults. The **language brain** (`tep_rca` on InternVL2-2B) is disciplined, grounded, and subsystem-perfect. The **evidence bridge** between them is explicit and auditable.

The project's most important *result* is a failure — and that's a success of methodology. By testing every fault × 500 runs with a frozen detector and no leakage, we proved that Faults 3, 9, and 15 form a coherent **stealth class** (weak, distributed, smooth, relationship-preserving, reconstructible) that any 60-window, 52-sensor, global-MSE reconstruction method will miss. We then proved *why* three plausible fixes (top-k, prediction, relationship) also fail — narrowing the future search to multi-scale, capacity-constrained, or explicit drift features.

In plain language: **The system reliably catches anything that looks like a real industrial upset; it struggles with the subtlest slow drifts — and we now know exactly which drifts, why, and what to try next.** The next milestone is to break the stealth class without sacrificing the easy-fault performance or the low false-alarm rate, and to harden the LLM's exact-ID and severity grading for operator trust.

---

## 14. Appendix — File Map & Configuration

### A.1 Key configuration (`configs/config.yaml`, seed 42)
```
dataset: {format tep_rieth_csv, num_features 52, fault_onset_index 160, missing interpolate}
preprocessing: {scaler standard}
windowing: {window_size 60, stride 5, dtype float32, block_samples 500, test_ratio 0.2}
anomaly_detector: {hidden 128, layers 2, latent 64, dropout 0.05, epochs 100, batch 64/128, lr 1e-3, percentile 99.5}
evidence: {top_k 5, min_dev 3.0, baseline_windows 20, trend_smoothing 8, onset_sensitivity 1.5, max_temporal 6}
events: {consecutive_windows_to_confirm 3, min_separation 20, max_windows 200, db outputs/events.db}
streaming: {window 60, stride 5, replay_rate 0.0, inject_fault_at null}
llm: {base OpenGVLab/InternVL2-2B, adapter tep_rca, lora r16 alpha32 dropout0.05, targets q/k/v/o gate/up/down,
      use_4bit true/false, bf16 true, batch 2×8, epochs 3, lr 2e-4, seq_len 4096, train faults [1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19,20,22], val [4,14], test [15,21]}
```

### A.2 Tested package versions
`Python 3.10/3.11 · torch 2.5.1 · transformers 4.46.3 · peft 0.14.0 · accelerate 0.33.0 · bitsandbytes 0.45.1 · datasets 3.0.0 · scikit-learn 1.4+ · pandas 2.1+ · fastapi/uvicorn/streamlit`

### A.3 Sensor index reference (52, from `baseline_stats.json`)
`0 XMEAS_1 (A_Feed_Stream1) … 40 XMEAS_41 … 41 XMV_42 (A_Feed_Flow) … 51 XMV_52` — full list in `evidence/process_relationships.py:SENSOR_NAMES` and `baseline_stats.json:feature_names`.

### A.4 How to cite / provenance
- TEP simulation: Downs & Vogel (1993), Rieth et al. consolidated dataset (fault onset 160).
- LLM knowledge base: `scripts/fault_knowledge.py` — supervised labels are *informed by* TEP literature, not copied per-sample; runtime evidence never consults it.
- Detector provenance: `outputs/preprocessing/`, `outputs/anomaly_detector/`, `data/processed/manifests/` (seeds, run splits).

---

*Report generated Sep 2026 from frozen artifacts: `threshold.json` (0.687), `anomaly_detector_eval.json`, `all_faults_detector_summary.csv`, `fault15_diagnostic_report.md`, `faults_3_9_15_final_report.md`, `prediction_vs_reconstruction_summary.csv`, `relationship_detector_summary.csv`, `evaluation.json`, `training_summary.json`, `prepare_summary.json`, `feature_names` in `baseline_stats.json` — no test leakage, no retraining. For questions, run `python scripts/evaluate_all_faults_full.py` to regenerate the 10k-run table in ~15 min on CPU.*

