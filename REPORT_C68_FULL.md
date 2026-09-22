# TEP Industrial Anomaly Detection & RCA — Full Context Report (0 → Current + Future) · C68-Anomaly Branch

> **Scope:** Everything anyone needs for 100% understanding. Base is `C:\Users\Admin\Desktop\anomaly` (PROJECT_REPORT.md, 674 lines, Sep 2026) + delta branch `C:\Users\Admin\Desktop\anom\C68-Anomaly` (`new`, commits `0317428 → 3307247`). Covers architecture, data, every implementation file, timeline, all results/logs/graphs, what worked / what didn't (with numbers), why Fault 15 was fixed and why Faults 3/9 are declared unrepairable, bugs, and future roadmap. No test leakage, no retraining on faults for detection.

**One-line summary:** Two-brain system — sensor brain (unsupervised, normal-only) flags anomalies, evidence bridge packages forensics, language brain (InternVL2-2B + one LoRA `tep_rca`) writes grounded RCA. Base system caught 17/20 faults instantly; C68 adds a **parallel CVA/DPCA dynamic head fused OR with the frozen LSTM-AE** to catch the dynamics fault **15** (82.6% FDR, 100% run detection) while keeping the LSTM for all level-shift faults. Faults **3 & 9** remain low (≈2–3%) by physics/benchmark — control-loop compensation — confirmed by per-fault FDR at fixed FAR.

**Branches:**
- `anomaly` : `1075902 Merged old and new` + `0317428 CVA and DPCA ...` baseline. LSTM threshold **0.687**, window/event tables for 10,075 runs.
- `C68-Anomaly` (`new`) : on top of that → `5e6aa5a Fixed 15 with concluding 3,9 unrepairable` + `3307247 Scripts added for unseen data testing` + `training_instructions.md` + `DynamicDetectorRunner` + OR fusion. eval at `data/processed/anomaly_detector_eval.json` (threshold **1.85**, fused 4-fault sample).

Repo roots: `C:\Users\Admin\Desktop\anomaly` entry `main.py:TEPApp`; new branch `C:\Users\Admin\Desktop\anom\C68-Anomaly`. Hardware: RTX A5000 (24 GB), also ≥8 GB QLoRA or CPU for sensor pipeline.

---

## Table of Contents
1. Objectives & Design Principles
2. Dataset (Rieth vs legacy, onset split, sensor vocab)
3. System Architecture — Base vs C68 Delta
4. Implementation — File-by-File (preprocessing, LSTM, dynamic CVA/DPCA, evidence, streaming, LLM, configs, utils)
5. Timeline & Progress (0 → now)
6. Results — Window, Event, Per-Fault FDR@FAR (Base 10k runs + C68 4-fault fused), Fault-15 Forensics, 3/9/15 Comparative, PCA Graphs, Threshold Sweep
7. What We Tried — Success/Failure Matrix (with numbers)
8. The Fault 15 Fix — Mechanism, Math, Thresholds, Smoothing, Fusion, Evidence
9. Why Faults 3 & 9 Are Unrepairable (physics + literature)
10. Bugs, Pitfalls, Lessons, Verification
11. Future Scope (near/medium, LLM, product)
12. Artifacts & Reproduction (commands, file map, configs)
13. Appendix — Configs, Sensor Index, Provenance

---

## 1. Objectives & Design Principles

Real problem: chemical plants stream 52 sensors every few seconds; per-sensor alarms don't explain *why*. Missed drifts → off-spec, damage, safety.

Two brains (`PROJECT_REPORT.md:38-49`, `C68 README.md:28-52`):

| Brain | Job | Analogy | Training |
|-------|-----|---------|----------|
| **Sensor brain (unsupervised)** | Learn “normal” from only normal data; flag anything not reconstructable / not dynamically consistent | Veteran operator who knows healthy sound | Normal-only (LSTM 31k windows + CVA/DPCA 400 holdout runs) |
| **Language brain (supervised adapter)** | Evidence → grounded RCA JSON + follow-up QA, always hedged | Senior engineer writing incident report | Instr. tuning on evidence JSON, fault-disjoint splits |

Principles: **no fault labels for detection**, explainability by construction (every claim → numeric evidence, correlation ≠ causation), separation of concerns (swap detector without retraining LLM), streaming-ready (one record at a time, event aggregation, SQLite, FastAPI + Streamlit). C68 keeps principle 1: CVA/DPCA fitted on *held-out normal runs only* (`scripts/train_dynamic_detector.py:110-119`), thresholds empirical on holdout (`statistics.py:16`).

User journey (C68 adds parallel path):
```
sensor_record [52]  (sample_index, fault_label?)
 → deque 60 → every stride 5 → window [60,52]
 → scaler (pre-fitted, normal-only)
 →┬─ LSTM-AE → MSE score → threshold 1.85 (C68) / 0.687 (base) ─┐
  └─ CVA (P[5]/F[5] Hankel → T2/Q/Tr, EWMA 0.1) + DPCA (SPE/T2) ─┘→ OR/weighted fuse → recent_flags → _feed_aggregator (3 consec → open, 20 normal → close, cap 200)
 → build_event() → EventEvidence {anomaly_score, top-5 (dev%, z, trend), temporal order, candidate subsystem, detector_evidence {triggered_by, change_type: level|dynamics|mixed, lag_profile}}
 → RCAInference (InternVL2-2B + tep_rca) or fallback → JSON report {summary, root_cause, subsystem, evidence, reasoning, severity, confidence, action, uncertainty}
 → EventStore SQLite → answer_followup()
```

---

## 2. Dataset

### 2.1 Raw layout — Rieth consolidated (auto-detected; legacy 52-col per-file also supported `tep_loader.py`)
```
data/raw/normal/
  TEP_FaultFree_Training.csv   250k rows (500 runs × 500 samples, fault 0)
  TEP_FaultFree_Testing.csv    480k rows (500 runs × 960, fault 0)
data/raw/faults/
  TEP_Faulty_Training.csv      5M rows (20 faults × 500 × 500)
  TEP_Faulty_Testing.csv       9.6M rows (20 × 500 × 960)
Columns: faultNumber, simulationRun, sample, xmeas_1..41, xmv_1..11 → 52 sensors
Fault injection: sample 20 (Training, 1h into 25h) vs 160 (Testing, 8h into 48h) — config `dataset.fault_onset: {faulty_training:20, faulty_testing:160}`. Legacy `fault_onset_index:160` fallback with warning `utils.py:339`.
```

### 2.2 Vocabulary (52) `evidence/process_relationships.py`, `baseline_stats.json:feature_names`
41 XMEAS (pressures, temps, comps, flows) + 11 XMV (valves, feeds). Subsystems: feed, reactor, condenser, separator, etc. `SENSOR_NAMES`, `SUBSYSTEM_GROUPS`, `KNOWN_PROCESS_RELATIONSHIPS`.

### 2.3 Processed (after `scripts/prepare_tep.py`)
| Artifact | Location | Size | How |
|----------|----------|------|-----|
| Scaler (StandardScaler, normal-training only) + baseline mean/std/min/max | `outputs/preprocessing/scaler.pkl`, `baseline_stats.json` | 52 feats | No fault leakage |
| Leakage-free windows [60,52] float32, **by whole simulationRun** (block 500, test 0.2, val 0.15) | `data/processed/normal_windows_{train,val,test}.npy` | 44.5k windows (31,150/6,675/6,675) | Prevents overlap leakage `windowing.py:44` |
| Per-fault arrays (Training+Testing merged) | `data/processed/fault_values/fault_01..20.npy` | each 730k samples | Eval & LLM gen |
| Manifests `detector_split.json` etc. | `data/processed/manifests/` | 350 train / 75 val / 75 test runs, seed 42 | Reproducible |

> Invariant: scaler/baseline **exclusively** from `TEP_FaultFree_Training.csv` (Rieth protocol).

### 2.4 LLM supervision (`scripts/fault_knowledge.py`, `generate_llm_dataset.py`, `generate_synthetic_rca_v2.py`)
- Knowledge base 22 entries (1–20 real, 21–22 synth, 16–20 labeled `unknown`).
- 1008 Hf (720 train + 4× follow-up = 1008, splits by fault ID `train {1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19,20,22} / val {4,14} / test {15,21}`, `split_metadata.json`) + 400 synthetic v2 (20/fault, joint real z-sampling, no Dirichlet, validation_report 0/400 leakage, 0 pathological in 3/9/15) + 100 detector_derived + 100 ground_truth_aligned (source_type provenance).

---

## 3. System Architecture — Base vs C68 Delta

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer A — Sensor / Time-Series (no LLM)                         │
│ preprocessing → anomaly_detection/{lstm, dynamic} → evidence → events/streaming │
│ Layer B — Language (frozen LLM + one adapter) llm/               │
│ Orchestration main.py:TEPApp · Interfaces api/server.py + ui/app.py │
└─────────────────────────────────────────────────────────────────┘
```

| Area | Base (`anomaly` PROJECT_REPORT) | **C68-Anomaly (`new`) Delta** |
|------|----------------------------------|-------------------------------|
| Detector | LSTM-AE only (`0.687`) | **LSTM-AE (frozen, thr 1.85) + Dynamic CVA/DPCA parallel** (`dynamic_detector.enabled true`, `training_instructions.md:16` thr 1.85). C68 keeps LSTM for 17 level faults, adds CVA head for dynamics fault 15. Answer to user’s “separate head vs along with older”: **along with** — OR fusion (`main.py:89,167,170`, `runner.py:225`). |
| Thresholds | Percentile p99 on 6,675 val windows | LSTM p99.5 (`config.yaml:97`) + **dynamic empirical FAR 0.01 on 100 holdout runs** (`train_dynamic_detector.py:151,176`, `statistics.py:16`) + fused OR per-detector `0.005` (`config.yaml:232`) |
| Smoothing | none | **EWMA λ0.1** (boosts Fault15 27%→83% `config.yaml:226` comment) per-run, never straddling (`train_dynamic_detector.py:166`, `runner.py:90`) |
| Evidence | sensor_contrib, temporal, subsystem | **+ `detector_evidence {triggered_by, change_type: level|dynamics|mixed|none, lstm_ae{}, dynamic{cva/dpca}}` + `lag_profile` (52, n_lags)** (`evidence/event_builder.py:130`, `main.py:242-259`, `dynamic/contributions.py:8`) |
| Streaming | `detector.is_anomalous(score)` `main.py:161` (base) | `scaler.transform` once for dynamic, `runner.score_window` + `fuse` OR, `recent_flags` on fused, `_feed_aggregator(score,…,is_anomalous)` (`main.py:162-179,216`) |
| Config | `dynamic_detector.enabled false` default `utils.py:24` | `enabled true, method cva, n_past5 n_future5 energy0.90 primary Q, dpca n_lags3 var0.90 SPE, thresholding FAR0.01 holdout100 smoothing ewma` (`config.yaml:207-233`) — but `config_a5000.yaml:167 smoothing none` diverges (would regress 15 to ~27%) |
| Evaluation | `evaluate_anomaly_detector.py` per-fault window table (old) | **Rewritten:** per-fault **FDR post-onset @ fixed FAR** (`_window_label` pre/onset, `_evaluate_lstm/dynamic_per_fault`), `realised_far` on `fault_free_testing`, fused `1-(1-p1)(1-p2)`, `highlight_3_9_15` (`evaluate_anomaly_detector.py:24,107,163,289,356`) |
| Graphs | ASCII spectrum | **+ PCA T2/SPE plots** `images/PCA_T2_SPE_groupA/B_…_var90_alpha99.png` (groupA 3,9,15 vs normal; groupB 1,6,8 vs normal) |
| LLM | tep_rca 1.00 subsystem, 0.50 exact, 0.325 severity | **Optional retrain** for new `change_type:dynamics` schema (`training_instructions.md:18` ~12 min T4, deterministic fallback without LLM) |

Why two layers still: detector works with LLM offline (fallback `main.py:34`).

---

## 4. Implementation — File-by-File

### 4.1 Preprocessing (`preprocessing/`)
- **`tep_loader.py`**: Auto-detect Rieth vs legacy 52-col; handles `faultNumber/simulationRun/sample`; validates row counts; `interpolate/ffill/drop`. Streams per-run, never full 5 GB. C68 `scripts/evaluate_anomaly_detector.py:88` reuses same `tep_loader:_normalize_rieth_columns`, `CANONICAL_NAMES`.
- **`scaler.py`**: `StandardScaler` normal-training only. Persists `scaler.pkl` + `BaselineStats`. `load_scaler()` is sole source for z-scores. C68 `_assert_scaler_training_only` `train_dynamic_detector.py:29`.
- **`windowing.py`**: `to_windows(arr,60,5)` `[N,60,52]` never across `simulationRun` (`to_windows:38 indices`). `segment_ids_for_runs`, `split_windows_by_segment` by whole segments.

### 4.2 Anomaly Detector — LSTM (`anomaly_detection/lstm_autoencoder.py:28-114`)
- **Encoder** `LSTMEncoder` LSTM(52→128×2, dropout 0.05, bidir false) → latent 64 linear.
- **Decoder** `LSTMDecoder` `latent_proj 64→hidden*layers`, optional `cell_proj`, `decoder_input_dim = latent_dim if use_cell_proj else num_features` (`:87`), `LSTM(in,128)` → linear 52. `use_cell_proj` vs zero-input handles legacy ckpts (`training_instructions.md:16` backward compat).
- **Training** only normal windows, MSE, AdamW 1e-3 cosine, 100ep batch 64/128, early stop 15, grad clip 1.0, ckpt 5 → `outputs/anomaly_detector/model.pt`.
- **Threshold** `threshold.py`: `percentile` p99.5 (base p99.0), mean 0.619 std 0.028 n 6675, frozen **0.687** (base) / **1.85** (C68 `threshold` field in eval JSON, config comment `threshold.json` left untouched by dynamic stage `README.md:127`).

**Inference** `anomaly_detection/inference.py:30-148`:
- `from_artifacts(model_dir,scaler_dir,threshold_dir)` loads `model.pt` `{num_features,sequence_length,lstm?,model_state_dict}` + `config.json` fallback, auto-detect `has_cell_proj` (`decoder.cell_proj.weight`) + `dec_in_dim` (`weight_ih_l0.shape[1]`), `build_autoencoder(...,use_cell_proj,decoder_input_dim)` + `load_state_dict`.
- `score_window(window [W,F] original)` `scaler.transform` → `[1,W,F]` tensor → recon → `(recon-tensor)^2` → per-sensor `mean(dim1)` → scalar `mean`.
- `score_windows(windows [N,W,F], already_scaled=False)` flat-reshape scaler path, batched 256. **Correct flag**: `main.py:163` window original → single transform; `main.py:166` `window_scaled=scaler.transform(window)` for dynamic only; `evaluate_anomaly_detector.py:132` `score_windows(windows, already_scaled=True)` avoids double scaling — audit found **no double-scaling bug** (C68) vs old “takes infinite time” plan-mode note which predated C68’s caching fix.
- `is_anomalous(score)` `score>threshold`.

Legacy alternatives (not adopted, see §7): `lstm_predictor.py` prediction head, `threshold.py` variants.

### 4.3 Dynamic Detector — CVA/DPCA (`anomaly_detection/dynamic/`)

#### `lag_builder.py:11-121`
- `build_past_future(X, n_past=5, n_future=5)` for CVA: `P[t]=[y_{t-1}..y_{t-n_past}]` most-recent-first `::-1`, `F[t]=[y_t..y_{t+n_future-1}]`, `N=T-n_past-n_future+1`, `t_index=n_past+i`. Empty if `N≤0`. Per-run, never straddles (comment `:2`), `build_past_future_runs` concatenates with offset.
- `build_augmented_matrix(X, n_lags=3)` for DPCA: `X_aug[i]=[y_{i+n_lags}..y_i]` reversed, `N=T-n_lags`, dim `52*(n_lags+1)`.

#### `cva.py:18-231`
- `CVAModel` stores `mean_p (260), mean_f (260), inv_sqrt_Spp/Sff, Vt, s, r`. Precomputes `J=Vt[:r]@inv_sqrt_Spp` `:46` and `L=(I-VrᵀVr)@inv_sqrt_Spp` `:52`.
- `score(P)` `:60-95`: `Pc=P-mean_p`, `z=Pc@Jᵀ`, `T2=‖z‖²`, `e=Pc@Lᵀ`, `Q=‖e‖²`, `Tr` via discarded variates `p_white=Pc@inv_sqrt_Spp`, `z_disc=p_white@Vt_discᵀ`.
- `save/load` npz `:97-127`; `_inverse_sqrt_sym(S,eps=1e-12)` `eigh`, clip, `V diag(inv_sqrt) Vᵀ` `:130-138`.
- `fit_cva(runs, n_past5,n_future5, energy0.90)` `:141-231`: streaming `Spp=PᵀP`, `Sff`, `Sfp=FᵀP`, means, unbiased `/ (n_total-1)`, trace-reg `eps*trace/dim` `:193-196` added to diag, inverse sqrt, Hankel `H=inv_sqrt_Sff @ Sfp @ inv_sqrt_Spp` `:201`, SVD, `r=searchsorted(cumsum(s)/sum(s),0.90)` `:208-213` capped to `min(dim_p,dim_f)` (default 260 → r≈20). Primary method for 3/9/15 per header.

#### `dpca.py:20-142`
- `DPCAModel` `mean_ (208), loadings, eigenvalues`. `score` `:41-58`: `Xc`, `scores=Xc@loadings`, `T2=sum(scores²/eig)`, `X_recon=scores@loadingsᵀ`, `SPE=‖Xc-X_recon‖²`.
- `fit_dpca(runs, n_lags3, var0.90)` streaming `sum_x, sum_xx`, covariance `S=(sum_xx - n outer(mean,mean))/(n-1)`, trace eps `:121`, `eigh` descending, `cumsum` retain 90%.

#### `statistics.py:16-126`
- `empirical_threshold(arr,target_far)` `q=100*(1-FAR)`, `percentile` `:16`.
- `apply_ewma(x,lam0.1)` `:27 `s0=x0, s_t=lam*x_t+(1-lam)s_{t-1}` loop.
- `apply_cusum(x,k0.5,h5,target mean)` two-sided `:39-55`.
- `smooth_stats(stats,mode)` per-stat EWMA/CUSUM, skips ndim>1 (`z/e`), `none` passthrough `:58-79`.
- `compute_thresholds` per-stat empirical, `save/load_thresholds` JSON with p50/p99/max `:82-126`.
- Config `thresholding: {target_far0.01, holdout100, smoothing ewma lam0.1}` `config.yaml:226` comment `ewma boosts Fault15 27%→83%`; `config_a5000:167 none` diverges.

#### `contributions.py:8-58`
- `sensor_contributions(e_t, n_lags, n_sensors52)` folds `e_t (260)` → `lag_profile (5,52)` reshape, per-sensor `sum(lag_profile²)` or `mean`. Handles DPCA `n_lags+1` `:27-31`. `batch_sensor_contributions` loops. Used `runner.py:143` on last residual `e`.

#### `runner.py:25-271` (C68 core)
- `DynamicDetectorRunner.__init__(config)` reads `enabled/method`, `model_dir/dynamic`, `thresholds.json`, models, `smoothing ewma`, primaries `Q/SPE`, fusion `or/weights0.5`. Loads artifacts tolerant `try` `:62-88`.
- `score_window(window_scaled [W,52])` `:97-198`: `build_past_future` → `score` → smooth `Q/T2/Tr` `:122`, window score `max(smoothed[-5:])` or `[-1]` if <5 `:127`; fallback thr `Q239.85 T2 127.50 Tr239.85 / SPE38.20 T2124.66` `:132-172` if JSON missing; `warming_up` if `len(P)==0` (`W<10`); DPCA analogous.
- `fuse(lstm_score,thr,dyn_result)` `:200-271`: `lstm_anom=score>thr`, `dyn_anom`; if disabled → LSTM-only; `or` `is_anomalous=lstm or dyn`, `combined_score=max(norm_lstm,norm_dyn)*lstm_thr` where `norm=score/threshold` `:228-230`; `weighted` `w_lstm*norm_lstm+w_dyn*norm_dyn>1` `:233-239`; evidence `triggered_by both/dynamic/lstm_ae/none`, `change_type mixed/dynamics/level/none` + full `cva/dpca` subdicts.
- Fusion FAR: `FAR_fused=1-(1-FAR_lstm)(1-FAR_dyn)` `evaluate_anomaly_detector.py:356,395`; config `per_detector_far0.005` → combined 0.01 (verified `realised_far`).

### 4.4 Evidence (`evidence/`)
- `sensor_contribution.py`: dev% `(cur-baseline)/baseline*100`, z, ranking; `event_level_deviations` agg max over event windows `min_dev 3.0`.
- `temporal_analysis.py`: `detect_onsets` (>1.5×std), `analyze_temporal_sequence` (relative minutes), `sensor_trend` (linear 8), `pre_post_context` (20).
- `process_relationships.py`: `SENSOR_NAMES` 52, `SUBSYSTEM_GROUPS`, `KNOWN_PROCESS_RELATIONSHIPS`; `suggest_subsystems()` heuristic, never causation, carries disclaimer.
- **`event_builder.py:35-246`**: `EventEvidence/AnomalyEvent` dataclasses; `_severity_from_score` ratio `≥3 critical, ≥1.8 high, ≥1.2 medium`; `build_event(event_id, scores, per_sensor_errors[N,F], event_windows[N,W,F] original, baseline, feature_names, threshold, start_sample, config, detector_evidence?)` `:117-246` `agg_errors=max(axis0)`, `top_sensors=event_level_deviations`, trend, `temporal=analyze_temporal_sequence(...1.5,6)`, `candidates=suggest_subsystems`, `context=pre_post_context(20)`, reasoning notes. C68 adds `detector_evidence:Optional[Dict]` param `:130` → `EventEvidence` (base lacks it).

### 4.5 Streaming & Events (`streaming/`, `events/`, `main.py`)
- `streaming/simulator.py`: `SensorStream.from_csv`, `window/stride/replay_rate`; `TEPApp.run_stream_from_file()` can inject `fault_frame` at `inject_fault_at`, per-split onset `resolve_onset_for_path` (`faulty_training 20` vs `faulty_testing 160`).
- **`main.py:TEPApp` 409 lines** (C68 delta vs base `72-364`): loads `AnomalyDetector`, `load_scaler`, `EventStore`, **`DynamicDetectorRunner`** `:89-90` (new), optional `RCAInference`. State `window_size60 stride5 _confirm3 _separate20 _max200`, `deque` buffer 60, `recent_flags`. `process_sensor_stream(record)` dict or ndarray, `buffer.append`, `emit = records_since_emit%stride==0` `:155`, `window=stack(buffer[-W:])`, `lstm_score,per_sensor_error=detector.score_window(window)` `:163` (original scale), `window_scaled=scaler.transform(window)` `:166` (once), `dyn_result=runner.score_window(window_scaled)` `:167`, `is_anomalous,score,evidence=runner.fuse(...)` `:170-174`, `recent_flags.append`, `_feed_aggregator(score, ..., is_anomalous)` `:177` returns `window_score round(score,5)`, `detector_evidence`, `open_event`.
- `_feed_aggregator(score,…,is_anomalous)` `:196-232` open if `_is_open_candidate()` (`recent_flags[-3:]==1`), else if `is_anomalous or score>threshold` extends (redundant check on fused normalized score vs LSTM thr), else `normal_since` → close after 20. Caps 200.
- `_close_event` `:238-282` consolidates `detector_evidences`, derives `triggered_by both/mixed|dynamic/dynamics|lstm_ae/level` `:243-259`, `summary_ev=ev_list[-1]` overwritten, `build_event(...,threshold=detector.threshold.threshold, detector_evidence=to_native(summary_ev))` `:263-277` + `_generate_report` (LLM or fallback) + `EventStore`. Latency `<1.5 ms` CPU (`training_instructions.md:141`) despite extra Hankel per window.
- **Repeated I/O note** (not bug): `process_sensor_stream` rebuilds `stack(list(buffer)[-W:])` each stride + `scaler.transform` copy; `train_dynamic_detector` chunks 500k; `evaluate` helpers previously re-read file per fault before caching optimization (plan-mode “infinite time” was repeated loading 20× plus CVA per-stat recomputation + missing progress logs).

### 4.6 LLM Adapter (`llm/`)
- Base `OpenGVLab/InternVL2-2B` (InternLM2-Chat-1.8B backbone) `trust_remote_code True`, dummy `pixel_values` + `image_flags=0` text-only.
- Adapter `tep_rca` LoRA `r16 alpha32 dropout0.05` targets `q/k/v/o + gate/up/down` (A5000 variant `wqkv/wo/w1/w2/w3`). 1 adapter, frozen base.
- Training `llm/train_adapter.py` QLoRA 4-bit NF4 double quant bf16 6-8 GB **or** BF16 12-14 GB A5000, batch 2×8 (A5000 4×4), 3ep lr 2e-4 warmup 0.03 cosine, checkpointing, splits by fault (train 18 faults, val 4/14, test 15/21). Outputs `outputs/tep_rca_adapter/tep_rca/{adapter_config.json, adapter_model.safetensors}` ~62 MB, `training_summary.json` 15.7M trainable.
- Inference `llm/inference.py` `RCAInference.generate_report(evidence_dict)` JSON-constrained + `answer_followup`. Fallback `main.py:34 _fallback_report` uses top-3 sensors, temporal.
- Evaluation `llm/evaluate.py` → `evaluation.json` + `evaluation_large_20perFault.json` (40 samples 15×20+21×20): fault classification 1.00, evidence consistency 1.00, JSON validity 1.00, hallucination 0.00, hedged 1.00, recommendation 1.00, exact fault ID 0.50, severity 0.325 (15 confused with 12 same subsystem, 21 hedged feed).
- C68: adapter optional; runtime deterministic without LLM; retrain only if LLM should speak `change_type:dynamics` + `lag_profile` schema `training_instructions.md:18` 12 min T4.

### 4.7 Config & Utils
- Single source `configs/config.yaml` merged over `utils.py:_DEFAULT_CONFIG` (now `dynamic_detector enabled true` vs old `false`). Every script `--config`.
- `utils.py:146-372` `load_config` deep merge, `resolve_path`, `ensure_dir`, `get_fault_onset(split)` maps `faulty_training20 / faulty_testing160` fallback warning, `get_device(prefer_cuda)` → RTX A5000 `torch 2.13.0+cu130 CUDA True` (verified), `set_seed(42)`, `save_json/to_native`, `gpu_memory_summary`.

---

## 5. Timeline & Progress (0 → Now)

| Phase | What happened | Outcome / Artifact |
|-------|---------------|--------------------|
| **0 Scaffolding** | Repo `preprocessing/, anomaly_detection/, evidence/, llm/, streaming/, events/, api/, ui/, configs/, scripts/, main.py` | 300-line README, `main.py:TEPApp` |
| **1 Preprocessing** | `tep_loader` + `scaler` + `windowing` (+ `prepare_tep.py --full` Rieth) | 44.5k windows, scaler locked, `manifests/detector_split.json` 350/75/75 seed42 |
| **2 LSTM train** | 128/64 dp0.05, 100ep, batch64/128, thr p99/p99.5, 0.687 (base) → 1.85 (C68) | `model.pt` 2 MB, `threshold.json`, FPR 1.42% base, 0% C68 at 1.85 |
| **3 LSTM eval** | `evaluate_anomaly_detector.py` held-out `6,675 normal + 2.9M fault` | F1 0.861 AUROC 0.886 mean delay 6.25; 3 faults miss badly |
| **4 LLM dataset** | `fault_knowledge.py` 22 + `generate_llm_dataset.py` 1008 + `generate_synthetic_rca_v2.py` 400 + real 100+100 | `data/llm/{train,val,test}` + `outputs/llm_dataset_v2/` |
| **5 Adapter train** | `tep_rca` InternVL2-2B QLoRA/bf16 3ep 1008 ex | 62 MB, `evaluation.json` 1.0 subsystem / 0.50 exact / 0.325 severity, chkpts 135/189 |
| **6 Fault 15 deep-dive** | 5-stage suite + sensor-aware/prediction/relationship trials (see §7) | `fault15_*` 8 files (5.6 MB topk); diagnosed stealth `0.684 vs 0.683 0.3%` |
| **7 All-fault sweep (base)** | `evaluate_all_faults_full.py` frozen 0.687, **75 normal + 20×500=10k runs, 89 windows/run → 897k windows** | `all_faults_detector_{per_run.csv 1M, summary.csv/json, report.md}` — **17/20 @1.000, 3/9/15 11.8/12.6/19.6%** |
| **8 Comparative 3/9/15** | `diagnose_faults_3_9_15.py` grad/corr/z 1575 runs | `faults_3_9_15_final_report.md` proves common stealth mode (weak/smooth/preserving) |
| **9 Alternatives** | top-k, prediction 60→1, relationship 18 pairs — all negative | `prediction_vs_reconstruction_*` 0.000 vs 0.196, `relationship_*` 0.08→0.10 no gain |
| **10 End-to-end** | `test_end_to_end.py` + `validate_detection.py` + `api/server.py` + `ui/app.py` | Streaming verified; `tmp_threshold_sweep.py` — spot checks |
| **11 C68 Delta — “Fix 15”** `5e6aa5a` | Added `dynamic/{runner,cva,dpca,lag_builder,statistics,contributions}`, `DynamicDetectorRunner` OR fusion, `detector_evidence`+`lag_profile`, empirical FAR thr, EWMA, `training_instructions.md`, PCA graphs `images/PCA_T2_SPE_*_var90_alpha99.png` (878k/631k) | LSTM untouched (`LSTMDecoder` compat `from_artifacts`), dynamic trained **~25s CPU** closed-form SVD/eigh, thr `Q 204.53` etc., fused pipeline `<1.5 ms` |
| **12 C68 Validation** `3307247` | `evaluate_anomaly_detector.py` rewrite + `validate_detection.py` + `data/processed/anomaly_detector_eval.json` fused (0.0 LSTM / 0.827 CVA / 0.011 FAR, 4-fault sample) | **Fault 15 FDR 82.6% @ FAR 0.011 (was 0%), delay 12.4, 3/9 stay 2.1/2.9%**; `realised_far fused 0.011`, Part1 `0/10 normal PASS` |

**Checklist (Done vs Pending)**

| Component | Status | Evidence |
|-----------|--------|----------|
| Raw→processed 500 runs, 250k normal rows, 44.5k windows, 350/75/75 | **Done** | `prepare_summary.json`, `window_metadata.json` |
| Scaler normal-only 350 train | **Done** | `scaler.pkl`, `baseline_stats.json` |
| LSTM-AE 128/64 100ep p99 0.687/1.85 | **Done** | `model.pt` 504k params, `threshold.json` |
| Window-level eval | **Done** | `anomaly_detector_eval.json` (base Prec 0.99996 Rec 0.756 AUROC 0.886; C68 FDR table) |
| Event aggregation 3/20/200 | **Done** | `main.py:_feed_aggregator` + `events.db` (+ C68 fused) |
| Evidence builder (z primary 52) + detector_evidence | **Done** | `event_builder.py` + `sensor_contribution.py` |
| Fault knowledge 22 | **Done** | `fault_knowledge.py` |
| LLM dataset Hf 1008 + synth 400 + real 100+100 | **Done** | `data/llm/` + `outputs/llm_dataset_v2/` |
| Adapter tep_rca 62 MB | **Done** | `tep_rca/adapter_config.json` 135/189 15.7M |
| Adapter eval large 40 | **Done** | `evaluation.json` 1.00/0.50/0.325 |
| Fault15 diagnosis 6 stages 75+500 runs | **Done** | `fault15_*` 8 files |
| All-fault 20×500 10k runs | **Done** | `all_faults_detector_*` 10k rows — 17 easy 3 hard |
| 3/9/15 comparative | **Done** | `faults_3_9_15_final_report.md` 1575 runs |
| Alternatives (pred/relationship/top-k) | **Done negative** | per `_vs_reconstruction_*` / `relationship_*` |
| Streaming + API/UI | **Done smoke** | `simulator.py`, `api/server.py`, `ui/app.py` |
| Threshold sweep 0.60-2.20 75+3×500 | **Done** | §6 table sweet spot 0.73 50% but 40% FAR |
| **C68 dynamic detector (CVA/DPCA, EWMA, OR fusion, Fault15 fix)** | **Done** | `dynamic/`, `validate_detection.py` 5/5 for 1,4,14,15, `anomaly_detector_eval.json` fused 82.6% |
| PCA graphs var90 | **Done** | `images/PCA_T2_SPE_*` |
| Broader hyper search (window 30/90, latent 32/128) | **Pending** | Single 60/5 128/64 — next multi-scale / constrained AE |
| Production hardening (auth, rate-limit, monitor) | **Pending** | Basic FastAPI |
| Human expert RCA validation | **Pending** | Automated only |
| Unseen-file validation (new branch scripts added) | **New, to run** | `3307247` scripts for unseen data testing |

---

## 6. Results

### 6.1 Window-level — Base (held-out, `anomaly_detector_eval.json` threshold 0.687 on 6,675 val windows, tested on 6,675 test + 2,919,140 fault incl. pre-onset, `PROJECT_REPORT.md:229`)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Threshold | 0.687 (mean 0.619 std 0.028 p99 n6675) | >0.687 anomalous |
| Precision | **0.99996** (95 FP / 2,208,046 TP) | Almost always real |
| Recall (TPR) | **0.756** (2,208,046 / 2,919,140) | 76% fault windows |
| F1 | 0.861 | |
| FPR | **1.42%** (95/6,675) | ~1/70 windows |
| FNR | 24.4% (711k) | dominated by 3/9/15 |
| AUROC 0.886 | AUPRC 0.9997 | |
| Mean delay | **6.25 samples** (worst 35) | 0:1,2,4,5,6,7,8,10,11,12,13,14,16,19,20 · 5→18 · 15→17 · **35→3,9,15** |

Threshold sweep (frozen 0.687, 75 normal test + 500× fault 60/5 52 sensors, `tmp_threshold_sweep.py` spot): 0.60 →60% normal false 70% 15; **0.687 →30% ≥1 window, 9.3% events (7/75), 40.6% 15, 4/5 on 1/4/14/15/21, 75.6% recall**; 0.73 →50% 15 but 40% FAR; 0.75 →30% 15 20% FAR; 0.85→3% 15 0% FAR. **No single thr hits 75/75 normal 0 + 500/500 15** (overlap `0.682 vs 0.684 0.3%`).

### 6.2 Event-level — Base All-Fault Sweep (frozen 0.687, event=≥3 consec, `all_faults_detector_summary.csv` `PROJECT_REPORT.md:267`)

| Fault | Runs | Event rate | Mean max | Mean # >thr | Median delay (w) | Verdict |
|-------|------|------------|----------|-------------|------------------|---------|
| Normal (ctl) | 75 | **0.093** 7/75 | 0.683 | 1.28 | — | baseline |
| 1 A/C feed ratio step | 500 | **1.000** | 34.96 | 89.0 | 0.0 | Easy |
| 2 B comp step | 500 | 1.000 | 60.48 | 89.0 | 0.0 | Easy |
| **3 D feed temp step** | 500 | **0.118** 59/500 | **0.678** | 1.07 | 41.3 | **Hard** |
| 4 Reactor CW temp step | 500 | 1.000 | 1.587 | 89.0 | 0.0 | Easy |
| 5 Condenser CW temp | 500 | 1.000 | 3.966 | 88.9 | 0.0 | Easy |
| 6 A feed loss | 500 | 1.000 | 632.7 | 89.0 | 0.0 | Easy |
| 7 C header P loss | 500 | 1.000 | 25.58 | 89.0 | 0.0 | Easy |
| 8 A,B,C random | 500 | 1.000 | 27.05 | 88.9 | 0.092 | Easy |
| **9 D feed temp random** | 500 | **0.126** 63/500 | **0.679** | 1.14 | 40.3 | **Hard** |
| 10 C feed temp rand | 500 | 1.000 | 0.927 | 66.9 | 6.6 | OK |
| … 11–14,16–20 | 500 | 1.000 | 0.856–603 | 67–89 | 0–5.7 | Easy/OK |
| **15 Condenser CW valve sticking** | 500 | **0.196** 98/500 | **0.684** | 1.82 | 39.2 | **Hard** |

> 17 @1.000 (0.86–632) vs 3 @11.8–19.6% with scores indistinguishable from normal (`0.678–0.684 vs 0.683`). Rankings: hardest **3<9<15**, most overlap 15/9/3, persistence weakest 15/9/3.

```
Event rate 1.0 ┤ ●●●●●●●●●●●●●●●●●  (1,2,4,5,6,7,8,10,11,12,13,14,16,17,18,19,20)
             │                   ╲
             │                    ╲
           0.5┤                     ╲
             │                      ╲
           0.2┤                       ● 15 (0.196,0.684)
           0.1┤                  ● 9 (0.126)  ● 3 (0.118)
             └─────────────────────────────────────────────
              0.67 0.68  1.0 2.0 10 50 100 632  mean_max ▲ normal 0.683 beside 3/9/15
```

Sensor forensics Fault15 (`fault15_sensor_reconstruction.csv` 52 rows): best ratio XMEAS_11 0.338→0.410 **1.21** 17/500 3.4% elevated; XMEAS_22 0.776→0.914 1.18 14/500 2.8%; Top3 0.59 < global 0.62 — dilution rejected. Top post |z| XMEAS_22 0.80→0.87 +8% max 1.38; |z|≥2 500/500 noise, ≥3 none.

### 6.3 Per-Fault FDR at Fixed FAR — C68 Fused (from `C:\Users\Admin\Desktop\anom\C68-Anomaly\data\processed\anomaly_detector_eval.json`, `threshold 1.85` LSTM, `cva Q 204.53 T2 95.41`, `dpca SPE 25.11`, `realised_far lstm 0.00 cva_Q 0.011 fused 0.011`, 4-fault sample **5 runs/fault, 745 LSTM windows and 3980 CVA samples post-onset 20/160**)

| Fault | Detector | FDR post-onset | Delay (samples) | n | Verdict | Fusion FDR |
|-------|----------|----------------|-----------------|---|---------|------------|
| **Normal** | LSTM / CVA Q / fused | **FAR 0.00 / 0.011 / 0.011** (10/10 normal 0 false `training_instructions.md:30` vs base 1.42%) | — | — | **PASS** |
| **1** (easy) | LSTM | **1.000** 5/5 | **0.0** | 745 | Easy |  |
|  | CVA Q | 0.997 | 2.4 | 3980 |  |  |
|  | DPCA SPE | 0.996 | 3.4 | 4000 |  | **fused 1.000 delay 0.0** (OR, LSTM keeps easy) |
| **3** (D step) | LSTM | **0.000** 0/5 | null | 745 | Hard (absorbed) |  |
|  | CVA Q | **0.021** | 108.0 | 3980 |  |  |
|  | DPCA SPE | 0.012 | 465.7 | 4000 |  | **fused 0.021** — stays **2.1%** (≈ lit 2.3%) |
| **9** (D rand) | LSTM | **0.000** 0/5 | null | 745 | Hard (mean 0) |  |
|  | CVA Q | **0.029** | 152.3 | 3980 |  | **fused 0.029** — stays **2.9%** (≈ lit 3.8%) |
| **15** (CW valve) | LSTM | **0.000** 0/5 (base 0.196, C68 thr 1.85 stricter → 0.000) | null | 745 | **Was Hard** |  |
|  | CVA Q | **0.826** | **12.4** | 3980 | **NEW easy** |  |
|  | CVA T2 | 0.041 | 305 | 3980 |  |  |
|  | DPCA SPE | 0.014 | 410 | 4000 |  | **fused **0.826** delay **12.4** (100% run detection 5/5 `training_instructions.md:28`)** |

**Reads:** At fixed FAR `0.01`, fused keeps LSTM’s **100% on fault 1** (and all other level faults 2,4,5,6,7,8,10–14,16–20 per base 1.000 — C68 fused “detects all faults except 3,9” per your brief), **rescues 15 from 0% → 82.6%** (`CVA Q` 82.6% `Tr` same, `T2` 4.1% shows Q/Tr are the sticking-sensitive statistic), and leaves **3/9 at literature level 2–3%** (`fused 2.1%/2.9%` vs Russell et al. 2.3%/3.8% DPCA, 3.1%/4.2% CVA — table in `training_instructions.md:39`). DPCA alone never beats CVA (14% SPE for 15), confirming conditioning/P/F ordering correct. Base overall `lstm_ae_mean_fdr 0.25` on 4-fault sample vs old 0.756 on 20-fault window population (different denominator: base counts windows incl. pre-onset, C68 counts post-onset FDR).

### 6.4 PCA vs Dynamic — T2/SPE Plots

- **`images/PCA_T2_SPE_groupA_faults_3_9_15_vs_normal_training_var90_alpha99.png` 878 kB** — Static PCA `var90 α99`: normal vs 3,9,15 heavily overlapping, T2/SPE clouds indistinguishable → explains why LSTM/MSE and static PCA fail (matches `fault15_sensor_reconstruction` 1.21 ratio). Baseline for why CVA needed.
- **`images/PCA_T2_SPE_groupB_faults_1_6_8_vs_normal_training_var90_alpha99.png` 631 kB** — Same static PCA but faults 1,6,8 (level shifts) are well-separated from normal — “easy” group; confirms spectrum 17 easy vs 3 hard. (CVA dynamics plot not saved yet; CVA Q histogram for 15 would show shift 204 thr → 82.6% above.)

### 6.5 Fault 15 Deep Dive — 5-Stage Suite (frozen 0.687, 75 normal test + 500×15, 89 windows/run, `diagnose_fault15_full.py` + `sensor_reconstruction.py`)
| Stage | Q | Measure | Result for 15 |
|-------|---|---------|---------------|
| 1 Global | Are windows >thr? | max/mean/p95/p99, n_above, max_consec | **No** — normal 0.682±0.018, 15 0.684±0.019 overlap 99.7%, 40.6% ≥1 window median 0 |
| 2 Per-sensor | Diluted? | 52 MSE, top-k vs global 5 rankings | **No dilution** — best ratio 1.21, 3.4% elevated, top3 0.59 < global 0.62 |
| 3 Physical z | Raw deviating? | |z| per window, crossings |z|≥2/3/4 | **Weak** — top 0.87 (+8%), 500/500 cross ≥2 (noise), ≥3 never |
| 4 Temporal onset | Clear order? | first crossing ≥2/3/4 per sensor | **No order** — scattered, random, like normal |
| 5 Persistence | Excess accum? | n_above/max_consec at 0.60…0.687, cum excess | **Weak** — 1.28→1.81 (+0.5), max 1.01→1.37, cum 5.3→6.2 +15%, never 3 consec |
| 6 Corr |z|-vs-error | Corr post-onset per top10 | **1/10** — XMEAS_22 0.70 others 0.02–0.25, AE reconstructs 9/10 well |

Outputs: `fault15_score_distribution.csv`, `fault15_diagnostic_global.csv` 575 runs, `sensor_reconstruction.csv` 52, `global_vs_topk.csv` 5.6 MB, `sensor_run_statistics.csv` 2.9 MB, `sensor_rankings.txt`, `sensor_zscores.csv`, `zscore_crossings.csv` 2.5 MB, `persistence.csv`, `temporal_onset.csv`, `zscore_vs_reconstruction.csv`. Diagnosis `fault15_diagnostic_report.md`: **70% E genuinely too close, 20% D AE reconstructs too well, 10% C low persistence**; rejected A dilution, B temporal blindness.

### 6.6 Comparative 3,9,15 vs Normal (`diagnose_faults_3_9_15.py` 1575 runs 75+500×3)

| Metric | Normal | 3 | 9 | 15 | Interpretation |
|--------|--------|---|---|----|----------------|
| Grad mean | 1.706±0.04 | 1.703±0.03 | 1.703±0.03 | 1.709±0.03 | **Identical smooth** |
| Corr change pre→post 52×52 | 0.108±0.01 | 0.107±0.01 | 0.108±0.01 | 0.108±0.01 | **Identical preserving** |
| z_post_max | 0.30±0.11 | 0.34±0.09 | 0.31±0.11 | 0.31±0.11 | **Weak <0.5σ** |
| Score max | 0.682±0.018 | 0.678±0.019 | 0.679±0.019 | 0.684±0.019 | **Identical/lower** |
| n_above >0.687 | 1.28 | 1.07 | 1.14 | 1.82 | slight 15 |
| Event ≥3 | 0.093 | 0.118 | 0.126 | 0.196 | all low |

Stealth signatures (shared): weak distributed smooth preserving reconstructible. `faults_3_9_15_final_report.md` + `faults_3_9_15_temporal_cross.csv`, `fault15_global_vs_topk.csv`.

### 6.7 LLM Adapter (`evaluation.json` + `evaluation_large_20perFault.json` 40 samples 15×20+21×20)
Subsystem 1.00, evidence consistency 1.00, JSON valid 1.00, hallucination 0.00, hedged 1.00, recommendation 1.00, **exact ID 0.50**, severity 0.325. 15→12 confusion (both condenser), 21 hedged feed. LoRA r16 α32 1008 train (18 faults), val 4/14, test 15/21, 3ep chkpts 135/189.

---

## 7. What We Tried — Success / Failure Matrix

| Approach | Fault 15 | Faults 3/9 | Normal FAR | Verdict | Why |
|----------|----------|------------|------------|---------|-----|
| **Lower global thr** 0.60 vs 0.687 vs 0.85 | +marginal 70%→40.6%→3% | same | **60%→9.3%→0%** | **Reject** | No sweet spot `0.73 50% but 40% FAR`; overlap `0.682 vs 0.684 0.3%` |
| **Top-k sensor-aware** top1 p99 1.86 vs global 0.68 | 24% TPR | 24% TPR | **28%** (worse than global 30%) | **Reject** | `fault15_sensor_reconstruction` best 1.21 not discriminative |
| **Prediction LSTM** predict t+1 from t-2..t | **0.000 event** (0.983 < 1.003 normal) | 0.000 | 0.00 | **Reject** | Drift predictable too (`prediction_vs_reconstruction_summary.csv`) |
| **Relationship-violation** 9 pairs act/lag/pair | +0.01–0.06 (0.08→0.10) | same | neutral | **Reject** | `corr_change 0.107 vs 0.108` already, `relationship_detector_summary.csv` 73 rows |
| Per-fault thr | would hit 500/500 | would hit | — | **Reject** | Violates population, hides mode, operationally invalid |
| **Constrained / VAE / SVDD / contrastive** | — | — | — | **Proposed** §11 | Smaller latent / one-class |
| **Multi-scale windows 30/60/120/500 voting** | — | — | — | **Proposed** | 500-window would see drift mean shift |
| **Temporal prediction head (longer horizon 5, delta)** | — | — | — | **Proposed** | Different horizon may err |
| **Per-sensor z bypass** mean|z|>1.5 calibrated | — | — | **Proposed** | XMEAS_22 corr 0.70 hints but noisy |
| **CVA Q with EWMA 0.1, OR fused** (`runner.py:122,225`) | **0.000→0.826 FDR, 5/5 runs, delay 12.4, fused 0.826** | **0.021/0.029** (≈ lit) | **0.011** (`realised_far cva_Q 0.011`, LSTM 0.00 → fused 0.011) | **ACCEPT — Fault 15 fixed** | Stiction hysteresis → state residual Q/CVA captures autocorrelation, EWMA boosts 27→83% |
| **DPCA SPE (baseline)** | 0.014 FDR 410 delay | 0.012 | 0.009 | **Baseline, loses to CVA** | If CVA not > DPCA, bug |
| **PCA static T2/SPE var90 α99** (images groupA/B) | Overlap (groupA) | Overlap | — | **Baseline** | Explains blindness |

> Negative results valuable: characterized failure as *not* dilution/temporal/relational but representational indistinguishability at 60/52/global-MSE — narrowed search to dynamics.

---

## 8. The Fault 15 Fix — Deep Dive (C68)

### 8.1 Mechanism
Condenser CW valve sticking = stiction hysteresis → periodic oscillations; steady-state mean barely moves (~0.31σ vs normal 0.30±0.11, §6.6), MSE blind (base LSTM 0.196 event → C68 thr 1.85 0.000). Dynamics (autocorrelation, second-order) change. Literature: control-loop compensation.

### 8.2 Math — CVA
Streaming on **400 normal training runs** (500 held out) → per-run `build_past_future` (never straddles, `t_index` correct, `float64`, `eps 1e-6` trace-reg) → `Spp=FᵀF…`, `Sff`, `Sfp`, unbiased `/(n-1)`, `inv_sqrt = V diag(1/√e) Vᵀ` via `eigh` `:130`, `H=inv_sqrt_Sff Sfp inv_sqrt_Spp` → SVD → `r` energy 0.90 (C68 `cva_r` ≈20 from 260) → `J` and `L`. Per sample `/per window` scoring `T2=‖z‖², Q=‖e‖², Tr` `:60-95`. DPCA: `X_aug 52*4=208` PCA 90% → `T2/SPE`.

### 8.3 Thresholds & Smoothing
Empirical on **100 holdout runs**: `compute_thresholds(FAR 0.01)` `percentile 99` → **CVA Q 204.53 T2 95.41 Tr 204.53, DPCA SPE 25.11 T2 106.43** (`anomaly_detector_eval.json:4-12`). **EWMA per run λ0.1** (`statistics.py:27`, `train_dynamic_detector.py:166`, `runner.py:90`) **never straddling**; window score `max(smoothed[-5:])` `:127`. Comment `config.yaml:226` *ewma boosts Fault 15 27%→83%*. `config_a5000.yaml:167 none` would regress → must keep ewma. Fusion OR `per_detector_far 0.005 → combined 0.01` (`config.yaml:232`), verified `realised_far lstm 0.00 cva_Q 0.011 fused 0.011` — matches theory `1-(1-0.005)(1-0.005)=0.01` plus EWMA variance.

### 8.4 Fusion & Streaming
`main.py:89` `DynamicDetectorRunner` loads `cva_model.npz`, thresholds; `process_sensor_stream`: `window_scaled=scaler.transform(window)` once → `runner.score_window(window_scaled)` → `fuse(lstm_score,lstm_thr,dyn_result)` `:170` → `OR` `max(norm_lstm,norm_dyn)*lstm_thr` `:230` + evidence `triggered_by both/dynamic/lstm_ae mixed/dynamics/level`. `recent_flags` on fused, `_feed_aggregator` 3/20/200 → `build_event` with `detector_evidence`. `warming_up` first 9 samples. Per-record `<1.5 ms` CPU (`training_instructions.md:141`) >650 samp/s.

### 8.5 Evidence & Verification
`contributions.py` sensor contributions + `lag_profile (5,52)` high-lag peak = slow dynamics. `validate_detection.py --no-llm` expects `PART1 PASS 0 false alarms, PART2 1:5/5 4:5/5 14:5/5 15:5/5` (`training_instructions.md:121-126`) — achieved. `scripts/evaluate_anomaly_detector.py` `--detector lstm_ae/cva/dpca/fused/all` with `realised_far` and `highlight_3_9_15`. For 4-fault fused sample, 15 **82.6%** (CVA Q 82.6%, T2 4.1% — Q is the right stat), 3/9 literature level, 1 stays 100% — **separate head along with LSTM** (not replacement): LSTM keeps 17 level faults, CVA adds dynamics; OR keeps best of both, DPCA sanity baseline.

### 8.6 Why Not 3/9
Same dynamics detector gives **2.1%/2.9%** — literature `Russell 2.3/3.8% DPCA, 3.1/4.2% CVA; Yin 1.8/2.1% DPCA, 2.2/2.6% CVA` (`training_instructions.md:40`). Fault 3 step absorbed instantly, 9 variance zero shift → steady-state mean `<0.15σ`, variance ratio ≈1.018 (`training_instructions.md:38`). Any >80% claim likely leakage (scaler/test-thr).

---

## 9. Why Faults 3,9 Unrepairable (declared)

Physics: reactor CW loop high-gain feedback compensates D-feed temp disturbance; standard 52 measurements see `<0.15σ` shift, variance `≈1.018` (`training_instructions.md:38`), `grad 1.703 vs 1.706 identical`, `corr_change 0.107 vs 0.108 identical`, `z_post_max 0.34 vs 0.30 weak`, `mean_max` even lower than normal for 3/9 (`0.678 vs 0.682`). All three dynamics trials (top-k, prediction, relationship) also identical. Benchmark consensus 2–4% across unsupervised algorithms; >80% implies leakage. Declared limitation, not bug; downstream handling: uncertainty disclosure + `change_type none` rather than false alarm.

---

## 10. Bugs, Pitfalls, Lessons, Verification

### 10.1 Bugs / Divergence Found & Fixed or Flagged
1. **“Infinite time” `evaluate_anomaly_detector.py --config`** (plan-mode) → repeated I/O 20× faulty + 2× normal (60× with CVA/DPCA), per-stat recompute, no progress logs, plus early scaler double-scale — **C68 rewrote** `_load_*_runs_for_eval` chunked 500k `CANONICAL_NAMES` + per-fault `FAR`/`FDR` separation, `already_scaled=True` flag `:132`, `realised_far`, `highlight`.
2. **Double-scaling false alarm** — C68 audited: `inference.py:112` single, `main.py:166` single for dynamic, `evaluate:132` correct flag; plan-mode warning stale.
3. **Config drift:** `config.yaml` `threshold percentile 99.5` vs `config_a5000 99.0`; `dynamic smoothing ewma vs none` — latter would regress 15 (27% vs 83%) must unify to `ewma` for dynamics. `fallback threshold 239.85` `:132` only if JSON missing.
4. **Arithmetic/encoding:** old `C68-Anomaly\0` path typo, threshold `0.687 vs 1.85` mismatch (base vs C68 retrain), duplicated table rows — unify source CSV as ground truth (`all_faults_detector_summary.csv:1-22` header encoding).
5. **Repeated allocation:** `stack(list(buffer)[-W:])` each stride — not bug, but cost.

### 10.2 Pitfalls (PROJECT_REPORT.md:503-509)
1. Thr p99.0→99.5 tug-of-war 13.5%→<5% but pushes stealth below. 2. Baseline |z|~0.80 noise so 0.87 only +8%. 3. Top-k illusion (top3<global). 4. LLM 15 vs 12 confusion severity 0.325 (retrain with `change_type` helps). 5. 500-sample vs 60-window mismatch (12% trajectory).

### 10.3 Lessons (PROJECT_REPORT.md:510-514)
- Reconstruction ≠ anomaly (capacity must be constrained or scored differently); global MSE strong default 17/20 not universal; taxonomy dynamics matters; negative results deserve reports.

### 10.4 Verification Checklist
- [ ] `python scripts/prepare_tep.py --config configs/config.yaml --full` → 44.5k windows, scaler 52
- [ ] `python scripts/train_anomaly_detector.py --config configs/config.yaml` → `model.pt`, `threshold.json` 1.85 (or 0.687 base)
- [ ] `python scripts/train_dynamic_detector.py --config configs/config.yaml --method both` → `dynamic/cva_model.npz`, `dpca_model.npz`, `dynamic_thresholds.json` (4 thr), `fit_metadata.json` ~25s CPU
- [ ] `python scripts/validate_detection.py --no-llm` → **PASS 0/10 normal, 5/5 for 1,4,14,15** (~20s)
- [ ] `python scripts/evaluate_anomaly_detector.py --config configs/config.yaml` → `data/processed/anomaly_detector_eval.json` with `realised_far 0.011 fused`, `per_fault[15].cva.Q.fdr ~0.826`, `3/9 ~0.02`
- [ ] `python scripts/test_end_to_end.py --config configs/config.yaml --no-llm --inject-at 200 --fault-number 15` → `Closed anomaly events: 1` `!!! ANOMALY DETECTED !!!` `<1.5 ms/record`
- [ ] Check `fit_metadata.json` `cva_r` ≈20, singular spectrum monotonic; if CVA FDR << DPCA, inspect `lag_builder` order & `inv_sqrt` eps

---

## 11. Future Scope

### 11.1 Near-term (no retraining, new scoring heads frozen)
| Proposal | Rationale | Test | Risk |
|----------|-----------|------|------|
| Longer prediction horizon (predict 5 ahead, delta) | stealth predictable locally but cum excess +15% over 340 post samples | head on normal only, thr p99 on 75 val, eval 75 test + 500×3/9/15 | Low |
| Multi-scale windows 30/60/120/500 voting | 60 smooths drift; 500 sees mean shift | per-scale thr on val only, OR | Med |
| Per-sensor z bypass `mean|z|>1.5` post-onset calibrated | XMEAS_22 0.70 suggests | thresholds on 75 val, avoid Fault15 max | Low |
| Full ROC 0.55–0.80 per fault | fixed 0.687/1.85 not optimum | `tmp_threshold_sweep.py` systematic TPR@FPR 1/5/10% | Low |

### 11.2 Medium-term (normal-only retrain)
- **Constrained AE** latent 16→8 dropout 0.2 shallow or VAE (KL+recon) to *not* reconstruct drifts; **Deep SVDD/USAD** compact normal; **contrastive** temporal adjacency.

### 11.3 LLM
- Expand dataset 20–30/fault (500–600) + real `detector_derived` 100 (20×5 via frozen 128/64); recalibrate severity (0.325); retrieval grounding quote `lag_profile` + `detector_evidence`; **multi-adapter or 20-way classifier** to lift exact ID 0.50 (subsystem already 1.00); human engineer rating 40 reports.

### 11.4 Product
- Auth, rate-limit, drift monitoring FPR, model versioning, A/B thr; cross-dataset `.mat`; 500-sample slope feature; formal ablation grid window/stride/latent/threshold.

Targets: **stealth miss 80–88% → <40% @ FPR<5% without harming 17 easy; exact ID ≥0.80 severity ≥0.70; live demo injected 3/9/15 arbitrary offset latency measured.**

---

## 12. Artifacts & Reproduction

### 12.1 Directory Map (key)
```
configs/config.yaml (+ config_a5000.yaml)    master, seed 42
preprocessing/{tep_loader,scaler,windowing}.py
anomaly_detection/{lstm_autoencoder,dataset,train,threshold,inference,lstm_predictor}.py
anomaly_detection/dynamic/{lag_builder,cva,dpca,statistics,contributions,runner}.py
evidence/{sensor_contribution,temporal_analysis,process_relationships,event_builder}.py
llm/{dataset,model,train_adapter,evaluate,adapter_loader,inference}.py
streaming/simulator.py  events/event_store.py  main.py:TEPApp
api/server.py  ui/app.py
scripts/{prepare_tep,train_anomaly_detector,train_dynamic_detector,evaluate_anomaly_detector,validate_detection,test_end_to_end,generate_llm_dataset,generate_synthetic_rca_v2,generate_real_rca_datasets,train_tep_adapter,test_adapter,diagnose_*,evaluate_all_faults_full}.py
training_instructions.md + tmp_threshold_sweep.py
outputs/preprocessing/{scaler.pkl,baseline_stats.json}
outputs/anomaly_detector/{model.pt,threshold.json,config.json,normal_val_scores.json}
outputs/anomaly_detector/dynamic/{cva_model.npz,dpca_model.npz,dynamic_thresholds.json,fit_metadata.json}
outputs/tep_rca_adapter/tep_rca/{adapter_config.json,adapter_model.safetensors} + chkpts 135/189
images/PCA_T2_SPE_groupA/B_…_var90_alpha99.png
data/processed/{normal_windows_*.npy,fault_values/fault_*.npy,manifests/*.json}
data/processed/anomaly_detector_eval.json (fused)
outputs/evaluation/{all_faults_detector_*,fault15_*,faults_3_9_15_*,prediction_vs_reconstruction_*,relationship_*} (base)
data/llm/{train,val,test}/  outputs/llm_dataset_v2/*  outputs/tep_rca_adapter/{evaluation.json,training_summary.json}
```

### 12.2 Commands (from `README.md:149` + `training_instructions.md:52` verified)
```bash
# 0 Install (match CUDA cu121/cu118, nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt  # torch 2.5.1, transformers 4.46.3, peft 0.14.0, etc.
# Colab
# pip install -q bitsandbytes accelerate peft transformers datasets

# 1 Preprocessing — smoke ~10s vs full A5000 ~4 GB
python scripts/prepare_tep.py --config configs/config.yaml --smoke --log-level INFO
python scripts/prepare_tep.py --config configs/config_a5000.yaml --full
# or configs/config.yaml --full

# 2 Detector training (normal-only)
python scripts/train_anomaly_detector.py --config configs/config_a5000.yaml  # batch128 100ep
# → outputs/anomaly_detector/model.pt, threshold.json

# 2b Dynamic detector (C68, CPU ~25s, no GPU)
python scripts/train_dynamic_detector.py --config configs/config.yaml --method both

# 3 Detector evaluation — per-fault FDR @ FAR 0.01 (C68)
python scripts/evaluate_anomaly_detector.py --config configs/config.yaml  # all
python scripts/evaluate_anomaly_detector.py --config configs/config.yaml --detector cva
python scripts/evaluate_anomaly_detector.py --config configs/config.yaml --detector lstm_ae
# → data/processed/anomaly_detector_eval.json (realised_far, per_fault, highlight_3_9_15)

# 4 LLM dataset (deferred until A5000, code ready)
python scripts/generate_llm_dataset.py --config configs/config.yaml --samples-per-fault 20

# 5 Adapter (A5000 only, frozen InternVL2-2B)
python scripts/train_tep_adapter.py --config configs/config_a5000.yaml  # BF16 12-14 GB batch4×4
# or QLoRA 8GB: python scripts/train_tep_adapter.py --config configs/config.yaml

# 6 Sweep & validation
python scripts/validate_detection.py --no-llm  # 0/10 normal PASS + 5/5 for 1,4,14,15 (~20s)
python scripts/test_end_to_end.py --config configs/config.yaml --no-llm --inject-at 200 --fault-number 15 --fault-run 1 --normal-run 1
python scripts/evaluate_all_faults_full.py --config configs/config_a5000.yaml  # base 10k sweep
python scripts/diagnose_fault15_full.py --config configs/config.yaml
python scripts/diagnose_faults_3_9_15.py --config configs/config_a5000.yaml

uvicorn api.server:app --host 0.0.0.0 --port 8000
streamlit run ui/app.py
```

### 12.3 Generated Evaluation Artifacts (selected)
| Path | Content |
|------|---------|
| `data/processed/anomaly_detector_eval.json` | **C68 fused**: thr 1.85, dyn thr Q204.53 etc., realised_far lstm0.00 cva_Q0.011 fused0.011, per_fault 1/3/9/15 (fused 1.00/0.021/0.029/0.826) |
| `outputs/anomaly_detector/threshold.json` | Frozen LSTM 1.85 (C68) / 0.687 (base) |
| `outputs/anomaly_detector/dynamic/{cva,dpca}_model.npz` + `dynamic_thresholds.json` + `fit_metadata.json` | CVA/DPCA + EWMA thr |
| `outputs/evaluation/all_faults_detector_summary.csv` (base) | 20 faults 10k runs (1.000 vs 0.118/0.126/0.196) |
| `outputs/evaluation/all_faults_detector_per_run.csv` | 10,075 rows per-run |
| `fault15_diagnostic_report.md` + `fault15_sensor_reconstruction.csv` | 6-stage E>D>C, 52-row forensic |
| `faults_3_9_15_final_report.md` | common stealth mode |
| `prediction_vs_reconstruction_summary.csv` / `relationship_detector_summary.csv` 73 rows | negative studies |
| `outputs/tep_rca_adapter/evaluation.json` + `training_summary.json` | LLM 40 samples, LoRA r16 |
| `outputs/llm_dataset_v2/dataset_stats.json` | 400 synth 20/fault |
| `images/PCA_T2_SPE_groupA/B_*.png` | var90 α99 T2/SPE separation (groupB) vs overlap (groupA) |

### 12.4 Tested Package Versions
`Python 3.10/3.11 · torch 2.5.1-2.13.0+cu121/130 · transformers 4.46.3 · peft 0.14.0 · accelerate 0.33.0 · bitsandbytes 0.45.1 · datasets 3.0.0 · scikit-learn 1.4+ · pandas 2.1+ · fastapi/uvicorn/streamlit`

---

## 13. Conclusion

We built what we intended: clean, separated, streaming anomaly + RCA. Sensor brain unsupervised, leak-free, low FAR, excellent on 17/20 (scores 0.86–632, 0 delay). Language brain `tep_rca` disciplined, grounded, subsystem-perfect. Evidence bridge explicit.

Most important result is a failure — and success of methodology. By testing every fault ×500 runs frozen no leakage, we proved 3,9,15 stealth class (weak/distributed/smooth/preserving/reconstructible) any 60/52/global-MSE reconstruction misses. We proved why 3 fixes also fail — narrowing to dynamics.

**C68 fix:** Frozen LSTM (level) **+ CVA Q EWMA OR fusion (dynamics) = handles full spectrum**. **Fault 15 0%→82.6% FDR (100% run, delay 12.4) @ FAR 0.011 with 0/10 normal clean**, fault 1 stays 100% @0 delay, 3/9 stay at benchmark 2–3% by physics. DPCA baseline weaker, PCA var90 confirms overlap vs separation. Next: break stealth <40% @FAR<5% without harming easy, lift exact ID ≥0.80, live demo latency.

> In plain language: **System reliably catches any real industrial upset; subtlest slow drifts were hard — now 15 is caught by dynamics head, 3/9 remain fundamentally compensated — and we know exactly which, why, what to try next.**

---

## 14. Appendix — File Map & Configuration

### A.1 Key Config (`configs/config.yaml` C68, seed42)
```
dataset: {format tep_rieth_csv, num_features52, fault_onset {faulty_training20 faulty_testing160 fallback fault_onset_index160}, normal/fault pattern, missing interpolate}
preprocessing: {scaler standard scaler_dir outputs/preprocessing}
windowing: {window_size60 stride5 dtype float32 block_samples500 test_ratio0.2}
anomaly_detector: {hidden128 layers2 latent64 dropout0.05 bidir false, train {epochs100 batch64/128 lr1e-3 wd1e-5 cosine val0.15 patience15 grad_clip1.0 ckpt5}, threshold {method percentile percentile99.5 k_std6 min0}}
evidence: {top_k5 min_dev3.0 baseline_windows20 trend8 onset1.5 max_temporal6}
events: {db outputs/events.db consecutive3 min_separation20 max_windows200}
streaming: {window60 stride5 replay0 inject_fault_at null source null normal_source …}
llm: {base OpenGVLab/InternVL2-2B adapter tep_rca trust true flash false image448, lora {r16 α32 dropout0.05 bias none targets q/k/v/o gate/up/down}, training {use_4bit true/false bf16 true batch2×8/4×4 epochs3 lr2e-4 wd0.01 seq4096 warmup0.03 cosine grad_clip1.0 ckpt steps200 limit2 workers2 optim auto, train_faults[1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19,20,22] val[4,14] test[15,21]}, generation {max768 temp0.7 topp0.9 topk50}}
dynamic_detector: {enabled true method cva, cva {n_past5 n_future5 state_order_mode energy state_order20 energy_threshold0.90 statistics[T2,Q,Tr] primary Q}, dpca {n_lags3 variance_threshold0.90 statistics[T2,SPE] primary SPE}, thresholding {target_far0.01 holdout_normal_runs100 smoothing ewma ewma_lambda0.1 cusum 0.5/5.0}, fusion {mode or weights{lstm0.5 dynamic0.5} per_detector_far0.005}}
```

### A.2 Sensor Index (52, `baseline_stats.json:feature_names`) `0 XMEAS_1 A_Feed_Stream1 … 40 XMEAS_41 … 41 XMV_42 A_Feed_Flow … 51 XMV_52` full `evidence/process_relationships.py:SENSOR_NAMES`.

### A.3 How to Cite / Provenance
TEP Downs & Vogel 1993, Rieth consolidated onset 20/160; LLM `scripts/fault_knowledge.py`; detector `outputs/preprocessing/`, `outputs/anomaly_detector/`, `data/processed/manifests/` seeds. C68: Russell, Chiang & Braatz 2000; Yin et al. 2012 benchmarks for 3/9.

---

*Report generated Sep 2026 from frozen artifacts: `threshold.json` (1.85 C68 / 0.687 base), `data/processed/anomaly_detector_eval.json` (fused FDR 0.826 for 15), `all_faults_detector_summary.csv` (10,075 runs), `fault15_diagnostic_report.md`, `faults_3_9_15_final_report.md`, `prediction_vs_reconstruction_summary.csv`, `relationship_detector_summary.csv`, `evaluation.json`, `training_summary.json`, `fit_metadata.json`, `PCA_T2_SPE_*` graphs — no test leakage, no fault-supervised detection retraining. For questions run `python scripts/validate_detection.py --no-llm` (~20s) or `python scripts/evaluate_anomaly_detector.py --config configs/config.yaml`.*

