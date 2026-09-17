# TEP RCA System

Industrial sensor anomaly detection and **automatic root-cause analysis** for the
Tennessee Eastman Process (TEP) using an **LSTM autoencoder** (unsupervised) and
**InternVL2-2B** fine-tuned with **ONE LoRA/QLoRA adapter** named **`tep_rca`**.

```
Continuous sensor stream
   -> preprocessor -> unsupervised anomaly detector
   -> event aggregation -> evidence extraction
   -> InternVL2-2B + tep_rca adapter -> automatic RCA report
   -> event store -> user follow-up questions -> conversational answers
```

The trained artifact is:

```
ORIGINAL InternVL2-2B   (frozen, untouched)
+  tep_rca LoRA/QLoRA adapter   (independently saveable / loadable)
```

Everything else (scaler, LSTM autoencoder, threshold, evidence builder, event
store, streaming simulator, API, UI) remains **external** to the adapter.

---

## 1. Architecture

Three intelligence layers (dynamic added additive, LSTM untouched):

| Layer | Module(s) | Responsibility |
|-------|-----------|----------------|
| **A. Sensor / time-series pipeline** | `preprocessing/`, `anomaly_detection/lstm`, `anomaly_detection/dynamic`, `evidence/`, `streaming/`, `events/` | raw stream, normal-only scaler, LSTM AE + **CVA/DPCA lag-aware** (parallel), fusion, per-sensor + `lag_profile` evidence, event aggregation |
| **B. InternVL2-2B TEP adapter** | `llm/` | understands `detector_evidence` + `change_type` (level vs dynamics) + `lag_profile`, root-cause reasoning |
| **Dynamic details** | `anomaly_detection/dynamic/{lag_builder,cva,dpca,statistics,contributions}` | CVA primary, DPCA baseline, `T2/Q/Tr` + `SPE`, empirical `FAR 0.01` thresholds, `or` fusion |

The adapter weights contain **only** learned InternVL LoRA parameters. No sensor
logic is inside the adapter.

```
Pipeline data flow (with dynamic detector — runs in parallel, fused before aggregation)
------------------
sensor_stream
  -> preprocessor (missing values, validation, scaler fitted on NORMAL only)
  ->+- LSTM autoencoder (reconstruction MSE) -+
    |                                          |-> fusion (OR / weighted, per-detector FAR) -> event_aggregator
  ->+- DPCA / CVA (lag-aware, T2/Q/Tr) -------+   (never straddles simulationRun)
  -> evidence_generator (top sensors + detector_evidence {triggered_by, change_type: level|dynamics|mixed, lag_profile})
  -> InternVL2-2B + tep_rca adapter -> automatic report (JSON)
  -> event_store (SQLite) -> user_follow_up -> adapter -> answer
```
**What the dynamic stage does:** The LSTM AE sees *level* shifts (MSE). Faults 3 (D-feed temp step absorbed by controller), 9 (random variance, mean 0), and 15 (valve sticking, autocorrelation) change *temporal/second-order* structure, not level, so MSE is blind. CVA (primary) builds past `P=[y_{t-1}..y_{t-5}]` / future `F=[y_t..y_{t+4}]` Hankel per `simulationRun` (never straddling, `lag_builder.py`), fits `Spp/Sff/Sfp` streaming, `H=Sff^{-1/2}SfpSpp^{-1/2}`, SVD, keeps `r` via `energy 0.90` (`state_order 20`), then emits `T2=z^Tz`, `Q=e^Te`, `Tr` per sample. `Q` is most sensitive for 3/9/15 (literature). DPCA (`n_lags 3`, PCA `90% var`) is the cheap baseline — if CVA does not beat DPCA, the inverse-sqrt conditioning or past/future ordering is wrong. Thresholds are empirical percentiles on *held-out* fault-free `100` runs (not chi-square), with optional `EWMA 0.1`/`CUSUM` smoothing. `fusion` `or` (each at `per_detector_far 0.005` → combined `0.01`) or `weighted` (percentile-rank normalisation). Evidence now carries `detector_evidence` + `lag_profile` so the LLM can distinguish `level` vs `dynamics` change.


## 2. TEP dataset assumptions

The loader isolates all format-specific logic in `preprocessing/tep_loader.py` and auto-detects both formats.

* **Format `tep_rieth_csv` (current, default)** — Rieth et al. consolidated files:
  ```
  data/raw/normal/TEP_FaultFree_Training.csv  (250k rows, 500 runs × 500 samples, fault 0)
  data/raw/normal/TEP_FaultFree_Testing.csv   (480k rows, 500 runs × 960 samples, fault 0)
  data/raw/faults/TEP_Faulty_Training.csv     (5M rows, 20 faults × 500 runs × 500 samples)
  data/raw/faults/TEP_Faulty_Testing.csv      (9.6M rows, 20×500×960)
  Columns: faultNumber, simulationRun, sample, xmeas_1..41, xmv_1..11 (52 sensors)
  Fault injected at sample 20 (0-based) for Training (1 h into 25 h run) and
  at sample 160 (0-based) for Testing (8 h into 48 h run) — `dataset.fault_onset`
  mapping `faulty_training: 20, faulty_testing: 160` (fault-free splits: null).
  ```
* **Format `tep_52col_csv` (legacy)** — one file per run: `data/raw/normal/normal*.csv`, `data/raw/faults/fault_XX.csv` (T×52 matrix, no meta cols). Still supported.
* Missing values: `interpolate | ffill | drop` (`dataset.missing_value_strategy`).
* To add another public TEP format (e.g. `.mat`) add a `TEPDataFormat` member and a loader in `tep_loader.py`; nothing else changes.

## 3. How the system behaves (section 30 walkthrough)

1. **Sensor data enters** via `TEPApp.process_sensor_stream(record)` (one record
   at a time) or the `SensorStream` simulator which replays CSVs as a live
   stream with configurable `window_size` / `stride` / `replay_rate`.
2. **Anomaly detection** is an LSTM autoencoder trained **only on normal**
   windows (MSE reconstruction loss). The scaler is fitted on normal data only
   and saved separately (`outputs/preprocessing/`).
3. **Anomaly events** are created only after
   `events.consecutive_windows_to_confirm` consecutive anomalous windows; the
   run is grouped into a single event and closed after `min_separation_windows`
   normal windows (configurable). InternVL is NOT called per timestamp.
4. **Evidence extraction** (`evidence/`) computes per-sensor reconstruction
   error, % deviation from the normal baseline, sensor trend, onset ordering,
   pre/post context, and a candidate subsystem. It labels evidence type and
   never claims correlation implies causation.
5. **InternVL receives** a text-only structured prompt (anomaly score, sensor
   deviations, trends, temporal sequence, candidate subsystem, context). TEP is
   sensor-only, so no images are created. Text-only training follows the
   official InternVL pattern: dummy `pixel_values` + `image_flags=0`.
6. **Automatic report**: `RCAInference.generate_report()` asks the model for a
   full JSON report (summary, root cause, subsystem, evidence, reasoning,
   severity, confidence, recommended action, uncertainty).
7. **Follow-up questions**: `answer_followup(event_id, question)` sends the
   stored event, structured evidence, previous report, conversation history and
   the new question, so the model answers about *that* anomaly.
8. **Inside the adapter**: only the learned InternVL LoRA parameters.
9. **Outside the adapter**: scaler, autoencoder, threshold, evidence, event
   store, streaming, API/UI.
10. **Save adapter**: training writes `adapter_config.json` +
    `adapter_model.safetensors` (+ tokenizer/config + remote-code files).
11. **Load adapter**: `load_tep_adapter()` reloads a FRESH original InternVL2-2B
    and attaches only `tep_rca` (see `scripts/test_adapter.py`).
12. **Run**: commands below.

## 4. Training process (what is trained, how, and where)

The system has **two independent training stages** — they do not share weights and can be run in any order (preprocessing must come first).

```
Stage 0: PREPROCESSING (no learning)
  TEP_FaultFree/Faulty_*.csv
    → validation + missing-value handling
    → scaler fitted ONLY on normal data (data/raw/normal/TEP_FaultFree_*.csv)
    → per-run sliding windows [60,52] (never across simulationRun)
    → leakage-free split by whole simulationRun (not random windows)
  Outputs: outputs/preprocessing/scaler.pkl + baseline_stats.json
           data/processed/normal_values.npy, normal_windows_{train,val,test}.npy
           data/processed/fault_values/fault_XX.npy

Stage 1: ANOMALY DETECTOR (unsupervised LSTM Autoencoder — unchanged, not retrained)
  What is trained: LSTM Encoder (52→128×2→64 in A5000 config) → reconstruction MSE on normal windows only
  Threshold: empirical percentile on normal val (99.5 on A5000)
  Outputs: outputs/anomaly_detector/model.pt, threshold.json (left untouched by dynamic stage)

Stage 1b: DYNAMIC DETECTOR (CVA primary + DPCA baseline — NEW, additive, parallel)
  What is trained: CVA past/future Hankel (`n_past 5`, `n_future 5`, `float64`, streaming `Spp/Sff/Sfp`, `eps 1e-6`, `eigh` inv-sqrt, SVD, `r` via energy 0.90) → `J/Vt` + `L`; DPCA augmented `52*(n_lags 3+1)` PCA 90%. NOT straddling runs, per-run `t_index`.
  Loss: none (linear algebra, not deep learning, CPU-only minutes on A5000)
  Statistics: CVA `T2/Q/Tr` (Q most sensitive for 3/9/15), DPCA `T2/SPE`; thresholds empirical `FAR 0.01` on held-out 100 fault-free runs → `outputs/anomaly_detector/dynamic/dynamic_thresholds.json` (never overwrites LSTM `threshold.json`)
  Outputs: outputs/anomaly_detector/dynamic/cva_model.npz, dpca_model.npz, dynamic_thresholds.json, fit_metadata.json (run IDs, singular spectrum)
  Why: faults 3 (step absorbed), 9 (variance only), 15 (valve sticking, dynamics) are invisible to MSE but visible to CVA Q/Tr. DPCA must be beaten by CVA or bug in conditioning.
  Evidence: `sensor_contributions` folds `52*n_past` residual → `52` + `(n_lags,52)` lag_profile (high-lag peak = slow dynamics). `detector_evidence {triggered_by, lstm_ae {score,thr,alarmed}, cva {T2/Q/Tr, state_order}, change_type: level|dynamics|mixed|unknown}` added, `evidence_schema_version 2`.

Fusion: `dynamic_detector.fusion.mode or` (each at `per_detector_far 0.005` → combined `0.01`, verified on fault_free_testing) or `weighted` (percentile-rank normalisation). Fused alarm feeds existing `event_aggregator` unchanged; first `n_past+n_future-1` samples return `warming_up` not null alarm.

Stage 2: LLM ADAPTER (supervised instruction-tuning)
  What is trained: ONE LoRA/QLoRA adapter "tep_rca" on FROZEN InternVL2-2B
  Input:  structured evidence JSON (top sensors, deviations, temporal order)
  Target: JSON {summary, root_cause, affected_subsystem, evidence, reasoning,
           severity, confidence, recommended_action, uncertainty}
  Splits: by fault scenario (train 1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19,20,22
          / val 4,14 / test 15,21) — no leakage of overlapping windows
  Outputs: outputs/tep_rca_adapter/adapter_config.json + adapter_model.safetensors
```

**Full command sequence (RTX A5000 — see §5 for VRAM notes):**

```bash
# 0. Install (A5000: match CUDA — check nvidia-smi)
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121  # or cu118
pip install -r requirements.txt
# bitsandbytes Windows: needs prebuilt wheel — see https://github.com/bitsandbytes-foundation/bitsandbytes#windows
# Without it, use --no-4bit (FP16 LoRA)

# 1. Preprocessing — smoke test (5 runs/fault, ~10 sec, no 5GB load)
#    Smoke is ISOLATED: writes to data/processed_smoke + outputs/preprocessing_smoke,
#    production data/processed + outputs/preprocessing remain untouched for A5000.
python scripts/prepare_tep.py --config configs/config.yaml --smoke --log-level INFO
# Full for A5000 (250k normal Training only → 730k if you include Testing):
#   — scaler fitted ONLY on TEP_FaultFree_Training.csv (Rieth protocol, not Testing)
#   — writes ~4GB to data/processed + outputs/preprocessing (overwrites smoke)
python scripts/prepare_tep.py --config configs/config.yaml --full
# or: python scripts/prepare_tep.py --config configs/config_a5000.yaml --full
# Custom: --max-runs 20  (20 runs per fault)

# 2. Train LSTM autoencoder (only normal windows) — UNCHANGED, NOT retrained by dynamic stage
python scripts/train_anomaly_detector.py --config configs/config.yaml          # 100 epochs, batch 64 (A5000: 128)
# Resume / override: --resume --epochs 80

# 2b. Train dynamic detector (CVA/DPCA) — NEW, runs in parallel, CPU-only minutes, no GPU
python scripts/train_dynamic_detector.py --config configs/config.yaml                 # default method cva, uses scaler.pkl assert
python scripts/train_dynamic_detector.py --config configs/config_a5000.yaml --method both  # or --method dpca / cva
# → outputs/anomaly_detector/dynamic/cva_model.npz, dpca_model.npz, dynamic_thresholds.json, fit_metadata.json
# Logs singular spectrum + chosen r; first thing to inspect if CVA underperforms on 3/9/15

# 3. Evaluate detector — NEW per-fault table (FDR @ FAR 0.01, not aggregate)
python scripts/evaluate_anomaly_detector.py --config configs/config.yaml                 # all detectors
python scripts/evaluate_anomaly_detector.py --config configs/config_a5000.yaml --detector cva  # isolate
python scripts/evaluate_anomaly_detector.py --config configs/config_a5000.yaml --detector lstm_ae
# → data/processed/anomaly_detector_eval.json per-fault {lstm_ae, dpca, cva, fused} + highlight_3_9_15 + realised_far
# Rules enforced: FDR post-onset only (discard <160/20), FAR only on fault_free_testing, thresholds from holdout

# 4. Generate LLM instruction data (structured evidence → JSON report)
#    NOTE: SKIP for now — A5000 not in use. Code ready; run before adapter training.
python scripts/generate_llm_dataset.py --config configs/config.yaml --samples-per-fault 8
# → data/llm/train.jsonl (360), val.jsonl, test.jsonl + split_metadata.json
# Uses fault knowledge base, not fault_id→name mapping
# A5000 recommended: --samples-per-fault 20-30 for better category accuracy

# 5. Fine-tune InternVL2-2B adapter (RUN ON RTX A5000 — not in smoke env)
# QLoRA (any GPU ≥8GB, 6-8GB VRAM):
python scripts/train_tep_adapter.py --config configs/config.yaml
# A5000 BF16 LoRA (recommended, 12-14GB, ~2× faster):
python scripts/train_tep_adapter.py --config configs/config_a5000.yaml
# Explicit: --no-4bit --batch-size 4 --lr 2e-4 --lora-r 16 --resume
# Prints: Base model, Adapters loaded: NONE, Training adapter: tep_rca, total/trainable/%
# Fails if any adapter already on base; saves to outputs/tep_rca_adapter/
```

InternVL2-2B is loaded as `AutoModel.from_pretrained(..., trust_remote_code=True)` — LLM backbone is `InternLM2-Chat-1.8B`; text-only TEP uses dummy `pixel_values` + `image_flags=0` (official InternVL pattern).

### 4.7 Inference with fused detectors + follow-up

After both detectors trained, streaming fuses `lstm_ae OR cva` (or `weighted`) before `event_aggregator`. Evidence now includes `detector_evidence` + `change_type` + `lag_profile` for the LLM.

```bash
# Verify adapter on fresh base
python scripts/test_adapter.py --config configs/config.yaml

# End-to-end stream with fused alarm (dynamic in parallel)
python scripts/test_end_to_end.py --config configs/config.yaml --inject-at 800
python scripts/test_end_to_end.py --config configs/config.yaml --no-llm --inject-at 800  # deterministic fallback
# With explicit fault/run selection (per-split onset 20 vs 160 handled)
python scripts/test_end_to_end.py --config configs/config.yaml --fault-number 9 --fault-run 5 --normal-run 1

# Follow-up (now grounded in detector_evidence)
```
from main import TEPApp
from utils import load_config
app = TEPApp(load_config())  # loads LSTM + CVA/DPCA if dynamic_detector.enabled
app.process_sensor_stream({"values": arr})  # first 9 samples → warming_up
app.answer_followup("ANOM-0001", "Why is this dynamics not level?")
```

# API / UI (unchanged, fused alarm transparent)
uvicorn api.server:app --host 0.0.0.0 --port 8000
streamlit run ui/app.py
```

## 5. Hardware / memory guidance (RTX A5000 primary target)

This system was prepared for **NVIDIA RTX A5000 (24 GB, Ampere 8.6, CUDA 11.8/12.1)**.
Training is NOT run in this environment; run it directly on the A5000.

| Mode | Approx. VRAM | RTX A5000 config | Notes |
|------|-------------|------------------|-------|
| **4-bit QLoRA** (default, `configs/config.yaml`) | ~6–8 GB | `use_4bit: true`, bf16, batch 2 × 8 grad-accum, checkpointing ON | Safest, leaves headroom; use on any GPU ≥8 GB |
| **BF16 LoRA** (recommended on RTX A5000, `configs/config_a5000.yaml`) | ~12–14 GB | `use_4bit: false`, bf16, batch 4 × 4 grad-accum, checkpointing ON | **Fastest on 24 GB**; ~2× throughput over QLoRA on A5000 |
| **FP16 LoRA** | ~12–14 GB | same as BF16 but `bf16: false, fp16: true` | Use only if BF16 unavailable |

A5000 install hint:
```bash
# CUDA 12.1 build (check nvidia-smi → CUDA Version)
pip install torch --index-url https://download.pytorch.org/whl/cu121
# or CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# optional: flash-attn for speed (requires matching CUDA toolkit + ninja)
# pip install flash-attn --no-build-isolation  # then set llm.use_flash_attn: true
```

Both modes support gradient accumulation, gradient checkpointing, configurable
batch size / sequence length / FP16/BF16, and CPU fallback for the sensor
pipeline. Single-GPU only; no multi-GPU assumptions. For A5000 prefer
`configs/config_a5000.yaml` (`--config configs/config_a5000.yaml`).

## 6. Tested package versions

InternVL2-2B remote code requires `transformers >= 4.37.0`. This project was
written and validated against:

```
Python 3.10/3.11 · torch 2.5.1 · transformers 4.46.3 · peft 0.14.0
accelerate 0.33.0 · bitsandbytes 0.45.1 · datasets 3.0.0
scikit-learn 1.4+ · pandas 2.1+ · fastapi/uvicorn/streamlit (latest)
```

## 7. Project layout

```
configs/config.yaml (+ config_a5000.yaml)  # single source, now with dynamic_detector + per-split fault_onset
preprocessing/                # tep_loader (per-split onset, run-boundary aware), scaler, windowing
anomaly_detection/            # lstm/, dynamic/{lag_builder,cva,dpca,statistics,contributions,fusion}, threshold, inference
evidence/                     # contributions, temporal, relationships, event builder (now detector_evidence v2)
llm/                          # dataset (v2 consumes detector_evidence), train_adapter, evaluate
streaming/                    # SensorStream (per-run faultNumber/simulationRun, warming_up)
events/                       # SQLite event store (fused alarm)
api/server.py                 # FastAPI backend
ui/app.py                     # Streamlit UI
scripts/                      # prepare_tep (--smoke/--full), train_anomaly_detector, train_dynamic_detector,
                              # evaluate_anomaly_detector (--detector, per-fault 3/9/15), generate_* , test_*
main.py                       # TEPApp (fusion OR/weighted, DynamicDetectorRunner)
tests/                        # 6 tests for lag, CVA recovery, variance sensitivity, etc.
```

## 8. Uncertainty & limitations

* Sensor deviation and onset ordering are **temporal/statistical evidence**,
  not proof of causation. The model is trained to say so.
* TEP faults 16–20 are officially "unknown"; supervision labels them as
  unclassified rather than fabricating a cause.
* The LSTM autoencoder, threshold and evidence pipeline run fully without the
  LLM; the adapter adds grounded natural-language reasoning.
* **NEW:** CVA targets dynamic/second-order faults; faults 3/9/15 remain the hardest. Expect 9 to lift clearly, 15 moderately, 3 possibly <50% FDR at 1% FAR — if 3 jumps >80%, audit for leakage (scaler contamination, threshold on test, run straddling).