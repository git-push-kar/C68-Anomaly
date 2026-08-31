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

Two independent intelligence layers:

| Layer | Module(s) | Responsibility |
|-------|-----------|----------------|
| **A. Sensor / time-series pipeline** | `preprocessing/`, `anomaly_detection/`, `evidence/`, `streaming/`, `events/` | raw sensor stream, normalization, sliding windows, unsupervised anomaly detection, per-sensor contributions, temporal analysis, event aggregation, evidence JSON |
| **B. InternVL2-2B TEP adapter** | `llm/` | understands structured anomaly evidence, root-cause reasoning, severity, recommendations, follow-up QA |

The adapter weights contain **only** learned InternVL LoRA parameters. No sensor
logic is inside the adapter.

```
Pipeline data flow
------------------
sensor_stream
  -> preprocessor (missing values, validation, scaler fitted on NORMAL only)
  -> LSTM autoencoder (reconstruction error)
  -> anomaly threshold (learned from normal validation scores)
  -> event_aggregator (groups consecutive anomalous windows -> ONE event)
  -> evidence_generator (top sensors, % deviation, trends, temporal order,
                         candidate subsystem, pre/post context)
  -> InternVL2-2B + tep_rca adapter -> automatic report (JSON)
  -> event_store (SQLite) -> user_follow_up -> adapter -> answer
```

## 2. TEP dataset assumptions

The loader isolates all format-specific logic in `preprocessing/tep_loader.py` and auto-detects both formats.

* **Format `tep_rieth_csv` (current, default)** — Rieth et al. consolidated files:
  ```
  data/raw/normal/TEP_FaultFree_Training.csv  (250k rows, 500 runs × 500 samples, fault 0)
  data/raw/normal/TEP_FaultFree_Testing.csv   (480k rows, 500 runs × 960 samples, fault 0)
  data/raw/faults/TEP_Faulty_Training.csv     (5M rows, 20 faults × 500 runs × 500 samples)
  data/raw/faults/TEP_Faulty_Testing.csv      (9.6M rows, 20×500×960)
  Columns: faultNumber, simulationRun, sample, xmeas_1..41, xmv_1..11 (52 sensors)
  Fault injected at sample 160 (0-based, 161 1-based) — `dataset.fault_onset_index`.
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

Stage 1: ANOMALY DETECTOR (unsupervised LSTM Autoencoder)
  What is trained: LSTM Encoder (52→32×2) → latent 16 → LSTM Decoder → reconstruction
  Loss: MSE on normal windows only (scaler fitted only on TEP_FaultFree_Training.csv)
  Threshold: learned from normal validation scores (percentile 99.5 on A5000 — was 99.0 synthetic, now less aggressive to cut FPR 13.5% → <5%)
  Outputs: outputs/anomaly_detector/model.pt, threshold.json

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

# 2. Train LSTM autoencoder (only normal windows)
python scripts/train_anomaly_detector.py --config configs/config.yaml          # default 50 epochs, batch 64
python scripts/train_anomaly_detector.py --config configs/config_a5000.yaml    # A5000: batch 128, faster
# Resume / override: --resume --epochs 80

# 3. Evaluate detector (no training)
python scripts/evaluate_anomaly_detector.py --config configs/config.yaml
# → data/processed/anomaly_detector_eval.json (precision/recall/F1/FPR/AUROC/delay)

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

### 4.7 Inference without training (quick check)

After training, or with the smoke `outputs/`:

```bash
# Verify adapter on fresh base
python scripts/test_adapter.py --config configs/config.yaml

# End-to-end stream + automatic report
python scripts/test_end_to_end.py --config configs/config.yaml --inject-at 800
python scripts/test_end_to_end.py --config configs/config.yaml --no-llm --inject-at 800  # deterministic fallback

<### 4.9 Ask follow-up questions

```python
from main import TEPApp
from utils import load_config
app = TEPApp(load_config())
app.answer_followup("ANOM-0001", "Which sensors should I inspect?")
# or one-liner:
# python -c "from main import TEPApp; from utils import load_config; app=TEPApp(load_config()); print(app.answer_followup('ANOM-0001','Why cooling?'))"
```

# API / UI
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
configs/config.yaml           # single source of configuration
preprocessing/                # tep_loader, scaler, windowing
anomaly_detection/            # dataset, LSTM AE, train, threshold, inference
evidence/                     # contributions, temporal, relationships, event builder
llm/                          # dataset, model, train_adapter, evaluate, adapter_loader, inference
streaming/                    # SensorStream simulator
events/                       # SQLite event store
api/server.py                 # FastAPI backend
ui/app.py                     # Streamlit UI
scripts/                      # prepare_tep, train/eval detector, gen dataset,
                              # train adapter, test_adapter, test_end_to_end
main.py                       # TEPApp high-level interface
```

## 8. Uncertainty & limitations

* Sensor deviation and onset ordering are **temporal/statistical evidence**,
  not proof of causation. The model is trained to say so.
* TEP faults 16–20 are officially "unknown"; supervision labels them as
  unclassified rather than fabricating a cause.
* The LSTM autoencoder, threshold and evidence pipeline run fully without the
  LLM; the adapter adds grounded natural-language reasoning.


  Pipeline — code generation → training → evaluation → inference (A5000)
# 0. Code already generated (no LLM dataset now — deferred, code ready at generate_llm_dataset.py)

# 1. PREPROCESSING — smoke (isolated) or full A5000 (production, scaler on Training only)
python scripts/prepare_tep.py --config configs/config.yaml --smoke        # writes _smoke, safe to run anytime
python scripts/prepare_tep.py --config configs/config_a5000.yaml --full   # 250k normal + 5M faulty → data/processed (overwrites smoke prod)

# 2. DETECTOR TRAINING (unsupervised, normal only)
python scripts/train_anomaly_detector.py --config configs/config_a5000.yaml  # batch 128, 50 epochs → model.pt
# → threshold 99.5 learned on normal val → threshold.json

# 3. DETECTOR EVALUATION (good results targeted)
python scripts/evaluate_anomaly_detector.py --config configs/config_a5000.yaml
# → data/processed/anomaly_detector_eval.json (expect after fix: FPR <5%, FNR <10%, AUROC >0.95, delay ~4-8)

# 4. LLM DATASET — SKIP NOW (deferred, A5000 not in use)
# When ready: python scripts/generate_llm_dataset.py --config configs/config.yaml --samples-per-fault 20

# 5. ADAPTER TRAINING — on A5000 only (frozen InternVL2-2B)
python scripts/train_tep_adapter.py --config configs/config_a5000.yaml  # BF16 LoRA 12-14GB, batch 4×4

# 6. INFERENCE / ACTUAL USE
python scripts/test_adapter.py --config configs/config_a5000.yaml
python scripts/test_end_to_end.py --config configs/config_a5000.yaml --inject-at 800  # auto report + follow-up
uvicorn api.server:app --host 0.0.0.0 --port 8000; streamlit run ui/app.py
Production threshold 99.5 + consecutive_windows_to_confirm 3 (config.yaml:120) reduces normal-as-anomaly; larger LLM dataset (20/sample) will fix category when A5000 is available.