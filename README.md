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

The loader isolates all format-specific logic in
`preprocessing/tep_loader.py`.

* **Format** `tep_52col_csv`: normal data in `data/raw/normal/normal*.csv`;
  each fault in `data/raw/faults/fault_XX.csv` (XX = fault id 01..22).
* Each file is a **T x F matrix**; rows are chronological samples, columns are
  the process variables (52 where applicable: XMEAS 1–41, XMV 42–52).
* Optional columns: a timestamp column and a fault-onset column
  (configurable via `dataset.timestamp_column` / `dataset.fault_onset_column`).
* Without a header, columns are named `XMEAS_1..XMEAS_41, XMV_42..XMV_52`.
* Missing values: interpolate / ffill / drop (`dataset.missing_value_strategy`).
* Fault onset default is sample index 161 when unknown.
* To add another public TEP format (e.g. .mat) add a `TEPDataFormat` member and
  a loader function in `tep_loader.py`; nothing else changes.

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

## 4. Quick start

### 4.1 Install dependencies

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**bitsandbytes on Windows**: 4-bit QLoRA needs a prebuilt wheel
(no official PyPI wheel for older versions). Check the
[bitsandbytes Windows guide](https://github.com/bitsandbytes-foundation/bitsandbytes#windows)
or install a compatible build. Without bitsandbytes, use FP16 LoRA:
`--no-4bit` (see `scripts/train_tep_adapter.py`).

### 4.2 Prepare TEP data

Place CSVs under `data/raw/` (see section 2), then:

```bash
python scripts/prepare_tep.py --config configs/config.yaml
```

This fits + saves the normal-only scaler and baseline, saves processed arrays
and creates leakage-free window splits (splitting by the contiguous normal run,
never random window splitting).

### 4.3 Train the anomaly detector

```bash
python scripts/train_anomaly_detector.py --config configs/config.yaml
# optional: --resume --epochs 80
```

Saves `outputs/anomaly_detector/model.pt` + `threshold.json` (threshold is
estimated from normal validation scores — percentile / mean+std, configurable —
never hardcoded).

### 4.4 Evaluate the anomaly detector

```bash
python scripts/evaluate_anomaly_detector.py --config configs/config.yaml
```

Writes precision/recall/F1/FPR/FNR/AUROC/AUPRC/detection-delay to
`data/processed/anomaly_detector_eval.json`.

### 4.5 Generate the LLM instruction dataset

```bash
python scripts/generate_llm_dataset.py --config configs/config.yaml --samples-per-fault 8
```

Supervision maps **sensor evidence -> likely root cause + reasoning** (not
`fault_id -> fault_name`). Splits are by whole fault scenario (train/val/test
faults configured in `config.yaml` under `llm.training.*_faults`), so a fault
never leaks across splits. Writes to `data/llm/`.

### 4.6 Fine-tune InternVL2-2B (run on RTX A5000 only - not in this env)

```bash
# Default QLoRA (works on any GPU ≥8 GB):
python scripts/train_tep_adapter.py --config configs/config.yaml
# RTX A5000 optimized (BF16 LoRA, faster, 24 GB):
python scripts/train_tep_adapter.py --config configs/config_a5000.yaml
# Or explicitly without 4-bit on the base config:
python scripts/train_tep_adapter.py --config configs/config.yaml --no-4bit
```

* Prints the mandatory header: base model, `Adapters loaded: NONE`,
  `Training adapter: tep_rca`, total / trainable / percentage.
* **Fails** if any unexpected adapter is detected on the base model.
* Base is frozen; only `tep_rca` LoRA parameters train (targets: the InternLM2
  attention/MLP projections of InternVL2-2B).
* Resume: `--resume` (uses the last checkpoint).
* Save: the adapter + tokenizer are written under `outputs/tep_rca_adapter/`.

> InternVL2-2B is loaded with `AutoModel.from_pretrained(..., trust_remote_code=True)`.
> Its LLM backbone is InternLM2-Chat-1.8B (verified from the released
> `config.json`). `forward()` requires `pixel_values`; for the text-only TEP
> data we pass a dummy zero image with `image_flags=0`, which is the officially
> supported InternVL text-only training pattern.

### 4.7 Verify the adapter on a fresh base

```bash
python scripts/test_adapter.py --config configs/config.yaml
```

Loads a fresh original InternVL2-2B, attaches **only** `tep_rca`, builds a
sample anomaly event, and prints an automatic RCA report.

### 4.8 Run the continuous sensor simulation + automatic reporting

```bash
python scripts/test_end_to_end.py --config configs/config.yaml --inject-at 800
# without the LLM (deterministic reports only):
python scripts/test_end_to_end.py --config configs/config.yaml --no-llm --inject-at 800
```

### 4.9 Ask follow-up questions

Either via the end-to-end script (it asks one automatically) or:

```python
from main import TEPApp
from utils import load_config
app = TEPApp(load_config())
app.answer_followup("ANOM-0001", "Which sensors should I inspect?")
```

### 4.10 API server + UI

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/api/events

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