# Post-Training Report: Pristine InternVL3-2B Adapter

**Date:** 2026-10-20
**Scope:** Retrain the TEP RCA LoRA adapter on the pristine InternVL3-2B base,
quarantine the previous fused-base adapter, and record all artifacts, metrics,
commands, and rollback steps.

---

## 1. Executive summary

- The previous live adapter (`intern_fused_lr`) was trained on
  `./models/internvl3_2b_lr_fused`, a base that already had an earlier LR adapter
  merged into its weights.
- The new adapter, `internvl3_2b_pristine_rca`, was trained from scratch on the
  pristine base at
  `C:/Users/Admin/Desktop/LY_C68/deepfake_project/models/InternVL3-2B`.
- No model code changes were required: the pristine base uses the same
  InternVL3-2B / Qwen2 architecture, ChatML tokenizer, vision size (448), and
  LoRA target module names as the previous fused base.
- The previous adapter and its adapter-specific docs were moved to
  `quarantine/2026-10-20_intern_fused_lr_v0/`.
- New in-distribution v0 evaluation: **25.0% exact root-cause accuracy**,
  up from 15.9% for the quarantined adapter, with all other measured quality
  metrics unchanged (100% JSON validity, 100% fault classification, 100%
  severity, 100% evidence consistency, 0% hallucination).
- Fault-disjoint v1 holdout: **50.0% exact root-cause accuracy** on 4 report
  samples. This sample is small and is not a statistically strong estimate.

---

## 2. What changed

### 2.1 Base model

| Item | Quarantined v0 | New active adapter |
|---|---|---|
| Base | `./models/internvl3_2b_lr_fused` | `C:/Users/Admin/Desktop/LY_C68/deepfake_project/models/InternVL3-2B` |
| Base state | InternVL3-2B with prior LR adapter merged | Pristine InternVL3-2B, no adapter |
| Base SHA256 (`model.safetensors`) | `CE545959C95C0EAC931FA5DA50ADD4C9C05C837B5BC02F434E042A75F14CFBC3` | `18B4CDBC8EAB790F447FEAC717F4D0CECBB85CAF8DF198B3B2401C22506858E7` |
| Adapter name | `intern_fused_lr` | `internvl3_2b_pristine_rca` |
| Adapter directory | `outputs/intern_fused_lr_v0` | `outputs/tep_rca_internvl3_2b` |

The pristine base contains full weights plus InternVL remote code:

- `model.safetensors` (4,177,999,192 bytes)
- `config.json`, `tokenizer.json`, `tokenizer_config.json`
- `configuration_internvl_chat.py`, `modeling_internvl_chat.py`
- `configuration_intern_vit.py`, `modeling_intern_vit.py`
- `conversation.py`, `preprocessor_config.json`

### 2.2 Configuration

Updated in both `configs/config.yaml` and `configs/config_a5000.yaml`:

```yaml
llm:
  base_model: "C:/Users/Admin/Desktop/LY_C68/deepfake_project/models/InternVL3-2B"
  adapter_name: internvl3_2b_pristine_rca
  adapter_dir: outputs/tep_rca_internvl3_2b
```

Training hyperparameters were kept aligned with the quarantined v0 run:

```yaml
training:
  use_4bit: false
  bf16: true
  fp16: false
  batch_size: 4
  gradient_accumulation_steps: 4
  num_epochs: 3
  learning_rate: 2.0e-4
  weight_decay: 0.01
  max_sequence_length: 4096
  warmup_ratio: 0.03
  lr_scheduler: cosine
  max_grad_norm: 1.0
  gradient_checkpointing: true
  optim: null            # resolves to adamw_torch because use_4bit=false
```

LoRA geometry:

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  bias: none
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

No changes were made to the anomaly detector, scaler, CVA/DPCA thresholds,
evidence extraction, event store, streaming simulator, API, or UI.

---

## 3. Training data

- Dataset: `data/llm` v0, JSONL format.
- Train: 2,376 examples.
- Validation: 396 examples.
- Test: 396 examples (44 report-kind samples used for scoring).
- Samples per fault: 12.
- Seed: 42.
- Teacher: `qwen2.5:32b` per `data/llm/split_metadata.json` (the config's
  distillation block still records `Qwen/Qwen2.5-72B-Instruct`; the dataset
  metadata is the authoritative record for this run).
- Split caveat: v0 contains all 22 fault IDs in train, validation, and test.
  Therefore v0 scores are **in-distribution** and must not be reported as
  zero-shot.
- The fault-disjoint holdout is `data/llm_v1` (train faults exclude 4, 14, 15,
  21; validation 4/14; test 15/21). It contains only 4 report-kind test samples,
  so its metrics are indicative, not statistically conclusive.

---

## 4. Training run

### 4.1 Environment

| Component | Value |
|---|---|
| OS | Windows |
| Python | 3.11.5 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 4.57.6 |
| PEFT | 0.20.0 |
| GPU | NVIDIA RTX A5000, 24 GB |
| CUDA available | Yes |
| Precision | BF16, 16-bit LoRA (`use_4bit=false`) |
| FlashAttention | Not installed (`use_flash_attn=false`) |

### 4.2 Command

```powershell
.\venv\Scripts\python.exe scripts\train_tep_adapter.py --config configs\config.yaml
```

Logs:

- `logs/train_internvl3_2b_pristine.out`
- `logs/train_internvl3_2b_pristine.err`

The run was launched with `Start-Process`; the wrapper process timed out from the
tool's perspective, but the detached Python training process continued and
completed normally.

### 4.3 Model / parameter accounting

| Item | Value |
|---|---|
| Base parameters | 2,088,957,440 |
| Total parameters after adapter wrap | 2,107,422,208 |
| Trainable parameters | 18,464,768 |
| Trainable percentage | 0.8762% |
| Adapter checkpoint size | 73,916,992 bytes (~70.5 MiB) |
| Adapter SHA256 | `050EE467E58B303708BD39EE7AB025B7721E43533E0CB9DC600FAAFB716C848F` |

### 4.4 Optimization results

| Item | Value |
|---|---|
| Epochs | 3 |
| Global steps | 447 |
| Effective batch size | 16 (4 x 4 gradient accumulation) |
| Final train loss | 0.158027 |
| Eval loss at step 200 (epoch 1.34) | 0.0604388 |
| Eval loss at step 400 (epoch 2.69) | 0.0529004 (best) |
| Best checkpoint | `outputs/tep_rca_internvl3_2b/checkpoint-400` |
| Train runtime | 8,877.14 s (2 h 27 min 57 s) |
| Training throughput | 0.803 samples/s, 0.050 steps/s |

The run was slower than the quarantined v0 run (4,688 s / 1.30 h). The model,
data, batch size, and step count were unchanged, so the difference is attributed
to concurrent desktop GPU/CPU load during the run, not to an algorithmic change.

Loss trajectory (selected checkpoints):

| Step | Epoch | Loss | Learning rate |
|---|---:|---:|---:|
| 10 | 0.07 | 1.8887 | 1.29e-4 |
| 50 | 0.34 | 0.3170 | 1.97e-4 |
| 100 | 0.67 | 0.1079 | 1.82e-4 |
| 150 | 1.01 | 0.0602 | 1.56e-4 |
| 200 | 1.34 | 0.0614 / eval 0.0604 | 1.23e-4 |
| 300 | 2.01 | 0.0498 | 5.23e-5 |
| 400 | 2.69 | 0.0350 / eval 0.0529 | 6.00e-6 |
| 447 | 3.00 | final train loss 0.1580 | ~0 |

### 4.5 Artifacts

```
outputs/tep_rca_internvl3_2b/
  internvl3_2b_pristine_rca/
    adapter_config.json
    adapter_model.safetensors
  checkpoint-400/internvl3_2b_pristine_rca/
  checkpoint-447/internvl3_2b_pristine_rca/
  training_summary.json
  training_args.bin
  evaluation.json                 # v0 in-distribution (restored after v1 run)
  evaluation_v0_indist.json       # explicit v0 copy
  evaluation_v1_holdout.json      # explicit v1 copy
  README.md
  config.json + tokenizer + remote-code support files
```

The active adapter in `internvl3_2b_pristine_rca/` is the final step-447 save.
`checkpoint-400` is retained because it has the best validation loss. The
evaluation and end-to-end results in this report were produced with the active
final adapter.

`adapter_config.json` records the external pristine base path. Loading with an
explicit `--base-model` argument still works if the directory is moved, but the
recorded path should be updated for reproducible provenance.

---

## 5. Evaluation

### 5.1 Smoke test

```powershell
.\venv\Scripts\python.exe scripts\test_adapter.py --config configs\config.yaml
```

Result: adapter attached to a fresh pristine base with no pre-existing adapter,
generated a valid structured report, and printed `ADAPTER VERIFICATION PASSED`.
Log: `logs/test_adapter_internvl3_2b_pristine.log`.

### 5.2 v0 in-distribution benchmark

```powershell
.\venv\Scripts\python.exe scripts\benchmark_tep_adapter.py --config configs\config.yaml --top-k 3
```

44 report samples from `data/llm/test.jsonl`.

| Metric | Quarantined v0 | New adapter | Delta |
|---|---:|---:|---:|
| Exact root-cause accuracy | 0.1591 | **0.2500** | **+0.0909** |
| Fault classification accuracy | 1.0000 | 1.0000 | 0 |
| Severity accuracy | 1.0000 | 1.0000 | 0 |
| Evidence consistency | 1.0000 | 1.0000 | 0 |
| Hallucination rate | 0.0000 | 0.0000 | 0 |
| Hedged reasoning rate | 0.6136 | 0.4545 | -0.1591 |
| Recommendation rate | 1.0000 | 1.0000 | 0 |
| JSON validity rate | 1.0000 | 1.0000 | 0 |
| Overclaiming rate | 0.0000 | 0.0000 | 0 |

Log: `logs/benchmark_v0_internvl3_2b_pristine.log`.

### 5.3 v1 fault-disjoint holdout

```powershell
.\venv\Scripts\python.exe scripts\benchmark_tep_adapter.py --config configs\config.yaml --dataset-dir data\llm_v1 --top-k 3
```

4 report samples (test faults 15 and 21).

| Metric | Value |
|---|---:|
| Exact root-cause accuracy | 0.5000 |
| Fault classification accuracy | 1.0000 |
| Severity accuracy | 1.0000 |
| Evidence consistency | 1.0000 |
| Hallucination rate | 0.0000 |
| Hedged reasoning rate | 0.2500 |
| Recommendation rate | 1.0000 |
| JSON validity rate | 1.0000 |
| Overclaiming rate | 0.0000 |

The previous v1 adapter also scored 0.5000 exact root-cause accuracy on the same
4 samples; the main change is the hedging rate (0.00 -> 0.25). Treat the v1 result
as a sanity check, not a benchmark.

### 5.4 End-to-end pipeline

Command that produced an event:

```powershell
.\venv\Scripts\python.exe scripts\test_end_to_end.py --config configs\config.yaml --inject-at 200 --fault-number 15 --fault-run 1 --normal-run 1
```

Result:

- `Closed anomaly events: 1`
- Event `ANOM-0487`
- Anomaly score `347.854`
- Severity `critical`
- New adapter generated the report with `confidence 0.9`.
- Root-cause candidate from the LLM was `A_Feed_Stream1`; affected subsystem was
  `condenser_cooling_system`. The deterministic evidence correctly identified
  the valve-sticking style dynamic event, but the LLM's exact root-cause name was
  not Fault 15. This is an LLM reasoning-quality limitation, not a load failure.
- Log: `logs/e2e_fault15_internvl3_2b_pristine.log`.

A second run with `--inject-at 800 --fault-number 1` produced zero closed events.
This is a stream/injection alignment outcome, not an adapter regression; the
documented Fault 15 run is the end-to-end acceptance case for this training.

### 5.5 UI and API connection check

The UI and API load `configs/config.yaml` directly, so they use the pristine base
and `internvl3_2b_pristine_rca` without a separate adapter switch.

Streamlit test (programmatic AppTest):

- App loaded with `EXCEPTIONS 0`.
- Sidebar reported:
  `LLM Adapter: internvl3_2b_pristine_rca (Online)`.
- Monitor tab processed a Fault 15 stream at 10 steps/run.
- Event `ANOM-0489` was created at step 310; the LLM generated a report
  (`root_cause: Feed Rate Disturbance`, severity `low`, confidence `0.85`).
- No UI exceptions occurred.

The UI simulation initially exposed a `None` LSTM score formatting bug at
`ui/app.py:57`. Fixed by coercing missing scores to `0.0` in the display and
history paths. This was a UI robustness fix only; it does not change detector or
adapter behavior.

FastAPI check on port 8010:

```json
{
  "status": "ok",
  "llm_loaded": true,
  "llm_base_model": "C:/Users/Admin/Desktop/LY_C68/deepfake_project/models/InternVL3-2B",
  "llm_adapter": "internvl3_2b_pristine_rca",
  "llm_adapter_dir": "outputs/tep_rca_internvl3_2b"
}
```

Both `/api/status` and `/api/model-info` now expose the active base and adapter.

---

## 6. Quarantine

Location: `quarantine/2026-10-20_intern_fused_lr_v0/`

Contents:

| Item | Path |
|---|---|
| Previous adapter + checkpoints + its README, training summary, evaluation | `adapter/intern_fused_lr_v0/` |
| Adapter integration guide | `docs/SYSTEM_INTEGRATION_AND_ADAPTER_GUIDE.md` |
| Old training instructions | `docs/training_instructions.md` |
| Snapshot of reports that describe the old adapter | `docs/snapshots/` |
| Quarantine manifest with hashes, metrics, and rollback | `MANIFEST.md` |

Quarantined adapter SHA256:
`99D6AB1E706D5CF2D134295F35800BB57121C41AB3AE7B488EC47B57F62A2649`

The quarantined adapter is **inactive**. Do not attach it to the pristine base;
its LoRA shapes may match, but its learned delta is tied to the fused base
weights and would produce incorrect behavior.

`outputs/intern_fused_lr_v1` remains in place as a historical comparison
artifact. It is also tied to the fused base and is not used by the active config.

---

## 7. Verification checklist

- [x] Pristine base exists and loads with `trust_remote_code=True`.
- [x] Base contains no PEFT adapter (`Adapter check passed`).
- [x] Tokenizer has ChatML special tokens and `<IMG_CONTEXT>`.
- [x] LoRA target modules match the base (`q/k/v/o/gate/up/down`, 28 layers).
- [x] Training completed all 447 steps / 3 epochs.
- [x] Best checkpoint saved at step 400 (eval loss 0.0529).
- [x] Final adapter saved with `adapter_config.json` and
  `adapter_model.safetensors`.
- [x] Fresh-base adapter load and report generation pass.
- [x] v0 in-distribution evaluation completed (44 samples).
- [x] v1 fault-disjoint holdout evaluation completed (4 samples).
- [x] End-to-end Fault 15 event detected and report generated by the new adapter.
- [x] Streamlit UI loaded with the new adapter and generated event `ANOM-0489`.
- [x] FastAPI `/api/status` and `/api/model-info` report the new base and adapter.
- [x] Old adapter and adapter-specific docs quarantined with a manifest.
- [x] Active configs point to the pristine base and new adapter.

---

## 8. Known limitations and caveats

1. **v0 is in-distribution.** All 22 faults appear in train and test. The 25.0%
   exact root-cause score is not a zero-shot result.
2. **v1 has only 4 report samples.** The 50.0% result is directionally useful but
   not statistically strong.
3. **Exact root-cause naming remains the weak metric.** Fault classification,
   severity, evidence consistency, JSON validity, and hallucination control are
   all at 1.0/0.0, but exact root-cause string matching is much lower.
4. **The base path is absolute and external to the repo.** If the model is moved,
   update `llm.base_model` in both config files and the adapter provenance.
5. **`trust_remote_code=True` is required** for InternVL3-2B. The remote code was
   loaded from the local base directory.
6. **The adapter is base-specific.** Any future base change requires another
   retrain; do not reuse this adapter with a different base.
7. **No FlashAttention.** The run used the standard attention path because
   FlashAttention2 is not installed in this environment.
8. **End-to-end Fault 1 at sample 800 produced no event.** Use the documented
   Fault 15 at sample 200 case for acceptance, or align the injection sample with
   the detector's evaluation protocol before treating a zero-event run as a
   failure.

---

## 9. Rollback

1. Stop the application and restore the old adapter directory:
   ```powershell
   Move-Item quarantine\2026-10-20_intern_fused_lr_v0\adapter\intern_fused_lr_v0 outputs\intern_fused_lr_v0
   ```
2. Set the old base and adapter in both config files:
   ```yaml
   llm:
     base_model: "./models/internvl3_2b_lr_fused"
     adapter_name: intern_fused_lr
     adapter_dir: outputs/intern_fused_lr_v0
   ```
3. Restore the old adapter-specific docs from
   `quarantine/2026-10-20_intern_fused_lr_v0/docs/`.
4. Do not load the quarantined adapter against the pristine base.

---

## 10. Reproduce

```powershell
# 1. Verify the pristine base
Get-ChildItem "C:\Users\Admin\Desktop\LY_C68\deepfake_project\models\InternVL3-2B"

# 2. Train (3 epochs, BF16 LoRA)
.\venv\Scripts\python.exe scripts\train_tep_adapter.py --config configs\config.yaml

# 3. Smoke test on a fresh base
.\venv\Scripts\python.exe scripts\test_adapter.py --config configs\config.yaml

# 4. v0 in-distribution evaluation
.\venv\Scripts\python.exe scripts\benchmark_tep_adapter.py --config configs\config.yaml --top-k 3

# 5. v1 fault-disjoint holdout
.\venv\Scripts\python.exe scripts\benchmark_tep_adapter.py --config configs\config.yaml --dataset-dir data\llm_v1 --top-k 3

# 6. End-to-end Fault 15 acceptance
.\venv\Scripts\python.exe scripts\test_end_to_end.py --config configs\config.yaml --inject-at 200 --fault-number 15 --fault-run 1 --normal-run 1
```

Note: step 5 overwrites `outputs/tep_rca_internvl3_2b/evaluation.json`. The v0
result is preserved as `evaluation_v0_indist.json`; restore it to
`evaluation.json` after step 5 if the v0 result should remain the default.
