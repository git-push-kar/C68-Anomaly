# Training and Inference Guide: TEP Anomaly Detection System

**Current model date:** 2026-10-20
**Active LLM adapter:** `internvl3_2b_pristine_rca`
**Active base:** `C:/Users/Admin/Desktop/LY_C68/deepfake_project/models/InternVL3-2B`

The previous fused-base adapter (`intern_fused_lr`) and its adapter-specific
docs are quarantined in `quarantine/2026-10-20_intern_fused_lr_v0/`.
See `docs/POST_TRAINING_INTERNL3_2B_PRISTINE_2026-10-20.md` for the full
post-training report, metrics, hashes, and rollback steps.

---

## 1. What needs retraining after a base-model change?

LoRA deltas are tied to the exact base weights. Changing `llm.base_model` means
the adapter must be retrained, even when the architecture and tokenizer match.

| Component | Status | Retraining required? |
|---|---|---|
| Scaler and normal baseline (`outputs/preprocessing/`) | Intact | No |
| LSTM autoencoder (`outputs/anomaly_detector/model.pt`) | Intact | No |
| Dynamic detector CVA/DPCA (`outputs/anomaly_detector/dynamic/`) | Intact | No |
| InternVL3-2B LoRA adapter (`outputs/tep_rca_internvl3_2b/`) | Retrained 2026-10-20 | Yes, on every base change |
| Evidence / event / streaming / API / UI | Intact | No |

---

## 2. Current adapter status

| Item | Value |
|---|---|
| Base | Pristine InternVL3-2B (no merged adapter) |
| Adapter name | `internvl3_2b_pristine_rca` |
| Adapter directory | `outputs/tep_rca_internvl3_2b` |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05`, 7 Qwen2 projection modules |
| Trainable parameters | 18,464,768 (0.8762% of 2.107B) |
| Training data | `data/llm` v0, 2,376 train / 396 val |
| Training precision | BF16 16-bit LoRA (`use_4bit=false`) |
| Epochs / steps | 3 / 447 |
| Active adapter | Final step-447 save in `internvl3_2b_pristine_rca/` |
| Best checkpoint | `checkpoint-400` (eval loss 0.0529) |
| Adapter SHA256 | `050EE467E58B303708BD39EE7AB025B7721E43533E0CB9DC600FAAFB716C848F` |

---

## 3. Commands

### 3.1 Preprocessing and detector

The sensor pipeline is unchanged by the adapter retrain.

```powershell
python scripts/prepare_tep.py --config configs/config.yaml --full
python scripts/train_anomaly_detector.py --config configs/config.yaml
python scripts/train_dynamic_detector.py --config configs/config.yaml --method both
python scripts/evaluate_anomaly_detector.py --config configs/config.yaml
```

### 3.2 Train the LLM adapter

```powershell
.\venv\Scripts\python.exe scripts\train_tep_adapter.py --config configs\config.yaml
```

Options:

```powershell
# Resume from the latest checkpoint in the output directory
python scripts/train_tep_adapter.py --config configs/config.yaml --resume

# Low-VRAM QLoRA fallback
python scripts/train_tep_adapter.py --config configs/config.yaml --4bit

# Override LoRA rank / epochs / learning rate
python scripts/train_tep_adapter.py --config configs/config.yaml --lora-r 16 --epochs 3 --lr 2e-4
```

Outputs:

```
outputs/tep_rca_internvl3_2b/
  internvl3_2b_pristine_rca/adapter_config.json
  internvl3_2b_pristine_rca/adapter_model.safetensors
  checkpoint-400/
  checkpoint-447/
  training_summary.json
```

### 3.3 Validate and evaluate

```powershell
# Fresh-base load + one generated report
.\venv\Scripts\python.exe scripts\test_adapter.py --config configs\config.yaml

# v0 in-distribution benchmark (44 report samples)
.\venv\Scripts\python.exe scripts\benchmark_tep_adapter.py --config configs\config.yaml --top-k 3

# v1 fault-disjoint holdout (4 report samples)
.\venv\Scripts\python.exe scripts\benchmark_tep_adapter.py --config configs\config.yaml --dataset-dir data\llm_v1 --top-k 3

# End-to-end Fault 15 acceptance case
.\venv\Scripts\python.exe scripts\test_end_to_end.py --config configs\config.yaml --inject-at 200 --fault-number 15 --fault-run 1 --normal-run 1
```

Note: the v1 command overwrites `outputs/tep_rca_internvl3_2b/evaluation.json`.
The v0 result is preserved as `evaluation_v0_indist.json`; restore it after the
v1 run if v0 should remain the default evaluation artifact.

### 3.4 Live inference

```powershell
uvicorn api.server:app --host 0.0.0.0 --port 8000
streamlit run ui/app.py
```

Both load `configs/config.yaml`, so they automatically use the pristine base and
the new adapter.

---

## 4. Current evaluation summary

### v0 in-distribution (44 report samples)

| Metric | Quarantined v0 | New adapter |
|---|---:|---:|
| Exact root-cause accuracy | 0.1591 | **0.2500** |
| Fault classification accuracy | 1.0000 | 1.0000 |
| Severity accuracy | 1.0000 | 1.0000 |
| Evidence consistency | 1.0000 | 1.0000 |
| Hallucination rate | 0.0000 | 0.0000 |
| Hedged reasoning rate | 0.6136 | 0.4545 |
| Recommendation rate | 1.0000 | 1.0000 |
| JSON validity rate | 1.0000 | 1.0000 |
| Overclaiming rate | 0.0000 | 0.0000 |

### v1 fault-disjoint holdout (4 report samples)

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

The v1 set is too small for a strong claim. The v0 result is in-distribution and
must not be cited as zero-shot.

---

## 5. Quarantine and rollback

The old adapter and stale adapter-specific docs are here:

```
quarantine/2026-10-20_intern_fused_lr_v0/
  MANIFEST.md
  adapter/intern_fused_lr_v0/
  docs/SYSTEM_INTEGRATION_AND_ADAPTER_GUIDE.md
  docs/training_instructions.md
  docs/snapshots/
```

To roll back, restore the old adapter directory and set:

```yaml
llm:
  base_model: "./models/internvl3_2b_lr_fused"
  adapter_name: intern_fused_lr
  adapter_dir: outputs/intern_fused_lr_v0
```

Do not attach the old adapter to the pristine base.

---

## 6. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `No adapter_config.json` | Adapter not trained or wrong directory | Check `outputs/tep_rca_internvl3_2b/internvl3_2b_pristine_rca/` |
| CUDA OOM | 16-bit LoRA too large with other GPU users | Retrain with `--4bit`, or reduce `batch_size` and increase gradient accumulation |
| Adapter loads but output is incoherent | Adapter/base mismatch | Confirm `adapter_config.json` `base_model_name_or_path` matches `llm.base_model` |
| `Size mismatch` | LoRA rank/layer geometry differs | Retrain on the current base; do not reuse another base's adapter |
| No end-to-end event | Injection sample / detector threshold mismatch | Use the Fault 15 at sample 200 acceptance command above |
| `trust_remote_code` prompt | InternVL remote code not cached | Accept/allow the prompt or pin the local base directory |

---

## 7. References

- Full post-training report:
  `docs/POST_TRAINING_INTERNL3_2B_PRISTINE_2026-10-20.md`
- Quarantine manifest:
  `quarantine/2026-10-20_intern_fused_lr_v0/MANIFEST.md`
- Training logs:
  `logs/train_internvl3_2b_pristine.out`,
  `logs/train_internvl3_2b_pristine.err`
- Evaluation logs:
  `logs/benchmark_v0_internvl3_2b_pristine.log`,
  `logs/benchmark_v1_holdout_internvl3_2b_pristine.log`
- End-to-end logs:
  `logs/e2e_fault15_internvl3_2b_pristine.log`,
  `logs/e2e_internvl3_2b_pristine.log`
