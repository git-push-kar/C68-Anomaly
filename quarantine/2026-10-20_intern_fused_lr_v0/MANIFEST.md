# Quarantine: `intern_fused_lr_v0`

**Quarantined:** 2026-10-20
**Reason:** The live adapter was trained on the fused base
`./models/internvl3_2b_lr_fused` (InternVL3-2B with the prior LR adapter merged).
The project is being retrained on the pristine base
`C:\Users\Admin\Desktop\LY_C68\deepfake_project\models\InternVL3-2B`.
LoRA deltas are base-specific, so the old adapter is not loadable/correct on the
new base.

## Quarantined artifacts

| Artifact | Original path | Quarantined path |
|---|---|---|
| Active adapter `intern_fused_lr` (v0) + checkpoints 400/447 + its README, `training_summary.json`, `training_args.bin`, `evaluation.json` | `outputs/intern_fused_lr_v0/` | `adapter/intern_fused_lr_v0/` |
| Adapter integration / deployment guide | `SYSTEM_INTEGRATION_AND_ADAPTER_GUIDE.md` | `docs/SYSTEM_INTEGRATION_AND_ADAPTER_GUIDE.md` |
| Training instructions | `training_instructions.md` | `docs/training_instructions.md` |
| Superseded report snapshots (for audit only) | `README.md`, `PROJECT_REPORT.md`, `REPORT_C68_FULL.md`, `RESEARCH_PAPER_TECHNICAL_REPORT.md`, `docs/*.md`, `outputs/eval_grounding_*.json` | `docs/snapshots/` |

## Provenance

- **Base used by quarantined adapter:** `./models/internvl3_2b_lr_fused`
- **Base SHA256:** `CE545959C95C0EAC931FA5DA50ADD4C9C05C837B5BC02F434E042A75F14CFBC3`
- **Adapter weights SHA256:** `99D6AB1E706D5CF2D134295F35800BB57121C41AB3AE7B488EC47B57F62A2649`
- **Adapter config:** LoRA `r=16`, `alpha=32`, `dropout=0.05`, 7 Qwen2 targets
  (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Training data:** `data/llm` v0 (2,376 train / 396 val), teacher `qwen2.5:32b`,
  12 samples/fault, all 22 faults present in every split.
- **Training run:** 3 epochs, BF16 LoRA (`use_4bit=false`), effective batch 16,
  max length 4096, 447 steps, 4,688 s.
- **Retained v0 metrics (in-distribution, 44 report samples):**
  root-cause accuracy `0.1591`, fault classification `1.000`, severity `1.000`,
  evidence consistency `1.000`, hallucination `0.000`, JSON validity `1.000`,
  recommendation `1.000`, overclaiming `0.000`.

## Rollback

1. Restore the adapter directory:
   `Move-Item quarantine\2026-10-20_intern_fused_lr_v0\adapter\intern_fused_lr_v0 outputs\intern_fused_lr_v0`
2. Restore the old base pointer:
   `llm.base_model: ./models/internvl3_2b_lr_fused`
   `llm.adapter_name: intern_fused_lr`
   `llm.adapter_dir: outputs/intern_fused_lr_v0`
3. Do not attach this adapter to the pristine base. The LoRA tensor shapes match,
   but the delta was learned against different base weights, so serving it on the
   pristine base would produce incorrect outputs.

## Status

**INACTIVE / DO NOT SERVE.** The active adapter is now
`outputs/tep_rca_internvl3_2b/internvl3_2b_pristine_rca` (see post-training doc).
