# Training run 1 (Protocol A, seed 42) — live state note

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade`
**Written:** 2026-10-31 16:33 IST, mid-run. This note exists so a later session can
pick the run up from disk instead of from a conversation.

## What is running

```
venv_train/Scripts/python.exe -u scripts/train_tep_adapter.py \
    --config configs/config_v2_protocolA_seed42.yaml
```

Started 2026-10-31 16:02 IST as a detached process (`nohup`), so it does not depend
on any agent session. It survives the session that launched it; it does not survive
a reboot or a manual kill.

| | |
|---|---|
| Adapter dir | `outputs/tep_rca_v2_protocolA_seed42/` |
| Train file | `outputs/benchmark_v2/sft/train_mixed.jsonl`, SHA-256 `0f16b82b3fe18728d67476fc842160b4eeaa142031e09105ae03513d980a20d2` |
| Mix | PROTOCOL §12.1b per-case targets (961 dialogue rows + 1,919 report rows) |
| Steps | 180 / epoch, 540 total, effective batch 16 |
| Checkpoints | every 100 steps, all kept |

## Snapshot at the time of writing

| step | loss | s/step | GPU mem | temp | clock | power |
|---|---|---|---|---|---|---|
| 10 | 1.2126 | 29.6 | 24,056 MiB | 75 °C | 1,830 MHz | 190 W |
| 40 | 0.5375 | 23.1 | 24,060 MiB | 77 °C | 1,830 MHz | 190 W |
| 70 | 0.4546 | 26.2 | 24,023 MiB | 73 °C | 1,830 MHz | 161 W |

One step at ~70 measured 44.63 s, just under the 45 s hard stop (which needs three
in a row). Projection at the current rate: **~3.5 h remaining**, i.e. finishing
around 20:00 IST.

## Files the run leaves behind (all inside the adapter dir)

| File | What it is |
|---|---|
| `train_console.log` | stdout of the trainer: per-step progress, loss curve, save events |
| `train_telemetry.jsonl` | one JSON row per 10 steps: loss, lr, grad norm, s/step, GPU mem/util/temp/clock/power |
| `train_telemetry.log` | the same, human readable |
| `checkpoint-100 … checkpoint-500` | full trainer checkpoints (adapter + optimizer state) |
| `adapter_config.json`, `adapter_model.safetensors` | the final adapter (step 540) |
| `training_summary.json` | adapter name, base model, LoRA config, training config, `global_step`, `epochs_completed`, and **SHA-256 of every file** |

## Picking it up in a new session

Nothing has to be re-derived; the numbers come from files, not from memory.

1. **Is it done?** `tail -3 outputs/tep_rca_v2_protocolA_seed42/train_console.log`
   and `ls -d outputs/tep_rca_v2_protocolA_seed42/checkpoint-*`. Done means
   `training_summary.json` exists and the log ends with the save line.
2. **Did it stop badly?** A non-finite loss, a sustained >45 s step or >85 °C for two
   minutes raises inside the run and the traceback is in `train_console.log`. To
   resume: same command with `--resume` (it picks up the newest checkpoint).
3. **Then run Part B**, in this order (each writes JSON under
   `outputs/benchmark_v2/eval/`):

```bash
venv_train/Scripts/python.exe scripts/eval_protocolA.py parity        --items 40
venv_train/Scripts/python.exe scripts/eval_protocolA.py checkpoints   --per-checkpoint 300
venv_train/Scripts/python.exe scripts/eval_protocolA.py speed         --items 10 --cap 128
venv_train/Scripts/python.exe scripts/eval_protocolA.py generate      --items 50 --cap 512
venv_train/Scripts/python.exe scripts/eval_protocolA.py followup       --items 40 --controls 20
```

`checkpoints` applies the pre-registered rule (macro-F1 on a stratified 300-item
validation subsample, accuracy as tie-break, later step as second tie-break) and
writes `checkpoint_table.json` + `selected_checkpoint.json`; `speed`, `generate` and
`followup` read the selected checkpoint from that file, so they must run after it.

4. **Then write the report**, from those JSON files:
   `outputs/benchmark_v2/eval/checkpoint_table.json`, `selected_checkpoint.json`,
   `scoring_parity.json`, `decode_speed.json`, `generation_512.json`,
   `followup_sanity.json`.

## What is not yet done

* `docs/reports/training_run1_report.md` — **not written**. The run is data, not a
  verdict: everything the report needs is in the JSON above, plus this file's log
  paths.
* No test component has been opened or scored, and nothing will be until a separate
  instruction says so.
* `outputs/` is untracked build output: the checkpoints, logs and eval JSON live on
  this machine only. The *code, config and PROTOCOL* that produced them are committed
  (`ee6b8ce`, `676e21b`, `6031a9c`).

## Preconditions verified before launch (for the record)

* GPU idle: 831 MiB used by desktop processes, 31 °C, 210 MHz; `ollama stop
  qwen3:30b` issued, no model resident.
* Base model `…/InternVL3-2B/model.safetensors` = `18B4CDBC…58E7`, 4,177,999,192 bytes.
* `verify_training_env.py`: torch 2.13.0+cu130, CUDA 13.0, transformers 4.57.6,
  peft 0.20.0, timm 1.0.30, accelerate 1.15.0, 18,464,768 trainable parameters.
* Input pin `configs/benchmark_inputs.json` verified: `model.pt`, `cva_model.npz`,
  `dpca_model.npz`, `scaler.pkl` all match.
* `train_mixed.jsonl` SHA-256 matched the recorded value before the first step.
