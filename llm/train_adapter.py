"""Train the single ``tep_rca`` LoRA/QLoRA adapter on InternVL2-2B.

Flow: load ORIGINAL InternVL2-2B (frozen) -> assert no adapters -> wrap with
``tep_rca`` -> SFT on the TEP instruction dataset (text-only, dummy image +
image_flags=0) -> save ONLY the adapter (adapter_config.json +
adapter_model.safetensors) + tokenizer + the remote-code files needed to reload
it on a fresh base.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from llm.dataset import TEPInstructionDataset, make_collator
from llm.model import load_base_model, print_training_header, wrap_peft
from utils import ensure_dir, get_device, load_json, save_json, set_seed

logger = logging.getLogger(__name__)


def _load_tokenized_datasets(config: Dict, tokenizer):
    ds_dir = Path(config["paths"]["llm_dataset_path"])
    if not ds_dir.exists():
        raise FileNotFoundError(
            f"LLM dataset not found at {ds_dir}. Run scripts/generate_llm_dataset.py first."
        )
    training_cfg = config["llm"].get("training") or {}
    # Which file to read is a property of the run, not of the trainer: the mixed
    # file and the report-only file are both valid targets, and an eval split is
    # optional because a structural split carries no prose to score a loss on.
    # Defaults keep the original behaviour (train.jsonl + val.jsonl) unchanged.
    train_name = str(training_cfg.get("train_file") or "train.jsonl")
    if "val_file" in training_cfg:
        val_name = training_cfg.get("val_file")
        val_name = str(val_name) if val_name else None
    else:
        val_name = "val.jsonl"
    train_hf = None
    val_hf = None
    if (ds_dir / train_name).exists():
        logger.info("Loading instruction dataset from %s", ds_dir / train_name)
        train_hf = _read_jsonl(ds_dir / train_name)
        if val_name and (ds_dir / val_name).exists():
            logger.info("Loading eval dataset from %s", ds_dir / val_name)
            val_hf = _read_jsonl(ds_dir / val_name)
        elif val_name:
            logger.info("No eval file at %s; eval loss disabled", ds_dir / val_name)
    elif (ds_dir / "train").exists():
        try:
            from datasets import load_from_disk
            logger.info("Loading instruction dataset from HF disk format at %s", ds_dir / "train")
            train_hf = load_from_disk(str(ds_dir / "train"))
            val_hf = load_from_disk(str(ds_dir / "val")) if (ds_dir / "val").exists() else None
        except ImportError:
            pass

    max_len = int(config["llm"]["training"].get("max_sequence_length", 4096))
    dummy_size = int(config["llm"].get("dummy_image_size", 448))
    compute_dtype = torch.bfloat16 if config["llm"]["training"].get("bf16", False) else torch.float16
    train = TEPInstructionDataset(train_hf, tokenizer, max_length=max_len,
                                  dummy_image_size=dummy_size, pixel_dtype=compute_dtype)
    val = (
        TEPInstructionDataset(val_hf, tokenizer, max_length=max_len,
                              dummy_image_size=dummy_size, pixel_dtype=compute_dtype)
        if val_hf is not None else None
    )
    return train, val


def _gpu_sample() -> Dict[str, object]:
    """One ``nvidia-smi`` reading, or an error instead of an invented number."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,utilization.gpu,temperature.gpu,clocks.current.sm,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        mem, util, temp, clock, power = [float(part.strip()) for part in out.split(",")]
        return {"gpu_mem_mib": mem, "gpu_util_pct": util, "gpu_temp_c": temp,
                "gpu_clock_mhz": clock, "gpu_power_w": power}
    except Exception as exc:  # pragma: no cover - depends on the host
        return {"gpu_error": str(exc)[:120]}


def training_watch(output_dir: Path, logging_steps: int):
    """Telemetry and the hard stops of a long run, written next to the adapter.

    The trainer logs its own loss; what a multi-hour unattended run cannot tell
    you afterwards is how hot the card was, how far it clocked down, or whether a
    step was ten times slower than its neighbour. Those are the failure modes that
    corrupt a run quietly, so they are sampled and, past a threshold, raised as an
    exception rather than left in a log nobody reads until the end.
    """
    from transformers import TrainerCallback

    class Watch(TrainerCallback):
        def __init__(self) -> None:
            self.jsonl = Path(output_dir) / "train_telemetry.jsonl"
            self.text = Path(output_dir) / "train_telemetry.log"
            self.last_log = time.time()
            self.last_step = time.time()
            self.hot_since: Optional[float] = None
            self.slow_streak = 0
            self.pending: Dict[str, object] = {}

        def _write(self, row: Dict) -> None:
            with open(self.jsonl, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            with open(self.text, "a", encoding="utf-8") as handle:
                handle.write(" ".join(f"{key}={value}" for key, value in row.items()) + "\n")

        def on_step_end(self, args, state, control, **kwargs):
            now = time.time()
            step_seconds = now - self.last_step
            self.last_step = now
            sample = _gpu_sample()
            temp = sample.get("gpu_temp_c")
            if isinstance(temp, float) and temp > 85.0:
                self.hot_since = self.hot_since or now
                if now - self.hot_since > 120.0:
                    raise RuntimeError(
                        f"GPU at {temp:.0f} C for more than 2 minutes at step "
                        f"{state.global_step}; stopping")
            else:
                self.hot_since = None
            self.slow_streak = self.slow_streak + 1 if step_seconds > 45.0 else 0
            if self.slow_streak >= 3:
                raise RuntimeError(
                    f"sustained step time {step_seconds:.1f}s above 45s at step "
                    f"{state.global_step}; stopping")
            self.pending = {"step": int(state.global_step),
                            "step_seconds": round(step_seconds, 2), **sample}
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            row: Dict[str, object] = {"step": int(state.global_step),
                                      "epoch": round(float(state.epoch or 0.0), 3)}
            for key in ("loss", "learning_rate", "grad_norm"):
                if key in logs:
                    row[key] = logs[key]
            if "loss" in logs and not math.isfinite(float(logs["loss"])):
                self._write(row)
                raise RuntimeError(f"non-finite loss at step {state.global_step}: {logs['loss']}")
            now = time.time()
            row["seconds_per_step"] = round((now - self.last_log) / max(logging_steps, 1), 2)
            self.last_log = now
            row.update({key: value for key, value in self.pending.items() if key not in row})
            self._write(row)
            return control

    return Watch()


def _adapter_hashes(output_dir: Path) -> Dict[str, str]:
    """SHA-256 of every file the adapter package ships."""
    hashes = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in {"train_telemetry.jsonl", "train_telemetry.log"}:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[str(path.relative_to(output_dir)).replace("\\", "/")] = digest
    return hashes


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train_tep_adapter(
    config: Dict,
    resume: bool = False,
    output_dir: Optional[str] = None,
) -> Path:
    """Run QLoRA/LoRA SFT; returns the adapter output directory."""
    from transformers import Trainer, TrainingArguments

    set_seed(config.get("seed", 42))
    llm_cfg = config["llm"]
    training = llm_cfg["training"]
    output_dir = Path(output_dir or llm_cfg.get("adapter_dir"))
    output_dir = ensure_dir(output_dir)
    device = get_device()
    logger.info("GPU: %s", device)

    use_bf16 = bool(training.get("bf16", False))
    use_fp16 = bool(training.get("fp16", False))
    if device.type == "cpu" and (use_bf16 or use_fp16):
        logger.warning("No GPU available; forcing fp32 (bf16/fp16 disabled).")
        use_bf16 = False
        use_fp16 = False

    model = load_base_model(config, eval_mode=False)
    model = wrap_peft(model, config)
    print_training_header(config, model)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        llm_cfg["base_model"], trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, val_dataset = _load_tokenized_datasets(config, tokenizer)
    collator = make_collator(tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=int(training.get("num_epochs", 3)),
        per_device_train_batch_size=int(training.get("batch_size", 2)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        lr_scheduler_type=training.get("lr_scheduler", "cosine"),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        logging_steps=int(training.get("logging_steps", 10)),
        save_steps=int(training.get("save_steps", 200)),
        eval_steps=int(training.get("eval_steps", 200)),
        eval_strategy="steps" if val_dataset is not None else "no",
        # 0 or absent-with-zero means keep every checkpoint: the selection rule
        # needs the whole curve, and a pruned history cannot be re-read later.
        save_total_limit=(int(training.get("save_total_limit", 2)) or None),
        dataloader_num_workers=int(training.get("dataloader_num_workers", 2)),
        fp16=use_fp16,
        bf16=use_bf16,
        optim=training.get("optim") or ("paged_adamw_8bit" if training.get("use_4bit", True) and torch.cuda.is_available() else "adamw_torch"),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        report_to=[],
        save_strategy="steps",
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss",
    )

    callbacks = []
    if bool(training.get("telemetry", False)):
        callbacks.append(training_watch(output_dir, int(training.get("logging_steps", 10))))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks or None,
        tokenizer=tokenizer,
    )

    resume_ckpt = _latest_checkpoint(output_dir) if resume and _checkpoints_exist(output_dir) else None
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(output_dir))

    # Persist tokenizer + remote-code/config so the adapter package is self
    # contained except for the (frozen) base weights.
    tokenizer.save_pretrained(str(output_dir))
    _copy_base_support_files(llm_cfg["base_model"], output_dir)

    # Save a human-readable train summary + adapter description.
    save_json(
        output_dir / "training_summary.json",
        {
            "adapter_name": llm_cfg.get("adapter_name", "tep_rca"),
            "base_model": llm_cfg["base_model"],
            "lora": llm_cfg["lora"],
            "training": training,
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset) if val_dataset is not None else 0,
            "global_step": trainer.state.global_step,
            "epochs_completed": trainer.state.epoch,
            # Written here, not reconstructed later: the hash of what was saved is
            # the only thing that ties a metric back to a file.
            "files_sha256": _adapter_hashes(output_dir),
        },
    )
    logger.info("TEP adapter saved to %s", output_dir)
    return output_dir


def _checkpoints_exist(output_dir: Path) -> bool:
    return any(p.is_dir() and p.name.startswith("checkpoint-") for p in output_dir.iterdir())


def _latest_checkpoint(output_dir: Path) -> Optional[str]:
    """Return the newest ``checkpoint-<step>`` directory, or None."""
    ckpts = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not ckpts:
        return None
    return str(sorted(ckpts, key=lambda p: int(p.name.split("-")[-1]))[-1])


def _copy_base_support_files(base_model: str, output_dir: Path) -> None:
    """Copy remote-code .py files + config.json of the base into the adapter dir.

    These are needed because InternVL2 is loaded with trust_remote_code=True;
    keeping them alongside the adapter makes the artifact easier to reload.
    """
    base_path = Path(base_model)
    if not base_path.exists():
        logger.info("Base model is a hub id; remote code not copied locally.")
        return
    for pattern in ("*.py", "config.json", "tokenizer*", "special_tokens_map.json"):
        for f in base_path.glob(pattern):
            try:
                shutil.copy2(f, output_dir / f.name)
            except (shutil.SameFileError, OSError) as exc:
                logger.debug("Skip copying %s: %s", f, exc)