"""Train the single ``tep_rca`` LoRA/QLoRA adapter on InternVL2-2B.

Flow: load ORIGINAL InternVL2-2B (frozen) -> assert no adapters -> wrap with
``tep_rca`` -> SFT on the TEP instruction dataset (text-only, dummy image +
image_flags=0) -> save ONLY the adapter (adapter_config.json +
adapter_model.safetensors) + tokenizer + the remote-code files needed to reload
it on a fresh base.
"""
from __future__ import annotations

import logging
import shutil
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
    train_hf = None
    val_hf = None
    try:
        from datasets import load_from_disk

        if (ds_dir / "train").exists():
            train_hf = load_from_disk(str(ds_dir / "train"))
            val_hf = load_from_disk(str(ds_dir / "val")) if (ds_dir / "val").exists() else None
    except ImportError:
        pass
    if train_hf is None and (ds_dir / "train.jsonl").exists():
        train_hf = _read_jsonl(ds_dir / "train.jsonl")
        val_hf = _read_jsonl(ds_dir / "val.jsonl") if (ds_dir / "val.jsonl").exists() else None

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


def _read_jsonl(path: Path) -> List[Dict]:
    import json

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
    logger.info("GPU: %s", get_device())

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
        evaluation_strategy="steps" if val_dataset is not None else "no",
        save_total_limit=int(training.get("save_total_limit", 2)),
        dataloader_num_workers=int(training.get("dataloader_num_workers", 2)),
        fp16=bool(training.get("fp16", False)),
        bf16=bool(training.get("bf16", False)),
        optim=training.get("optim") or ("paged_adamw_8bit" if training.get("use_4bit", True) else "adamw_torch"),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        report_to=[],
        save_strategy="steps",
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=str(output_dir) if resume and _checkpoints_exist(output_dir) else None)
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
        },
    )
    logger.info("TEP adapter saved to %s", output_dir)
    return output_dir


def _checkpoints_exist(output_dir: Path) -> bool:
    return any(p.is_dir() and p.name.startswith("checkpoint-") for p in output_dir.iterdir())


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