"""Twenty optimizer steps of the report task, measured rather than assumed.

What this answers before a full run is authorised:

* does the SFT export actually train through the existing dataset/collator path;
* is the loss the trainer computes the *masked* cross-entropy we think it is
  (checked by recomputing it by hand from the same logits, on the same batch);
* how many tokens are supervised versus merely shown;
* how long a step takes and how much memory it peaks at, which together give the
  3-epoch projection;
* how fast the resulting adapter generates a validation answer.

It writes nothing to the adapter directory: the model is discarded afterwards, so
a smoke run cannot be mistaken for a trained checkpoint.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SFT_DIR = Path("outputs/benchmark_v2/sft")
OUT = Path("outputs/analysis/phase3a")


def load_rows(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    import torch
    import torch.nn.functional as F
    import yaml
    from transformers import AutoTokenizer

    from llm.dataset import TEPInstructionDataset, make_collator
    from llm.model import load_base_model, wrap_peft

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config_v2.yaml")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--generate", type=int, default=20,
                        help="validation items to generate after the steps")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training = config["llm"]["training"]
    max_len = int(training.get("max_sequence_length", 4096))
    batch_size = int(args.batch_size or training.get("batch_size", 4))
    grad_accum = int(training.get("gradient_accumulation_steps", 4))
    dummy = int(config["llm"].get("dummy_image_size", 448))
    dtype = torch.bfloat16 if training.get("bf16") else torch.float16

    train_rows = load_rows(SFT_DIR / "train.jsonl")
    # Evaluation splits have no teacher prose, so they are not SFT rows; their
    # questions still come from the dataset itself, which is what generation needs.
    val_rows = load_rows(SFT_DIR / "val.jsonl")
    if not val_rows:
        from llm.benchmark.prompt_v2 import SYSTEM_PROMPT_V2

        val_rows = [{"system": SYSTEM_PROMPT_V2, "question": item.get("prompt") or ""}
                    for item in load_rows(Path("outputs/benchmark_v2/dataset/val.jsonl"))]
    if not train_rows:
        raise SystemExit(f"no SFT rows in {SFT_DIR}; run `python -m llm.benchmark.to_sft --all`")

    tokenizer = AutoTokenizer.from_pretrained(
        config["llm"]["base_model"], trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ds = TEPInstructionDataset(train_rows, tokenizer, max_length=max_len,
                                     dummy_image_size=dummy, pixel_dtype=dtype)
    collator = make_collator(tokenizer.pad_token_id)

    torch.cuda.reset_peak_memory_stats()
    model = wrap_peft(load_base_model(config, eval_mode=False), config)
    model.train()

    # -- one batch, checked by hand ------------------------------------------
    batch = collator([train_ds[i] for i in range(batch_size)])
    batch = {key: value.to(model.device) for key, value in batch.items()}
    labels = batch["labels"]
    supervised = int((labels != -100).sum())
    shown = int((labels == -100).sum())
    with torch.no_grad():
        outputs = model(**batch, use_cache=False)
    logits = outputs.logits
    hand = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                           labels[:, 1:].reshape(-1), ignore_index=-100)
    check = {
        "model_loss": float(outputs.loss),
        "hand_loss": float(hand),
        "abs_diff": abs(float(outputs.loss) - float(hand)),
        "supervised_tokens_batch": supervised,
        "masked_prompt_tokens_batch": shown,
        "supervised_fraction": round(supervised / max(1, supervised + shown), 4),
        "answer_tokens_per_example": round(supervised / batch_size, 1),
        "prompt_tokens_per_example": round(shown / batch_size, 1),
    }
    del outputs, logits
    torch.cuda.empty_cache()

    # -- twenty steps ---------------------------------------------------------
    from transformers import Trainer, TrainingArguments

    log_dir = Path("outputs/_smoke_trainer")
    cli = TrainingArguments(
        output_dir=str(log_dir),
        max_steps=args.steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(training.get("learning_rate", 2e-4)),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        lr_scheduler_type=training.get("lr_scheduler", "cosine"),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        bf16=bool(training.get("bf16", True)),
        fp16=bool(training.get("fp16", False)),
        optim="adamw_torch",
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        seed=42,
    )
    trainer = Trainer(model=model, args=cli, train_dataset=train_ds,
                      data_collator=collator)
    started = time.time()
    result = trainer.train()
    seconds = time.time() - started
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    train_metrics = {
        "steps": args.steps,
        "seconds": round(seconds, 1),
        "seconds_per_step": round(seconds / args.steps, 2),
        "seconds_per_step_with_accum": round(seconds / args.steps, 2),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_trace": [round(value, 4) for value in losses],
        "peak_gpu_gb": round(peak, 2),
        "train_loss_reported": float(result.training_loss) if result else None,
        "effective_batch": batch_size * grad_accum,
    }

    # -- projection ----------------------------------------------------------
    steps_per_epoch = (len(train_rows) + batch_size * grad_accum - 1) // (batch_size * grad_accum)
    epochs = float(training.get("num_epochs", 3))
    per_epoch = steps_per_epoch * (seconds / args.steps)
    projection = {
        "train_rows": len(train_rows),
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "total_steps": int(steps_per_epoch * epochs),
        "hours_per_config": round(per_epoch * epochs / 3600, 2),
        "hours_3_configs_3_seeds": round(per_epoch * epochs * 9 / 3600, 1),
        "note": ("linear projection from the measured per-step time at this batch "
                 "shape; step time varies with sequence length, and eval/save "
                 "overhead is not included"),
    }

    # -- generation ----------------------------------------------------------
    generation: Dict[str, object] = {"items": 0}
    if args.generate and val_rows:
        model.eval()
        model.config.use_cache = True
        produced = 0
        seconds_list: List[float] = []
        tokens_out = 0
        started = time.time()
        for row in val_rows[: args.generate]:
            prompt = (f"<|im_start|>system\n{row['system']}<|im_end|>"
                      f"<|im_start|>user\n{row['question']}<|im_end|>"
                      "<|im_start|>assistant\n")
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=max_len).to(model.device)
            start = time.time()
            with torch.no_grad():
                out = model.generate(
                    input_ids=ids.input_ids, attention_mask=ids.attention_mask,
                    pixel_values=torch.zeros(1, 3, dummy, dummy, dtype=dtype,
                                             device=model.device),
                    image_flags=torch.tensor([0], device=model.device),
                    max_new_tokens=512, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                )
            seconds_list.append(time.time() - start)
            tokens_out += int(out.shape[1] - ids.input_ids.shape[1])
            produced += 1
        elapsed = time.time() - started
        generation = {
            "items": produced,
            "seconds_total": round(elapsed, 1),
            "seconds_per_item": round(statistics.fmean(seconds_list), 2) if seconds_list else None,
            "tokens_per_second": round(tokens_out / elapsed, 2) if elapsed else None,
            "mean_new_tokens": round(tokens_out / produced, 1) if produced else None,
            "note": "greedy, unbatched, unmerged adapter; not scored",
        }

    report = {"check": check, "train": train_metrics, "projection": projection,
              "generation": generation,
              "config": {"batch_size": batch_size, "grad_accum": grad_accum,
                         "max_length": max_len, "dtype": str(dtype)}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_train.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
