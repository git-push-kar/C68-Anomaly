"""3B-12 pre-flight (c): idle check and three real optimizer steps before the pilot."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/tep_rca_v2_prose_pilot_seed42"
TRAIN = ROOT / "outputs/benchmark_v2/sft/train_prose_v1.jsonl"


def gpu_state():
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
    procs = [line.strip() for line in rows.stdout.splitlines() if line.strip()]
    py = [line for line in procs if "python" in line.lower()]
    mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         timeout=15).stdout.strip().splitlines()[0]
    used, total, temp = [float(x) for x in mem.split(",")]
    return {"compute_processes": procs, "python_processes": py,
            "memory_used_mib": used, "memory_total_mib": total, "temp_c": temp}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    state = gpu_state()
    if state["python_processes"]:
        report = {"aborted": "GPU busy: python compute process present", "state": state}
        (OUT / "preflight_gpu.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2

    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments
    from llm.dataset import TEPInstructionDataset, make_collator
    from llm.model import load_base_model, wrap_peft
    from utils import load_config

    config = load_config(str(ROOT / "configs/config_prose_pilot_seed42.yaml"))
    training = dict(config["llm"]["training"])
    training.update({"train_file": "train_prose_v1.jsonl", "val_file": None})
    tok = AutoTokenizer.from_pretrained(config["llm"]["base_model"], trust_remote_code=True,
                                        use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(line) for line in TRAIN.open(encoding="utf-8") if line.strip()][:24]
    ds = TEPInstructionDataset(rows, tok, max_length=4096, dummy_image_size=448)
    model = wrap_peft(load_base_model(config, eval_mode=False), config)
    args = TrainingArguments(
        output_dir=str(OUT / "preflight"), overwrite_output_dir=True,
        max_steps=3, per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=float(training["learning_rate"]), weight_decay=0.01,
        warmup_ratio=0.03, lr_scheduler_type="cosine", max_grad_norm=1.0,
        logging_steps=1, save_strategy="no", eval_strategy="no", report_to=[],
        bf16=True, fp16=False, optim="adamw_torch", gradient_checkpointing=True,
        dataloader_num_workers=0, seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=make_collator(tok.pad_token_id), tokenizer=tok)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer.train()
    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated()
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory
    headroom = (total - peak) / total * 100.0
    after = gpu_state()
    report = {
        "state_before": state,
        "steps": int(trainer.state.global_step),
        "seconds": round(elapsed, 2),
        "seconds_per_step": round(elapsed / max(int(trainer.state.global_step), 1), 2),
        "sample_rows": len(rows),
        "max_memory_allocated_gib": round(peak / 1024 ** 3, 3),
        "device_total_gib": round(total / 1024 ** 3, 3),
        "headroom_pct": round(headroom, 2),
        "headroom_floor_pct": 1.5,
        "temp_after_c": after["temp_c"],
        "projected_288_steps_hours": round(elapsed / max(int(trainer.state.global_step), 1) * 288 / 3600, 2),
        "passed": headroom >= 1.5,
    }
    (OUT / "preflight_gpu.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
