"""What train_prose_v1.jsonl costs the optimizer, in the base model's own tokens.

CPU only: the tokenizer is loaded, no model and no GPU. Run with the training interpreter.

Usage::  ./venv_train/Scripts/python.exe scripts/measure_prose_tokens.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_MODEL = "C:/Users/Admin/Desktop/LY_C68/deepfake_project/models/InternVL3-2B"
FILE = ROOT / "outputs/benchmark_v2/sft/train_prose_v1.jsonl"
OUT = ROOT / "outputs/benchmark_v2/sft/train_prose_v1_tokens.json"
CAP = 4096
COMPLETION = 512
BATCH, ACCUM, EPOCHS, SECONDS_PER_STEP = 4, 4, 3, 21.71


def main() -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    rows = [json.loads(line) for line in FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    total_prompt = total_answer = 0
    prompts, answers, supervised = [], [], []
    truncated = 0
    for row in rows:
        prompt = f"{row.get('system') or ''}\n{row['question']}"
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        answer_tokens = len(tokenizer(str(row["answer"]), add_special_tokens=False)["input_ids"])
        prompts.append(prompt_tokens)
        answers.append(answer_tokens)
        supervised.append(min(answer_tokens, COMPLETION))
        total_prompt += prompt_tokens
        total_answer += answer_tokens
        truncated += int(prompt_tokens > CAP)
    steps_per_epoch = -(-len(rows) // (BATCH * ACCUM))
    steps = steps_per_epoch * EPOCHS
    report = {
        "file": str(FILE), "rows": len(rows), "cap": CAP, "completion_cap": COMPLETION,
        "prompt_tokens": total_prompt, "answer_tokens": total_answer,
        "supervised_tokens_at_512": sum(supervised),
        "max_prompt_tokens": max(prompts), "mean_prompt_tokens": round(sum(prompts) / len(prompts), 1),
        "max_answer_tokens": max(answers),
        "rows_over_4096_prompt": truncated,
        "steps_per_epoch": steps_per_epoch, "steps": steps,
        "seconds_per_step": SECONDS_PER_STEP,
        "estimated_seconds": round(steps * SECONDS_PER_STEP, 1),
        "estimated_hours": round(steps * SECONDS_PER_STEP / 3600, 2),
        "note": "supervised tokens clip each answer at the 512-token completion budget; the estimate "
                "reuses run-1's measured 21.71 s/step rather than a measurement of this file",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
