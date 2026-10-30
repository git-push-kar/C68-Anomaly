"""Measure how long the benchmark training examples actually are.

The trainer truncates the prompt to ``max_sequence_length`` and then gives the
answer whatever budget is left. Neither step warns, so a prompt that is too long
silently loses its evidence tail and, past the limit, leaves an answer with zero
tokens -- an example that trains the model to emit nothing. Token counts are
therefore a precondition for a smoke run, not a diagnostic after it.

CPU only: loads the tokenizer, never the model. The answer is rendered in the
schema-shaped form the adapter is expected to use, built from the deterministic
target plus the teacher prose, so the estimate is the text the student must emit
rather than a guess.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm.dataset import SYSTEM_PROMPT, build_chat_prompt  # noqa: E402

DATASET = Path("outputs/benchmark_v2/dataset")
OUT = Path("outputs/analysis/phase3a")

#: Keys the student emits. ``severity`` is deliberately absent: it is computed in
#: code from the detector statistics and is not a model output.
ANSWER_ORDER = (
    "fault_id", "verdict", "abstain", "primary_subsystem", "ranked_hypotheses",
    "mitigation", "table_selection", "suggested_followups", "clarifying_question",
    "physical_reasoning", "narrative",
)


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_answer(item: Dict) -> str:
    """The answer the student is trained to produce, as compact JSON."""
    target = item.get("target") or {}
    prose = item.get("prose") or {}
    payload = {
        "fault_id": target.get("fault_id"),
        "verdict": target.get("verdict"),
        "abstain": target.get("abstain"),
        "primary_subsystem": target.get("primary_subsystem"),
        "ranked_hypotheses": [hyp.get("fault_id") for hyp in (target.get("ranked_hypotheses") or [])],
        "mitigation": target.get("mitigation"),
        "table_selection": target.get("table_selection"),
        "suggested_followups": target.get("suggested_followups") or [],
        "clarifying_question": target.get("clarifying_question"),
        "physical_reasoning": prose.get("physical_reasoning"),
        "narrative": prose.get("narrative"),
    }
    return json.dumps({key: payload[key] for key in ANSWER_ORDER}, ensure_ascii=False)


def percentile(values: List[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def measure(rows: List[Dict], tokenizer, max_length: int, system: Optional[str],
            with_system: bool) -> Dict:
    prompt_lengths: List[int] = []
    answer_lengths: List[int] = []
    truncated_prompts = 0
    truncated_answers = 0
    empty_answers = 0
    over_long = 0
    for row in rows:
        text = row.get("prompt") or ""
        prompt = (build_chat_prompt(system, [("user", text)])
                  if with_system else f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
        prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True,
                               max_length=max_length).input_ids
        answer_ids = tokenizer(render_answer(row), add_special_tokens=False).input_ids
        answer_ids = answer_ids + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
        budget = max_length - len(prompt_ids)
        prompt_lengths.append(len(prompt_ids))
        answer_lengths.append(len(answer_ids))
        if len(prompt_ids) >= max_length:
            truncated_prompts += 1
        if len(answer_ids) > budget:
            truncated_answers += 1
        if budget <= 0:
            empty_answers += 1
        if len(prompt_ids) + len(answer_ids) > max_length:
            over_long += 1
    total = [p + a for p, a in zip(prompt_lengths, answer_lengths)]
    return {
        "rows": len(rows),
        "max_length": max_length,
        "system_prompt": "default SYSTEM_PROMPT" if with_system else "none",
        "prompt_tokens": {
            "p50": percentile(prompt_lengths, 0.50), "p90": percentile(prompt_lengths, 0.90),
            "p95": percentile(prompt_lengths, 0.95), "p99": percentile(prompt_lengths, 0.99),
            "max": max(prompt_lengths) if prompt_lengths else 0,
            "mean": round(statistics.fmean(prompt_lengths), 1) if prompt_lengths else 0,
        },
        "answer_tokens": {
            "p50": percentile(answer_lengths, 0.50), "p90": percentile(answer_lengths, 0.90),
            "p95": percentile(answer_lengths, 0.95), "p99": percentile(answer_lengths, 0.99),
            "max": max(answer_lengths) if answer_lengths else 0,
            "mean": round(statistics.fmean(answer_lengths), 1) if answer_lengths else 0,
        },
        "total_tokens": {
            "p50": percentile(total, 0.50), "p95": percentile(total, 0.95),
            "max": max(total) if total else 0,
        },
        "prompts_at_limit": truncated_prompts,
        "answers_truncated": truncated_answers,
        "answers_with_zero_budget": empty_answers,
        "rows_over_limit": over_long,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=(
        "C:/Users/Admin/Desktop/LY_C68/deepfake_project/models/InternVL3-2B"))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=False)
    report: Dict[str, Dict] = {}
    for split in ("train", "val", "test"):
        rows = load_jsonl(DATASET / f"{split}.jsonl")
        if not rows:
            continue
        report[f"{split}_with_system"] = measure(rows, tokenizer, args.max_length,
                                                 SYSTEM_PROMPT, True)
        report[f"{split}_no_system"] = measure(rows, tokenizer, args.max_length, None, False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "token_lengths.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    headers = ["split / system", "rows", "prompt p50", "prompt p95", "prompt max",
               "answer p50", "answer p95", "answer max", "total p95", "total max",
               "answers truncated", "zero budget"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for name, row in report.items():
        lines.append("| " + " | ".join(str(v) for v in [
            name, row["rows"], row["prompt_tokens"]["p50"], row["prompt_tokens"]["p95"],
            row["prompt_tokens"]["max"], row["answer_tokens"]["p50"],
            row["answer_tokens"]["p95"], row["answer_tokens"]["max"],
            row["total_tokens"]["p95"], row["total_tokens"]["max"],
            row["answers_truncated"], row["answers_with_zero_budget"]]) + " |")
    text = "\n".join(lines)
    (out / "token_lengths.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
