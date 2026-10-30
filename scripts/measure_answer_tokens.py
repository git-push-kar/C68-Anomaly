"""Answer-length cost of the full target versus the slim one, with the real tokenizer.

The slimming argument is about generation cost as much as about truth: six of the
eleven answer fields were strings the render path already knows, and every one of
them is tokens the student has to produce on every item. This measures the
difference with the model's own tokenizer so the claim is a number, not an
intuition.

Run inside the training venv (it has the pinned transformers):

    venv_train/Scripts/python scripts/measure_answer_tokens.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark.to_sft import ANSWER_KEYS, load_split, to_row  # noqa: E402

DEFAULT_MODEL = "C:/Users/Admin/Desktop/LY_C68/deepfake_project/models/InternVL3-2B"

#: The answer keys as they were before the slimming, for the comparison.
FULL_KEYS = (
    "fault_id", "verdict", "abstain", "primary_subsystem", "ranked_hypotheses",
    "mitigation", "table_selection", "suggested_followups", "clarifying_question",
    "physical_reasoning", "narrative",
)


def full_body(item: dict) -> dict:
    """The pre-slimming answer body, rebuilt from the item's target and prose."""
    target = item.get("target") or {}
    prose = item.get("prose") or {}
    return {
        "fault_id": target.get("fault_id"),
        "verdict": target.get("verdict"),
        "abstain": bool(target.get("abstain")),
        "primary_subsystem": target.get("primary_subsystem"),
        "ranked_hypotheses": [h.get("fault_id") for h in (target.get("ranked_hypotheses") or [])],
        "mitigation": target.get("mitigation"),
        "table_selection": target.get("table_selection"),
        "suggested_followups": list(target.get("suggested_followups") or []),
        "clarifying_question": target.get("clarifying_question"),
        "physical_reasoning": prose.get("physical_reasoning"),
        "narrative": prose.get("narrative"),
    }


def percentiles(values):
    if not values:
        return {}
    values = sorted(values)
    def pick(q):
        index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        return values[index]
    return {"n": len(values), "min": values[0], "p50": pick(0.50), "p90": pick(0.90),
            "p99": pick(0.99), "max": values[-1],
            "mean": round(statistics.fmean(values), 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="outputs/benchmark_v2/sft/token_budget.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    items = [item for item in load_split(args.split) if item.get("prose")]
    if args.limit:
        items = items[: args.limit]

    prompt_tokens, full_tokens, slim_tokens = [], [], []
    for item in items:
        row = to_row(item)
        prompt_tokens.append(len(tokenizer(row["question"])["input_ids"]))
        full_tokens.append(len(tokenizer(json.dumps(full_body(item), ensure_ascii=False))["input_ids"]))
        slim_tokens.append(len(tokenizer(row["answer"])["input_ids"]))

    result = {
        "model": args.model,
        "split": args.split,
        "keys_before": list(FULL_KEYS),
        "keys_after": list(ANSWER_KEYS),
        "prompt_tokens": percentiles(prompt_tokens),
        "answer_tokens_before": percentiles(full_tokens),
        "answer_tokens_after": percentiles(slim_tokens),
        "sequence_tokens_before": percentiles([p + a for p, a in zip(prompt_tokens, full_tokens)]),
        "sequence_tokens_after": percentiles([p + a for p, a in zip(prompt_tokens, slim_tokens)]),
        "answer_token_reduction": {
            "mean": round(statistics.fmean(full_tokens) - statistics.fmean(slim_tokens), 1),
            "share": round(1 - statistics.fmean(slim_tokens) / statistics.fmean(full_tokens), 4),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
