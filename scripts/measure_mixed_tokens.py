#!/usr/bin/env python
"""What the mixed SFT file actually costs the optimizer, in real tokens.

The trainer masks everything before the assistant header, so "answer tokens" is
not "tokens the answer would take on its own": the prompt is truncated first and
the answer only gets what is left of ``max_sequence_length``. This measures that
split with the base model's own tokenizer, per component (report / dialogue) and
per dialogue case type, and reports optimizer steps at the configured effective
batch size.

Run with the training interpreter (it needs ``transformers``)::

    venv_train/Scripts/python.exe scripts/measure_mixed_tokens.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MIXED = Path("outputs/benchmark_v2/sft/train_mixed.jsonl")
FOLLOWUP_TRAIN = Path("outputs/benchmark_v2/followup/train.jsonl")
OUT = Path("outputs/benchmark_v2/sft/token_share.json")


def load_rows(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def component_of(item_id: str) -> str:
    return "dialogue" if str(item_id).startswith("FD") else "report"


def case_types(followup_path: Path) -> Dict[str, str]:
    """dialogue_id -> case_type, so a dialogue row can be attributed to its case."""
    if not followup_path.exists():
        return {}
    return {row["dialogue_id"]: row["case_type"]
            for row in load_rows(followup_path)}


def turn_case_types(followup_path: Path) -> Dict[str, str]:
    """``dialogue_id#turn`` -> that turn's own case type.

    A dialogue's case type names its first turn; later turns can be a different
    case (a repeat, a clarification). Attributing a row to the dialogue's type
    makes the template sample look over-represented, so the per-turn map is used
    for the row counts.
    """
    if not followup_path.exists():
        return {}
    mapping: Dict[str, str] = {}
    for row in load_rows(followup_path):
        for index, turn in enumerate(row["turns"]):
            mapping[f"{row['dialogue_id']}#{index}"] = turn.get("case_type") or row["case_type"]
    return mapping


def percentiles(values: List[int]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(p: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        return float(ordered[index])

    return {
        "mean": round(statistics.fmean(values), 1),
        "p50": pick(0.50),
        "p90": pick(0.90),
        "p99": pick(0.99),
        "max": float(ordered[-1]),
    }


def main() -> int:
    import yaml
    from transformers import AutoTokenizer

    from llm.dataset import EOS, build_chat_prompt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default=str(MIXED))
    ap.add_argument("--followup", default=str(FOLLOWUP_TRAIN))
    ap.add_argument("--config", default="configs/config_v2.yaml")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training = config["llm"]["training"]
    max_len = int(training.get("max_sequence_length", 4096))
    batch = int(training.get("batch_size", 4))
    accum = int(training.get("gradient_accumulation_steps", 4))
    effective = batch * accum
    epochs = int(training.get("num_epochs", 3))

    tokenizer = AutoTokenizer.from_pretrained(
        config["llm"]["base_model"], trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_rows(Path(args.rows))
    cases = case_types(Path(args.followup))

    stats: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: {"prompt": [], "supervised": [], "full": []})
    truncated_prompt = 0
    truncated_answer = 0
    for row in rows:
        system = row.get("system") or ""
        prompt = build_chat_prompt(system, [("user", row["question"])])
        prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True,
                               max_length=max_len).input_ids
        full_answer = tokenizer(row["answer"] + EOS, add_special_tokens=False).input_ids
        budget = max_len - len(prompt_ids)
        answer_ids = full_answer[:max(0, budget)]
        if len(answer_ids) < len(full_answer):
            truncated_answer += 1
        if len(prompt_ids) >= max_len:
            truncated_prompt += 1
        key = component_of(row["item_id"])
        bucket = stats[key]
        bucket["prompt"].append(len(prompt_ids))
        bucket["supervised"].append(len(answer_ids))
        bucket["full"].append(len(prompt_ids) + len(answer_ids))
        if key == "dialogue":
            dialogue_id = str(row["item_id"]).split("#")[0]
            bucket = stats[f"dialogue:{cases.get(dialogue_id, 'unknown')}"]
            bucket["prompt"].append(len(prompt_ids))
            bucket["supervised"].append(len(answer_ids))
            bucket["full"].append(len(prompt_ids) + len(answer_ids))

    report = {
        "rows": len(rows),
        "max_sequence_length": max_len,
        "batch_size": batch,
        "gradient_accumulation_steps": accum,
        "effective_batch": effective,
        "components": {},
        "truncation": {
            "answers_cut_by_the_length_budget": truncated_answer,
            "prompts_at_the_length_cap": truncated_prompt,
        },
    }
    total_supervised = sum(len(v["supervised"]) and sum(v["supervised"]) or 0
                           for k, v in stats.items() if ":" not in k)
    for key in sorted(stats):
        bucket = stats[key]
        supervised_total = sum(bucket["supervised"])
        report["components"][key] = {
            "rows": len(bucket["supervised"]),
            "prompt_tokens": percentiles(bucket["prompt"]),
            "supervised_tokens": percentiles(bucket["supervised"]),
            "total_tokens": percentiles(bucket["full"]),
            "supervised_token_total": supervised_total,
            "supervised_share_of_all": round(supervised_total / max(1, total_supervised), 4)
            if ":" not in key else None,
        }

    steps_per_epoch = -(-len(rows) // effective)
    report["steps"] = {
        "per_epoch": steps_per_epoch,
        "for_num_epochs": steps_per_epoch * epochs,
        "num_epochs": epochs,
        "note": "ceil(rows / effective batch); one optimizer step per accumulation window",
    }

    # Accepted train turns the mix leaves out, by case type: the surplus exists
    # so a later 1:1 experiment needs no new teacher calls.
    per_turn = turn_case_types(Path(args.followup))
    accepted = Counter()
    used = Counter()
    if Path(args.followup).exists():
        for row in load_rows(Path(args.followup)):
            for index, turn in enumerate(row["turns"]):
                if turn.get("accepted") and turn.get("answer_source") == "teacher":
                    accepted[turn.get("case_type") or row["case_type"]] += 1
        for row in rows:
            if component_of(row["item_id"]) == "dialogue":
                used[per_turn.get(str(row["item_id"]), "unknown")] += 1
    report["unused_accepted_turns"] = {
        "accepted_total": sum(accepted.values()),
        "used_total": sum(used.values()),
        "by_case_type": {
            case: {
                "accepted": accepted.get(case, 0),
                "used": used.get(case, 0),
                "unused": accepted.get(case, 0) - used.get(case, 0),
            }
            for case in sorted(set(accepted) | set(used))
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"rows {report['rows']}  effective batch {effective}  "
          f"steps/epoch {steps_per_epoch}  3 epochs {steps_per_epoch * epochs}")
    print(f"{'component':24s} {'rows':>5s} {'prompt p50':>10s} {'sup p50':>8s} "
          f"{'sup mean':>9s} {'sup total':>10s} {'share':>7s}")
    for key, block in report["components"].items():
        print(f"{key:24s} {block['rows']:5d} {block['prompt_tokens']['p50']:10.0f} "
              f"{block['supervised_tokens']['p50']:8.0f} {block['supervised_tokens']['mean']:9.1f} "
              f"{block['supervised_token_total']:10d} "
              f"{(block['supervised_share_of_all'] or 0):7.3f}")
    print(f"truncated answers {truncated_answer}  prompts at cap {truncated_prompt}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
