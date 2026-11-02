"""Does the adapter see the evidence the classifier reads? Token budget, per segment.

The classifier's features are parsed from the rendered prompt (``scripts/evidence_classifier.py``),
so in principle the two halves read the same page. The first thing that can break that claim is
the 4,096-token training cap: a prompt that does not fit is truncated from the right, and the
right-hand end of this prompt is the candidate menu, the instruction and the schema line — which
would silently turn a menu question into an open-ended one.

This script answers only that question, with the tokenizer the training run used:

* prompt and answer tokens for every training row and every validation row, against the cap;
* the headroom (cap − prompt − answer) and the count of rows with no supervised token left;
* the token count at each block boundary, so that if a row *were* cut, the report can say which
  block it cut;
* the same numbers for the 48 menu-present items the step-500 checkpoint got wrong, because a
  truncation that only affects mistakes would hide inside an aggregate.

CPU only: the tokenizer is loaded, no model and no GPU.

Usage::

    ./venv_train/Scripts/python.exe scripts/prompt_budget.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DATASET = ROOT / "outputs" / "benchmark_v2" / "dataset"
SFT = ROOT / "outputs" / "benchmark_v2" / "sft"
PREDS = ROOT / "outputs/analysis/phase3a/instrument_check/preds_decode_step500_full.jsonl"
OUT = ROOT / "outputs/analysis/phase3a/fit"
BASE_MODEL = "C:/Users/Admin/LY_C68" if False else "C:/Users/Admin/Desktop/LY_C68/deepfake_project/models/InternVL3-2B"

#: The blocks of a v2 report prompt, in the order they appear. The cut, if any, falls in the
#: first block whose end lies past the cap.
BLOCKS: Tuple[Tuple[str, str], ...] = (
    ("header", "A new anomaly has been detected"),
    ("detectors", "DETECTORS"),
    ("sensors", "TOP DEVIATING SENSORS"),
    ("residual", "DYNAMIC RESIDUAL ATTRIBUTION"),
    ("onset", "ONSET ORDER"),
    ("menu", "CANDIDATE MENU"),
    ("instruction", "Choose the single best entry"),
    ("schema", "Return one JSON object"),
)


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def percentile(values: List[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def block_offsets(question: str) -> List[Tuple[str, int]]:
    """Character offset of each block's first occurrence, in order."""
    out: List[Tuple[str, int]] = []
    for name, marker in BLOCKS:
        index = question.find(marker)
        if index >= 0:
            out.append((name, index))
    return out


def fault_of(item_id: str, prompt: str) -> str:
    """The class a row is *about*, for the per-class table.

    Taken from the prompt's own text only where it is safe: the row id carries the fault for
    benchmark items (``EVT-train-f04-r123``), which is metadata, not model input. Dialogue rows
    carry the event they are about in the same way.
    """
    text = str(item_id or "")
    parts = text.split("-")
    for part in parts:
        if part.startswith("f") and part[1:].isdigit():
            return f"FAULT_{int(part[1:]):02d}"
    return "OTHER"


def measure(rows: List[Dict], tokenizer, cap: int, max_length: int,
            with_system: bool) -> Dict:
    prompt_tokens: List[int] = []
    answer_tokens: List[int] = []
    headroom: List[int] = []
    over_cap = 0
    zero_budget = 0
    cut_blocks: Dict[str, int] = {}
    per_class: Dict[str, List[int]] = {}
    samples: List[Dict] = []
    for row in rows:
        question = row.get("question") or row.get("prompt") or ""
        system = row.get("system") or ""
        from llm.dataset import build_chat_prompt

        prompt = (build_chat_prompt(system, [("user", question)]) if with_system
                  else build_chat_prompt("", [("user", question)]))
        answer = row.get("answer") or ""
        prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False).input_ids
        answer_ids = tokenizer(answer, add_special_tokens=False, truncation=False).input_ids
        prompt_tokens.append(len(prompt_ids))
        answer_tokens.append(len(answer_ids))
        room = cap - len(prompt_ids)
        headroom.append(room - len(answer_ids))
        if len(prompt_ids) > cap:
            over_cap += 1
            # Which block does token `cap` fall in? Tokenise block prefixes and take the last
            # boundary at or before the cut.
            offsets = block_offsets(question)
            prefix_of_question = prompt[: len(prompt) - len(question)] if question else prompt
            base = len(tokenizer(prefix_of_question, add_special_tokens=True).input_ids) if prefix_of_question else 0
            cut_block = "header"
            for name, offset in offsets:
                count = base + len(tokenizer(question[:offset], add_special_tokens=False).input_ids)
                if count <= cap:
                    cut_block = name
            cut_blocks[cut_block] = cut_blocks.get(cut_block, 0) + 1
        if room <= 0:
            zero_budget += 1
        klass = fault_of(str(row.get("item_id") or ""), question)
        per_class.setdefault(klass, []).append(len(prompt_ids))
        if len(samples) < 3:
            samples.append({"item_id": row.get("item_id"), "prompt_tokens": len(prompt_ids),
                            "answer_tokens": len(answer_ids), "headroom": room - len(answer_ids)})
    return {
        "rows": len(rows),
        "max_length": max_length,
        "cap": cap,
        "prompt_tokens": {"p50": percentile(prompt_tokens, 0.5),
                          "p95": percentile(prompt_tokens, 0.95),
                          "p99": percentile(prompt_tokens, 0.99),
                          "max": max(prompt_tokens) if prompt_tokens else 0,
                          "mean": round(statistics.fmean(prompt_tokens), 1) if prompt_tokens else 0},
        "answer_tokens": {"p95": percentile(answer_tokens, 0.95),
                          "max": max(answer_tokens) if answer_tokens else 0},
        "headroom_tokens": {"min": min(headroom) if headroom else 0,
                            "p50": percentile(headroom, 0.5)},
        "rows_over_cap": over_cap,
        "rows_with_zero_answer_budget": zero_budget,
        "cut_block_counts": cut_blocks,
        "per_class_prompt_max": {key: {"rows": len(values), "p50": percentile(values, 0.5),
                                       "max": max(values)}
                                 for key, values in sorted(per_class.items())},
        "samples": samples,
    }


def error_set() -> List[Dict]:
    """The 48 menu-present validation items the step-500 checkpoint got wrong."""
    import json as _json

    preds = [_json.loads(line) for line in PREDS.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    return [row for row in preds
            if row.get("variant") == "menu" and not row.get("unobservable")
            and row.get("pred") != row.get("gold")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True,
                                              use_fast=False)
    mixed = load_jsonl(SFT / "train_mixed.jsonl")
    report_rows = [row for row in mixed if str(row.get("item_id", "")).startswith(("EVT-", "NORM-"))]
    dialogue_rows = [row for row in mixed if row not in report_rows]
    val_rows = load_jsonl(DATASET / "val.jsonl")

    errors = error_set()
    by_id = {row["item_id"]: row for row in val_rows}
    error_rows = [{"item_id": item["item_id"],
                   "question": by_id[item["item_id"]]["prompt"],
                   "system": None} for item in errors if item["item_id"] in by_id]

    report = {
        "tokenizer": {"path": args.base_model,
                      "truncation_side": getattr(tokenizer, "truncation_side", "unknown"),
                      "model_max_length": getattr(tokenizer, "model_max_length", None),
                      "padding_side": getattr(tokenizer, "padding_side", "unknown")},
        "train_mixed_report": measure(report_rows, tokenizer, args.max_length, args.max_length, True),
        "train_mixed_dialogue": measure(dialogue_rows, tokenizer, args.max_length,
                                        args.max_length, True),
        "val_report": measure([{"item_id": row["item_id"], "prompt": row["prompt"],
                                "system": None} for row in val_rows],
                              tokenizer, args.max_length, args.max_length, True),
        "val_step500_errors": measure(error_rows, tokenizer, args.max_length, args.max_length, True),
        "error_items": len(error_rows),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt_budget.json").write_text(json.dumps(report, indent=2, default=str),
                                            encoding="utf-8")
    summary = {key: {"rows": value["rows"], "prompt_max": value["prompt_tokens"]["max"],
                     "p95": value["prompt_tokens"]["p95"], "over_cap": value["rows_over_cap"],
                     "headroom_min": value["headroom_tokens"]["min"],
                     "zero_answer_budget": value["rows_with_zero_answer_budget"]}
               for key, value in report.items() if isinstance(value, dict) and "rows" in value}
    print(json.dumps({"truncation_side": report["tokenizer"]["truncation_side"],
                      "summary": summary}, indent=2, default=str))
    print(f"written: {out / 'prompt_budget.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
