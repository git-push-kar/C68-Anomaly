"""How much of the training signal is the answer field, and where is the loss?

Two questions that decide whether "92 % validation macro-F1" says anything about the
prose the deployment wants:

``share``  Of the supervised tokens in a report row, how many belong to ``fault_id``?
           If the label field is a rounding error, the closed-set metric measures a
           capability the gradient barely optimised, and improving it needs a
           different objective rather than more data.

``loss``   Teacher-forced per-field loss of the trained adapter. Splitting the answer
           into fields turns one number into five: a model can be excellent at the
           menu choice and still be inventing its reasoning.

Validation has no teacher prose (the export was built for structural splits), so the
validation rows carry only the deterministic fields. That is reported rather than
worked around: it is exactly why eval-loss was switched off for this run.

Usage::

    venv_train/Scripts/python.exe scripts/label_loss_share.py share
    venv_train/Scripts/python.exe scripts/label_loss_share.py loss --rows 200
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

MIXED = ROOT / "outputs/benchmark_v2/sft/train_mixed.jsonl"
VAL_ITEMS = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
OUT = ROOT / "outputs/analysis/phase3a/label_loss"
TOKENIZER_PATH = "C:/Users/Admin/Desktop/LY_C68/deepfake_project/models/InternVL3-2B"
FIELDS = ("fault_id", "verdict", "ranked_hypotheses", "physical_reasoning", "narrative")
EOS = "<|im_end|>"


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def is_report_row(row: Dict) -> bool:
    return not str(row.get("item_id", "")).startswith("FD-")


def field_spans(answer: str) -> Dict[str, Dict[str, int]]:
    """Character span of each field's *value* inside the answer JSON string.

    Parsed by hand rather than by ``json.loads`` on re-serialisation: the string being
    tokenised is the exact string in the file, and re-serialising it (key order,
    spacing) would map spans onto a different string.
    """
    spans: Dict[str, Dict[str, int]] = {}
    position = 0
    for field in FIELDS:
        key = f'"{field}": '
        start = answer.find(key, position)
        if start < 0:
            continue
        value_start = start + len(key)
        cursor = value_start
        if cursor < len(answer) and answer[cursor] == '"':
            cursor += 1
            while cursor < len(answer):
                if answer[cursor] == "\\":
                    cursor += 2
                    continue
                if answer[cursor] == '"':
                    cursor += 1
                    break
                cursor += 1
        elif cursor < len(answer) and answer[cursor] in "[{":
            opening, closing = answer[cursor], "]" if answer[cursor] == "[" else "}"
            depth = 0
            while cursor < len(answer):
                if answer[cursor] == opening:
                    depth += 1
                elif answer[cursor] == closing:
                    depth -= 1
                    if depth == 0:
                        cursor += 1
                        break
                cursor += 1
        else:
            while cursor < len(answer) and answer[cursor] not in ",}":
                cursor += 1
        spans[field] = {"start": value_start, "end": cursor}
        position = cursor
    return spans


def token_char_spans(tokenizer, ids: Sequence[int], text: str) -> Optional[List[tuple]]:
    """Character span of every token, or ``None`` if the pieces do not concatenate.

    The base tokeniser is a slow (Python) one, so ``return_offsets_mapping`` is not
    available. Decoding each token on its own and summing lengths gives exact spans for
    ASCII text, which the exported answers are, and the concatenation check is what makes
    that safe to rely on: if a future answer broke it the function says so instead of
    returning spans that are off by a character.
    """
    pieces = [tokenizer.decode([int(token_id)]) for token_id in ids]
    if "".join(pieces) != text:
        return None
    spans = []
    cursor = 0
    for piece in pieces:
        spans.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return spans


def answer_for_item(item: Dict) -> str:
    """The deterministic answer a validation item supports (no prose exists)."""
    target = item.get("target") or {}
    body = {
        "fault_id": target.get("fault_id"),
        "verdict": target.get("verdict"),
        "ranked_hypotheses": [entry.get("fault_id") for entry in
                              (target.get("ranked_hypotheses") or [])],
        "physical_reasoning": "",
        "narrative": "",
    }
    return json.dumps(body)


def share(seed: int = 42) -> Dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    rows = [row for row in load_jsonl(MIXED) if is_report_row(row)]
    per_row = []
    totals = defaultdict(int)
    for row in rows:
        answer = str(row["answer"])
        supervised = tokenizer(answer + EOS, add_special_tokens=False)["input_ids"]
        spans = field_spans(answer)
        counts = {}
        for field, span in spans.items():
            counts[field] = len(tokenizer(answer[span["start"]:span["end"]],
                                          add_special_tokens=False)["input_ids"])
        # The field as the model writes it: the key, the value and the terminator.
        prefix_end = spans["fault_id"]["end"] + 1 if spans.get("fault_id") else 0
        counts["fault_id_with_key"] = len(
            tokenizer(answer[:prefix_end], add_special_tokens=False)["input_ids"])
        per_row.append({"item_id": row["item_id"], "supervised": len(supervised),
                        **counts})
        totals["supervised"] += len(supervised)
        for field in ("fault_id", "fault_id_with_key", "verdict", "ranked_hypotheses",
                      "physical_reasoning", "narrative"):
            totals[field] += counts.get(field, 0)
    supervised = [row["supervised"] for row in per_row]
    fault_tokens = [row.get("fault_id_with_key", 0) for row in per_row]
    return {
        "rows": len(per_row),
        "supervised_tokens_total": totals["supervised"],
        "supervised_tokens_mean": round(statistics.mean(supervised), 2) if supervised else None,
        "supervised_tokens_median": statistics.median(supervised) if supervised else None,
        "fault_id_tokens_mean": round(statistics.mean(fault_tokens), 3) if fault_tokens else None,
        "fault_id_share_of_supervised": round(totals["fault_id_with_key"]
                                              / max(totals["supervised"], 1), 5),
        "fault_id_value_only_share": round(totals["fault_id"]
                                           / max(totals["supervised"], 1), 5),
        "per_field_share": {field: round(totals[field] / max(totals["supervised"], 1), 5)
                            for field in ("fault_id", "fault_id_with_key", "verdict",
                                          "ranked_hypotheses", "physical_reasoning",
                                          "narrative")},
        "per_field_tokens": {field: totals[field] for field in
                             ("fault_id", "fault_id_with_key", "verdict",
                              "ranked_hypotheses", "physical_reasoning", "narrative")},
    }


def loss(step: str, rows: int, seed: int) -> Dict:
    import torch
    from peft import PeftModel

    from llm.dataset import build_chat_prompt
    from eval_protocolA import adapter_root, bare, load_base

    model, tokenizer, config, _applied = load_base(
        ROOT / "configs/config_v2_protocolA_seed42.yaml")
    path = adapter_root(ROOT / f"outputs/tep_rca_v2_protocolA_seed42/checkpoint-{step}")
    wrapped = PeftModel.from_pretrained(model, str(path), adapter_name="loss")
    wrapped.set_adapter("loss")
    base_model = bare(wrapped)
    dummy = int(config["llm"].get("dummy_image_size", 448))

    rng = random.Random(seed)
    train_rows = [row for row in load_jsonl(MIXED) if is_report_row(row)]
    rng.shuffle(train_rows)
    train_rows = train_rows[:rows]
    val_items = [item for item in load_jsonl(VAL_ITEMS)
                 if item.get("prompt") and not item.get("unobservable")]
    rng.shuffle(val_items)
    val_items = val_items[:rows]

    def measure(pairs: Sequence[Dict]) -> Dict:
        per_field: Dict[str, List[float]] = defaultdict(list)
        overall = []
        skipped = [0, 0]  # spans not derivable, rows over the sequence budget
        started = time.time()
        with torch.no_grad():
            for index, pair in enumerate(pairs, start=1):
                prompt = build_chat_prompt(pair["system"], [("user", pair["question"])])
                answer = pair["answer"]
                prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
                answer_ids = tokenizer(answer + EOS, add_special_tokens=False)["input_ids"]
                offsets = token_char_spans(tokenizer, answer_ids, answer + EOS)
                if offsets is None:
                    skipped[0] += 1
                    continue
                if len(prompt_ids) + len(answer_ids) > int(config["llm"]["training"]
                                                          .get("max_sequence_length", 4096)):
                    skipped[1] += 1
                    continue
                ids = torch.tensor([prompt_ids + answer_ids], device=base_model.device)
                labels = torch.tensor([[-100] * len(prompt_ids) + answer_ids],
                                      device=base_model.device)
                image = torch.zeros(1, 3, dummy, dummy,
                                    dtype=next(base_model.parameters()).dtype,
                                    device=base_model.device)
                flags = torch.zeros((1,), dtype=torch.long, device=base_model.device)
                logits = base_model(pixel_values=image, input_ids=ids,
                                    attention_mask=torch.ones_like(ids),
                                    image_flags=flags, use_cache=False).logits
                shift_logits = logits[:, :-1, :].float()
                shift_labels = labels[:, 1:]
                token_loss = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1), reduction="none",
                    ignore_index=-100).reshape(shift_labels.shape)[0]
                mask = shift_labels[0] != -100
                values = token_loss[mask].tolist()
                overall.extend(values)
                # Token i of the answer is predicted from position i-1, so its loss is
                # stored at index i-1 of the shifted tensor.
                supervised_positions = [pos for pos, keep in enumerate(mask.tolist()) if keep]
                for field, span in field_spans(answer).items():
                    # ``values[k]`` is the loss of predicting input token
                    # ``supervised_positions[k] + 1``, which is answer token
                    # ``supervised_positions[k] + 1 - len(prompt_ids)``. A token
                    # belongs to the field whose character span it starts in.
                    hits = []
                    for k, position in enumerate(supervised_positions):
                        answer_index = position + 1 - len(prompt_ids)
                        if not 0 <= answer_index < len(offsets):
                            continue
                        if span["start"] <= offsets[answer_index][0] < span["end"]:
                            hits.append(values[k])
                    per_field[field].extend(hits)
                if index % 50 == 0:
                    print(json.dumps({"progress": index, "of": len(pairs)}), flush=True)
        seconds = time.time() - started
        return {"seconds": round(seconds, 1),
                "rows_measured": len(pairs) - skipped[0] - skipped[1],
                "skipped_no_spans": skipped[0],
                "skipped_over_length": skipped[1],
                "supervised_tokens": len(overall),
                "overall_mean_loss": round(statistics.mean(overall), 4) if overall else None,
                "per_field": {field: {"tokens": len(values),
                                      "mean_loss": round(statistics.mean(values), 4),
                                      "perplexity": round(2.718281828 ** min(statistics.mean(values), 20), 2)}
                              for field, values in sorted(per_field.items()) if values}}

    train_pairs = [{"system": row["system"], "question": row["question"],
                    "answer": row["answer"]} for row in train_rows]
    val_pairs = []
    for item in val_items:
        from llm.benchmark.prompt_v2 import SYSTEM_PROMPT_V2

        val_pairs.append({"system": SYSTEM_PROMPT_V2, "question": item["prompt"],
                          "answer": answer_for_item(item)})
    return {"checkpoint": step, "rows": rows,
            "train": measure(train_pairs), "validation": measure(val_pairs)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    share_cmd = sub.add_parser("share")
    share_cmd.add_argument("--seed", type=int, default=42)
    loss_cmd = sub.add_parser("loss")
    loss_cmd.add_argument("--step", default="500")
    loss_cmd.add_argument("--rows", type=int, default=200)
    loss_cmd.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "share":
        result = share(args.seed)
        (OUT / "share.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        result = loss(args.step, args.rows, args.seed)
        (OUT / f"loss_step{args.step}.json").write_text(json.dumps(result, indent=2),
                                                        encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
