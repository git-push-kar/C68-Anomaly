"""Closed-set scoring in one forward pass per item.

Generation is the wrong instrument for "which fault is this". The answer is a menu
choice, so the quantity worth measuring is the probability the model assigns to
each menu id at the first answer field -- one prefill plus a few cheap cached steps
per candidate, instead of a 256-token decode that ends in a string that then has to
be parsed.

This is also where confidence comes from when a calibrator exists: the probability
vector here, isotonic-calibrated on validation. No calibrator is fitted yet, so the
values are raw model probabilities and the renderer says "uncalibrated".

Usage::

    venv_train/Scripts/python -m llm.benchmark.label_scorer --items 20 --generate
    venv_train/Scripts/python -m llm.benchmark.label_scorer --split test --limit 200
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark.menu import UNCLASSIFIED, build_menu  # noqa: E402
from llm.benchmark.prompt_v2 import SYSTEM_PROMPT_V2  # noqa: E402

DATASET = ROOT / "outputs" / "benchmark_v2" / "dataset"
DEFAULT_CONFIG = ROOT / "configs" / "config_v2.yaml"

#: The answer's first field, as the exporter writes it. The scorer reads the
#: distribution right after this prefix, so it must match the training format.
ANSWER_PREFIX = '{"fault_id": "'


def candidates() -> List[str]:
    """Every legal answer, in menu order."""
    return [entry["fault_id"] for entry in build_menu()]


def prompt_for(question: str) -> str:
    from llm.dataset import build_chat_prompt

    return build_chat_prompt(SYSTEM_PROMPT_V2, [("user", question)])


def _suffix_ids(tokenizer, label: str) -> List[int]:
    """Tokens a label adds after ``ANSWER_PREFIX`` (closing quote included)."""
    prefix = tokenizer(ANSWER_PREFIX, add_special_tokens=False)["input_ids"]
    whole = tokenizer(ANSWER_PREFIX + label + '"', add_special_tokens=False)["input_ids"]
    if whole[: len(prefix)] != prefix:
        raise RuntimeError("answer prefix does not tokenise as a stable prefix")
    return whole[len(prefix):]


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    top = max(scores.values())
    weights = {label: math.exp(value - top) for label, value in scores.items()}
    norm = sum(weights.values()) or 1.0
    return {label: round(value / norm, 6) for label, value in weights.items()}


def score_batch(model, tokenizer, questions: Sequence[str], dummy_size: int,
                labels: Optional[Sequence[str]] = None,
                batch: int = 1) -> List[Dict]:
    """Score every question against every label, ``batch`` items at a time.

    Batching is over *items*, not labels: the prompt lengths differ and the label
    suffixes do not, so one padded prefill serves the whole batch and each label
    costs a few cheap cached steps for all rows at once. That is where the saving
    is -- a label scan is prefill-bound, and a batch of 8 fills the card.
    """
    import torch

    from llm.benchmark.decode_bench import _step

    labels = list(labels or candidates())
    suffixes = {label: _suffix_ids(tokenizer, label) for label in labels}
    results: List[Dict] = []
    for start in range(0, len(questions), batch):
        chunk = list(questions[start:start + batch])
        encoded = [tokenizer(prompt_for(question) + ANSWER_PREFIX,
                             add_special_tokens=True)["input_ids"] for question in chunk]
        width = max(len(ids) for ids in encoded)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        inputs, masks = [], []
        for ids in encoded:
            pad = width - len(ids)
            inputs.append(torch.tensor([pad_id] * pad + list(ids)))
            masks.append(torch.tensor([0] * pad + [1] * len(ids)))
        ids = torch.stack(inputs).to(model.device)
        mask = torch.stack(masks).to(model.device)
        image = torch.zeros(len(chunk), 3, dummy_size, dummy_size,
                            dtype=next(model.parameters()).dtype, device=model.device)
        flags = torch.zeros((len(chunk),), dtype=torch.long, device=model.device)

        started = time.time()
        with torch.no_grad():
            prefill = model(pixel_values=image, input_ids=ids, attention_mask=mask,
                            image_flags=flags, use_cache=True)
            base_past = prefill.past_key_values
            base_logits = prefill.logits[:, -1, :]
            scores = {row: {} for row in range(len(chunk))}
            for label in labels:
                suffix = suffixes[label]
                totals = torch.zeros(len(chunk), dtype=torch.float32, device=model.device)
                step_past, step_logits, step_mask = base_past, base_logits, mask
                for token in suffix:
                    log_probs = torch.log_softmax(step_logits.float(), dim=-1)
                    totals += log_probs[:, token]
                    step_mask = torch.cat(
                        [step_mask, torch.ones((len(chunk), 1), dtype=step_mask.dtype,
                                               device=step_mask.device)], dim=1)
                    # ``image_flags=None`` is rejected by the remote model, so the
                    # same fallback the decode path uses applies here.
                    step, _used = _step(
                        model,
                        torch.full((len(chunk), 1), token, device=ids.device),
                        step_mask, step_past, image, flags)
                    step_past = step.past_key_values
                    step_logits = step.logits[:, -1, :]
                for row, value in enumerate(totals.tolist()):
                    scores[row][label] = value
        seconds = time.time() - started
        for row in range(len(chunk)):
            probs = _softmax(scores[row])
            best = max(scores[row], key=scores[row].get) if scores[row] else None
            results.append({
                "label": best,
                "probability": probs.get(best) if best else None,
                "probs": probs,
                "seconds": round(seconds / len(chunk), 4),
                "n_labels": len(scores[row]),
            })
    return results


def score_item(model, tokenizer, question: str, dummy_size: int,
               labels: Optional[Sequence[str]] = None) -> Dict:
    """One question, batch of one (kept for tests and for the agreement check)."""
    return score_batch(model, tokenizer, [question], dummy_size, labels, batch=1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1,
                        help="items per prefill; batching is over items, not labels")
    parser.add_argument("--generate", action="store_true",
                        help="also generate an answer for the first N items and compare labels")
    parser.add_argument("--generate-items", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch
    import yaml
    from transformers import AutoTokenizer

    from llm.benchmark.decode_bench import build_model

    items = [json.loads(line) for line in
             (DATASET / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    items = [item for item in items if item.get("prompt")][: args.limit]
    model, tokenizer, config, applied = build_model(Path(args.config), args.adapter, merged=False)
    dummy_size = int(config["llm"].get("dummy_image_size", 448))

    started = time.time()
    rows = []
    correct = 0
    scored_list = score_batch(model, tokenizer, [item["prompt"] for item in items],
                              dummy_size, batch=args.batch)
    for item, scored in zip(items, scored_list):
        truth = (item.get("target") or {}).get("fault_id")
        rows.append({
            "item_id": item.get("item_id"),
            "fault": item.get("fault"),
            "variant": item.get("variant"),
            "truth": truth,
            "label": scored["label"],
            "probability": scored["probability"],
            "correct": scored["label"] == truth,
            "seconds": scored["seconds"],
        })
        correct += int(scored["label"] == truth)
    seconds = time.time() - started

    result = {
        "split": args.split,
        "model": applied,
        "batch": args.batch,
        "items": len(items),
        "seconds": round(seconds, 2),
        "seconds_per_item": round(seconds / max(len(items), 1), 3),
        "accuracy": round(correct / max(len(items), 1), 4),
        "rows": rows,
    }
    if args.generate:
        from llm.benchmark.decode_bench import decode_kv

        pairs = []
        for item in items[: args.generate_items]:
            generated = decode_kv(model, tokenizer, item["prompt"], args.max_new_tokens, dummy_size)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            produced = None
            for label in candidates():
                if label in text:
                    produced = label
                    break
            scored = next(row for row in rows if row["item_id"] == item["item_id"])
            pairs.append({"item_id": item.get("item_id"), "generated_label": produced,
                          "scored_label": scored["label"],
                          "agree": produced == scored["label"]})
        result["generation_agreement"] = {
            "items": len(pairs),
            "agree": sum(1 for pair in pairs if pair["agree"]),
            "pairs": pairs,
        }
    out = Path(args.out) if args.out else ROOT / "outputs" / "benchmark_v2" / "sft" / f"label_scan_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {key: result[key] for key in
               ("split", "model", "batch", "items", "seconds_per_item", "accuracy")}
    print(json.dumps(summary, indent=2))
    if "generation_agreement" in result:
        print(json.dumps(result["generation_agreement"]["agree"], indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
