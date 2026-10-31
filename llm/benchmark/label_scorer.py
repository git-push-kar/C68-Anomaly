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
import copy
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


#: The dataset's non-fault answer. It is deliberately *not* a menu entry -- there
#: is no fault to point at, and an open-set episode must not be able to name one --
#: but it is a legal training target (111 of the 1,919 report rows), so the closed
#: set the scorer reads has to contain it. Without it every NORMAL_TRANSIENT item is
#: unscoreable by construction: the argmax can never be that string, and its recall
#: is pinned at 0 for a reason that has nothing to do with the model.
NORMAL_TRANSIENT = "NORMAL_TRANSIENT"


def candidates() -> List[str]:
    """Every legal answer: the menu in order, then the non-fault class."""
    return [entry["fault_id"] for entry in build_menu()] + [NORMAL_TRANSIENT]


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


def normalised_score(total_logp: float, n_tokens: int) -> float:
    """Per-token mean log-probability of one label string.

    The candidates do not all tokenise to the same length once a non-``FAULT_nn``
    answer is in the set: the four label suffixes are 5 tokens for ``FAULT_01``,
    4 for ``NORMAL_TRANSIENT`` and 6 for ``UNCLASSIFIED_OPEN_SET``. Comparing the
    *sums* therefore compares a sum with a shorter sum, and every extra token
    contributes a negative term, so the shortest label wins almost regardless of
    the model: with the sum in use, adding ``NORMAL_TRANSIENT`` made 287 of 300
    validation items answer it, at every checkpoint. Dividing by the token count
    is the standard way to compare strings of different length, and it is the
    number the argmax and the probability vector are built from.
    """
    return float(total_logp) / max(int(n_tokens), 1)


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    top = max(scores.values())
    weights = {label: math.exp(value - top) for label, value in scores.items()}
    norm = sum(weights.values()) or 1.0
    return {label: round(value / norm, 6) for label, value in weights.items()}


def _causal_mask(valid, q_len: int, kv_len: int, dtype):
    """The additive mask the model would have built, for left-padded rows.

    Passing ``position_ids`` to this transformers version makes
    ``create_causal_mask`` take the packing path and build a mask for a different
    length than the batch actually has. So the mask is built here, per row: a
    query at cache position ``p`` may see every key at position ``<= p`` that is
    real in that row.
    """
    import torch

    keys = torch.arange(kv_len, device=valid.device).view(1, 1, 1, -1)
    queries = torch.arange(kv_len - q_len, kv_len, device=valid.device).view(1, 1, -1, 1)
    allowed = (keys <= queries) & valid.view(valid.shape[0], 1, 1, kv_len).bool()
    mask = torch.zeros(valid.shape[0], 1, q_len, kv_len, device=valid.device, dtype=dtype)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)


def _forward(model, ids, valid, position_ids, past, image, flags, via: str):
    """One call, wrapper or language model only, with an explicit mask."""
    mask = {"full_attention": _causal_mask(valid, ids.shape[1], valid.shape[1],
                                          next(model.parameters()).dtype)}
    if via == "language_model":
        embeds = model.language_model.get_input_embeddings()(ids)
        return model.language_model(inputs_embeds=embeds, attention_mask=mask,
                                    position_ids=position_ids, past_key_values=past,
                                    use_cache=True, return_dict=True)
    return model(pixel_values=image, input_ids=ids, attention_mask=mask,
                 image_flags=flags, position_ids=position_ids, past_key_values=past,
                 use_cache=True, return_dict=True)


def score_batch(model, tokenizer, questions: Sequence[str], dummy_size: int,
                labels: Optional[Sequence[str]] = None,
                batch: int = 1, via: str = "wrapper") -> List[Dict]:
    """Score every question against every label, ``batch`` items at a time.

    Batching is over *items*, not labels: the prompt lengths differ and the label
    suffixes do not, so one padded prefill serves the whole batch and each label
    costs a few cheap cached steps for all rows at once. That is where the saving
    is -- a label scan is prefill-bound, and a batch of 8 fills the card.
    """
    import torch

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
        valid = torch.stack(masks).to(model.device)
        # Left padding is only safe if every row is told its own positions. With
        # ``position_ids=None`` this transformers version numbers tokens from
        # ``cache_position``, so a row's first real token would inherit the width
        # of the longest prompt in the batch and one item's label probabilities
        # would depend on its batch mates -- measured before the fix: max |dp|
        # 0.73 and 36/40 argmax agreement against batch 1.
        position_ids = (valid.long().cumsum(-1) - 1).clamp(min=0)
        image = torch.zeros(len(chunk), 3, dummy_size, dummy_size,
                            dtype=next(model.parameters()).dtype, device=model.device)
        flags = torch.zeros((len(chunk),), dtype=torch.long, device=model.device)

        started = time.time()
        with torch.no_grad():
            prefill = _forward(model, ids, valid, position_ids, None, image, flags, via)
            base_past = prefill.past_key_values
            base_logits = prefill.logits[:, -1, :]
            scores = {row: {} for row in range(len(chunk))}
            raw = {row: {} for row in range(len(chunk))}
            for label in labels:
                suffix = suffixes[label]
                totals = torch.zeros(len(chunk), dtype=torch.float32, device=model.device)
                # ``DynamicCache`` grows in place, so a label that used the same
                # object would start from the previous label's length. The copy
                # is small (a few tens of MB) and is dropped at the next label.
                step_past = copy.deepcopy(base_past)
                step_logits, step_valid = base_logits, valid
                step_positions = position_ids
                for token in suffix:
                    log_probs = torch.log_softmax(step_logits.float(), dim=-1)
                    totals += log_probs[:, token]
                    step_valid = torch.cat(
                        [step_valid, torch.ones((len(chunk), 1), dtype=step_valid.dtype,
                                                device=step_valid.device)], dim=1)
                    step_positions = (step_positions[:, -1:] + 1).to(step_positions.dtype)
                    # ``image_flags=None`` is rejected by the remote model, so the
                    # same fallback the decode path uses applies here.
                    step = _forward(
                        model,
                        torch.full((len(chunk), 1), token, device=ids.device),
                        step_valid, step_positions, step_past, image, flags, via)
                    step_past = step.past_key_values
                    step_logits = step.logits[:, -1, :]
                for row, value in enumerate(totals.tolist()):
                    # Per token, not in total: see ``normalised_score``. The raw
                    # sum is kept so a reader can see the length effect instead of
                    # having to take the normalisation on trust.
                    scores[row][label] = normalised_score(value, len(suffix))
                    raw[row][label] = value
        seconds = time.time() - started
        for row in range(len(chunk)):
            probs = _softmax(scores[row])
            best = max(scores[row], key=scores[row].get) if scores[row] else None
            results.append({
                "label": best,
                "probability": probs.get(best) if best else None,
                "probs": probs,
                "mean_logp": scores[row],
                "total_logp": raw[row],
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
