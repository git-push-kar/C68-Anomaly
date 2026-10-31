#!/usr/bin/env python
"""Decode through the language model alone, skipping the vision tower.

Every text-only call currently runs InternVL's ``forward``: it extracts a ViT
feature map for a dummy image on *every* step, then discards it because
``image_flags`` is zero. That is pure overhead here. This calls
``model.language_model`` with ``inputs_embeds`` from the text tokens and a KV
cache, and measures what changes:

* label probabilities against the wrapper's own scorer (cosine, argmax), so a
  faster path cannot quietly score differently -- both use the same masking, so
  the only difference is whether the ViT runs;
* seconds per item for both paths at a 512-token cap, and how often a complete
  JSON object actually fits in 512 tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET = Path("outputs/benchmark_v2/dataset")
OUT = Path("outputs/benchmark_v2/sft/decode_lm_direct.json")


def load_items(limit: int) -> List[Dict]:
    rows = [json.loads(line) for line in
            (DATASET / "val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("prompt")][:limit]


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(a.get(k, 0.0) ** 2 for k in keys))
    nb = math.sqrt(sum(b.get(k, 0.0) ** 2 for k in keys))
    return round(dot / (na * nb), 6) if na and nb else 0.0


def generate_direct(model, tokenizer, prompt: str, max_new: int) -> Tuple[str, float]:
    """Greedy decode through the language model, one unbatched prompt at a time.

    No padding is involved, so no position bookkeeping is needed: the sequence
    starts at zero and grows by one token per step.
    """
    import torch

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(model.device)
    mask = enc["attention_mask"].to(model.device)
    stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    started = time.time()
    produced: List[int] = []
    with torch.no_grad():
        embeds = model.language_model.get_input_embeddings()(ids)
        out = model.language_model(inputs_embeds=embeds, attention_mask=mask,
                                   use_cache=True, return_dict=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
        for _ in range(max_new):
            token = int(torch.argmax(logits, dim=-1))
            if token in stop:
                break
            produced.append(token)
            mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)],
                             dim=1)
            step = model.language_model(input_ids=torch.tensor([[token]], device=model.device),
                                        attention_mask=mask, past_key_values=past,
                                        use_cache=True, return_dict=True)
            past, logits = step.past_key_values, step.logits[:, -1, :]
    return tokenizer.decode(produced, skip_special_tokens=True), round(time.time() - started, 3)


def complete_json(text: str) -> bool:
    body = (text or "").strip()
    try:
        json.loads(body)
        return True
    except ValueError:
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end <= start:
            return False
        try:
            json.loads(body[start:end + 1])
            return True
        except ValueError:
            return False


def main() -> int:
    import torch
    import yaml

    from llm.benchmark.decode_bench import build_model, decode_kv
    from llm.benchmark.label_scorer import candidates, score_batch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16, help="batch for the label-scan comparison")
    ap.add_argument("--generate-items", type=int, default=None,
                    help="items to decode per path (default: all)")
    ap.add_argument("--config", default="configs/config_v2.yaml")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    items = load_items(args.items)
    model, tokenizer, config, applied = build_model(Path(args.config), None, merged=False)
    dummy = int(config["llm"].get("dummy_image_size", 448))
    labels = candidates()
    torch.cuda.reset_peak_memory_stats()

    questions = [item["prompt"] for item in items]
    print(f"{len(items)} items, {len(labels)} labels, batch {args.batch}")
    started = time.time()
    wrapper = score_batch(model, tokenizer, questions, dummy, labels=labels, batch=args.batch)
    wrapper_seconds = round((time.time() - started) / len(items), 4)
    started = time.time()
    direct = score_batch(model, tokenizer, questions, dummy, labels=labels,
                         batch=args.batch, via="language_model")
    direct_seconds = round((time.time() - started) / len(items), 4)

    comparisons = []
    for item, w_row, d_row in zip(items, wrapper, direct):
        d_best = max(d_row["probs"], key=d_row["probs"].get)
        comparisons.append({
            "item_id": item["item_id"],
            "cosine": cosine(w_row["probs"], d_row["probs"]),
            "wrapper_label": w_row["label"],
            "direct_label": d_best,
            "agree": w_row["label"] == d_best,
            "max_abs_delta": round(max(abs(w_row["probs"].get(k, 0.0) - d_row["probs"].get(k, 0.0))
                                       for k in set(w_row["probs"]) | set(d_row["probs"])), 6),
        })

    generation: Dict[str, List[Dict]] = {"wrapper": [], "direct": []}
    to_generate = items[: args.generate_items or len(items)]
    for item in to_generate:
        started = time.time()
        produced_ids = decode_kv(model, tokenizer, item["prompt"], args.max_new_tokens, dummy)
        seconds = round(time.time() - started, 3)
        text = tokenizer.decode(produced_ids, skip_special_tokens=True)
        generation["wrapper"].append({
            "item_id": item["item_id"], "seconds": seconds,
            "tokens": len(produced_ids),
            "complete_json": complete_json(text),
        })
    for item in to_generate:
        text, seconds = generate_direct(model, tokenizer, item["prompt"], args.max_new_tokens)
        generation["direct"].append({
            "item_id": item["item_id"], "seconds": seconds,
            "tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"]),
            "complete_json": complete_json(text),
        })

    def summarise(rows: List[Dict]) -> Dict:
        seconds = [row["seconds"] for row in rows] or [0.0]
        return {
            "items": len(rows),
            "seconds_per_item": round(sum(seconds) / len(seconds), 3),
            "mean_tokens": round(sum(row["tokens"] for row in rows) / max(1, len(rows)), 1),
            "complete_json": sum(1 for row in rows if row["complete_json"]),
            "complete_json_rate": round(sum(1 for row in rows if row["complete_json"])
                                        / max(1, len(rows)), 3),
            "hit_cap": sum(1 for row in rows if row["tokens"] >= args.max_new_tokens),
        }

    report = {
        "items": len(items),
        "batch": args.batch,
        "labels": len(labels),
        "label_scan": {
            "wrapper_seconds_per_item": wrapper_seconds,
            "direct_seconds_per_item": direct_seconds,
            "speedup": round(wrapper_seconds / direct_seconds, 2) if direct_seconds else None,
            "argmax_agreement": sum(1 for c in comparisons if c["agree"]),
            "min_cosine": min((c["cosine"] for c in comparisons), default=None),
            "max_abs_delta": max((c["max_abs_delta"] for c in comparisons), default=None),
        },
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "wrapper": summarise(generation["wrapper"]),
            "direct": summarise(generation["direct"]),
        },
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "comparisons": comparisons,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
