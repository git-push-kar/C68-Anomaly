#!/usr/bin/env python
"""Is the label scorer's probability vector independent of the batch it lands in?

Scoring is a prefill plus a few cached steps per label, so it should depend only
on the question. Left padding plus a growing attention mask can break that if the
model derives positions from the *tensor width* instead of the mask, which would
make a label's log-probability depend on the longest prompt in its batch -- and a
batch-16 evaluation would then not be comparable with a batch-1 one.

Reports, per item, the largest absolute difference of the label log-probability
vector against batch 1 and whether the argmax agrees, plus accuracy and seconds
per item at every batch size.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET = Path("outputs/benchmark_v2/dataset")
OUT = Path("outputs/benchmark_v2/sft/label_batch_invariance.json")


def load_items(split: str, limit: int) -> List[Dict]:
    rows = [json.loads(line) for line in
            (DATASET / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return [row for row in rows if row.get("prompt")][:limit]


def compare(reference: Dict[str, Dict], other: Dict[str, Dict], items: Sequence[Dict]) -> Dict:
    deltas: List[float] = []
    agree = 0
    flips: List[Dict] = []
    for item in items:
        item_id = item["item_id"]
        ref_probs, other_probs = reference[item_id], other[item_id]
        labels = set(ref_probs) | set(other_probs)
        delta = max(abs(ref_probs.get(label, 0.0) - other_probs.get(label, 0.0))
                    for label in labels)
        deltas.append(delta)
        ref_best = max(ref_probs, key=ref_probs.get)
        other_best = max(other_probs, key=other_probs.get)
        if ref_best == other_best:
            agree += 1
        else:
            flips.append({"item_id": item_id, "batch1": ref_best, "batched": other_best})
    deltas.sort()
    return {
        "items": len(items),
        "argmax_agreement": agree,
        "argmax_agreement_rate": round(agree / max(1, len(items)), 4),
        "max_abs_prob_delta": round(deltas[-1], 6) if deltas else None,
        "p50_abs_prob_delta": round(deltas[len(deltas) // 2], 6) if deltas else None,
        "flips": flips,
    }


def main() -> int:
    import torch
    import yaml

    from llm.benchmark.decode_bench import build_model
    from llm.benchmark.label_scorer import score_batch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 4, 8, 16])
    ap.add_argument("--config", default="configs/config_v2.yaml")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    items = load_items(args.split, args.limit)
    model, tokenizer, config, applied = build_model(Path(args.config), None, merged=False)
    dummy = int(config["llm"].get("dummy_image_size", 448))
    torch.cuda.reset_peak_memory_stats()

    probs: Dict[int, Dict[str, Dict]] = {}
    timing: Dict[str, Dict] = {}
    accuracies: Dict[int, float] = {}
    for batch in args.batches:
        started = time.time()
        scored = score_batch(model, tokenizer, [item["prompt"] for item in items],
                             dummy, batch=batch)
        seconds = time.time() - started
        probs[batch] = {item["item_id"]: row["probs"] for item, row in zip(items, scored)}
        correct = sum(1 for item, row in zip(items, scored)
                      if row["label"] == (item.get("target") or {}).get("fault_id"))
        accuracies[batch] = round(correct / max(1, len(items)), 4)
        timing[str(batch)] = {"seconds": round(seconds, 2),
                             "seconds_per_item": round(seconds / max(1, len(items)), 3),
                             "accuracy": accuracies[batch]}
        print(f"batch {batch:2d}: {timing[str(batch)]['seconds_per_item']:6.3f} s/item  "
              f"accuracy {accuracies[batch]:.3f}")

    reference = probs[args.batches[0]]
    report = {
        "split": args.split,
        "items": len(items),
        "reference_batch": args.batches[0],
        "timing": timing,
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "comparisons": {
            str(batch): compare(reference, probs[batch], items)
            for batch in args.batches[1:]
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for batch, block in report["comparisons"].items():
        print(f"batch {batch:>2s} vs 1: max|dp| {block['max_abs_prob_delta']}  "
              f"argmax {block['argmax_agreement']}/{block['items']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
