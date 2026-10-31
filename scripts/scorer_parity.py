"""Do the label scorer and the decoder measure the same string?

Both instruments read the answer's first field. The scorer appends a candidate to
``{"fault_id": "`` and sums its token log-probabilities; the decoder generates the
value and sums the log-probabilities along the path it walked. If those two sums are
not the same number for the same string, then the "agreement" reported between them
in ``run1_instrument_report.md`` compared two different events, and the confidence a
reader is given depends on which instrument produced it.

There are exactly two ways they can differ, and the two subcommands test them
separately:

``tokens``  tokenisation. A BPE segment is not compositional: ``tokenizer(label)``,
            ``tokenizer(prefix + label)`` and the tokens that follow the *real*
            prompt can all differ, and a scorer that assumes otherwise reads the
            wrong logits. Checked by tokenising the whole continuation for every
            candidate on a sample of prompts and comparing the tail.

``parity``  the terminator. The scorer and the decoder must score the same string,
            which means the same terminator token. This is where the defect was:
            the scorer ended at the bare quote (id 1) while the model writes ``",``
            (id 497) as one token, so the scorer measured a continuation the model
            cannot produce. The first run of this script measured that gap at
            22.80 nats mean absolute difference on all 300 items; after the fix it
            should be zero up to bf16 noise.

Usage::

    python scripts/scorer_parity.py tokens --prompts 100
    venv_train/Scripts/python.exe scripts/scorer_parity.py parity --step 500
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.benchmark.label_scorer import (  # noqa: E402
    ANSWER_PREFIX,
    CLOSING_QUOTE,
    FIELD_TERMINATOR,
    candidates,
    prompt_for,
    suffix_ids,
)

VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
SUBSET = ROOT / "outputs/benchmark_v2/eval/val_subsample_ids.json"
OUT = ROOT / "outputs/analysis/phase3a/scorer_parity"
TOKENIZER_PATH = "C:/Users/Admin/Desktop/LY_C68/deepfake_project/models/InternVL3-2B"


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def subsample(items: Sequence[Dict]) -> List[Dict]:
    wanted = json.loads(SUBSET.read_text(encoding="utf-8"))
    by_id = {item["item_id"]: item for item in items}
    return [by_id[item_id] for item_id in wanted if item_id in by_id]


# --------------------------------------------------------------------------- #
# tokenisation audit
# --------------------------------------------------------------------------- #
def audit_tokens(prompts: int, seed: int = 42) -> Dict:
    """Where the note is added, relative to the prompt's own tokens.

    Three tokenisations per candidate, which is the whole point: *alone* is what a
    naive scorer would use, *with the prefix* is what ``label_scorer`` uses, and
    *in context* is what the model is actually asked to continue.
    """
    import random
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    items = [item for item in load_jsonl(VAL) if item.get("prompt")]
    rng = random.Random(seed)
    sample = rng.sample(items, min(prompts, len(items)))
    labels = candidates()

    prefix = tokenizer(ANSWER_PREFIX, add_special_tokens=False)["input_ids"]
    quote_id = tokenizer(CLOSING_QUOTE, add_special_tokens=False)["input_ids"]
    terminator = tokenizer(FIELD_TERMINATOR, add_special_tokens=False)["input_ids"]
    rows = []
    mismatches = []
    for item in sample:
        context = prompt_for(item["prompt"]) + ANSWER_PREFIX
        ctx_ids = tokenizer(context, add_special_tokens=True)["input_ids"]
        tail_ok = ctx_ids[-len(prefix):] == prefix
        for label in labels:
            whole = tokenizer(context + label + FIELD_TERMINATOR,
                              add_special_tokens=True)["input_ids"]
            # The ids the model would be asked to produce, from the real context.
            common = 0
            limit = min(len(ctx_ids), len(whole))
            while common < limit and ctx_ids[common] == whole[common]:
                common += 1
            in_context = whole[common:]
            from_prefix = suffix_ids(tokenizer, label)
            alone = tokenizer(label, add_special_tokens=False)["input_ids"]
            row = {"item_id": item["item_id"], "label": label,
                   "alone": alone, "from_prefix": from_prefix,
                   "in_context": in_context, "common_prefix": common,
                   "context_length": len(ctx_ids), "prompt_tail_is_prefix": tail_ok,
                   "surface": [tokenizer.decode([token]) for token in in_context]}
            rows.append(row)
            if in_context != from_prefix or not tail_ok:
                mismatches.append(row)

    by_label: Dict[str, Dict] = {}
    for label in labels:
        subset = [row for row in rows if row["label"] == label]
        by_label[label] = {
            "suffix_ids": subset[0]["from_prefix"],
            "surface": subset[0]["surface"],
            "n_tokens": len(subset[0]["from_prefix"]),
            "alone_ids": subset[0]["alone"],
            "ends_with_terminator": subset[0]["from_prefix"][-1:] == terminator,
            "ends_with_bare_quote": subset[0]["from_prefix"][-1:] == quote_id,
            "in_context_matches_prefix_tokenisation": all(
                row["in_context"] == row["from_prefix"] for row in subset),
        }

    lengths = sorted({entry["n_tokens"] for entry in by_label.values()})
    return {
        "prompts_checked": len(sample),
        "candidates": len(labels),
        "pairs_checked": len(rows),
        "closing_quote_id": quote_id,
        "terminator_id": terminator,
        "terminator_surface": tokenizer.decode(terminator),
        "suffix_token_counts": lengths,
        "mismatched_pairs": len(mismatches),
        "mismatches": mismatches[:10],
        "per_label": by_label,
        "finding": (
            "every candidate ends with the same in-context terminator "
            f"{tokenizer.decode(terminator)!r} (id {terminator[0]}), not the bare "
            f"quote (id {quote_id[0]}); every candidate's tokens after the real "
            "prompt equal its tokens after the prefix alone"
            if not mismatches else
            f"{len(mismatches)} candidate/prompt pairs tokenise differently in context"
        ),
    }


# --------------------------------------------------------------------------- #
# probability parity
# --------------------------------------------------------------------------- #
def audit_parity(step: str, batch: int, decode_batch: int, cap: int,
                 limit: Optional[int] = None) -> Dict:
    """Decoded chain vs scorer sum, for the string each item actually produced."""
    import time

    from instrument_check import decode_first_field, score_detail
    from eval_protocolA import adapter_root, load_base

    from peft import PeftModel

    model, tokenizer, config, _applied = load_base(
        ROOT / "configs/config_v2_protocolA_seed42.yaml")
    dummy = int(config["llm"].get("dummy_image_size", 448))
    path = adapter_root(ROOT / f"outputs/tep_rca_v2_protocolA_seed42/checkpoint-{step}")
    wrapped = PeftModel.from_pretrained(model, str(path), adapter_name=f"p{step}")
    wrapped.set_adapter(f"p{step}")
    from eval_protocolA import bare

    base_model = bare(wrapped)

    items = subsample([item for item in load_jsonl(VAL) if not item.get("unobservable")])
    if limit:
        items = items[:limit]
    questions = [item["prompt"] for item in items]

    started = time.time()
    detail = score_detail(base_model, tokenizer, questions, dummy, batch)
    decoded = decode_first_field(base_model, tokenizer, questions, dummy, decode_batch, cap)
    seconds = time.time() - started

    rows = []
    for item, scored, walked in zip(items, detail, decoded):
        label = walked["label"]
        record = {"item_id": item["item_id"], "gold": (item.get("target") or {}).get("fault_id"),
                  "decoded": walked["text"], "decoded_label": label,
                  "decode_chain_logp": walked["chain_logp"]}
        if label and label in scored["labels"]:
            path_row = scored["labels"][label]
            logps = path_row["logps"]
            record.update({
                "scorer_joint": path_row["joint"],
                "scorer_sum_label_only": sum(logps[:-1]),
                "terminator_logp": logps[-1] if len(logps) > 1 else None,
                "n_suffix_tokens": len(logps),
                "surfaces": path_row["tokens"],
                "delta_full": walked["chain_logp"] - path_row["joint"],
                "delta_label_only": walked["chain_logp"] - sum(logps[:-1]),
            })
        rows.append(record)

    scored_rows = [row for row in rows if "delta_full" in row]
    deltas_full = [abs(row["delta_full"]) for row in scored_rows]
    deltas_label = [abs(row["delta_label_only"]) for row in scored_rows]
    terminator = [row["terminator_logp"] for row in scored_rows
                  if row["terminator_logp"] is not None]
    surfaces = {tuple(row["surfaces"][-1:]) for row in scored_rows}
    return {
        "checkpoint": step,
        "items": len(rows),
        "scored": len(scored_rows),
        "seconds": round(seconds, 1),
        "terminator_surfaces": [list(surface) for surface in sorted(surfaces)][:3],
        "delta_full": {
            "mean_abs": round(statistics.mean(deltas_full), 6) if deltas_full else None,
            "max_abs": round(max(deltas_full), 6) if deltas_full else None,
            "over_0.01_nats": sum(1 for value in deltas_full if value > 0.01),
            "over_1e-3_nats": sum(1 for value in deltas_full if value > 1e-3),
        },
        "delta_label_only": {
            "mean_abs": round(statistics.mean(deltas_label), 6) if deltas_label else None,
            "max_abs": round(max(deltas_label), 6) if deltas_label else None,
            "over_0.01_nats": sum(1 for value in deltas_label if value > 0.01),
        },
        "terminator_logp": {
            "mean": round(statistics.mean(terminator), 4) if terminator else None,
            "min": round(min(terminator), 4) if terminator else None,
            "max": round(max(terminator), 4) if terminator else None,
        },
        "rows": rows,
    }


def audit_residual(items_count: int, step: str,
                   item_ids: Optional[Sequence[str]] = None) -> Dict:
    """Where the remaining 1e-2 nats come from: the batch, or the code path?

    With the same terminator on both sides the two sums are the same quantity, so any
    difference is numerical. The two candidates are the batch (left padding changes
    the mask width and so the bf16 accumulation order) and the path (the scorer's own
    loop against the decoder's). Measuring both batchings of both instruments
    separates them: if batch 1 agrees across paths but batch 16 does not, it is
    batching.
    """
    from peft import PeftModel

    from eval_protocolA import adapter_root, bare, load_base
    from instrument_check import decode_first_field, score_detail

    model, tokenizer, config, _applied = load_base(
        ROOT / "configs/config_v2_protocolA_seed42.yaml")
    dummy = int(config["llm"].get("dummy_image_size", 448))
    path = adapter_root(ROOT / f"outputs/tep_rca_v2_protocolA_seed42/checkpoint-{step}")
    wrapped = PeftModel.from_pretrained(model, str(path), adapter_name="res")
    wrapped.set_adapter("res")
    base_model = bare(wrapped)

    pool = subsample([item for item in load_jsonl(VAL) if not item.get("unobservable")])
    if item_ids:
        wanted = set(item_ids)
        items = [item for item in pool if item["item_id"] in wanted]
    else:
        items = pool[:items_count]
    questions = [item["prompt"] for item in items]
    decoded_1 = decode_first_field(base_model, tokenizer, questions, dummy, 1, 16)
    decoded_8 = decode_first_field(base_model, tokenizer, questions, dummy, 8, 16)
    detail_1 = score_detail(base_model, tokenizer, questions, dummy, 1)
    detail_16 = score_detail(base_model, tokenizer, questions, dummy, 16)

    def joint(detail, index: int, label: str) -> Optional[float]:
        row = detail[index]["labels"].get(label or "")
        return row["joint"] if row else None

    pairs = {"scorer_b1_vs_decode_b1": [], "scorer_b16_vs_decode_b8": [],
             "scorer_b1_vs_scorer_b16": [], "decode_b1_vs_decode_b8": []}
    for index, walked in enumerate(decoded_1):
        label = walked["label"] or walked["raw"]
        first = joint(detail_1, index, label)
        sixteenth = joint(detail_16, index, label)
        if first is None:
            continue
        pairs["scorer_b1_vs_decode_b1"].append(abs(first - walked["chain_logp"]))
        pairs["scorer_b16_vs_decode_b8"].append(
            abs(sixteenth - decoded_8[index]["chain_logp"]))
        pairs["scorer_b1_vs_scorer_b16"].append(abs(first - sixteenth))
        pairs["decode_b1_vs_decode_b8"].append(
            abs(walked["chain_logp"] - decoded_8[index]["chain_logp"]))
    return {"items": len(items), "checkpoint": step,
            "comparisons": {name: {"mean_abs": round(statistics.mean(values), 6)
                                   if values else None,
                                   "max_abs": round(max(values), 6) if values else None,
                                   "over_1e-3": sum(1 for value in values if value > 1e-3),
                                   "n": len(values)}
                            for name, values in pairs.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    tokens = sub.add_parser("tokens")
    tokens.add_argument("--prompts", type=int, default=100)
    tokens.add_argument("--seed", type=int, default=42)

    parity = sub.add_parser("parity")
    parity.add_argument("--step", default="500")
    parity.add_argument("--batch", type=int, default=16)
    parity.add_argument("--decode-batch", dest="decode_batch", type=int, default=8)
    parity.add_argument("--cap", type=int, default=16)
    parity.add_argument("--limit", type=int, default=None)

    residual = sub.add_parser("residual")
    residual.add_argument("--items", type=int, default=20)
    residual.add_argument("--item-id", dest="item_id", action="append")
    residual.add_argument("--step", default="500")

    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "tokens":
        result = audit_tokens(args.prompts, args.seed)
        (OUT / "tokens.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({key: result[key] for key in
                          ("prompts_checked", "pairs_checked", "mismatched_pairs",
                           "closing_quote_id", "suffix_token_counts", "finding")}, indent=2))
        print(json.dumps({label: {"n_tokens": entry["n_tokens"], "surface": entry["surface"],
                                  "alone_ids": entry["alone_ids"]}
                          for label, entry in list(result["per_label"].items())[:3]}, indent=2))
    elif args.command == "residual":
        result = audit_residual(args.items, args.step, args.item_id)
        (OUT / f"residual_step{args.step}.json").write_text(json.dumps(result, indent=2),
                                                            encoding="utf-8")
        print(json.dumps(result, indent=2))
    else:
        result = audit_parity(args.step, args.batch, args.decode_batch, args.cap, args.limit)
        (OUT / f"parity_step{args.step}.json").write_text(json.dumps(result, indent=2),
                                                          encoding="utf-8")
        print(json.dumps({key: result[key] for key in
                          ("checkpoint", "items", "scored", "terminator_surfaces",
                           "delta_full", "delta_label_only", "terminator_logp")}, indent=2))
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
