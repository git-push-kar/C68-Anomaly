"""Generate the pilot's validation answers raw (PROTOCOL 25.4). Read only.

Generation goes through model.language_model (no dummy-image ViT pass, PROTOCOL 15.5) in left-padded
batches, which is what makes 1,739 generations affordable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_protocolA import adapter_root, bare, load_base  # noqa: E402
from llm.benchmark.prompt_v2 import SYSTEM_PROMPT_V2  # noqa: E402
from llm.dataset import build_chat_prompt  # noqa: E402
from llm.followup.teacher import SYSTEM_PROMPT_FOLLOWUP  # noqa: E402
from scripts.prose_pilot_prompts import build_orders, load_jsonl  # noqa: E402

EXPORT = ROOT / "outputs/benchmark_v2/eval/prose_pilot"
FOLLOWUPS = EXPORT / "prose_eval_followups_179.jsonl"


def decode_batch_direct(model, tokenizer, prompts, max_new: int, batch: int):
    """Left-padded greedy decode through model.language_model; token ids per prompt."""
    import torch

    language_model = getattr(model, "language_model", None)
    if language_model is None:
        from llm.benchmark.decode_bench import decode_batch
        return decode_batch(model, tokenizer, prompts, max_new, 448, batch)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    stops = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    stops = {int(token) for token in stops if token is not None and int(token) >= 0}
    results = []
    for start in range(0, len(prompts), batch):
        chunk = list(prompts[start:start + batch])
        encoded = [tokenizer(text, add_special_tokens=True)["input_ids"] for text in chunk]
        width = max(len(row) for row in encoded)
        ids = torch.tensor([[pad_id] * (width - len(row)) + list(row) for row in encoded],
                           device=model.device)
        mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in encoded],
                            device=model.device)
        rows = [[] for _ in chunk]
        done = [False] * len(chunk)
        with torch.no_grad():
            embeds = language_model.get_input_embeddings()(ids)
            position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
            out = language_model(inputs_embeds=embeds, attention_mask=mask,
                                 position_ids=position_ids, use_cache=True, return_dict=True)
            past, logits = out.past_key_values, out.logits[:, -1, :]
            for _ in range(max_new):
                nxt = torch.argmax(logits, dim=-1)
                for row, token in enumerate(nxt.tolist()):
                    if done[row]:
                        continue
                    rows[row].append(int(token))
                    if int(token) in stops:
                        done[row] = True
                if all(done):
                    break
                mask = torch.cat([mask, torch.ones((len(chunk), 1), dtype=mask.dtype,
                                                   device=mask.device)], dim=1)
                position_ids = (mask.cumsum(-1) - 1).clamp(min=0)[:, -1:]
                step = language_model(input_ids=nxt.view(-1, 1), attention_mask=mask,
                                      position_ids=position_ids, past_key_values=past,
                                      use_cache=True, return_dict=True)
                past, logits = step.past_key_values, step.logits[:, -1, :]
        results.extend(rows)
    return results


def batch_decode(model, tokenizer, prompts, cap: int, batch: int):
    started = time.time()
    produced = decode_batch_direct(model, tokenizer, prompts, cap, batch)
    elapsed = round(time.time() - started, 2)
    return [(tokenizer.decode(ids, skip_special_tokens=True), len(ids)) for ids in produced], elapsed


def reports(model, tokenizer, rows, order, args, handle):
    per_item = []
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        prompts = [build_chat_prompt(SYSTEM_PROMPT_V2, [("user", row["prompts"][order])])
                   for row in chunk]
        decoded, wall = batch_decode(model, tokenizer, prompts, args.cap, args.batch)
        per_item.append(round(wall / len(chunk), 2))
        for row, (text, tokens) in zip(chunk, decoded):
            handle.write(json.dumps({"item_id": row["item_id"], "order": order,
                                     "gold_fault_id": row["gold_fault_id"],
                                     "gold_verdict": row["gold_verdict"],
                                     "variant": row["variant"], "tokens": tokens,
                                     "text": text}) + chr(10))
        handle.flush()
        done = start + len(chunk)
        if done % (args.batch * 10) == 0 or done == len(rows):
            mean = sum(per_item) / len(per_item)
            print(json.dumps({"order": order, "done": done, "of": len(rows),
                              "seconds_per_item": round(mean, 2),
                              "eta_minutes": round(mean * (len(rows) - done) / 60, 1)}), flush=True)
    return per_item


def followup_turns(model, tokenizer, turns, args, handle):
    per_item = []
    for start in range(0, len(turns), args.batch):
        chunk = turns[start:start + args.batch]
        prompts = [build_chat_prompt(SYSTEM_PROMPT_FOLLOWUP, [("user", str(turn["question"]))])
                   for turn in chunk]
        decoded, wall = batch_decode(model, tokenizer, prompts, args.cap, args.batch)
        per_item.append(round(wall / len(chunk), 2))
        for offset, (turn, (text, tokens)) in enumerate(zip(chunk, decoded)):
            handle.write(json.dumps({"index": turn.get("index", start + offset),
                                     "dialogue_id": turn["dialogue_id"], "kind": turn["kind"],
                                     "in_scope": turn.get("in_scope"), "tokens": tokens,
                                     "text": text}) + chr(10))
        handle.flush()
        if (start + len(chunk)) % (args.batch * 4) == 0 or start + len(chunk) == len(turns):
            print(json.dumps({"followups": start + len(chunk), "of": len(turns)}), flush=True)
    return per_item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config_prose_pilot_seed42.yaml")
    parser.add_argument("--adapter", default="outputs/tep_rca_v2_prose_pilot_seed42")
    parser.add_argument("--out", default=str(EXPORT))
    parser.add_argument("--mode", default="both", choices=("reports", "followups", "both"))
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--orders", default="kb,shuffled-1,shuffled-2")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--followup-limit", dest="followup_limit", type=int, default=0)
    parser.add_argument("--cap", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    from peft import PeftModel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model, tokenizer, config, _applied = load_base(ROOT / args.config)
    path = adapter_root(ROOT / args.adapter)
    wrapped = PeftModel.from_pretrained(model, str(path), adapter_name=args.tag)
    wrapped.set_adapter(args.tag)
    inner = bare(wrapped)
    print(json.dumps({"adapter": str(path),
                      "load_seconds": round(time.time() - started, 1)}), flush=True)
    written, per_item = {}, []
    if args.mode in ("reports", "both"):
        rows = build_orders()
        if args.limit:
            rows = rows[:args.limit]
        for order in [name for name in args.orders.split(",") if name]:
            target = out / ("raw_" + args.tag + "_" + order + ".jsonl")
            with target.open("w", encoding="utf-8") as handle:
                per_item += reports(inner, tokenizer, rows, order, args, handle)
            written[order] = {"path": str(target), "rows": len(rows)}
    if args.mode in ("followups", "both"):
        turns = load_jsonl(FOLLOWUPS)
        if args.followup_limit:
            turns = turns[:args.followup_limit]
        target = out / ("raw_" + args.tag + "_followups.jsonl")
        with target.open("w", encoding="utf-8") as handle:
            per_item += followup_turns(inner, tokenizer, turns, args, handle)
        written["followups"] = {"path": str(target), "rows": len(turns)}
    summary = {"adapter": str(path), "tag": args.tag, "cap": args.cap, "batch": args.batch,
               "written": written,
               "mean_seconds_per_item": round(sum(per_item) / max(len(per_item), 1), 2),
               "total_seconds": round(time.time() - started, 1)}
    (out / ("raw_" + args.tag + "_run.json")).write_text(json.dumps(summary, indent=2),
                                                        encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
