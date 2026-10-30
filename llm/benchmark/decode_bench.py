"""Measure how fast the student can decode, before spending a night on evaluation.

`llm/inference.py` decodes with repeated forward passes and a KV-cache fast path
that silently falls back to full recompute when the remote model rejects the
cache inputs. A silent fallback is four orders of magnitude of cost difference on
a long evaluation, so this measures each variant separately and records what
actually happened instead of trusting the code path.

Modes (one per process, so only one model is resident at a time):

``forward_only``  full-sequence forward pass per token, no cache -- the fallback
``kv_cache``      prefill once, then one token per step with ``past_key_values``
``merged``        ``merge_and_unload`` on a *copy* of the adapter, then ``kv_cache``
``batch4/8/16``   left-padded batch decode with ``kv_cache``

The adapter is never modified: ``merged`` loads a second model and merges into
that. Without an adapter on disk the merged row is the base model and says so.

Usage::

    venv_train/Scripts/python -m llm.benchmark.decode_bench --mode kv_cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "outputs" / "benchmark_v2" / "dataset"
DEFAULT_CONFIG = ROOT / "configs" / "config_v2.yaml"
MODES = ("forward_only", "kv_cache", "merged", "batch4", "batch8", "batch16")


def load_prompts(limit: int, split: str = "val") -> List[str]:
    rows = [json.loads(line) for line in
            (DATASET / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("prompt")][:limit]
    return [row["prompt"] for row in rows]


def _dummy(model, size: int, dtype):
    import torch
    return torch.zeros(1, 3, size, size, dtype=dtype, device=model.device)


def _encode(tokenizer, prompt: str):
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=True)


def _stop_ids(tokenizer) -> set:
    ids = {tokenizer.eos_token_id}
    for token in ("<|im_end|>", "</s>", "<|endoftext|>"):
        value = tokenizer.convert_tokens_to_ids(token)
        if isinstance(value, int) and value > 0:
            ids.add(value)
    ids.discard(None)
    return ids


def decode_forward_only(model, tokenizer, prompt: str, max_new: int, dummy_size: int) -> List[int]:
    """A full forward pass per token: the slow path, kept as the baseline."""
    import torch

    enc = _encode(tokenizer, prompt)
    ids = enc["input_ids"].to(model.device)
    mask = enc["attention_mask"].to(model.device)
    image = _dummy(model, dummy_size, next(model.parameters()).dtype)
    flags = torch.zeros((ids.shape[0],), dtype=torch.long, device=model.device)
    stops = _stop_ids(tokenizer)
    out: List[int] = []
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(pixel_values=image, input_ids=ids, attention_mask=mask,
                           image_flags=flags, use_cache=False).logits[:, -1, :]
            nxt = int(torch.argmax(logits, dim=-1))
            out.append(nxt)
            if nxt in stops:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
            mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)], dim=1)
    return out


def _step(model, ids, mask, past, image, flags):
    """One cached step, with the fallback ladder the live path also needs.

    The remote model rejects ``image_flags=None`` (``'NoneType' object has no
    attribute 'squeeze'``), so the first attempt without image inputs is retried
    with the dummy image. That is exactly what ``llm/inference.py`` does; the
    difference is that this module records which form worked instead of leaving it
    in a log line.
    """
    try:
        out = model(pixel_values=None, input_ids=ids, attention_mask=mask,
                    image_flags=None, past_key_values=past, use_cache=True)
        return out, "no_image_inputs"
    except Exception:
        out = model(pixel_values=image, input_ids=ids, attention_mask=mask,
                    image_flags=flags, past_key_values=past, use_cache=True)
        return out, "dummy_image"


def decode_kv(model, tokenizer, prompt: str, max_new: int, dummy_size: int) -> List[int]:
    """Prefill once, then one token per step: what the fast path is supposed to do."""
    import torch

    enc = _encode(tokenizer, prompt)
    ids = enc["input_ids"].to(model.device)
    mask = enc["attention_mask"].to(model.device)
    image = _dummy(model, dummy_size, next(model.parameters()).dtype)
    flags = torch.zeros((ids.shape[0],), dtype=torch.long, device=model.device)
    stops = _stop_ids(tokenizer)
    out: List[int] = []
    with torch.no_grad():
        prefill = model(pixel_values=image, input_ids=ids, attention_mask=mask,
                        image_flags=flags, use_cache=True)
        past = getattr(prefill, "past_key_values", None)
        if past is None:
            raise RuntimeError("model returned no past_key_values; capture is not supported")
        logits = prefill.logits[:, -1, :]
        step_image = None
        for _ in range(max_new):
            nxt = int(torch.argmax(logits, dim=-1))
            out.append(nxt)
            if nxt in stops:
                break
            step_id = torch.tensor([[nxt]], device=ids.device)
            mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)], dim=1)
            step, used = _step(model, step_id, mask, past, image, flags)
            step_image = used
            past = step.past_key_values
            logits = step.logits[:, -1, :]
    decode_kv.last_step_form = step_image
    return out


def decode_batch(model, tokenizer, prompts: Sequence[str], max_new: int,
                 dummy_size: int, batch: int) -> List[List[int]]:
    """Left-padded batch decode. Padding goes on the left so the last column is a
    real token for every row, which is the only way one argmax works for all."""
    import torch

    outputs: List[List[int]] = []
    stops = _stop_ids(tokenizer)
    for start in range(0, len(prompts), batch):
        chunk = prompts[start:start + batch]
        encoded = [_encode(tokenizer, prompt)["input_ids"][0] for prompt in chunk]
        width = max(int(ids.numel()) for ids in encoded)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        inputs, masks = [], []
        for ids in encoded:
            pad = width - int(ids.numel())
            inputs.append(torch.cat([torch.full((pad,), pad_id, dtype=ids.dtype), ids]))
            masks.append(torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(int(ids.numel()), dtype=torch.long)]))
        ids = torch.stack(inputs).to(model.device)
        mask = torch.stack(masks).to(model.device)
        image = _dummy(model, dummy_size, next(model.parameters()).dtype).expand(len(chunk), -1, -1, -1)
        flags = torch.zeros((len(chunk),), dtype=torch.long, device=model.device)
        rows: List[List[int]] = [[] for _ in chunk]
        done = [False] * len(chunk)
        with torch.no_grad():
            prefill = model(pixel_values=image, input_ids=ids, attention_mask=mask,
                            image_flags=flags, use_cache=True)
            past = prefill.past_key_values
            logits = prefill.logits[:, -1, :]
            for _ in range(max_new):
                next_tokens = torch.argmax(logits, dim=-1)
                for row, token in enumerate(next_tokens.tolist()):
                    if done[row]:
                        continue
                    rows[row].append(int(token))
                    if int(token) in stops:
                        done[row] = True
                if all(done):
                    break
                step_ids = next_tokens.view(-1, 1)
                mask = torch.cat([mask, torch.ones((len(chunk), 1), dtype=mask.dtype, device=mask.device)], dim=1)
                step, _used = _step(model, step_ids, mask, past, image, flags)
                past = step.past_key_values
                logits = step.logits[:, -1, :]
        outputs.extend(rows)
    return outputs


def build_model(config_path: Path, adapter: Optional[str], merged: bool):
    import yaml

    from llm.model import load_base_model

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = load_base_model(config, use_4bit=False, eval_mode=True)
    model.eval()
    applied = "base"
    if adapter and Path(adapter).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        applied = "adapter"
        if merged:
            model = model.merge_and_unload()
            applied = "merged"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["llm"]["base_model"],
                                              trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, config, applied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=list(MODES))
    parser.add_argument("--items", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--identity-check", action="store_true",
                        help="compare the first 5 sequences against forward_only")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch

    merged = args.mode == "merged"
    model, tokenizer, config, applied = build_model(Path(args.config), args.adapter, merged)
    dummy_size = int(config["llm"].get("dummy_image_size", 448))
    prompts = load_prompts(args.items)

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    generated: List[List[int]] = []
    error = None
    try:
        if args.mode == "forward_only":
            generated = [decode_forward_only(model, tokenizer, p, args.max_new_tokens, dummy_size)
                         for p in prompts]
        elif args.mode in ("kv_cache", "merged"):
            generated = [decode_kv(model, tokenizer, p, args.max_new_tokens, dummy_size)
                         for p in prompts]
        else:
            batch = int(args.mode.replace("batch", ""))
            generated = decode_batch(model, tokenizer, prompts, args.max_new_tokens,
                                     dummy_size, batch)
    except Exception as exc:  # a mode that cannot run must be reported, not faked
        error = f"{type(exc).__name__}: {exc}"
    seconds = time.time() - started

    tokens = sum(len(row) for row in generated)
    result = {
        "mode": args.mode,
        "model": applied,
        "adapter": args.adapter,
        "items": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "tokens_generated": tokens,
        "seconds": round(seconds, 3),
        "seconds_per_item": round(seconds / max(len(prompts), 1), 3),
        "tokens_per_second": round(tokens / seconds, 3) if seconds > 0 else None,
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "step_form": getattr(decode_kv, "last_step_form", None),
        "error": error,
    }

    if args.identity_check and args.mode == "kv_cache" and not error:
        # Greedy must be identical: a cache changes the cost, never the argmax.
        reference = [decode_forward_only(model, tokenizer, p, args.max_new_tokens, dummy_size)
                     for p in prompts[:5]]
        result["identity_check"] = {
            "items": len(reference),
            "identical": [a == b for a, b in zip(reference, generated[:5])],
        }
        result["identity_check"]["all_identical"] = all(result["identity_check"]["identical"])

    out = Path(args.out) if args.out else ROOT / "outputs" / "benchmark_v2" / "sft" / f"decode_{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"written: {out}")
    return 0 if not error else 1


if __name__ == "__main__":
    raise SystemExit(main())
