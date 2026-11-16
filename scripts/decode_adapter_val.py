"""Decode the validation prompts with the pristine adapter, so the 0.8330 baseline can be regenerated.

The stored `preds_val_full_400.jsonl` has no preserved producer, and its two candidate sources disagree
(449/497 vs 394/497). This runs the adapter over the exact `prompt` text stored in `val.jsonl`, greedily, and
writes one JSONL row per item with the raw completion and the latency — resumable, because the full pass is
about five hours at the measured 36.5 s/item.

    python scripts/decode_adapter_val.py --limit 4 --out outputs/.../decode_probe.jsonl
    python scripts/decode_adapter_val.py --out outputs/.../preds_adapter_val497.jsonl     # full pass

Greedy decoding (`do_sample=False`, the configured default) is what makes a determinism check meaningful at all;
sampling would make two passes differ by construction.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.dataset import SYSTEM_PROMPT, build_chat_prompt  # noqa: E402
from llm.inference import REPORT_SYSTEM_PROMPT, RCAInference  # noqa: E402
from utils import load_config  # noqa: E402

DEFAULT_VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
DEFAULT_OUT = ROOT / "outputs/benchmark_v2/eval/preds_adapter_val497.jsonl"


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=pathlib.Path, default=DEFAULT_VAL)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="0 = every item")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768,
                        help="matches generate_report's budget; 512 truncated a third of the replies mid-JSON")
    parser.add_argument("--variant", default="menu", help="menu | openset | all (default: menu, the 503-item baseline)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.val)
    if args.variant != "all":
        rows = [row for row in rows if row.get("variant") == args.variant]
    rows = rows[args.offset:(args.offset + args.limit) if args.limit else None]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict] = {}
    if args.out.exists():
        done = {row["item_id"]: row for row in read_jsonl(args.out)}
    pending = [row for row in rows if row["item_id"] not in done]
    print(json.dumps({"items": len(rows), "already": len(done), "pending": len(pending)}), flush=True)
    if not pending:
        return 0

    config = load_config(args.config) if args.config else load_config()
    inference = RCAInference.from_adapter(config=config)
    print("model loaded", flush=True)

    with args.out.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(pending, start=1):
            started = time.perf_counter()
            # The adapter is trained and served through the chat template with the report system prompt;
            # feeding the bare question produces a degenerate, truncated answer. Same construction as
            # ``RCAInference.generate_report``.
            prompt = build_chat_prompt(REPORT_SYSTEM_PROMPT + " " + SYSTEM_PROMPT,
                                       [("user", row["prompt"])])
            raw = inference._generate(prompt, max_new_tokens=args.max_new_tokens,
                                      do_sample=False, temperature=0.0)
            elapsed = time.perf_counter() - started
            record = {"item_id": row["item_id"], "gold": row.get("fault"),
                      "variant": row.get("variant"), "raw": raw, "seconds": round(elapsed, 2)}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps({"done": index, "of": len(pending), "item_id": row["item_id"],
                              "seconds": round(elapsed, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
