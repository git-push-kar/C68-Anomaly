"""Test-split prose rows: the 3,168 held-out items with their three rebuilt prompts.

`prose_pilot_prompts.build_orders` is hard-wired to `val.jsonl` and the pilot export, so the test split
had no rows and the one-shot pass could not be pointed at it at all. This script mirrors that builder
over `test.jsonl`, whose rows carry the same schema (prompt, target, variant, menu_ids).

Nothing is generated here. The output is the input for
`python scripts/prose_pilot_run.py --rows <this file> --orders kb --limit 0 --batch <n>`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prose_pilot_prompts import (DATASET, MENU_LINE, ORDERS, _order_only, assemble,
                                         load_jsonl, shuffled, supplied_of, top2_of)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(ROOT / "outputs/benchmark_v2/eval/prose_test/rows_test.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = load_jsonl(DATASET / ("%s.jsonl" % args.split))
    if args.limit:
        items = items[:args.limit]
    built, no_menu = [], 0
    for item in items:
        supplied = supplied_of(item)
        raw = str(item.get("prompt") or "")
        _field, _marker, rest = raw.partition("CANDIDATE MENU")
        lines = MENU_LINE.findall(rest)
        if not lines:
            no_menu += 1
            continue
        prompts = {}
        for name, seed in ORDERS:
            order = lines if seed is None else shuffled(lines, seed, str(item["item_id"]))
            prompts[name] = assemble(raw, supplied, top2_of(item.get("probs")), order)
        built.append({"item_id": item["item_id"], "gold_fault_id": supplied["fault_id"],
                      "gold_verdict": supplied["verdict"], "variant": item.get("variant"),
                      "menu_ids": item.get("menu_ids") or [], "menu_lines": len(lines),
                      "prompts": prompts, "order_only": _order_only(prompts)})

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in built:
            handle.write(json.dumps(row) + "\n")
    order_only = sum(1 for row in built if row["order_only"])
    print(json.dumps({"split": args.split, "rows": len(built), "skipped_no_menu": no_menu,
                      "order_only_rows": order_only,
                      "gold_abstain": sum(1 for row in built
                                          if str(row["gold_fault_id"]) == "UNCLASSIFIED_OPEN_SET"),
                      "out": str(out)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
