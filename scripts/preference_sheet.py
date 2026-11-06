"""Blind N-condition preference sheet with Latin-square side assignment.

Input JSONL rows: {"item_id": ..., "condition": ..., "text": ...}.

Example:
    python scripts/preference_sheet.py --texts texts.jsonl --pairs C-A,C-B,M-B \
        --items 120 --seed 43 --out outputs/preference_2026-11-05

Writes `sheet_rater.csv` (pair_id, item_id, text_left, text_right, choice, notes) and
`key.csv` (pair_id, item_id, left, right). Neither sheet names a condition; the side
assignment alternates by (item index + pair index) so no condition sits on one side.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--pairs", default="C-A,C-B,M-B")
    ap.add_argument("--items", type=int, default=120)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="",
                    help="file of item_ids to keep out of the draw (one per line, or a CSV "
                         "with an item_id column): items already read once cannot carry a "
                         "fresh preference claim")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.texts, encoding="utf-8") if line.strip()]
    by_item: dict[str, dict[str, str]] = {}
    for row in rows:
        by_item.setdefault(row["item_id"], {})[row["condition"]] = str(row["text"])

    pairs = [tuple(p.split("-")) for p in args.pairs.split(",") if p.strip()]
    excluded: set[str] = set()
    if args.exclude:
        raw = pathlib.Path(args.exclude).read_text(encoding="utf-8-sig")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if lines and lines[0].lower().replace("\ufeff", "") == "item_id":
            lines = lines[1:]
        excluded = {line.split(",")[0].strip() for line in lines}
    usable = sorted(i for i, v in by_item.items()
                    if i not in excluded and all(a in v and b in v for a, b in pairs))
    if excluded:
        print("excluded %d items from the draw" % len(excluded), file=sys.stderr)
    random.Random(args.seed).shuffle(usable)
    chosen = usable[: args.items]
    if len(chosen) < args.items:
        print("only %d items carry every named condition" % len(chosen), file=sys.stderr)
    if not chosen:
        return 2

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sheet_path, key_path = out / "sheet_rater.csv", out / "key.csv"
    with sheet_path.open("w", newline="", encoding="utf-8") as sheet, \
            key_path.open("w", newline="", encoding="utf-8") as key:
        sw = csv.DictWriter(sheet, fieldnames=["pair_id", "item_id", "text_left",
                                               "text_right", "choice", "notes"])
        kw = csv.DictWriter(key, fieldnames=["pair_id", "item_id", "left", "right"])
        sw.writeheader()
        kw.writeheader()
        for i, item in enumerate(chosen):
            for j, (a, b) in enumerate(pairs):
                flip = (i + j) % 2 == 1
                left, right = (b, a) if flip else (a, b)
                pid = "%s-%s" % (item, "-".join(sorted((a, b))))
                sw.writerow({"pair_id": pid, "item_id": item,
                             "text_left": by_item[item][left],
                             "text_right": by_item[item][right],
                             "choice": "", "notes": ""})
                kw.writerow({"pair_id": pid, "item_id": item, "left": left, "right": right})

    print("wrote %d pairs over %d items -> %s"
          % (len(chosen) * len(pairs), len(chosen), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
