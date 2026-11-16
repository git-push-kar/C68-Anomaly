"""Populate the mechanism-validity sheet from condition C, then B, to reach the limit.

tag_role flags carry {kind, claimed, tag, canonical} and no sentence, so each flag's
sentence is located in the composed text by the tag token it names. C carries 19 such
sentences on this checkout, fewer than the 30 the plan wants, so the remaining rows are
drawn from B and labelled by condition rather than silently padded.

The human marks supported / not against llm/benchmark/kb_causes.py. Nothing here decides
validity; the count of `supported` is rung 3 of the selection rule.

Once the sheet is marked, `--from-sheet` converts it into the rung-3 input directly:

    python scripts/mechanism_sheet.py --from-sheet outputs/overnight_2026-11-05/n5_examples_audited.csv

which writes `{cell: {supported, marked, items}}` for `scripts/select_condition.py --mechanism`.
A row counts as marked only when `supported` holds a value; blank is not "no", because an unread
row must not become an unsupported claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.overnight_run import (EXPORT, HYBRID, _a_text, build_orders, compose,
                                   load_jsonl, load_overlay, parse_prose, parse_store,
                                   sentences)
from scripts.pilot_rechecks import tag_role_flags

TRUTHY = {"true", "t", "yes", "y", "1", "supported"}
FALSY = {"false", "f", "no", "n", "0", "unsupported", "not_supported"}
MARKED_FLOOR, ITEMS_FLOOR = 15, 10


def refuse(message: str) -> None:
    """Exit 2 with the reason on stderr, matching select_condition.py's refusals."""
    print(message, file=sys.stderr)
    raise SystemExit(2)


def rung3_from_sheet(path: str) -> dict[str, dict]:
    """Marked sheet -> the rung-3 input: {cell: {supported, marked, items}}."""
    with pathlib.Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        refuse("refusing: %s has no rows" % path)
    if "supported" not in rows[0]:
        refuse("refusing: %s has no supported column" % path)

    counts: dict[str, dict] = {}
    for number, row in enumerate(rows, 2):
        cell = str(row.get("condition") or "").strip()
        if not cell:
            refuse("refusing: row %d names no condition" % number)
        mark = str(row.get("supported") or "").strip().lower()
        if not mark:
            continue
        if mark in TRUTHY:
            value = 1
        elif mark in FALSY:
            value = 0
        else:
            refuse("refusing: row %d has supported=%r; use yes or no" % (number, mark))
        entry = counts.setdefault(cell, {"supported": 0, "marked": 0, "items": set()})
        entry["supported"] += value
        entry["marked"] += 1
        entry["items"].add(str(row.get("item_id") or "").strip())

    if not counts:
        refuse("refusing: no row carries a supported value, so nothing is marked")
    return {cell: {"supported": entry["supported"], "marked": entry["marked"],
                   "items": len(entry["items"])}
            for cell, entry in sorted(counts.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "outputs/overnight_2026-11-05/n5_examples.md"))
    ap.add_argument("--csv", default=str(ROOT / "outputs/overnight_2026-11-05/n5_examples.csv"))
    ap.add_argument("--from-sheet", default=None,
                    help="marked CSV to convert into the rung-3 input instead of building a sheet")
    ap.add_argument("--rung3-out",
                    default=str(ROOT / "outputs/overnight_2026-11-05/rung3_input.json"))
    args = ap.parse_args()

    if args.from_sheet:
        counts = rung3_from_sheet(args.from_sheet)
        out = pathlib.Path(args.rung3_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
        for cell, entry in counts.items():
            eligible = entry["marked"] >= MARKED_FLOOR and entry["items"] >= ITEMS_FLOOR
            print("%-3s supported %2d / marked %2d over %2d items  %s"
                  % (cell, entry["supported"], entry["marked"], entry["items"],
                     "eligible" if eligible
                     else "below floor: needs %d marked from %d items" % (MARKED_FLOOR, ITEMS_FLOOR)))
        print("wrote", out)
        return 0

    built = build_orders()
    rows_b = load_jsonl(EXPORT / "raw_pilot_kb.jsonl")
    probs = {r["item_id"]: r.get("probs") for r in load_jsonl(HYBRID / "crossfit_val_rows.jsonl")}
    overlay = load_overlay(ROOT / "configs/kb_overlay_v1.json")
    a_text = _a_text(built, probs, overlay)
    prompts = {r["item_id"]: r["prompts"]["kb"] for r in built}

    texts: dict[str, dict[str, str | int]] = {}
    for row in rows_b:
        item = row["item_id"]
        store = parse_store(prompts[item])
        parsed = parse_prose(row["text"])
        b_text = parsed["prose"] if parsed["schema_valid"] else str(row["text"])
        c_text, _ = compose(a_text[item], b_text, store)
        texts[item] = {"B": b_text, "C": c_text, "fault": row.get("gold_fault_id")}

    ledger: list[dict] = []
    for condition in ("C", "B"):
        for item, entry in texts.items():
            if len(ledger) >= args.limit:
                break
            body = str(entry[condition])
            parts = sentences(body)
            def sentence_for(flag: dict) -> str:
                """The sentence the flag's window came from, not merely one that holds the tag.

                The flag carries a window of prose, not a sentence index. Choosing by tag alone
                pairs that window with a different clause whenever the tag is mentioned twice,
                which makes the row impossible to judge as printed.
                """
                claimed = str(flag.get("claimed") or "").strip()
                by_claimed = next((s for s in parts if claimed and claimed in s), "")
                by_tag = next((s for s in parts
                               if flag.get("tag") and str(flag["tag"]) in s), "")
                return by_claimed or by_tag

            for flag in tag_role_flags(body):
                if len(ledger) >= args.limit:
                    break
                tag = str(flag.get("tag") or "")
                ledger.append({"item_id": item, "condition": condition,
                               "gold_fault_id": entry["fault"],
                               "kind": flag.get("kind", ""), "tag": tag,
                               "claimed": flag.get("claimed", ""),
                               "canonical": flag.get("canonical", ""),
                               "sentence": sentence_for(flag)})
        if len(ledger) >= args.limit:
            break

    with open(args.csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "condition", "gold_fault_id",
                                                    "kind", "tag", "claimed", "canonical",
                                                    "sentence", "supported", "note"])
        writer.writeheader()
        writer.writerows(ledger)

    by_condition = {c: sum(1 for r in ledger if r["condition"] == c) for c in ("C", "B")}
    lines = ["# Mechanism validity: sentences awaiting the hand check", "",
             "Drawn in item order from `outputs/overnight_2026-11-05/n5_examples.csv` "
             "(%d from C, %d from B; C holds only 19 tag-role sentences on this checkout)."
             % (by_condition["C"], by_condition["B"]), "",
             "For each row: does the sentence's causal claim hold as a path in "
             "`llm/benchmark/kb_causes.py`? Mark `supported` = yes / no. The count of "
             "`yes` is rung 3 of the selection rule; nothing in the pipeline decides it.", "",
             "| # | condition | item | kind | tag | claimed | canonical | sentence | supported | note |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, row in enumerate(ledger, 1):
        sentence = row["sentence"].replace("|", "\\|")
        lines.append("| %d | %s | `%s` | %s | %s | %s | %s | %s |  |  |"
                     % (i, row["condition"], row["item_id"], row["kind"], row["tag"],
                        row["claimed"], row["canonical"], sentence))
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote %d sentences (%s) -> %s" % (len(ledger), by_condition, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
