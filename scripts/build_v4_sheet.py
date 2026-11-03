"""Build the populated v4 A/B sheet (PROTOCOL 25.6) from the pilot's kb-order prose.

Both sides of every row are prose only: the pilot adapter's own kb-order generation and fallback
v3.1 on the same supplied classification. The rater never sees family, diagnosis, verdict, source or
the assignment; those live in the key, where the balance is exactly 50/50 inside each stratum.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.fallback_v31 import fallback_v31  # noqa: E402
from scripts.fallback_v3 import load_overlay  # noqa: E402
from scripts.fallback_v2 import load_jsonl, parse_store  # noqa: E402
from scripts.make_ab_sheet_v4 import STRATA, build_sheet, leak_report  # noqa: E402
from scripts.prose_pilot_prompts import EXPORT, build_orders  # noqa: E402

FIT = ROOT / "outputs/analysis/phase3a/fit"
DATASET = ROOT / "outputs/benchmark_v2/dataset"
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
SCORES = ["correct_A", "grounded_A", "consistent_A", "useful_A",
          "correct_B", "grounded_B", "consistent_B", "useful_B", "preference"]
SHEET_COLUMNS = ["item_id", "text_A", "text_B", *SCORES, "notes"]
KEY_COLUMNS = ["item_id", "family", "diagnosis", "verdict", "A", "B", "seed"]


def render(answer: dict) -> str:
    reasoning = str((answer or {}).get("physical_reasoning") or "").strip()
    narrative = str((answer or {}).get("narrative") or "").strip()
    return "Reasoning: " + reasoning + chr(10) + chr(10) + "Narrative: " + narrative


def stratum_of(row: dict) -> str:
    if str(row["gold_fault_id"]) == "UNCLASSIFIED_OPEN_SET":
        return "abstention_menu" if str(row["variant"]) == "menu" else "abstention_removed"
    return "normal"


def choose(pool, count: int, rng) -> list:
    ordered = sorted(pool, key=lambda row: row["item_id"])
    return rng.sample(ordered, count) if len(ordered) > count else ordered


FAMILY = {"abstention_menu": "open_set", "abstention_removed": "removed", "normal": "normal"}


def main() -> int:
    import argparse
    import ast as _ast

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(FIT))
    args = parser.parse_args()
    rng = random.Random(args.seed)

    raw = {row["item_id"]: row["text"]
           for row in load_jsonl(EXPORT / "raw_pilot_kb.jsonl")}
    orders = build_orders()
    val = {row["item_id"]: row for row in load_jsonl(DATASET / "val.jsonl")}
    probs = {row["item_id"]: row.get("probs")
             for row in load_jsonl(HYBRID / "crossfit_val_rows.jsonl")}
    overlay = load_overlay(ROOT / "configs/kb_overlay_v1.json")

    candidates = []
    for stratum, count in STRATA.items():
        pool = [row for row in orders if stratum_of(row) == stratum and row["item_id"] in raw]
        chosen = choose(pool, count, rng)
        if len(chosen) < count:
            raise SystemExit("not enough " + stratum + " items: " + str(len(chosen)))
        for row in chosen:
            item = val.get(row["item_id"]) or {}
            store = parse_store(str(item.get("prompt") or ""))
            raw_probs = probs.get(row["item_id"]) or {}
            if isinstance(raw_probs, str):
                raw_probs = _ast.literal_eval(raw_probs)
            answer = fallback_v31(str(row["gold_fault_id"]), str(row["gold_verdict"]),
                                  row["menu_ids"], store, raw_probs, overlay)
            candidates.append({"item_id": row["item_id"], "stratum": stratum,
                               "family": FAMILY[stratum],
                               "diagnosis": row["gold_fault_id"],
                               "verdict": row["gold_verdict"],
                               "menu_present": stratum != "abstention_removed",
                               "text_adapter": raw[row["item_id"]],
                               "text_fallback": render(answer)})

    rows, key = build_sheet(candidates, seed=args.seed)
    report = leak_report(rows, key)
    if not report["passed"]:
        raise SystemExit("leak test failed: " + json.dumps(report))
    meta_by_item = {entry["item_id"]: entry for entry in candidates}
    side_by_item = {entry["item_id"]: entry for entry in key}

    sheet_rows = [{column: ("" if column in SCORES or column == "notes" else row.get(column, ""))
                   for column in SHEET_COLUMNS} for row in rows]
    key_rows = [{"item_id": entry["item_id"],
                 "family": meta_by_item[entry["candidate"]]["family"],
                 "diagnosis": meta_by_item[entry["candidate"]]["diagnosis"],
                 "verdict": meta_by_item[entry["candidate"]]["verdict"],
                 "A": side_by_item[entry["item_id"]]["side_A_source"],
                 "B": side_by_item[entry["item_id"]]["side_B_source"],
                 "seed": args.seed} for entry in key]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "audit_ab_100_v4.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(sheet_rows)
    with (out / "audit_ab_100_v4_key.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEY_COLUMNS)
        writer.writeheader()
        writer.writerows(key_rows)
    meta = {
        "sheet": "audit_ab_100_v4.csv", "key": "audit_ab_100_v4_key.csv",
        "rows": len(sheet_rows), "distinct_items": len({row["item_id"] for row in sheet_rows}),
        "seed": args.seed,
        "strata": dict(Counter(entry["stratum"] for entry in candidates)),
        "item_map": [{"item_id": entry["item_id"], "corpus_item_id": entry["candidate"]}
                     for entry in key],
        "verdicts": dict(Counter(row["verdict"] for row in key_rows)),
        "adapter_on_A": sum(1 for row in key_rows if row["A"] == "adapter"),
        "adapter_on_A_by_stratum": report["per_stratum"],
        "columns_visible": SHEET_COLUMNS,
        "scoring_cells_blank": sum(1 for row in sheet_rows for column in SCORES + ["notes"]
                                   if row[column] == ""),
        "scoring_cells_total": len(sheet_rows) * (len(SCORES) + 1),
        "leak_report": report,
        "rows_where_the_two_texts_are_identical": sum(1 for row in rows
                                                      if row["text_A"] == row["text_B"]),
        "comparator": "fallback_v31 (scripts/fallback_v31.py)",
        "adapter_source": "raw_pilot_kb.jsonl (pilot, kb order)",
        "checks": {"no_family_diagnosis_verdict_source_columns":
                   not ({"family", "diagnosis", "verdict", "source"} & set(SHEET_COLUMNS)),
                   "key_carries_assignment": set(KEY_COLUMNS) == {"item_id", "family", "diagnosis",
                                                                  "verdict", "A", "B", "seed"},
                   "exactly_50_50_per_stratum":
                   all(info["adapter_on_A"] * 2 == info["rows"]
                       for info in report["per_stratum"].values())}}
    (out / "audit_ab_100_v4_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
