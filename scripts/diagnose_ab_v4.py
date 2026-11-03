"""Diagnose every row of the invalid v4 sheet (PROTOCOL 26.4), and re-check the 25 evaluation.

Writes a per-row CSV naming the item each text was generated for, whether the two texts describe
the same event, whether each text's style matches the key's label, and the stratum the row really
belongs to. Also measures whether the section 25 gate numbers could have been joined positionally.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.ab_sheet_checks import FIT, Sources, audit_sheet, read_rows, top_sensor  # noqa: E402
from scripts.build_prose_dataset import load_jsonl  # noqa: E402
from scripts.fallback_v2 import parse_store  # noqa: E402
from scripts.prose_pilot_prompts import EXPORT  # noqa: E402

COLUMNS = ["item_id", "key_family", "key_diagnosis", "key_verdict", "key_A", "key_B",
           "text_A_item", "text_A_source", "text_A_style", "text_B_item", "text_B_source",
           "text_B_style", "same_item", "tag_A", "z_A", "tag_B", "z_B", "same_event",
           "style_matches_key", "stratum_of_row", "defect"]


def evaluation_anchor_rate() -> dict:
    """Do the stored generations belong to the item they are filed under?

    The run script builds each prompt from the row it also writes out, so the join is by item_id.
    As an external check, count how often a row's prose names the top sensor of *its own* prompt:
    a positional misalignment would drive that rate towards the rate for a random other item.
    """
    val = {row["item_id"]: row for row in load_jsonl(ROOT / "outputs/benchmark_v2/dataset/val.jsonl")}
    rows = list(load_jsonl(EXPORT / "raw_pilot_kb.jsonl"))
    own = other = 0
    for index, row in enumerate(rows):
        item = val.get(row["item_id"]) or {}
        tag, z = top_sensor(str(item.get("prompt") or ""))
        if not tag:
            continue
        text = row["text"]
        own += int(tag in text and f"{z:.2f}" in text)
        neighbour = rows[(index + 1) % len(rows)]
        other_item = val.get(neighbour["item_id"]) or {}
        other_tag, other_z = top_sensor(str(other_item.get("prompt") or ""))
        other += int(other_tag in text and f"{other_z:.2f}" in text)
    return {"rows": len(rows), "own_prompt_top_sensor_and_z": own,
            "next_rows_top_sensor_and_z": other,
            "own_rate": round(own / max(len(rows), 1), 4),
            "other_rate": round(other / max(len(rows), 1), 4)}


def main() -> int:
    sources = Sources()
    report = audit_sheet(FIT / "audit_ab_100_v4.csv", FIT / "audit_ab_100_v4_key.csv", sources)
    out = FIT / "ab_v4_diagnosis.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in report["per_row"]:
            record = {column: row.get(column, "") for column in COLUMNS}
            record["defect"] = "; ".join(row.get("defect") or [])
            writer.writerow(record)
    stratum = {}
    for row in read_rows(FIT / "audit_ab_100_v4_key.csv"):
        stratum[row["verdict"]] = stratum.get(row["verdict"], 0) + 1
    summary = {"sheet": str(out), "rows": len(report["per_row"]),
               "rows_pairing_different_items": report["rows_with_different_items"],
               "rows_with_style_mismatch": report["rows_with_style_mismatch"],
               "classified_rows_in_non_classified_stratum": report["classified_rows"],
               "key_verdict_counts": stratum,
               "root_cause": {
                   "different_events": "scripts/make_ab_sheet_v4.py:51 puts "
                                       "b['text_adapter'] on row_a beside a['text_adapter'], and "
                                       "line 55 puts a['text_fallback'] on row_b beside "
                                       "b['text_fallback']: the pair is two items, not two "
                                       "explainers of one item",
                   "style_vs_key": "same two lines: the key calls the pair adapter/fallback while "
                                   "both texts of a row come from one generator",
                   "classified_in_normal": "scripts/build_v4_sheet.py:45 returns 'normal' for "
                                           "every row whose gold fault id is not "
                                           "UNCLASSIFIED_OPEN_SET, so the 19 classified "
                                           "(menu, FAULT_nn) items enter the NORMAL pool"},
               "evaluation_join": evaluation_anchor_rate()}
    (FIT / "ab_v4_diagnosis_summary.json").write_text(json.dumps(summary, indent=2),
                                                      encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
