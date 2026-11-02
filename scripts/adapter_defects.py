"""Adapter output defects: empty narrative, stub narrative, empty reasoning — counted by family.

3B-7 printed one item whose adapter narrative was empty (`EVT-val-f01-r306::menu`) and described
it as "empty in this generation row". One example is not a rate, and a prose-only training contract
(§20.3 outcome A, §20.4's empty-narrative gate of 0 %) turns on whether that example is isolated.
This counts the three defects across everything the adapter has written for validation:

* the **242 main-batch** answers,
* the **58 counterfactual** answers,
* the **100 adapter texts on the rebuilt A/B sheet** (`audit_ab_100_v2.csv`), whose adapter side is
  assembled by `make_ab_sheet_v2.adapter_texts()` so the count and the sheet cannot disagree.

Broken down by family and by batch, and with the specific item 3B-7 named listed beside its peers.

CPU only, no model, no generation.

Usage::

    python scripts/adapter_defects.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.fallback_baseline import load_jsonl  # noqa: E402
from scripts.make_ab_sheet_v2 import OUT, adapter_texts  # noqa: E402

HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
NAMED_BY_3B7 = "EVT-val-f01-r306::menu"


def defects(entry: Dict) -> Dict[str, bool]:
    narrative = str(entry.get("narrative") or "").strip()
    reasoning = str(entry.get("physical_reasoning") or "").strip()
    return {
        "empty_narrative": not narrative,
        "narrative_under_5_words": 0 < len(narrative.split()) < 5,
        "empty_physical_reasoning": not reasoning,
    }


def bucket(entries: List[Dict]) -> Dict:
    counts = {"n": len(entries), "empty_narrative": 0, "narrative_under_5_words": 0,
              "empty_physical_reasoning": 0}
    by_family: Dict[str, Dict[str, int]] = {}
    empty_ids: List[str] = []
    for entry in entries:
        found = defects(entry)
        for key, value in found.items():
            counts[key] += int(value)
        family = str(entry.get("family") or "unknown")
        slot = by_family.setdefault(family, {"n": 0, "empty_narrative": 0,
                                             "narrative_under_5_words": 0,
                                             "empty_physical_reasoning": 0})
        slot["n"] += 1
        for key, value in found.items():
            slot[key] += int(value)
        if found["empty_narrative"]:
            empty_ids.append(entry["item_id"])
    counts["rate"] = {key: round(counts[key] / counts["n"], 4) if counts["n"] else 0.0
                      for key in ("empty_narrative", "narrative_under_5_words",
                                  "empty_physical_reasoning")}
    counts["by_family"] = dict(sorted(by_family.items()))
    counts["empty_narrative_items"] = sorted(empty_ids)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    rows = load_jsonl(HYBRID / "generation_rows.jsonl")
    adapters = adapter_texts()

    main_entries = [{"item_id": row["item_id"], "family": row.get("family"),
                     **adapters[row["item_id"]]}
                    for row in rows if row.get("kind") == "main"]
    counter_entries = [{"item_id": row["item_id"], "family": row.get("family"),
                        **adapters[row["item_id"]]}
                       for row in rows if row.get("kind") == "counterfactual"]

    sheet_path = Path(args.out) / "audit_ab_100_v2.csv"
    if not sheet_path.exists():
        raise SystemExit(f"{sheet_path} is missing — run scripts/make_ab_sheet_v2.py first")
    sheet_ids = [row["item_id"] for row in csv.DictReader(sheet_path.open(encoding="utf-8"))]
    ab_entries = [{"item_id": item_id, "family": adapters[item_id]["family"],
                   **adapters[item_id]} for item_id in sheet_ids]

    everything = main_entries + counter_entries + ab_entries
    named = next((entry for entry in everything if entry["item_id"] == NAMED_BY_3B7), None)
    named_defects = defects(named) if named else None

    report = {
        "main_242": bucket(main_entries),
        "counterfactual_58": bucket(counter_entries),
        "ab_100": bucket(ab_entries),
        "generation_300": bucket(main_entries + counter_entries),
        "everything_400": bucket(everything),
        "the_item_3b7_named": {
            "item_id": NAMED_BY_3B7,
            "defects": named_defects,
            "narrative_words": len(str((named or {}).get("narrative") or "").split()),
            "is_isolated": (named is not None and named_defects["empty_narrative"]
                            and len(bucket(everything)["empty_narrative_items"]) == 1),
            "peers_with_empty_narrative": [item for item in
                                           bucket(everything)["empty_narrative_items"]
                                           if item != NAMED_BY_3B7],
        },
    }
    (Path(args.out) / "adapter_defects.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({key: {inner: report[key][inner] for inner in
                            ("n", "empty_narrative", "narrative_under_5_words",
                             "empty_physical_reasoning", "rate")}
                      for key in ("main_242", "counterfactual_58", "ab_100", "generation_300",
                                  "everything_400")}, indent=2))
    print(json.dumps({"named": report["the_item_3b7_named"],
                      "empty_narrative_items": report["everything_400"]["empty_narrative_items"]},
                     indent=2))
    print(f"written: {Path(args.out) / 'adapter_defects.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
