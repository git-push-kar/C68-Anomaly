"""3B-9 step 4: the fresh 40-row A/B sheet, blank, on items the first 100 never showed.

Selection, seed 42, all deterministic:

* **20 classified faults**, round-robin across fault labels, classifier errors first (rows where the
  supplied diagnosis is not the gold label) so the sheet can test obedience to a wrong prefix;
* **10 abstention items**: 5 `open_set` on a menu, 5 menu-removed;
* **10 `NORMAL_TRANSIENT`** rows;
* every item drawn from the **242 main batch**, and **none of the 100** already scored.

The adapter is one side of each row (its stored answer, rendered the same way the v2 sheet rendered
it); **fallback v3** is the other. No check flag, score or hint appears in the sheet — `text_A` and
`text_B` carry prose only, and the nine scoring columns are blank.

Usage::  python scripts/make_ab_sheet_v3.py [--seed 42] [--out DIR]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.fallback_v2 import load_jsonl, parse_store  # noqa: E402
from scripts.fallback_v3 import fallback_v3, load_overlay  # noqa: E402

DATASET = ROOT / "outputs/benchmark_v2/dataset"
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
FIT = ROOT / "outputs/analysis/phase3a/fit"
SCORE_COLUMNS = ["correct_A", "grounded_A", "consistent_A", "useful_A",
                 "correct_B", "grounded_B", "consistent_B", "useful_B", "preference"]
HEADER = ["item_id", "family", "diagnosis", "verdict", "source", "text_A", "text_B",
          *SCORE_COLUMNS, "notes"]


def render(answer: Dict) -> str:
    """One side's text, in the same shape the v2 sheet used."""
    ranked = answer.get("ranked_hypotheses") or []
    ranked_text = ", ".join(str(entry) for entry in ranked) if isinstance(ranked, list) else str(ranked)
    return (f"Diagnosis: {answer.get('fault_id')} (verdict {answer.get('verdict')})\n"
            f"Ranked: {ranked_text}\n"
            f"Reasoning: {answer.get('physical_reasoning')}\n"
            f"Narrative: {answer.get('narrative')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(FIT))
    args = parser.parse_args()
    rng = random.Random(args.seed)

    used = {row["item_id"] for row in csv.DictReader((FIT / "audit_ab_100_v2.csv")
                                                    .open(encoding="utf-8-sig"))}
    val = {row["item_id"]: row for row in load_jsonl(DATASET / "val.jsonl")}
    probs_by_item = {}
    for prediction in load_jsonl(HYBRID / "crossfit_val_rows.jsonl"):
        want = prediction["item_id"].replace("::menu", "").replace("::removed", "")
        probs_by_item[prediction["item_id"]] = prediction.get("probs")

    main_rows = []
    for row in load_jsonl(HYBRID / "generation_rows.jsonl"):
        if row.get("kind") != "main" or row["item_id"] in used:
            continue
        answer = row["answer"] if isinstance(row["answer"], dict) else json.loads(row["answer"])
        main_rows.append({"row": row, "answer": answer, "family": row.get("family"),
                          "verdict": answer.get("verdict"),
                          "gold": row.get("gold") or row.get("truth"),
                          "diagnosis": row.get("diagnosis")})
    by_family: Dict[str, List[Dict]] = collections.defaultdict(list)
    for entry in main_rows:
        by_family[str(entry["family"])].append(entry)

    def take(pool: List[Dict], count: int) -> List[Dict]:
        ordered = sorted(pool, key=lambda item: item["row"]["item_id"])
        if len(ordered) <= count:
            return ordered
        return sorted(rng.sample(ordered, count), key=lambda item: item["row"]["item_id"])

    taken: List[Dict] = []
    # 20 classified faults, round-robin by label, classifier errors first
    faults = [entry for entry in by_family["fault_menu"] if entry["verdict"] == "classified"]
    errors = [entry for entry in faults if entry["gold"] and entry["diagnosis"]
              and str(entry["diagnosis"]) != str(entry["gold"])]
    clean = [entry for entry in faults if entry not in errors]
    buckets: Dict[str, List[Dict]] = collections.defaultdict(list)
    for entry in errors + clean:
        buckets[str(entry["diagnosis"])].append(entry)
    chosen_faults: List[Dict] = []
    labels = sorted(buckets)
    while len(chosen_faults) < 20 and any(buckets.values()):
        for label in labels:
            if buckets[label] and len(chosen_faults) < 20:
                chosen_faults.append(buckets[label].pop(0))
    taken += chosen_faults
    abst = take([e for e in by_family["open_set"] if e["row"]["item_id"].endswith("::menu")], 5)
    abst += take([e for e in by_family["removed"]], 5)
    taken += abst
    normal = take([e for e in by_family["normal"] if e["verdict"] == "no_fault"], 10)
    taken += normal

    overlay = load_overlay(ROOT / "configs/kb_overlay_v1.json")
    rows_out = []
    key_out = []
    for entry in taken:
        item_id = entry["row"]["item_id"]
        item = val.get(item_id) or {}
        store = parse_store(item.get("prompt") or "")
        probs = probs_by_item.get(item_id) or entry["row"].get("confidence") or {}
        if not isinstance(probs, dict):
            probs = {}
        probs = {str(label): float(value) for label, value in probs.items()} if probs else {}
        v3 = fallback_v3(str(entry["diagnosis"]), str(entry["verdict"]),
                         item.get("menu_ids") or [], store, probs, overlay)
        adapter_first = rng.random() < 0.5
        text_a = render(entry["answer"]) if adapter_first else render(v3)
        text_b = render(v3) if adapter_first else render(entry["answer"])
        rows_out.append({"item_id": item_id, "family": entry["family"],
                         "diagnosis": entry["diagnosis"], "verdict": entry["verdict"],
                         "source": "adapter" if adapter_first else "fallback_v3",
                         "text_A": text_a, "text_B": text_b,
                         **{column: "" for column in SCORE_COLUMNS}, "notes": ""})
        key_out.append({"item_id": item_id, "A": "adapter" if adapter_first else "fallback_v3",
                        "B": "fallback_v3" if adapter_first else "adapter", "seed": args.seed})

    out = Path(args.out)
    with (out / "audit_ab_100_v3.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows_out)
    with (out / "audit_ab_100_v3_key.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "A", "B", "seed"])
        writer.writeheader()
        writer.writerows(key_out)

    families = collections.Counter(str(row["family"]) for row in rows_out)
    availability = {family: len(pool) - sum(1 for row in rows_out if str(row["family"]) == family)
                    for family, pool in by_family.items()}
    meta = {
        "sheet": "audit_ab_100_v3.csv", "rows": len(rows_out),
        "distinct_items": len({row["item_id"] for row in rows_out}),
        "seed": args.seed,
        "families": dict(families),
        "verdicts": dict(collections.Counter(str(row["verdict"]) for row in rows_out)),
        "adapter_side_A": sum(1 for row in key_out if row["A"] == "adapter"),
        "comparator": "fallback v3 (scripts/fallback_v3.py)",
        "excluded_items": len(used),
        "classifier_error_rows_included": sum(
            1 for entry in taken if entry["gold"] and entry["diagnosis"]
            and str(entry["diagnosis"]) != str(entry["gold"])),
        "unused_pool_by_family": availability,
        "scoring_columns_filled": 0,
        "note": ("every item is from the 242 main batch and absent from audit_ab_100_v2.csv; "
                 "no check flag appears anywhere in the sheet"),
        "supersedes": [],
    }
    (out / "audit_ab_100_v3_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"written: {out / 'audit_ab_100_v3.csv'} and its key")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
