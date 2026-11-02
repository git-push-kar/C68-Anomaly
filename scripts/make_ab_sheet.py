"""Build the blinded A/B sheet §19.2 is decided on — one row per item, no repeats.

The first version of this sheet (written inside `fallback_baseline.py`) copied the
100 rows of `audit_hybrid_100.csv` straight through. That sheet is *not* 100 distinct
items: it stratifies by family, and the six items that appear both as `fault_menu` rows
and as `counterfactual` rows were counted in both, so the A/B sheet had 94 distinct
items and six duplicates — the same item shown twice, once with its own diagnosis and
once with a forced wrong one. For scoring prose quality that is a repeated measure
dressed up as two independent judgements, and it skews whichever family it lands in.

This rebuilds it properly: the 94 distinct items from the audit sheet, plus six more
`fault_menu` items drawn (seeded) from the generated set, for **100 rows over 100
distinct items**. Everything else is unchanged — same two sources, same one presentation
format, same seeded A/B assignment, key in a separate file.

CPU only. No model, no generation, no test component.

Usage::

    python scripts/make_ab_sheet.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.fallback_baseline import (  # noqa: E402
    DATASET, HYBRID, OUT, adapter_answer, evidence_from_prompt, load_jsonl, render,
)
from llm.hybrid.generate import fallback_answer  # noqa: E402

WANTED = 100


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=WANTED)
    args = parser.parse_args()

    val = {row["item_id"]: row for row in load_jsonl(DATASET / "val.jsonl")}
    predicted = {row["item_id"]: row for row in load_jsonl(HYBRID / "crossfit_val_rows.jsonl")}
    generated = {row["item_id"]: row for row in load_jsonl(HYBRID / "generation_rows.jsonl")}
    audit = list(csv.DictReader((HYBRID / "audit_hybrid_100.csv").open(encoding="utf-8")))

    entries: List[Dict] = []
    seen = set()
    duplicates = []
    for entry in audit:
        item_id = entry["item_id"]
        if item_id in seen:
            duplicates.append(item_id)
            continue
        seen.add(item_id)
        entries.append({"item_id": item_id, "family": entry.get("family"),
                        "diagnosis": entry.get("forced_diagnosis") or "UNCLASSIFIED_OPEN_SET",
                        "prose": entry.get("prose")})

    rng = random.Random(args.seed)
    pool = sorted(item_id for item_id, row in generated.items()
                  if item_id not in seen and val.get(item_id) is not None
                  and row.get("family") == "fault_menu")
    rng.shuffle(pool)
    for item_id in pool:
        if len(entries) >= args.rows:
            break
        seen.add(item_id)
        entries.append({"item_id": item_id, "family": "fault_menu",
                        "diagnosis": (predicted.get(item_id) or {}).get("pred")
                        or "UNCLASSIFIED_OPEN_SET",
                        "prose": None})
    if len(entries) < args.rows:
        raise SystemExit(f"only {len(entries)} distinct items available, want {args.rows}")
    topped_up = len(entries) - (len(audit) - len(duplicates))

    sheet_rows, key_rows = [], []
    for entry in entries:
        item_id = entry["item_id"]
        item = val[item_id]
        menu_ids = item.get("menu_ids") or []
        supply = entry["diagnosis"]
        verdict = (predicted.get(item_id) or {}).get("verdict") or "classified"
        if item_id in {row["item_id"] for row in load_jsonl(HYBRID / "crossfit_val_rows.jsonl")}:
            verdict = (predicted[item_id].get("verdict") or "classified")
        fallback = fallback_answer(supply, verdict, menu_ids, evidence_from_prompt(item["prompt"]))
        adapter = adapter_answer(generated.get(item_id, {}))
        adapter_text = render(item, {
            "fault_id": supply, "verdict": verdict,
            "ranked_hypotheses": adapter.get("ranked_hypotheses"),
            "physical_reasoning": entry["prose"] or adapter.get("physical_reasoning"),
            "narrative": adapter.get("narrative"),
        })
        fallback_text = render(item, fallback)
        adapter_is_a = rng.random() < 0.5
        sheet_rows.append({
            "item_id": item_id, "family": entry["family"], "diagnosis": supply,
            "text_A": adapter_text if adapter_is_a else fallback_text,
            "text_B": fallback_text if adapter_is_a else adapter_text,
            "correct_A": "", "grounded_A": "", "consistent_A": "", "useful_A": "",
            "correct_B": "", "grounded_B": "", "consistent_B": "", "useful_B": "",
            "preference": "", "notes": "",
        })
        key_rows.append({"item_id": item_id, "A": "adapter" if adapter_is_a else "fallback",
                         "B": "fallback" if adapter_is_a else "adapter", "seed": args.seed})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sheet = out / "audit_ab_100.csv"
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheet_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sheet_rows)
    key = out / "audit_ab_100_key.csv"
    with key.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)

    report = {
        "rows": len(sheet_rows),
        "distinct_items": len({row["item_id"] for row in sheet_rows}),
        "dropped_duplicates": sorted(set(duplicates)),
        "topped_up_from_fault_menu": topped_up,
        "adapter_as_A": sum(1 for row in key_rows if row["A"] == "adapter"),
        "seed": args.seed,
        "sheet": str(sheet), "key": str(key),
        "why_rebuilt": "audit_hybrid_100.csv counts the counterfactual items twice; "
                       "100 rows must cover 100 distinct items",
    }
    (out / "audit_ab_100_meta.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "distinct_items", "adapter_as_A",
                                             "dropped_duplicates")}, indent=2))
    print(f"written: {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
