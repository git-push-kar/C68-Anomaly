"""Price the display-name direction check over the cache, with no model calls.

`filters._direction_conflict` searches for the canonical **tag** (`XMEAS_7`). The
teacher writes **display names** (`Reactor_Pressure`) most of the time, so the
rejection rule judges only part of the prose it claims to judge. This script
counts the rest: sentences that name a sensor by its display name and attach a
direction word contradicting the stored sign.

It is a measurement, not a rule. Nothing here rejects an answer; the output is the
number an owner needs before deciding whether the advisory flag should become a
rejection.

Usage::

    python scripts/audit_direction_display.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark.filters import (  # noqa: E402
    direction_inconsistent,
    direction_inconsistent_display,
)
from scripts.build_benchmark_dataset import (  # noqa: E402
    evidence_of,
    load_records,
    load_teacher_cache,
)

DATASET = ROOT / "outputs" / "benchmark_v2" / "dataset"
EVENTS = ROOT / "outputs" / "benchmark_v2" / "events"


def audit() -> Dict:
    records = load_records(EVENTS)
    flat = {record["event_id"]: record for split in records.values() for record in split}
    items = [json.loads(line) for line in
             (DATASET / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = load_teacher_cache(DATASET / "teacher_cache.jsonl")

    per_fault: Counter = Counter()
    per_kind: Counter = Counter()
    per_basis: Counter = Counter()
    examples: List[Dict] = []
    seen_sentence: set = set()
    accepted = 0
    items_flagged = 0
    for item in items:
        entry = cache.get(item["item_id"])
        if not entry or not entry.get("accepted") or not entry.get("prose"):
            continue
        record = flat.get(item["event_id"])
        if record is None:
            continue
        accepted += 1
        evidence = evidence_of(record)
        prose = entry["prose"]
        flags = direction_inconsistent_display(prose, evidence)
        items_flagged += 1 if flags else 0
        for flag in flags:
            fault = item.get("fault")
            per_fault[str(fault)] += 1
            per_kind[str(flag.get("tag_kind"))] += 1
            per_basis[str(flag.get("basis"))] += 1
            key = flag["sentence"].strip().lower()
            if key not in seen_sentence and len(examples) < 10:
                seen_sentence.add(key)
                examples.append({
                    "item_id": item["item_id"],
                    "fault": fault,
                    "sensor": flag["sensor"],
                    "tag_kind": flag.get("tag_kind"),
                    "stored": flag.get("stored"),
                    "basis": flag.get("basis"),
                    "words": flag.get("words"),
                    "sentence": flag["sentence"],
                })
    return {
        "accepted_items": accepted,
        "items_with_display_flag": items_flagged,
        "flags": sum(per_fault.values()),
        "by_fault": dict(sorted(per_fault.items(), key=lambda kv: int(kv[0]))),
        "by_tag_kind": dict(per_kind),
        "by_basis": dict(per_basis),
        "examples": examples,
        "note": "advisory only: nothing here rejects an answer",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATASET / "direction_display_audit.json"))
    args = parser.parse_args()
    result = audit()
    # Items, not sentences, is the number that matters for a scope decision.
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("accepted_items", "items_with_display_flag", "flags",
                       "by_fault", "by_tag_kind", "by_basis")},
                     indent=2))
    for row in result["examples"]:
        print(f"- [{row['fault']}] {row['sensor']} stored={row['stored']} "
              f"({row['basis']}): {row['sentence'][:140]}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
