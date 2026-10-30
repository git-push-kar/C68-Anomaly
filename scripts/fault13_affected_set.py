"""Which cached answers the fault-13 direction edit can reach.

The KB says fault 13's initiating variable (XMEAS_7 / Reactor_Pressure) *decreases*
first. The stored evidence says it increases in 102 of 102 train events, so the
menu line is wrong and has to change. Changing it re-renders one line of the
candidate menu, which is present in nearly every prompt -- but "present in the
prompt" is not the same as "the answer depended on it". This script draws the line
where it can be defended:

``fault13``
    Every accepted train answer for fault 13. Its own mechanism is what the edited
    line describes, so the prose was written against the claim that changed.
``named_with_direction``
    Accepted answers for *other* faults that name FAULT_13 (or fault 13) and make
    a direction claim about its initiating variable in the same breath. These are
    differential arguments quoting the line that moved.
``other_only``
    Everything else: the line is in the prompt, but the answer never used it. Those
    records are marked ``menu_revision_unaffected`` with that reason instead of
    being re-called.

The union of the first two is the AFFECTED set. The brief caps it at 160 items;
the script reports the size and refuses to write a plan above the cap, so an
under-estimate cannot silently become an expensive run.

Usage::

    python scripts/fault13_affected_set.py            # counts only
    python scripts/fault13_affected_set.py --write    # + item-id plan file
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark.filters import DECREASE_WORDS, INCREASE_WORDS  # noqa: E402

#: The audit reads the vocabulary the patch will leave in force, gerunds
#: included. Reading the pre-patch lists missed exactly the answers that say
#: "pressure is increasing" -- the ones the new rule is going to reject.
_UP = tuple(INCREASE_WORDS) + ("increasing", "rising")
_DOWN = tuple(DECREASE_WORDS) + ("decreasing", "falling")
from scripts.build_benchmark_dataset import (  # noqa: E402
    evidence_of,
    load_records,
    load_teacher_cache,
    replay_cache,
)

DATASET = ROOT / "outputs" / "benchmark_v2" / "dataset"
EVENTS = ROOT / "outputs" / "benchmark_v2" / "events"
FAULT = 13
CAP = 160

#: Every way an answer can refer to the fault whose line moved.
_FAULT_REF = re.compile(r"\bfault[ _-]?0?13\b", re.IGNORECASE)
_INITIATING = re.compile(r"xmeas[ _-]?7\b|reactor[ _-]?pressure", re.IGNORECASE)


def _direction_claim(text: str) -> Optional[str]:
    """Which way the answer says the initiating variable moves, if it says.

    Sentence-scoped on purpose: a direction word elsewhere in a paragraph about a
    different sensor is not a claim about XMEAS_7. The fault reference is allowed
    anywhere in the answer, because a differential sentence ("fault 13 would need
    the pressure to fall") often names the fault before it names the variable.
    """
    body = text or ""
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        if not _INITIATING.search(sentence):
            continue
        low = sentence.lower()
        up = any(word in low for word in _UP)
        down = any(word in low for word in _DOWN)
        if up or down:
            return "increase" if up and not down else ("decrease" if down and not up else "mixed")
    return None


def classify() -> Dict:
    records = load_records(EVENTS)
    flat = {record["event_id"]: record for split in records.values() for record in split}
    items = [json.loads(line) for line in
             (DATASET / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = load_teacher_cache(DATASET / "teacher_cache.jsonl")
    # The filter hash the cache was written under is whatever the last run used;
    # replaying with a placeholder hash only recomputes the ``replayed`` marker.
    valid = replay_cache(items, flat, cache, filter_sha="", )
    accepted = {item_id for item_id, entry in valid.items() if entry.get("prose")}

    own: List[str] = []
    named: List[Dict] = []
    for item in items:
        item_id = item["item_id"]
        if item_id not in accepted:
            continue
        if item.get("fault") == FAULT:
            own.append(item_id)
            continue
        prose = cache[item_id].get("prose") or {}
        text = f"{prose.get('physical_reasoning') or ''}\n{prose.get('narrative') or ''}"
        if not _FAULT_REF.search(text):
            continue
        direction = _direction_claim(text)
        if direction is None:
            continue
        named.append({"item_id": item_id, "fault": item.get("fault"),
                      "direction": direction, "variant": item.get("variant")})

    affected = sorted(set(own) | {row["item_id"] for row in named})
    return {
        "fault13_item_ids": sorted(own),
        "named_with_direction": named,
        "affected": affected,
        "by_fault": dict(sorted(Counter(
            str(row["fault"]) for row in named).items(), key=lambda kv: int(kv[0]))),
        "by_direction": dict(Counter(row["direction"] for row in named)),
        "counts": {
            "fault13_accepted": len(own),
            "named_with_direction": len(named),
            "union": len(affected),
            "cap": CAP,
            "over_cap": len(affected) > CAP,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="write the AFFECTED item-id plan file")
    parser.add_argument("--out", default=str(DATASET / "fault13_affected.txt"))
    args = parser.parse_args()
    result = classify()
    print(json.dumps({key: result[key] for key in
                      ("counts", "by_fault", "by_direction")}, indent=2))
    if args.write:
        if result["counts"]["over_cap"]:
            print(f"REFUSING to write a plan of {result['counts']['union']} items "
                  f"(cap {CAP})")
            return 2
        Path(args.out).write_text("\n".join(result["affected"]) + "\n", encoding="utf-8")
        print(f"written: {args.out} ({len(result['affected'])} item ids)")
    out = DATASET / "fault13_affected_set.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
