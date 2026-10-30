#!/usr/bin/env python
"""Which accepted train answers a fault-13 direction edit can actually change.

Flipping the KB's fault-13 ``initiating.direction`` from ``decrease`` to
``increase`` changes one line of the candidate menu, and that line is rendered
into nearly every prompt (5,731 of 5,870). Almost none of those answers are about
fault 13, though: re-rendering a prompt does not make an answer that never
mentions the variable wrong. So the re-call set is defined by *prose content*, not
by prompt membership.

The AFFECTED set is the union of three groups:

``mentions``
    accepted answers whose prose names FAULT_13, "fault 13", XMEAS_7,
    Reactor_Pressure or "reactor pressure" -- the only answers whose wording can
    disagree with the edited claim;
``contradicting``
    answers the new rule (flipped sign + gerund vocabulary) would now reject,
    i.e. prose asserting XMEAS_7 falls;
``gerund_rejected``
    answers the gerund rule would reject *while the KB still says decrease*, i.e.
    prose asserting XMEAS_7 rises. These are correct today under the old wording
    and must be re-judged once the sign flips.

Everything is read from disk: the item file for the prose actually published, the
cache for the raw text a stricter filter would see. No model calls.

Usage::

    python scripts/count_fault13_affected.py            # count and write JSON
    python scripts/count_fault13_affected.py --ids      # also write the id list
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark import filters as F  # noqa: E402
from llm.benchmark.menu import UNCLASSIFIED  # noqa: E402
from llm.benchmark.teacher import parse_teacher_json  # noqa: E402
from llm.response.allowed_set import build_allowed_set  # noqa: E402
from scripts.build_benchmark_dataset import evidence_of, load_records  # noqa: E402
from scripts.fault_knowledge import FAULT_KNOWLEDGE  # noqa: E402
from scripts.replay_fault13_change import (  # noqa: E402
    GERUNDS_DOWN,
    GERUNDS_UP,
    _judge,
    _with_patch,
)

DATASET = ROOT / "outputs" / "benchmark_v2" / "dataset"
EVENTS = ROOT / "outputs" / "benchmark_v2" / "events"
FAULT = 13

#: The prose patterns that put an answer inside the blast radius. Fault 13's
#: initiating tag is XMEAS_7 (Reactor_Pressure), so any of these names the
#: variable the edited menu line describes.
MENTION_PATTERNS = (
    (r"\bfault[_ ]?0?13\b", "fault_13"),
    (r"\bxmeas[_ ]?0?7\b", "xmeas_7"),
    (r"\breactor[_ ]pressure\b", "reactor_pressure"),
)


def _load_items(split: str = "train") -> list:
    with (DATASET / f"{split}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _load_cache() -> dict:
    cache = {}
    with (DATASET / "teacher_cache.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cache[row["item_id"]] = row
    return cache


def _prose_text(item: dict) -> str:
    prose = item.get("prose") or {}
    return " ".join(str(prose.get(key) or "")
                    for key in ("physical_reasoning", "narrative"))


def _mentions(text: str) -> list:
    low = " ".join(str(text or "").lower().split())
    found = []
    for pattern, label in MENTION_PATTERNS:
        if re.search(pattern, low):
            found.append(label)
    return found


def _class_of(item: dict) -> str:
    label = str((item.get("target") or {}).get("fault_id") or "")
    if label == UNCLASSIFIED:
        return "unclassified"
    if label == "NORMAL_TRANSIENT":
        return "normal"
    return "classified"


def count() -> dict:
    items = _load_items("train")
    records = load_records(EVENTS)
    flat = {record["event_id"]: record for split in records.values() for record in split}
    cache = _load_cache()

    accepted = [item for item in items if item.get("prose")]
    current = _judge(items, flat, cache)
    fault_ids = {item["item_id"] for item in accepted if item.get("fault") == FAULT}

    # Both scenarios are needed: the contradicted answers only appear once the
    # gerunds exist, and the ones that *become* correct only differ from today
    # while the KB still carries the old sign.
    vocab_only = _with_patch(None, True, items, flat, cache)
    flip_and_vocab = _with_patch("increase", True, items, flat, cache)

    def newly_rejected(verdicts) -> list:
        return [item_id for item_id in fault_ids
                if not current.get(item_id, {}).get("codes")
                and verdicts.get(item_id, {}).get("codes")]

    contradicting = newly_rejected(flip_and_vocab)
    gerund_rejected = newly_rejected(vocab_only)

    by_id = {item["item_id"]: item for item in accepted}
    mention_ids = [item["item_id"] for item in accepted if _mentions(_prose_text(item))]

    groups = {
        "mentions": mention_ids,
        "contradicting": contradicting,
        "gerund_rejected": gerund_rejected,
    }
    affected = sorted(set().union(*[set(v) for v in groups.values()]))

    def breakdown(ids: list) -> dict:
        rows = [by_id[item_id] for item_id in ids if item_id in by_id]
        return {
            "items": len(rows),
            "by_class": dict(sorted(Counter(_class_of(row) for row in rows).items())),
            "by_fault": dict(sorted(Counter(
                "normal" if row.get("fault") is None else int(row["fault"])
                for row in rows).items(), key=lambda kv: str(kv[0]))),
            "by_variant": dict(sorted(Counter(row.get("variant") for row in rows).items())),
        }

    return {
        "dataset": str(DATASET / "train.jsonl"),
        "train_items": len(items),
        "accepted_items": len(accepted),
        "cache_entries": len(cache),
        "fault13_accepted": len(fault_ids),
        "groups": {name: breakdown(sorted(set(ids))) for name, ids in groups.items()},
        "affected_total": len(affected),
        "affected": breakdown(affected),
        # The overlap is the interesting part: the two rule-derived groups should
        # be contained in the mention group, or the patterns are too narrow.
        "overlap": {
            "contradicting_not_in_mentions": sorted(set(contradicting) - set(mention_ids)),
            "gerund_not_in_mentions": sorted(set(gerund_rejected) - set(mention_ids)),
        },
        "estimate": {
            "items": len(affected),
            "seconds_per_item": 10.8,
            "expected_calls_at_1_12_per_item": round(len(affected) * 1.12, 1),
            "worst_case_calls": len(affected) * 2,
            "minutes": round(len(affected) * 10.8 * 1.12 / 60, 1),
        },
        "affected_ids": affected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", action="store_true",
                        help="write the affected item-id list next to the JSON")
    args = parser.parse_args()
    out = count()
    path = ROOT / "outputs" / "evaluation" / "fault13_affected.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.ids:
        (DATASET / "fault13_affected_items.txt").write_text(
            "\n".join(out["affected_ids"]) + "\n", encoding="utf-8")
    print(json.dumps({key: out[key] for key in
                      ("train_items", "accepted_items", "fault13_accepted", "groups",
                       "affected_total", "affected", "overlap", "estimate")},
                     indent=2, ensure_ascii=False))
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
