"""Bring the teacher cache's provenance up to date without a model call.

A filter or menu edit changes what the cache records mean. Two things have to
happen before the next teacher run, or it will re-call answers it already has:

1. **validation** -- every cached answer that still passes the *current* filters
   gets ``validated_filter_sha256`` / ``validated_at`` / ``validation_method``.
   This is a re-judgement, not a claim: ``replay_cache`` applies the real rules.
2. **exemption** -- a menu revision that changed one line cannot make an answer
   about a different fault stale, so records outside the affected set are marked
   ``menu_revision_unaffected`` with the revision and a reason. Explicitly, per
   record, never as a global assumption.

``--affected-file`` is what keeps that honest: an item in the file is never
exempted, so it stays pending and the teacher is asked again.

Both writes go to the cache sidecar; the answer file itself is not rewritten.

Usage::

    python scripts/sync_cache_provenance.py --affected-file <ids.txt>
    python scripts/sync_cache_provenance.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark.provenance import (  # noqa: E402
    MENU_REVISION,
    menu_provenance,
    reuse_decision,
    validation_stamp,
)
from scripts.build_benchmark_dataset import (  # noqa: E402
    append_cache_meta,
    filter_code_hash,
    load_cache_meta,
    load_records,
    load_teacher_cache,
    replay_cache,
)

DATASET = ROOT / "outputs" / "benchmark_v2" / "dataset"
EVENTS = ROOT / "outputs" / "benchmark_v2" / "events"

#: Why a record can be exempted from a menu revision. Stated once, quoted in the
#: manifest, so a reader can disagree with the claim instead of guessing at it.
EXEMPT_REASON = ("the edited menu line is fault 13's; this answer was written about "
                 "another fault and never restates it")


def sync(affected: Optional[set] = None, dry_run: bool = False) -> Dict:
    provenance = {"filter_sha256": filter_code_hash(), **menu_provenance()}
    items = [json.loads(line) for line in
             (DATASET / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    records = load_records(EVENTS)
    flat = {record["event_id"]: record for split in records.values() for record in split}
    cache_path = DATASET / "teacher_cache.jsonl"
    cache = load_teacher_cache(cache_path)

    before: Counter = Counter()
    for item in items:
        entry = cache.get(item["item_id"])
        if entry is None:
            before["not_called"] += 1
            continue
        reusable, reason = reuse_decision(entry, provenance)
        before["reusable" if reusable else reason] += 1

    valid = replay_cache(items, flat, cache, provenance["filter_sha256"])
    if not dry_run:
        append_cache_meta(cache_path, [
            {"item_id": item_id, **validation_stamp(provenance["filter_sha256"], "replay")}
            for item_id in sorted(valid)])

    exempt: List[Dict] = []
    for item in items:
        item_id = item["item_id"]
        if item_id in (affected or set()):
            continue
        if item_id not in valid:
            continue
        exempt.append({
            "item_id": item_id,
            "menu_revision_unaffected": {
                "revisions": [MENU_REVISION],
                "reason": EXEMPT_REASON,
                "from_revision": (cache.get(item_id) or {}).get("menu_revision") or "r1",
            },
        })
    if not dry_run:
        append_cache_meta(cache_path, exempt)

    after = Counter()
    fresh = load_teacher_cache(cache_path) if not dry_run else cache
    for item in items:
        entry = cache.get(item["item_id"]) if dry_run else fresh.get(item["item_id"])
        if entry is None:
            after["not_called"] += 1
            continue
        if dry_run and item["item_id"] in valid:
            entry = dict(entry, **validation_stamp(provenance["filter_sha256"], "replay"))
            if item["item_id"] not in (affected or set()):
                entry["menu_revision_unaffected"] = [MENU_REVISION]
        reusable, reason = reuse_decision(entry, provenance)
        after["reusable" if reusable else reason] += 1

    return {
        "filter_sha256": provenance["filter_sha256"],
        "menu_revision": provenance["menu_revision"],
        "menu_sha256": provenance["menu_sha256"],
        "items": len(items),
        "replay_valid": len(valid),
        "stamped": 0 if dry_run else len(valid),
        "exempted": 0 if dry_run else len(exempt),
        "affected_held_pending": len((affected or set()) & set(valid)),
        "reusable_before": dict(sorted(before.items())),
        "reusable_after": dict(sorted(after.items())),
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affected-file", default=None,
                        help="item ids that must stay pending (never exempted)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    affected = None
    if args.affected_file:
        affected = {line.strip() for line in
                    Path(args.affected_file).read_text(encoding="utf-8").splitlines()
                    if line.strip()}
    result = sync(affected, args.dry_run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
