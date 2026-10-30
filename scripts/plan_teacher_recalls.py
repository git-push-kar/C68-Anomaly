"""Which cached teacher answers the *current* filters reject, and why.

A filter edit makes every cached answer stale by hash, but staleness is not
failure: an answer can be re-judged against the same text and only the ones that
now fail need a model call. This script does that re-judgement, groups the
failures by rule and family, and writes the id list the builder takes with
``--teacher-items``.

It is deliberately read-only: no model call, no dataset write. The call budget is
spent by the builder, under ``--max-calls``.

Usage:
    python scripts/plan_teacher_recalls.py                 # report only
    python scripts/plan_teacher_recalls.py --write-items    # write the id list
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm.benchmark.filters import check_answer  # noqa: E402
from llm.benchmark.teacher import parse_teacher_json  # noqa: E402
from llm.response.allowed_set import build_allowed_set  # noqa: E402
from scripts.build_benchmark_dataset import evidence_of, load_records  # noqa: E402

DATASET = Path("outputs/benchmark_v2/dataset")
EVENTS = Path("outputs/benchmark_v2/events")


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_cache() -> Dict[str, Dict]:
    """The last line written for each item, whatever its hash."""
    out: Dict[str, Dict] = {}
    for record in load_jsonl(DATASET / "teacher_cache.jsonl"):
        out[record["item_id"]] = record
    return out


def family(fault, variant) -> str:
    if variant == "removed":
        return "OPEN_SET"
    if fault is None:
        return "NORMAL"
    return f"FAULT {int(fault):02d}"


def plan(records: Dict[str, Dict]) -> Dict:
    items = load_jsonl(DATASET / "train.jsonl")
    cache = latest_cache()
    rows: List[Dict] = []
    for item in items:
        record = cache.get(item["item_id"])
        if record is None:
            rows.append({**item, "state": "never_called", "codes": []})
            continue
        event = records.get(item["event_id"])
        if event is None:
            continue
        evidence = evidence_of(event)
        allowed = build_allowed_set(
            evidence, extra_faults=[event["fault"]] if event.get("fault") else [])
        answer, parse_error = parse_teacher_json(record.get("raw_text") or "")
        codes = check_answer(answer, item["target"], evidence, allowed,
                             parse_error=parse_error, prompt=item.get("prompt"),
                             menu_ids=item.get("menu_ids"))
        rows.append({
            "item_id": item["item_id"],
            "fault": item.get("fault"),
            "variant": item.get("variant"),
            "family": family(item.get("fault"), item.get("variant")),
            "was_accepted": bool(record.get("accepted")),
            "state": "valid" if not codes else "needs_call",
            "codes": codes,
        })

    valid = [r for r in rows if r["state"] == "valid"]
    needs = [r for r in rows if r["state"] == "needs_call"]
    by_rule = Counter(code.split(":")[0] for r in needs for code in r["codes"])
    by_family = Counter(r["family"] for r in needs)
    reasons_by_family: Dict[str, Counter] = defaultdict(Counter)
    for row in needs:
        for code in row["codes"]:
            reasons_by_family[row["family"]][code.split(":")[0]] += 1
    newly_bad = [r for r in needs if r["was_accepted"]]
    still_bad = [r for r in needs if not r["was_accepted"]]
    return {
        "items": len(rows),
        "valid": len(valid),
        "needs_call": len(needs),
        "newly_invalid": len(newly_bad),
        "still_dropped": len(still_bad),
        "by_rule": dict(by_rule),
        "by_family": dict(by_family),
        "reasons_by_family": {k: dict(v) for k, v in reasons_by_family.items()},
        "needs": needs,
        "newly_invalid_ids": [r["item_id"] for r in newly_bad],
        "still_dropped_ids": [r["item_id"] for r in still_bad],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-items", action="store_true",
                        help="write the ordered id list for --teacher-items")
    parser.add_argument("--out", default=str(DATASET))
    args = parser.parse_args()

    from scripts.build_benchmark_dataset import load_records as _lr

    records = {record["event_id"]: record
               for records in _lr(Path(EVENTS)).values() for record in records}
    result = plan(records)
    out = Path(args.out)
    payload = {k: v for k, v in result.items() if k != "needs"}
    (out / "recall_plan.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"items {result['items']}  valid {result['valid']}  need a call {result['needs_call']}")
    print(f"  newly invalid (were accepted): {result['newly_invalid']}")
    print(f"  still dropped:                 {result['still_dropped']}")
    print("by rule:", json.dumps(result["by_rule"], sort_keys=True))
    print("by family:", json.dumps(result["by_family"], sort_keys=True))

    if args.write_items:
        # Dropped-and-still-dropped first: those items have no prose at all, so a
        # recovery adds a row. Rejected-for-scaffolding second: those have prose
        # that would teach the student to talk about its prompt.
        ordered = result["still_dropped_ids"] + result["newly_invalid_ids"]
        (out / "recall_items.txt").write_text("\n".join(ordered) + "\n", encoding="utf-8")
        print(f"wrote {len(ordered)} ids -> {out / 'recall_items.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
