"""Count abstention items whose teacher prose still names the fault it abstains from.

An open-set episode removes the true fault's menu entry; faults 16-20 never had
one. Either way the expected answer is UNCLASSIFIED_OPEN_SET, so prose that names
the removed fault is prose that answers the unanswerable question -- and would
teach a model to recover the label from text rather than evidence.

This script is the audit, not the filter: it reads the teacher cache (which keeps
the raw answer even for rejected items) and reports, per batch, how many accepted
answers name the fault. Run it before and after a prompt/filter change to get a
before/after count instead of an assertion.

Usage:
    python scripts/scan_open_set_leaks.py [--cache PATH] [--items DIR] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark.filters import names_removed_fault  # noqa: E402
from llm.benchmark.menu import UNCLASSIFIED, build_menu  # noqa: E402

DEFAULT_CACHE = ROOT / "outputs" / "benchmark_v2" / "dataset" / "teacher_cache.jsonl"
DEFAULT_ITEMS = ROOT / "outputs" / "benchmark_v2" / "dataset"
ABSTENTION_FAMILIES = {"open_set", "fault_16", "fault_17", "fault_18", "fault_19", "fault_20"}
_ID = re.compile(r"f(\d{2})-r(\d{3})")


def fault_of(record: Dict) -> Optional[int]:
    """The generating fault number, from the family or the item id."""
    family = str(record.get("family") or "")
    match = re.fullmatch(r"fault_(\d{2})", family)
    if match:
        return int(match.group(1))
    match = _ID.search(str(record.get("item_id") or ""))
    if match:
        return int(match.group(1))
    return None


def is_abstention(record: Dict) -> bool:
    family = str(record.get("family") or "")
    if family in ABSTENTION_FAMILIES:
        return True
    return str(record.get("item_id") or "").endswith("::openset")


def scan(records: Iterable[Dict], verbose: bool = False) -> Dict:
    counts: Counter = Counter()
    findings: List[Dict] = []
    for record in records:
        if not is_abstention(record):
            continue
        number = fault_of(record)
        counts["abstention_records"] += 1
        counts["accepted" if record.get("accepted") else "rejected"] += 1
        if not record.get("accepted"):
            continue
        if number is None:
            counts["unparsed_fault"] += 1
            continue
        # Faults 16-20 were never on the menu; the removed-variant menu is the one
        # for 1-15. Both are reconstructed here so the shared-prefix rule (faults
        # 5 and 12) is applied exactly as the filter applies it.
        menu = build_menu(remove=[number])
        target = {"fault_id": UNCLASSIFIED, "truth_fault_number": number}
        prose = record.get("prose") or {}
        text = "\n".join(str(prose.get(key) or "") for key in ("physical_reasoning", "narrative"))
        if not text.strip():
            text = str(record.get("raw_text") or "")
        reason = names_removed_fault(text, target, [entry["fault_id"] for entry in menu])
        if reason:
            counts["violations"] += 1
            findings.append({"item_id": record.get("item_id"), "fault": number,
                             "reason": reason, "batch": str(record.get("filter_sha256"))[:8]})
        else:
            counts["clean"] += 1
    if verbose:
        for row in findings:
            print(f"  VIOLATION {row['item_id']} fault={row['fault']} {row['reason']} "
                  f"batch={row['batch']}")
    return {"counts": dict(sorted(counts.items())), "findings": findings}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--items", default=str(DEFAULT_ITEMS),
                    help="dataset directory (only used to report which ids are live)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    path = Path(args.cache)
    if not path.exists():
        print(f"no cache at {path}")
        return 1
    # The cache is append-only, so a re-run item appears more than once. The last
    # record for an item is the one that counts -- the same rule the builder uses.
    latest: Dict[str, Dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        latest[str(record.get("item_id"))] = record
    records = list(latest.values())
    result = scan(records, verbose=args.verbose)
    counts = result["counts"]
    print(f"unique items: {len(records)}")
    print(f"abstention records: {counts.get('abstention_records', 0)} "
          f"(accepted {counts.get('accepted', 0)}, rejected {counts.get('rejected', 0)})")
    print(f"accepted answers naming the removed fault: {counts.get('violations', 0)}")
    print(f"accepted and clean: {counts.get('clean', 0)}")
    by_fault = Counter(row["fault"] for row in result["findings"])
    if by_fault:
        print("violations by fault:", dict(sorted(by_fault.items())))
    batches = Counter(row["batch"] for row in result["findings"])
    if batches:
        print("violations by filter batch:", dict(sorted(batches.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
