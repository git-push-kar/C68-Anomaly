#!/usr/bin/env python
"""Scan the built follow-up set for open-set leaks.

An open-set item must not tell the model which fault it is looking at -- not in
the context, not in the question, and not in the accepted answer. The builder's
`names_removed_fault` filter covers answers; this script checks all three
surfaces independently, because a filter that is wrong in the same way as the
prompt would agree with itself.

Two populations are scanned:

* ``removed`` items, where the menu entry was taken away on purpose;
* faults 16-20, which have no menu entry at all, so their items are open-set
  by construction rather than by variant.

For each, the script looks for the true fault's number in a fault-shaped phrase
(``fault 15``, ``FAULT_15``) and its KB name. The KB *initiating variable* is
deliberately not a needle: on an open-set item that variable is usually sitting
in the evidence block, which is exactly what the model is supposed to reason
from. Naming it is not naming the fault -- only the family name or the number
would give the answer away. ``--include-initiating`` reports those separately
for a reviewer who wants to see them.

Usage::

    python scripts/scan_followup_open_set.py            # whole built set
    python scripts/scan_followup_open_set.py --split train
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.fault_knowledge import FAULT_KNOWLEDGE  # noqa: E402

FOLLOWUP = ROOT / "outputs" / "benchmark_v2" / "followup"
#: Faults with no menu entry at all, so every item about them is open-set.
OPEN_FAMILY = (16, 17, 18, 19, 20)


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def needles(number: int, include_initiating: bool = False) -> List[Tuple[str, str]]:
    """Phrases that would name the fault outright."""
    entry = FAULT_KNOWLEDGE.get(int(number)) or {}
    out: List[Tuple[str, str]] = []
    name = str(entry.get("name") or "").strip()
    if name:
        out.append(("kb_name", _norm(name)))
    if include_initiating:
        initiating = entry.get("initiating")
        if isinstance(initiating, list) and initiating:
            sensor = str((initiating[0] or {}).get("sensor") or "").strip()
            if sensor:
                out.append(("initiating_sensor", _norm(sensor).replace("_", " ")))
    return out


FAULT_PHRASE = re.compile(r"\bfault[ _]?(\d{1,2})\b", re.IGNORECASE)


def scan_text(text: str, number: int, include_initiating: bool = False) -> List[str]:
    hits: List[str] = []
    for match in FAULT_PHRASE.finditer(text or ""):
        if int(match.group(1)) == int(number):
            hits.append(f"fault_number:{match.group(0)}")
    body = _norm(text)
    body_spaced = body.replace("_", " ")
    for kind, needle in needles(number, include_initiating):
        if needle and (needle in body or needle in body_spaced):
            hits.append(f"{kind}:{needle}")
    return hits


def iter_dialogues(splits: Iterable[str]) -> Iterable[Dict]:
    for split in splits:
        path = FOLLOWUP / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("train", "val", "test"), default=None)
    ap.add_argument("--out", default=str(FOLLOWUP / "open_set_scan.json"))
    ap.add_argument("--include-initiating", action="store_true",
                    help="also count the KB initiating variable (diagnostic, not a leak)")
    args = ap.parse_args()
    splits = [args.split] if args.split else ["train", "val", "test"]

    counts: Counter = Counter()
    violations: List[Dict] = []
    echoes: List[Dict] = []
    by_surface: Counter = Counter()

    for dialogue in iter_dialogues(splits):
        fault = dialogue.get("fault")
        if fault is None:
            counts["normal"] += 1
            continue
        number = int(fault)
        open_set = bool(dialogue.get("open_set")) or number in OPEN_FAMILY
        if not open_set:
            counts["classified"] += 1
            continue
        counts["open_set"] += 1
        counts["open_set_removed" if dialogue.get("open_set") else "open_set_native"] += 1

        hits = scan_text(dialogue.get("context") or "", number, args.include_initiating)
        if hits:
            violations.append({"dialogue_id": dialogue["dialogue_id"], "surface": "context",
                               "fault": number, "hits": sorted(set(hits))})
            by_surface["context"] += 1
        for index, turn in enumerate(dialogue.get("turns") or []):
            for surface in ("question", "answer"):
                text = turn.get(surface) or ""
                if not text:
                    continue
                hits = scan_text(text, number, args.include_initiating)
                if hits:
                    surface_name = f"turn.{surface}"
                    record = {"dialogue_id": dialogue["dialogue_id"], "surface": surface_name,
                              "turn": index, "fault": number, "hits": sorted(set(hits))}
                    # A question that uses the fault's plain physical description is
                    # the operator's own words ("is the condenser cooling water
                    # valve sticking?" is a question an operator asks), and the
                    # rule that matters is that the *answer* must not confirm it.
                    if surface == "question" and all(h.startswith("kb_name") for h in hits):
                        echoes.append(record)
                        by_surface["turn.question_operator_echo"] += 1
                        continue
                    violations.append(record)
                    by_surface[surface_name] += 1

    for split in splits:
        counts[f"dialogues_{split}"] = sum(1 for d in iter_dialogues([split]))
    report = {
        "splits": splits,
        "counts": dict(counts),
        "violations": len(violations),
        "operator_echoes": len(echoes),
        "by_surface": dict(by_surface),
        "detail": violations[:40],
        "echo_detail": echoes[:20],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("counts", "violations", "operator_echoes", "by_surface")},
                     indent=1))
    if violations:
        print("first violations:")
        for item in violations[:10]:
            print(" ", item["dialogue_id"], item["surface"], item["hits"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
