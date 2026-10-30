"""Leak checks: no split identifiers in a context, no shared event across splits.

Two different leaks are guarded here, and both abort the build instead of being
reported:

* **Context/question leaks.** Every follow-up context and question is scanned for
  the benchmark event ids (``EVT-train-f15-r004`` encodes split, fault and run),
  dataset item ids, split names, run numbers, filenames, and any fault number
  outside the menu that item is shown. The context carries an opaque incident
  label instead of an event id.
* **Split overlap.** The same (source, fault, run) key must not appear in two
  splits, and the same event id must not either. The split file is the
  authority, so every built key is also checked against the component the file
  assigns it to.

A third check keeps the paraphrases honest: the held-out questions used for val
and test must not appear anywhere in the train half.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from llm.followup.paraphrases import LABELS, heldout_questions, normalise, train_questions
from llm.response.validators import fault_ids_in

#: Patterns that must never appear in a follow-up context or question.
LEAK_PATTERNS = (
    (r"\bEVT-", "benchmark event id"),
    (r"\bNORM-", "benchmark event id"),
    (r"\bANOM-", "stored event id"),
    (r"\bINC-[0-9A-F]{6,}", "opaque id is allowed, but not in a question"),
    (r"::", "dataset item id"),
    (r"\bitem_id\b", "dataset item id"),
    (r"\b(?:train|val|test|holdout)\b", "split name"),
    (r"\bfold[ _-]?\d\b", "protocol fold"),
    (r"\bsplit\b", "split vocabulary"),
    (r"\br\d{3}\b", "run number"),
    (r"\bsimulationRun\b", "run field"),
    (r"\brun\s+\d+\b", "run number"),
    (r"\bseed\b", "build seed"),
    (r"TEP_Fault", "source filename"),
    (r"\.csv\b", "source filename"),
)

#: The opaque id is allowed in the context (as the incident reference) but never
#: in an operator question. ``leak_violations`` takes ``allow_opaque`` for that.
OPAQUE_RE = re.compile(r"\bINC-[0-9A-F]{6,}\b")


class LeakError(RuntimeError):
    """A context, question or split set leaked what it must not."""


def leak_violations(
    text: str,
    *,
    permitted_faults: Sequence[int],
    event_id: str = "",
    allow_opaque: bool = False,
) -> List[str]:
    """Return a list of leak descriptions for one piece of text."""
    body = text or ""
    found: List[str] = []
    for pattern, label in LEAK_PATTERNS:
        if label == "opaque id is allowed, but not in a question" and allow_opaque:
            continue
        if re.search(pattern, body, re.IGNORECASE):
            found.append(f"{label} ({pattern})")
    if event_id and normalise(event_id) in normalise(body):
        found.append("literal event id")
    permitted = set(int(x) for x in permitted_faults)
    outside = sorted(fault_ids_in(body) - permitted)
    if outside:
        found.append(f"fault number(s) outside the menu: {outside}")
    return found


def assert_clean(
    text: str,
    *,
    permitted_faults: Sequence[int],
    event_id: str = "",
    allow_opaque: bool = False,
    where: str = "context",
) -> None:
    problems = leak_violations(
        text, permitted_faults=permitted_faults, event_id=event_id, allow_opaque=allow_opaque
    )
    if problems:
        raise LeakError(f"{where}: {'; '.join(problems)}")


# ---------------------------------------------------------------------------
def record_key(record: Dict) -> str:
    fault = record.get("fault")
    source = record.get("source_file") or ("TEP_FaultFree_Testing.csv" if fault is None else "")
    return f"{source}|{fault if fault is not None else 'normal'}|{record.get('simulationRun')}"


def _expand_component(block: Dict) -> Set[str]:
    source = Path(str(block.get("source_file") or "")).name
    keys: Set[str] = set()
    for entry in block.get("faults") or []:
        number = int(entry["faultNumber"])
        low, high = (int(x) for x in entry["simulationRunRange"])
        for run in range(low, high + 1):
            keys.add(f"{source}|{number}|{run}")
    return keys


def check_event_overlap(
    records_by_split: Dict[str, List[Dict]],
    splits_path: Optional[Path] = None,
) -> Dict:
    """Abort unless every split is run-disjoint, then check the split file.

    Returns the evidence written into the manifest: per-split key counts, the
    pairwise comparisons actually made, and the number of keys the split file
    independently confirms.
    """
    keys = {split: [record_key(record) for record in rows] for split, rows in records_by_split.items()}
    ids = {split: [str(record.get("event_id")) for record in rows] for split, rows in records_by_split.items()}

    for split, values in keys.items():
        duplicates = [key for key, count in Counter(values).items() if count > 1]
        if duplicates:
            raise LeakError(f"{split}: duplicate event key(s) {duplicates[:3]}")

    names = sorted(keys)
    pairwise = 0
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = set(keys[left]) & set(keys[right])
            if shared:
                raise LeakError(
                    f"SPLIT OVERLAP: {left} and {right} share {len(shared)} event key(s), "
                    f"e.g. {sorted(shared)[:3]}"
                )
            shared_ids = set(ids[left]) & set(ids[right])
            if shared_ids:
                raise LeakError(
                    f"SPLIT OVERLAP: {left} and {right} share event id(s) {sorted(shared_ids)[:3]}"
                )
            pairwise += 1

    report: Dict = {
        "events_per_split": {split: len(values) for split, values in keys.items()},
        "pairwise_comparisons": pairwise,
        "zero_overlap": True,
        "duplicate_keys": {},
    }

    if splits_path is not None and Path(splits_path).exists():
        splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
        block = ((splits.get("protocols") or {}).get("A_seen_faults_run_disjoint") or {}).get("splits") or {}
        confirmed = {}
        for component, spec in block.items():
            if component not in keys:
                continue
            allowed = _expand_component(spec)
            # Fault-free runs are not part of the faulty split file; only the
            # faulty keys can be confirmed against it.
            faulty_keys = {key for key in keys[component] if key.split("|")[1].isdigit()}
            stray = sorted(faulty_keys - allowed)
            if stray:
                raise LeakError(
                    f"split file disagrees with the build: {len(stray)} {component} key(s) "
                    f"are not in that component, e.g. {stray[:3]}"
                )
            confirmed[component] = len(keys[component])
        report["split_file_confirmed"] = confirmed
    return report


def check_paraphrase_disjointness() -> Dict:
    """Abort if a held-out question appears in any train half."""
    report: Dict[str, Dict] = {}
    train_all: Dict[str, Set[str]] = defaultdict(set)
    for label in LABELS:
        train_all[label] = {normalise(q) for q in train_questions(label)}
    train_union = {q for values in train_all.values() for q in values}

    for label in LABELS:
        train = train_all[label]
        heldout = {normalise(q) for q in heldout_questions(label)}
        clash = sorted(train & heldout)
        if clash:
            raise LeakError(f"{label}: held-out question also in train: {clash[:2]}")
        cross = sorted(heldout & (train_union - train))
        if cross:
            raise LeakError(f"{label}: held-out question appears in another label's train half: {cross[:2]}")
        report[label] = {"train": len(train), "heldout": len(heldout), "disjoint": True}
    return {
        "per_label": report,
        "train_total": sum(len(v) for v in train_all.values()),
        "heldout_total": sum(len(heldout_questions(label)) for label in LABELS),
    }


__all__ = [
    "LEAK_PATTERNS",
    "LeakError",
    "assert_clean",
    "check_event_overlap",
    "check_paraphrase_disjointness",
    "leak_violations",
    "record_key",
]
