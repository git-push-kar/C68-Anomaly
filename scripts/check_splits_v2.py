"""Validate data/splits_v2.json against the benchmark's split contract.

Contract
--------
* unique key: ``(source_file, faultNumber, simulationRun)``;
* every expanded key is unique inside a protocol/fold;
* no two components of a protocol/fold share a key -- so nothing from train
  appears in test, and nothing from train appears in ``test_seen``;
* protocols that declare ``constraints.fault_disjoint_across_splits`` (the
  leave-faults-out folds) also require the held-out ``test`` faults to be
  disjoint from every other component's faults;
* ``test_seen``, when present, may only contain faults that are in ``train``;
* headline splits contain real faults only; synthetic-only faults 21/22 never
  appear in a split;
* each protocol/fold records a SHA-256 of its expanded sorted key list, and the
  recorded value must match a fresh recomputation.

Usage
-----
    python scripts/check_splits_v2.py
    python scripts/check_splits_v2.py --verify-source   # slow: streams the CSV

Exit code 0 means every check passed; any violation prints ``FAIL`` and exits 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = REPO_ROOT / "data" / "splits_v2.json"
EXPECTED_KEY = ["source_file", "faultNumber", "simulationRun"]
SPLIT_NAMES = ("train", "val", "test")
#: Optional extra components. ``test_seen`` holds runs of the *seen* faults so
#: that false abstention can be measured without breaking fault disjointness.
OPTIONAL_SPLITS = ("test_seen",)
SYNTHETIC_FAULTS = [21, 22]
REAL_FAULTS = list(range(1, 21))

Key = Tuple[str, int, int]


def expand_split(block: Dict) -> List[Key]:
    """Expand a split's declared run ranges into explicit keys."""
    keys: List[Key] = []
    source = str(block.get("source_file", ""))
    for entry in block.get("faults") or []:
        fault = int(entry["faultNumber"])
        start, end = (int(v) for v in entry["simulationRunRange"])
        for run in range(start, end + 1):
            keys.append((source, fault, run))
    return keys


def key_digest(keys: Iterable[Key]) -> str:
    """SHA-256 over the sorted, de-duplicated keys -- the recorded fingerprint."""
    lines = sorted({f"{source}|{fault:02d}|{run}" for source, fault, run in keys})
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _units(doc: Dict) -> List[Tuple[str, Dict, Dict]]:
    """Flatten the document into (label, splits, constraints) validation units."""
    units: List[Tuple[str, Dict, Dict]] = []
    for protocol_name, protocol in (doc.get("protocols") or {}).items():
        constraints = protocol.get("constraints") or {}
        if "splits" in protocol:
            units.append((protocol_name, protocol["splits"], constraints))
        for fold_name, fold in (protocol.get("folds") or {}).items():
            units.append(
                (f"{protocol_name}/{fold_name}", fold["splits"], {**constraints, **(fold.get("constraints") or {})})
            )
    return units


def check_document(doc: Dict, verify_source: bool = False) -> List[str]:
    failures: List[str] = []

    if doc.get("unique_key") != EXPECTED_KEY:
        failures.append(f"unique_key is {doc.get('unique_key')!r}, expected {EXPECTED_KEY!r}")
    source = doc.get("source") or {}
    source_file = source.get("file")
    low, high = int(source.get("simulationRunMin", 1)), int(source.get("simulationRunMax", 500))
    per_fault = int(source.get("runsPerFault", high - low + 1))

    declared_scope = doc.get("fault_scope") or {}
    if sorted(declared_scope.get("headline") or []) != REAL_FAULTS:
        failures.append(f"fault_scope.headline is {declared_scope.get('headline')!r}, expected faults 1-20")
    if sorted(declared_scope.get("synthetic_only") or []) != SYNTHETIC_FAULTS:
        failures.append(
            f"fault_scope.synthetic_only is {declared_scope.get('synthetic_only')!r}, expected {SYNTHETIC_FAULTS}"
        )

    units = _units(doc)
    if not units:
        failures.append("no protocols/folds declared")
        return failures

    for label, splits, constraints in units:
        for name in SPLIT_NAMES:
            if name not in splits:
                failures.append(f"{label}: missing required split {name!r}")
        if any(name not in splits for name in SPLIT_NAMES):
            continue

        present = [name for name in SPLIT_NAMES + OPTIONAL_SPLITS if name in splits]
        keys_by_split = {name: expand_split(splits[name]) for name in present}

        for name, keys in keys_by_split.items():
            duplicates = [key for key, count in Counter(keys).items() if count > 1]
            if duplicates:
                failures.append(f"{label}/{name}: {len(duplicates)} duplicate keys, e.g. {duplicates[:3]}")

        # cross-split key overlap, reported per pair so train/test is explicit
        for left, right in combinations(present, 2):
            shared = set(keys_by_split[left]) & set(keys_by_split[right])
            if shared:
                failures.append(
                    f"{label}: {len(shared)} keys appear in both {left} and {right}, e.g. {sorted(shared)[:3]}"
                )

        # run-level overlap on the same fault, even if the source string differs
        pair_splits: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        for name, keys in keys_by_split.items():
            for _, fault, run in keys:
                pair_splits[(fault, run)].add(name)
        shared_pairs = {pair: names for pair, names in pair_splits.items() if len(names) > 1}
        if shared_pairs:
            failures.append(
                f"{label}: {len(shared_pairs)} (faultNumber, simulationRun) pairs in more than one split, "
                f"e.g. {sorted(shared_pairs)[:3]}"
            )

        # fault disjointness where the protocol demands it
        if constraints.get("fault_disjoint_across_splits"):
            held = {fault for _, fault, _ in keys_by_split["test"]}
            reference = {
                fault
                for name in present
                if name != "test"
                for _, fault, _ in keys_by_split[name]
            }
            overlap = held & reference
            if overlap:
                failures.append(
                    f"{label}: held-out test faults {sorted(overlap)} also appear in "
                    f"train/val/test_seen (fault_disjoint_across_splits=true)"
                )

        # test_seen may only contain faults the model has actually seen in train
        if "test_seen" in present:
            seen_in_test = {fault for _, fault, _ in keys_by_split["test_seen"]}
            train_faults = {fault for _, fault, _ in keys_by_split["train"]}
            stray = sorted(seen_in_test - train_faults)
            if stray:
                failures.append(
                    f"{label}: test_seen contains faults not present in train: {stray} "
                    f"(test_seen must be a subset of the train faults)"
                )

        # scope, bounds, source
        for name, splits_block in splits.items():
            if splits_block.get("source_file") != source_file:
                failures.append(
                    f"{label}/{name}: source_file {splits_block.get('source_file')!r} != declared source {source_file!r}"
                )
            for entry in splits_block.get("faults") or []:
                fault = int(entry["faultNumber"])
                start, end = (int(v) for v in entry["simulationRunRange"])
                count = int(entry.get("runCount", end - start + 1))
                if fault in SYNTHETIC_FAULTS:
                    failures.append(f"{label}/{name}: synthetic-only fault {fault} appears in a headline split")
                if fault not in REAL_FAULTS:
                    failures.append(f"{label}/{name}: unknown fault {fault}")
                if start < low or end > high or start > end:
                    failures.append(f"{label}/{name}: fault {fault} range [{start}, {end}] outside [{low}, {high}]")
                if count != end - start + 1:
                    failures.append(f"{label}/{name}: fault {fault} runCount {count} != range size {end - start + 1}")

        # recorded fingerprints
        declared = splits[SPLIT_NAMES[0]].get("sha256")
        if not isinstance(declared, dict):
            failures.append(f"{label}: missing sha256 block on the train split")
        else:
            for name in present:
                expected = key_digest(keys_by_split[name])
                recorded = declared.get(name)
                if recorded != expected:
                    failures.append(f"{label}/{name}: sha256 drift (recorded {recorded}, computed {expected})")
            union = [key for name in present for key in keys_by_split[name]]
            if declared.get("all") != key_digest(union):
                failures.append(f"{label}: sha256 'all' does not match the union of the three splits")

    synthetic_block = doc.get("synthetic") or {}
    if sorted(synthetic_block.get("faults") or []) != SYNTHETIC_FAULTS:
        failures.append(f"synthetic.faults is {synthetic_block.get('faults')!r}, expected {SYNTHETIC_FAULTS}")
    if synthetic_block.get("headline_eligible") is not False:
        failures.append("synthetic.headline_eligible must be false")
    if synthetic_block.get("label_required") != "synthetic":
        failures.append("synthetic.label_required must be 'synthetic'")

    # cross-check the declared scope against the enum module
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from scripts.fault_ids import OPEN_SET_FAULTS, UNOBSERVABLE_FAULTS, validate as validate_ids

        if sorted(declared_scope.get("open_set") or []) != sorted(OPEN_SET_FAULTS):
            failures.append(f"fault_scope.open_set != enum OPEN_SET_FAULTS {OPEN_SET_FAULTS}")
        if sorted(declared_scope.get("unobservable") or []) != sorted(UNOBSERVABLE_FAULTS):
            failures.append(f"fault_scope.unobservable != enum UNOBSERVABLE_FAULTS {UNOBSERVABLE_FAULTS}")
        failures.extend(f"fault_ids: {problem}" for problem in validate_ids())
    except ImportError as exc:  # pragma: no cover - import guard
        failures.append(f"cannot import scripts.fault_ids: {exc}")

    if verify_source:
        failures.extend(_verify_against_source(doc, units, low, high))
    return failures


def _scan_source(path: Path, low: int, high: int) -> Set[Tuple[int, int]]:
    import csv

    pairs: Set[Tuple[int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                pair = (int(row[0]), int(row[1]))
            except ValueError:
                continue
            if low <= pair[1] <= high:
                pairs.add(pair)
    return pairs


def _verify_against_source(doc: Dict, units, low: int, high: int) -> List[str]:
    failures: List[str] = []
    source = (REPO_ROOT / str((doc.get("source") or {}).get("file", ""))).resolve()
    if not source.is_file():
        return [f"source file not found: {source}"]
    print(f"  scanning {source.name} ...", flush=True)
    available = _scan_source(source, low, high)
    seen: Set[Tuple[int, int]] = set()
    for label, splits, _ in units:
        for block in splits.values():
            for key in expand_split(block):
                _, fault, run = key
                seen.add((fault, run))
    missing = sorted(seen - available)
    if missing:
        failures.append(f"{len(missing)} declared keys are absent from {source.name}, e.g. {missing[:5]}")
    return failures


def summarise(doc: Dict) -> None:
    print(f"unique key : {tuple(doc['unique_key'])}")
    src = doc.get("source") or {}
    print(f"source     : {src.get('file')} ({src.get('samplesPerRun')} samples/run, "
          f"onset {src.get('faultOnsetIndex')}, {src.get('runsPerFault')} runs/fault)")
    print()
    for label, splits, constraints in _units(doc):
        print(f"{label}  (fault_disjoint={bool(constraints.get('fault_disjoint_across_splits'))})")
        print(f"  {'split':<10}{'faults':<8}{'runs':<7}sha256[:12]")
        present = [name for name in SPLIT_NAMES + OPTIONAL_SPLITS if name in splits]
        for name in present:
            block = splits[name]
            n_faults = len(block.get("faults") or [])
            n_runs = sum(int(e.get("runCount", 0)) for e in block.get("faults") or [])
            digest = (block.get("sha256") or {}).get(name, "")[:12]
            print(f"  {name:<10}{n_faults:<8}{n_runs:<7}{digest}")
        faults = ", ".join(
            f"{name}: {[e['faultNumber'] for e in splits[name].get('faults') or []]}" for name in present
        )
        print(f"  faults -> {faults}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--verify-source", action="store_true",
                        help="also confirm the runs exist in the source CSV (slow)")
    args = parser.parse_args(argv)

    path = args.splits if args.splits.is_absolute() else (REPO_ROOT / args.splits)
    if not path.is_file():
        print(f"FAIL: splits file not found: {path}")
        return 1
    with path.open(encoding="utf-8") as handle:
        doc = json.load(handle)

    failures = check_document(doc, verify_source=args.verify_source)
    print(f"splits file: {path}\n")
    summarise(doc)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"\n{len(failures)} check(s) failed.")
        return 1
    print("PASS: unique keys, no train/val/test overlap, fault-disjoint where required, "
          "faults 1-20 only, recorded sha256 matches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
