"""Export the evaluation items as prompt+label rows, with no answer text.

``train.jsonl`` is a training file: it carries teacher prose, event ids and the
full target. Evaluation must not: scoring needs the prompt, the expected label and
which component the item belongs to, and a reference answer in the same file is how
a reranker ends up "measuring" something it was given.

So each row is exactly ``item_id, system, question, label, variant, component``
plus ``unobservable`` where it applies. ``val.jsonl``/``test.jsonl`` keep their
role as dataset records; these files are the evaluation view of them.

Component assignment comes from ``data/splits_v2.json``: an item is in a component
when its ``(source_file, fault, run)`` key is inside that component's run ranges.
Protocol A ``val``/``test`` are the dataset splits themselves; the Protocol B folds
add ``val``, ``test`` (held-out faults) and ``test_seen`` (kept faults).

Usage::

    python scripts/build_eval_exports.py
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

from llm.benchmark.prompt_v2 import SYSTEM_PROMPT_V2  # noqa: E402

DATASET = ROOT / "outputs" / "benchmark_v2" / "dataset"
SPLITS = ROOT / "data" / "splits_v2.json"
OUT = DATASET


def _key(item: Dict) -> str:
    fault = item.get("fault")
    source = item.get("source_file") or ("TEP_FaultFree_Testing.csv" if fault is None else "")
    return f"{source}|{fault if fault is not None else 'normal'}|{item.get('simulationRun')}"


def _expand(block: Dict) -> set:
    source = Path(str(block.get("source_file") or "")).name
    keys = set()
    for entry in block.get("faults") or []:
        number = int(entry["faultNumber"])
        low, high = (int(x) for x in entry["simulationRunRange"])
        for run in range(low, high + 1):
            keys.add(f"{source}|{number}|{run}")
    return keys


def components() -> Dict[str, set]:
    """component name -> the run keys it covers."""
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    out: Dict[str, set] = {}
    protocol_a = (splits["protocols"]["A_seen_faults_run_disjoint"] or {}).get("splits") or {}
    out["A/val"] = _expand(protocol_a.get("val") or {})
    out["A/test"] = _expand(protocol_a.get("test") or {})
    for fold, block in (splits["protocols"]["B_leave_faults_out"]["folds"] or {}).items():
        short = "fold_1" if fold.startswith("fold_1") else "fold_2"
        for name, value in (block.get("splits") or {}).items():
            out[f"{short}/{name}"] = _expand(value or {})
    return out


def _row(item: Dict, component: str, source_dataset: Optional[str] = None) -> Dict:
    row = {
        "item_id": item.get("item_id"),
        "system": SYSTEM_PROMPT_V2,
        "question": item.get("prompt"),
        "label": (item.get("target") or {}).get("fault_id"),
        "variant": item.get("variant"),
        "component": component,
    }
    if item.get("unobservable"):
        row["unobservable"] = True
    # Only set when a row is scored without being in the dataset split its
    # component names: a fold's held-out fault has test runs in ``test`` but its
    # train-range runs live in the *train* dataset, untouched by that fold.
    if source_dataset:
        row["source_dataset"] = source_dataset
    return row


def fold_held_out_rows(comps: Dict[str, set]) -> Dict[str, List[Dict]]:
    """Held-out faults' train-range items, for the fold's ``test`` component.

    A fold trains on the kept faults (runs 1-300) and is tested on the held-out
    faults' test runs (401-500). The held-out faults' own runs 1-300 are used by
    no fold training set, so they are free evaluation material and roughly
    triple the unseen-fault test n. They are marked ``source_dataset: train`` so
    a reader can tell a train-range item from a test-range one.
    """
    train_path = DATASET / "train.jsonl"
    if not train_path.exists():
        return {}
    train_items = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    out: Dict[str, List[Dict]] = {}
    for fold, block in (json.loads(SPLITS.read_text(encoding="utf-8"))
                        ["protocols"]["B_leave_faults_out"]["folds"] or {}).items():
        short = "fold_1" if fold.startswith("fold_1") else "fold_2"
        test_keys = comps.get(f"{short}/test") or set()
        train_keys = comps.get(f"{short}/train") or set()
        held = {int(key.split("|")[1]) for key in test_keys
                if key.split("|")[1] not in ("normal", "None")}
        if not held:
            continue
        rows: List[Dict] = []
        for item in train_items:
            fault = item.get("fault")
            if fault is None or int(fault) not in held:
                continue
            key = _key(item)
            assert key not in train_keys, f"{short}: held-out item {key} sits in the fold's train set"
            rows.append(_row(item, f"{short}/test", source_dataset="train"))
        out[f"{short}/test"] = rows
    return out


def build() -> Dict:
    comps = components()
    summary: Dict[str, Dict] = {}
    for split, protocol_component in (("val", "A/val"), ("test", "A/test")):
        items = [json.loads(line) for line in
                 (DATASET / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        rows: List[Dict] = []
        per_component: Counter = Counter()
        for item in items:
            key = _key(item)
            member = [name for name, keys in comps.items() if key in keys]
            # The dataset split decides the Protocol A membership; the split file
            # decides the fold ones. A key can be in both (a kept fault's test run
            # is A/test and a fold's test_seen), and it is scored in both.
            if protocol_component not in member:
                member = [protocol_component] + member
            for name in member:
                rows.append(_row(item, name))
                per_component[name] += 1
        path = OUT / f"{split}_eval.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[split] = {
            "dataset_items": len(items),
            "rows": len(rows),
            "by_component": dict(sorted(per_component.items())),
            "protocol_a_rows": per_component.get(protocol_component, 0),
            "matches_dataset": per_component.get(protocol_component, 0) == len(items),
            "path": str(path),
        }

    # Fold ``test`` also carries the held-out faults' train-range items, so it is
    # written after the split files and counted separately.
    extra = fold_held_out_rows(comps)
    if extra:
        test_path = OUT / "test_eval.jsonl"
        rows = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        added: Dict[str, int] = {}
        seen = {(row["item_id"], row["component"]) for row in rows}
        for component, new_rows in extra.items():
            added[component] = 0
            for row in new_rows:
                if (row["item_id"], component) in seen:
                    continue
                rows.append(row)
                seen.add((row["item_id"], component))
                added[component] += 1
        with open(test_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        by_component = Counter(row["component"] for row in rows)
        summary["test"]["rows"] = len(rows)
        summary["test"]["by_component"] = dict(sorted(by_component.items()))
        summary["train_range_fold_test_rows"] = dict(sorted(added.items()))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(OUT / "eval_export_summary.json"))
    args = parser.parse_args()
    summary = build()
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    splits = [summary[name] for name in ("val", "test") if name in summary]
    return 0 if all(part["matches_dataset"] for part in splits) else 1


if __name__ == "__main__":
    raise SystemExit(main())
