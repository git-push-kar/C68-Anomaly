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


def _row(item: Dict, component: str) -> Dict:
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
    return row


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
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(OUT / "eval_export_summary.json"))
    args = parser.parse_args()
    summary = build()
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if all(part["matches_dataset"] for part in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
