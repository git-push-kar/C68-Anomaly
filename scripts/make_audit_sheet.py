#!/usr/bin/env python
"""Build the human-audit sheet from accepted teacher outputs.

Sheet v1 (``audit_sample.csv``) sampled items from the dataset and left the human
columns blank; it contains no teacher text at all, so an auditor could not see
what the teacher wrote. This script samples only items whose prose was accepted,
and puts that prose on the row next to a short evidence summary.

Columns are what an auditor needs to judge one row without opening the dataset:

    item id (opaque)  family  evidence summary  menu variant  target label
    teacher reasoning  narrative  correct  grounded  direction_consistent
    useful  notes

The last five are deliberately blank; this script does not score anything.

Usage:
    python scripts/make_audit_sheet.py                     # 100 rows
    python scripts/make_audit_sheet.py --rows 200 --out .../audit_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark.prompt_v2 import opaque_event_id  # noqa: E402


def load_items(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_events(events_dir: Path) -> Dict[str, Dict]:
    """Event records by event id, for the evidence summary."""
    out: Dict[str, Dict] = {}
    for path in sorted((events_dir / "faulty").glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        out[record.get("event_id")] = record
    for path in sorted((events_dir / "normal").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        out[record.get("event_id")] = record
    return out


def _fmt(value: object, decimals: int = 2) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def evidence_summary(record: Dict) -> str:
    """One line of the numbers an auditor needs to judge grounding."""
    if not record:
        return ""
    event = record.get("event") or {}
    evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else event
    det = evidence.get("detector_evidence") or {}
    parts: List[str] = []
    lstm = det.get("lstm_ae") or {}
    if "score" in lstm:
        thr = float(lstm.get("threshold") or 0.0)
        ratio = float(lstm.get("score") or 0.0) / thr if thr else 0.0
        parts.append(f"LSTM {_fmt(lstm.get('score'), 4)}/{_fmt(thr, 3)} ({_fmt(ratio)}x)")
    dyn = det.get("dynamic") or {}
    if "score" in dyn:
        thr = float(dyn.get("threshold") or 0.0)
        ratio = float(dyn.get("score") or 0.0) / thr if thr else 0.0
        parts.append(f"CVA {_fmt(dyn.get('score'), 1)}/{_fmt(thr, 1)} ({_fmt(ratio)}x)")
    sensors = []
    for sensor in (evidence.get("top_anomalous_sensors") or [])[:3]:
        if isinstance(sensor, dict):
            sensors.append(
                f"{sensor.get('display_name') or sensor.get('name')} "
                f"z={_fmt(sensor.get('z_score'))} dev={_fmt(sensor.get('deviation_percent'))}%"
            )
    if sensors:
        parts.append("top: " + "; ".join(sensors))
    lag = (det.get("lag_attribution") or {}).get("top_variables") or []
    lag_bits = [f"{v.get('variable')} {_fmt(v.get('share'), 3)}" for v in lag[:2] if isinstance(v, dict)]
    if lag_bits:
        parts.append("lag: " + ", ".join(lag_bits))
    return " | ".join(parts)


def family_of(item: Dict) -> str:
    fault = item.get("fault")
    if fault is None:
        return "normal_transient"
    if item.get("variant") == "removed":
        return "open_set"
    return f"FAULT_{int(fault):02d}"


def stratified(items: List[Dict], rows: int, seed: int) -> List[Dict]:
    """Round-robin across families, so rare families are represented."""
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for item in items:
        buckets[family_of(item)].append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda i: i["item_id"])
        rng.shuffle(bucket)
    order: List[Dict] = []
    index = 0
    while len(order) < min(rows, len(items)):
        progressed = False
        for key in sorted(buckets):
            bucket = buckets[key]
            if index < len(bucket) and len(order) < rows:
                order.append(bucket[index])
                progressed = True
        if not progressed:
            break
        index += 1
    return order


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="outputs/benchmark_v2/dataset")
    ap.add_argument("--events", default="outputs/benchmark_v2/events")
    ap.add_argument("--out", default="outputs/benchmark_v2/dataset/audit_v2.csv")
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    items: List[Dict] = []
    for split in ("train", "val", "test"):
        path = dataset / f"{split}.jsonl"
        if path.exists():
            items.extend(load_items(path))
    events = load_events(Path(args.events))

    accepted = [item for item in items
                if (item.get("teacher") or {}).get("accepted") and item.get("prose")]
    if not accepted:
        raise SystemExit("no accepted teacher prose in this dataset; run the teacher first")

    chosen = stratified(accepted, args.rows, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["item_id", "family", "evidence_summary", "menu_variant", "target_label",
               "teacher_reasoning", "teacher_narrative", "correct", "grounded",
               "direction_consistent", "useful", "notes"]
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for item in chosen:
            prose = item.get("prose") or {}
            writer.writerow([
                opaque_event_id(item.get("event_id")),
                family_of(item),
                evidence_summary(events.get(item.get("event_id"))),
                item.get("variant"),
                (item.get("target") or {}).get("fault_id"),
                prose.get("physical_reasoning", ""),
                prose.get("narrative", ""),
                "", "", "", "", "",
            ])

    families = Counter(family_of(item) for item in chosen)
    print(f"accepted items available: {len(accepted)} of {len(items)}")
    print(f"wrote {len(chosen)} rows -> {out_path}")
    print("families:", dict(sorted(families.items())))
    for required in ("open_set", "normal_transient"):
        print(f"  includes {required}: {families.get(required, 0)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
