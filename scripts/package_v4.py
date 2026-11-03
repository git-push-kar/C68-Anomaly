"""Package the judging artefacts (3B-13 step 5). Copies only small files, never weights."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FIT = ROOT / "outputs/analysis/phase3a/fit"
EXPORT = ROOT / "outputs/benchmark_v2/eval/prose_pilot"
LIMIT = 5 * 1024 * 1024


def main() -> int:
    meta = json.loads((FIT / "audit_ab_100_v4_meta.json").read_text(encoding="utf-8"))
    mapping = {entry["item_id"]: entry["corpus_item_id"] for entry in meta["item_map"]}
    ids = [row["item_id"] for row in
           csv.DictReader((FIT / "audit_ab_100_v4_key.csv").open(encoding="utf-8"))]
    raw = {row["item_id"]: row for row in
           (json.loads(line) for line in (EXPORT / "raw_pilot_kb.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip())}
    with (FIT / "pilot_sheet_raw_generations.jsonl").open("w", encoding="utf-8") as handle:
        for item_id in ids:
            row = raw.get(mapping[item_id])
            if row is None:
                raise SystemExit("missing raw generation for " + item_id)
            handle.write(json.dumps({"item_id": item_id, "corpus_item_id": mapping[item_id],
                                     "order": "kb",
                                     "gold_fault_id": row["gold_fault_id"],
                                     "gold_verdict": row["gold_verdict"],
                                     "tokens": row["tokens"], "text": row["text"]}) + chr(10))
    wanted = ["audit_ab_100_v4.csv", "audit_ab_100_v4_key.csv", "audit_ab_100_v4_meta.json",
              "pilot_eval_summary.json", "pilot_gate_table.md", "pilot_training_summary.json",
              "pilot_sheet_raw_generations.jsonl"]
    report = {}
    for name in wanted:
        path = FIT / name
        if not path.exists():
            report[name] = "MISSING"
            continue
        size = path.stat().st_size
        if size >= LIMIT:
            raise SystemExit("too large to commit: " + name)
        report[name] = {"bytes": size}
    for name in ("raw_pilot_kb.jsonl", "raw_pilot_shuffled-1.jsonl", "raw_pilot_shuffled-2.jsonl",
                 "raw_pilot_followups.jsonl", "raw_run1_followups.jsonl"):
        path = EXPORT / name
        if path.exists():
            report["export_" + name] = {"bytes": path.stat().st_size,
                                        "under_5MB": path.stat().st_size < LIMIT}
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
