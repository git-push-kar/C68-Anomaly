#!/usr/bin/env python
"""Write a small *fixture* event set in the Phase 2B record contract.

Why this exists: the follow-up builder must be exercised end to end without the
model, and the Phase 2B event set is not reproducible in this checkout -- the
detector checkpoint here is hidden=32/latent=16 while ``configs/*.yaml`` and
``outputs/anomaly_detector/config.json`` declare 128/64, so
``build_benchmark_events.py`` cannot load the model (see
``docs/reports/phase_2c_report.md`` §2).

The fixtures are therefore **not benchmark data**: each record borrows the
evidence block of one of the three stored response fixtures (a classified
fault-15 valve-sticking episode, an unclassified episode, a normal transient) and
uses a real ``(source_file, fault, run)`` key from ``data/splits_v2.json``, so the
split-overlap and leak checks are meaningful. Every record is marked
``"fixture": true`` and names the evidence it borrowed.

Usage::

    python scripts/make_followup_fixture_events.py --out outputs/benchmark_v2/followup_fixture_events
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures" / "response"
SPLITS = ROOT / "data" / "splits_v2.json"
EVIDENCE = {
    "classified": "classified.json",
    "unclassified": "unclassified.json",
    "normal": "normal_excursion.json",
}


def _runs_for(split: str) -> Dict[int, List[int]]:
    """Run numbers per fault for one Protocol A component, from the split file."""
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    block = ((splits.get("protocols") or {}).get("A_seen_faults_run_disjoint") or {}).get("splits") or {}
    out: Dict[int, List[int]] = {}
    for entry in block.get(split, {}).get("faults") or []:
        number = int(entry["faultNumber"])
        low, high = (int(x) for x in entry["simulationRunRange"])
        out[number] = list(range(low, high + 1))
    return out


def _record(kind: str, split: str, event_id: str, fault, run: int, source: str) -> Dict:
    base = json.loads((FIXTURES / EVIDENCE[kind]).read_text(encoding="utf-8"))
    evidence = copy.deepcopy(base["evidence"])
    evidence["event_id"] = event_id
    event = {
        "event_id": event_id,
        "start_sample": base.get("start_sample", 0),
        "end_sample": base.get("end_sample", 0),
        "n_windows": base.get("n_windows", 1),
        "max_anomaly_score": base.get("max_anomaly_score"),
        "mean_anomaly_score": base.get("mean_anomaly_score"),
        "evidence": evidence,
    }
    return {
        "event_id": event_id,
        "found": True,
        "fault": fault,
        "simulationRun": run,
        "split": split,
        "source_file": source,
        "seed": 42,
        "n_windows": event["n_windows"],
        "event_open_sample": base.get("start_sample", 0),
        "fixture": True,
        "evidence_source": f"tests/fixtures/response/{EVIDENCE[kind]}",
        "label": None if fault is not None else "NORMAL_TRANSIENT",
        "event": event,
    }


def build(out: Path, *, per_split: int = 30, normal_per_split: int = 4) -> Dict:
    faulty_source = "TEP_Faulty_Testing.csv"
    normal_source = "TEP_FaultFree_Testing.csv"
    written = 0
    counts: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        runs = _runs_for(split)
        faulty_dir = out / "faulty" / split
        normal_dir = out / "normal"
        faulty_dir.mkdir(parents=True, exist_ok=True)
        normal_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        # Classified episodes: the valve-sticking evidence, fault 15 (classifiable).
        for run in runs.get(15, [])[:per_split]:
            event_id = f"EVT-{split}-f15-r{run:03d}"
            record = _record("classified", split, event_id, 15, run, faulty_source)
            (faulty_dir / f"f15_r{run:03d}.json").write_text(json.dumps(record), encoding="utf-8")
            n += 1
            written += 1
        # Unclassified episodes: same contract, open-set by protocol (fault 17).
        for run in runs.get(17, [])[:3]:
            event_id = f"EVT-{split}-f17-r{run:03d}"
            record = _record("unclassified", split, event_id, 17, run, faulty_source)
            (faulty_dir / f"f17_r{run:03d}.json").write_text(json.dumps(record), encoding="utf-8")
            n += 1
            written += 1
        # Normal transients.
        for run in runs.get(1, [])[:normal_per_split]:
            event_id = f"NORM-{split}-r{run:03d}"
            record = _record("normal", split, event_id, None, run, normal_source)
            (normal_dir / f"{split}_r{run:03d}.json").write_text(json.dumps(record), encoding="utf-8")
            n += 1
            written += 1
        # Train-only: a fault the builder must exclude from train dialogues.
        if split == "train":
            for run in runs.get(3, [])[:2]:
                event_id = f"EVT-{split}-f03-r{run:03d}"
                record = _record("classified", split, event_id, 3, run, faulty_source)
                (faulty_dir / f"f03_r{run:03d}.json").write_text(json.dumps(record), encoding="utf-8")
                n += 1
                written += 1
        counts[split] = n
    return {"events_per_split": counts, "events_total": written, "fixture": True}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "benchmark_v2" / "followup_fixture_events"))
    ap.add_argument("--per-split", type=int, default=30)
    ap.add_argument("--normal-per-split", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    summary = build(out, per_split=args.per_split, normal_per_split=args.normal_per_split)
    summary["note"] = (
        "FIXTURE EVENTS: evidence borrowed from tests/fixtures/response/*, keys taken "
        "from data/splits_v2.json. Not benchmark data."
    )
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
