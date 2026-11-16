"""Run the wired report path on a stored event, end to end.

Loads one real ``AnomalyEvent`` out of ``events.db``, reconstructs it, builds ``TEPApp`` with ``enable_llm=False``
and the hybrid flag on, and prints what ``_generate_report`` actually produced: the diagnosis verdict and the
prose that accompanies it.

    python scripts/run_live_event.py                 # newest event that carries detector evidence
    python scripts/run_live_event.py --event-id ...  # a specific event
    python scripts/run_live_event.py --legacy        # newest schema-v1 event, to see the guard refuse it

Nothing is written back to the database.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence.event_builder import AnomalyEvent, EventEvidence  # noqa: E402
from main import TEPApp  # noqa: E402
from utils import load_config  # noqa: E402

DEFAULT_DB = ROOT / "outputs/events.db"
DEFAULT_ARTIFACTS = ROOT / "outputs/analysis/phase3a/hybrid/vocabfix"


def pick_event(db: Path, event_id: str | None, legacy: bool) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT event_id, event_json FROM events ORDER BY rowid DESC").fetchall()
    finally:
        con.close()
    for stored_id, payload in rows:
        record = json.loads(payload)
        has_detector = (record.get("evidence") or {}).get("detector_evidence") is not None
        if event_id and stored_id != event_id:
            continue
        if event_id or (has_detector != legacy):
            return record
    raise SystemExit("no matching event in the database")


def reconstruct(record: dict) -> AnomalyEvent:
    payload = record["evidence"]
    evidence = EventEvidence(
        event_id=payload["event_id"],
        anomaly_score=payload["anomaly_score"],
        severity=payload["severity"],
        pre_anomaly_context=payload.get("pre_anomaly_context") or {},
        top_anomalous_sensors=payload.get("top_anomalous_sensors") or [],
        temporal_sequence=payload.get("temporal_sequence") or [],
        candidate_subsystem=payload.get("candidate_subsystem") or "",
        candidate_subsystem_score=payload.get("candidate_subsystem_score") or 0.0,
        reasoning_notes=payload.get("reasoning_notes") or [],
        evidence_type=payload.get("evidence_type", "model_derived_evidence"),
        uncertainty=payload.get("uncertainty", ""),
        detector_evidence=payload.get("detector_evidence"),
    )
    return AnomalyEvent(
        event_id=record["event_id"],
        start_time=datetime.fromisoformat(record["start_time"]),
        detection_time=datetime.fromisoformat(record["detection_time"]),
        end_time=datetime.fromisoformat(record["end_time"]) if record.get("end_time") else None,
        start_sample=record.get("start_sample", 0),
        end_sample=record.get("end_sample"),
        n_windows=record.get("n_windows", 0),
        max_anomaly_score=record.get("max_anomaly_score", 0.0),
        mean_anomaly_score=record.get("mean_anomaly_score", 0.0),
        evidence=evidence,
        sensor_time_series=record.get("sensor_time_series"),
        report=record.get("report"),
        fault_label=record.get("fault_label"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--event-id", default=None)
    parser.add_argument("--legacy", action="store_true", help="pick a schema-v1 event (guard refusal)")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()

    record = pick_event(args.db, args.event_id, args.legacy)
    event = reconstruct(record)

    config = load_config()
    config["hybrid"] = {"enabled": True, "artifact_dir": str(args.artifacts)}
    app = TEPApp(config, enable_llm=False)
    assert app.rca is None, "this run is about the adapter-off path"

    report = app._generate_report(event)
    diagnosis = report.get("diagnosis") or {}

    print(f"event            {event.event_id}")
    print(f"fault_label      {event.fault_label}")
    print(f"detector scores  {sorted((event.evidence.detector_evidence or {}).keys())}")
    print(f"verdict          {diagnosis.get('verdict')}")
    print(f"reason           {diagnosis.get('reason')}")
    print(f"fault_id         {diagnosis.get('fault_id')}")
    print(f"predicted_class  {diagnosis.get('predicted_class')}")
    print(f"confidence       {diagnosis.get('confidence')}")
    print(f"threshold        {diagnosis.get('threshold')}")
    print(f"artifacts        {diagnosis.get('artifacts')}")
    print(f"root_cause       {report.get('root_cause')}")
    print(f"summary          {report.get('summary')}")
    print(f"report keys      {sorted(report)}")


if __name__ == "__main__":
    main()
