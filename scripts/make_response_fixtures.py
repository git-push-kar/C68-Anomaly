"""Build the three response-layer fixtures from stored events.

Fixtures are copied from `outputs/events.db` so the renderer is tested on real
evidence shapes, not invented ones. Only ``report_json`` is rewritten (the old
free-text ``root_cause`` becomes a benchmark label) — evidence is untouched.

    python scripts/make_response_fixtures.py

Writes tests/fixtures/response/{classified,unclassified,normal_excursion}.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB = ROOT / "outputs" / "events.db"
OUT = ROOT / "tests" / "fixtures" / "response"


def _event(db: sqlite3.Connection, where: str, params: tuple = ()) -> dict:
    db.row_factory = sqlite3.Row
    row = db.execute(f"select * from events where {where} order by rowid desc limit 1", params).fetchone()
    if row is None:
        raise SystemExit(f"no event matches {where}")
    event = json.loads(row["event_json"])
    event["report"] = json.loads(row["report_json"] or "{}")
    event["fault_label"] = row["fault_label"]
    return event


def _report(fault: str, subsystem: str, hypotheses: list) -> dict:
    return {
        "root_cause": fault,
        "affected_subsystem": subsystem,
        "summary": "A sustained deviation was detected in the cooling-water loop. "
                   "The pattern is consistent with a sticking valve actuator rather than a sensor failure.",
        "evidence": [
            "CVA Q-statistic stayed above its threshold for the whole event window",
            "The condenser cooling-water flow sensor is the largest single contributor",
        ],
        "reasoning": "Reduced cooling-water flow raises separator pressure and temperature together, "
                     "which is a flow-side disturbance rather than a measurement fault.",
        "reasoning_steps": "1. Compare detector margins. 2. Rank sensor contributions. 3. Match the KB mechanism.",
        "recommended_action": "Inspect the condenser cooling-water valve actuator and verify the positioner response.",
        "uncertainty": "A fouled condenser would produce a similar signature; on-site verification is required.",
        # No report-level ``confidence``: it is not a model field. The renderer
        # prints "uncalibrated" until a calibrator is fitted.
        "hypotheses": hypotheses,
        "table": "detectors",
        "citations": ["detector.dynamic.score", "detector.dynamic.threshold", "subsystem.candidate"],
    }


def main() -> None:
    if not DB.exists():
        raise SystemExit(f"event store not found: {DB}")
    OUT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB)

    classified = _event(db, "fault_label = ?", (15,))
    classified["report"] = _report(
        "FAULT_15",
        "condenser_cooling_system",
        [
            {"fault": "FAULT_15", "confidence": 0.71, "rationale": "Valve stiction matches the lag profile.", "supporting_tags": ["XMV_52", "XMEAS_22"]},
            {"fault": "FAULT_14", "confidence": 0.22, "rationale": "Reactor cooling stiction also shows lag energy.", "supporting_tags": ["XMV_51"]},
        ],
    )
    (OUT / "classified.json").write_text(json.dumps(classified, indent=1), encoding="utf-8")

    # distinct from the normal-excursion fixture: a fused (both-detector) event
    unclassified = _event(
        db,
        "fault_label is null and json_extract(evidence_json, '$.detector_evidence.triggered_by') = 'both'",
        (),
    )
    unclassified["report"] = _report(
        "UNCLASSIFIED_OPEN_SET",
        "condenser_cooling_system",
        [
            {"fault": "UNCLASSIFIED_OPEN_SET", "confidence": 0.55, "rationale": "The lag signature does not match a known family.", "supporting_tags": ["XMV_52"]},
        ],
    )
    (OUT / "unclassified.json").write_text(json.dumps(unclassified, indent=1), encoding="utf-8")

    normal = _event(
        db,
        "fault_label is null and json_extract(evidence_json, '$.detector_evidence.triggered_by') = 'dynamic'",
        (),
    )
    normal["report"] = _report(
        "NORMAL_TRANSIENT",
        "condenser_cooling_system",
        [],
    )
    (OUT / "normal_excursion.json").write_text(json.dumps(normal, indent=1), encoding="utf-8")

    for name in ("classified", "unclassified", "normal_excursion"):
        data = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
        print(name, data.get("event_id"), "sensors", len(data["evidence"].get("top_anomalous_sensors", [])))


if __name__ == "__main__":
    main()
