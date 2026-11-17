"""Run provenance on the event record.

An event that cannot name its simulation run cannot be scored, so the three fields are checked end to end: on the
dataclass, through ``build_event``'s close path, and in SQLite both for a fresh database and for one created
before the columns existed.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from evidence.event_builder import AnomalyEvent, EventEvidence
from events.event_store import EventStore

PROVENANCE = ("TEP_Faulty_Testing.csv", 123, 15)


def _event(**overrides) -> AnomalyEvent:
    evidence = EventEvidence(
        event_id="ANOM-0001",
        anomaly_score=3.0,
        severity="high",
        pre_anomaly_context={},
        top_anomalous_sensors=[],
        temporal_sequence=[],
        candidate_subsystem="feed",
        candidate_subsystem_score=0.5,
        reasoning_notes=[],
    )
    kwargs = dict(
        event_id="ANOM-0001",
        start_time=datetime(2026, 1, 1),
        detection_time=datetime(2026, 1, 1),
        end_time=None,
        start_sample=580,
        end_sample=910,
        n_windows=66,
        max_anomaly_score=3.0,
        mean_anomaly_score=2.0,
        evidence=evidence,
        source_file=PROVENANCE[0],
        simulation_run=PROVENANCE[1],
        fault_number=PROVENANCE[2],
    )
    kwargs.update(overrides)
    return AnomalyEvent(**kwargs)


def test_to_dict_carries_provenance() -> None:
    data = _event().to_dict()
    assert (data["source_file"], data["simulation_run"], data["fault_number"]) == PROVENANCE


def test_provenance_is_optional() -> None:
    data = _event(source_file=None, simulation_run=None, fault_number=None).to_dict()
    assert data["source_file"] is None
    assert data["simulation_run"] is None
    assert data["fault_number"] is None


def test_store_persists_provenance_as_columns_and_json(tmp_path) -> None:
    db = tmp_path / "events.db"
    store = EventStore(str(db))
    store.store_event(_event())
    store.close()

    con = sqlite3.connect(db)
    try:
        row = con.execute(
            "SELECT source_file, simulation_run, fault_number FROM events"
        ).fetchone()
        payload = con.execute("SELECT event_json FROM events").fetchone()[0]
    finally:
        con.close()

    assert row == PROVENANCE
    assert json.loads(payload)["fault_number"] == PROVENANCE[2]


def test_database_created_before_provenance_gains_the_columns(tmp_path) -> None:
    db = tmp_path / "legacy.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY, start_time TEXT, detection_time TEXT, end_time TEXT,
            max_score REAL, mean_score REAL, severity TEXT, fault_label INTEGER,
            event_json TEXT, report_json TEXT, evidence_json TEXT, created_at TEXT
        );
        """
    )
    con.commit()
    con.close()

    store = EventStore(str(db))
    store.store_event(_event())
    store.close()

    con = sqlite3.connect(db)
    try:
        assert con.execute("SELECT fault_number FROM events").fetchone()[0] == PROVENANCE[2]
    finally:
        con.close()


def test_close_event_stamps_the_latest_known_run(monkeypatch) -> None:
    import main as main_mod

    captured: dict = {}

    class FakeEvent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.report = None
            self.event_id = kwargs["event_id"]
            self.max_anomaly_score = 1.2
            self.evidence = SimpleNamespace(severity="high")

        def to_dict(self) -> dict:
            return dict(self.kwargs, report=self.report)

    def fake_build(**kwargs):
        captured.update(kwargs)
        return FakeEvent(**kwargs)

    monkeypatch.setattr(main_mod, "build_event", fake_build)

    app = main_mod.TEPApp.__new__(main_mod.TEPApp)
    app.config = {"windowing": {"stride": 5, "window_size": 60}, "evidence": {}}
    app._max_event_windows = 200
    # The event opened on the fault-free replay, then injection switched the source.
    app._provenance = {"source_file": "TEP_Faulty_Testing.csv", "simulation_run": 7,
                       "fault_number": 15}
    app.detector = SimpleNamespace(threshold=SimpleNamespace(threshold=1.0))
    app.scaler = SimpleNamespace(baseline=SimpleNamespace(feature_names=["x"]))
    app.event_store = SimpleNamespace(next_event_id=lambda: "ANOM-0001",
                                      store_event=lambda event: "ANOM-0001")
    app._generate_report = lambda event: {"diagnosis": None}
    app._open = {
        "scores": [1.2],
        "errors": [np.zeros(1, dtype=np.float32)],
        "windows": [np.zeros((60, 1), dtype=np.float32)],
        "detector_evidences": [],
        "start_sample": 580,
        "start_time": datetime(2026, 1, 1),
        "fault_label": 15,
        "provenance": {"source_file": "TEP_FaultFree_Training.csv", "simulation_run": None,
                       "fault_number": None},
        "normal_since": 0,
    }

    app._close_event()

    assert captured["source_file"] == "TEP_Faulty_Testing.csv"
    assert captured["simulation_run"] == 7
    assert captured["fault_number"] == 15
