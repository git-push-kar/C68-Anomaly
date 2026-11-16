"""The report path with the hybrid wired in and the adapter off.

The point of the flag is that a deployment with no GPU and no adapter can still get a diagnosis. These tests
build a real ``TEPApp`` with ``enable_llm=False``, turn the hybrid on, and run one report through it — first with
detector evidence, then with a pre-instrumentation record the guard must refuse.

The event is a stub rather than a full ``AnomalyEvent``: building one needs the detector pipeline, and this test
is about the report seam, not event construction. The stub carries exactly the attributes ``_fallback_report``
reads, which is the contract the seam depends on.
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from main import TEPApp, _fallback_report  # noqa: E402
from utils import load_config  # noqa: E402

ARTIFACTS = str(ROOT / "outputs/analysis/phase3a/hybrid/vocabfix")


def evidence_dict(with_detector: bool = True) -> dict:
    evidence = {
        "event_id": "ANOM-TEST",
        "anomaly_score": 118.627,
        "severity": "high",
        "candidate_subsystem": "Stripper",
        "candidate_subsystem_score": 0.81,
        "uncertainty": "moderate",
        "reasoning_notes": ["stub event for the report seam"],
        "temporal_sequence": [
            {"display_name": "A_Feed_Stream1", "sensor": "XMEAS_1", "relative_time_minutes": 0.0},
        ],
        "top_anomalous_sensors": [
            {"name": "XMEAS_1", "display_name": "A_Feed_Stream1", "z_score": 18.12,
             "deviation_percent": 223.37, "trend": "decreasing", "contribution": 0.31,
             "index": 0, "error": 1.2, "current_value": 1.0, "baseline_value": 0.8},
            {"name": "XMV_44", "display_name": "A_Feed_Flow", "z_score": 18.10,
             "deviation_percent": 223.21, "trend": "decreasing", "contribution": 0.28,
             "index": 43, "error": 1.1, "current_value": 1.0, "baseline_value": 0.8},
        ],
    }
    if with_detector:
        evidence["detector_evidence"] = {
            "triggered_by": "both",
            "change_type": "mixed",
            "lstm_ae": {"score": 17.5333, "threshold": 0.687, "alarmed": True},
            "dynamic": {"score": 10960.5, "threshold": 194.6, "alarmed": True,
                        "cva": {"score": 10960.5, "threshold": 194.6, "is_anomalous": True,
                                "sensor_contributions": [float(i % 7) for i in range(52)],
                                "dominant_lag": 4, "slow_drift_ratio": 0.51}},
        }
    return evidence


def stub_event(with_detector: bool = True) -> SimpleNamespace:
    payload = evidence_dict(with_detector)
    evidence = SimpleNamespace(**payload)
    # A schema-v1 record has no `detector_evidence` key at all; the attribute is present and None, which is
    # how `_fallback_report` reads it. `to_dict()` is the record itself — the guard must see the missing key.
    evidence.detector_evidence = payload.get("detector_evidence")
    evidence.to_dict = lambda: payload
    return SimpleNamespace(event_id="ANOM-TEST", evidence=evidence,
                           max_anomaly_score=118.627, report=None)


@pytest.fixture(scope="module")
def app(tmp_path_factory) -> TEPApp:
    config = load_config()
    config["events"]["db_path"] = str(tmp_path_factory.mktemp("events") / "events.db")
    config["hybrid"] = {"enabled": True, "artifact_dir": ARTIFACTS}
    return TEPApp(config, enable_llm=False)


def test_the_app_runs_with_the_adapter_off_and_the_hybrid_on(app: TEPApp) -> None:
    assert app.rca is None
    assert app.hybrid is not None
    assert app.hybrid.threshold == pytest.approx(1.0)


def test_report_keeps_its_prose_and_gains_a_verdict(app: TEPApp) -> None:
    report = app._generate_report(stub_event())
    assert set(report) >= {"summary", "root_cause", "severity", "confidence", "evidence"}
    diagnosis = report["diagnosis"]
    assert diagnosis["source"] == "classifier+menu_rule"
    assert diagnosis["verdict"] in {"classified", "unclassified_open_set", "no_fault"}
    assert diagnosis["fault_id"]
    assert 0.0 <= diagnosis["confidence"] <= 1.0


def test_fallback_report_still_renders_against_the_same_stub() -> None:
    """The stub satisfies the prose contract on its own, so the test above is not vacuous."""
    report = _fallback_report(stub_event())
    assert report["summary"].startswith("Anomaly detected")
    assert report["root_cause"]


def test_the_guard_refuses_a_pre_instrumentation_event(app: TEPApp) -> None:
    report = app._generate_report(stub_event(with_detector=False))
    diagnosis = report["diagnosis"]
    assert diagnosis["verdict"] == "unavailable"
    assert diagnosis["reason"] == "evidence_incomplete"
    assert diagnosis["fault_id"] is None
    assert diagnosis["abstained"] is True
    assert "detector_evidence" in diagnosis["detail"]
    # The report is still produced: a refusal is a diagnosis outcome, not an error.
    assert report["summary"]


def test_the_shipped_config_ships_the_flag_off() -> None:
    shipped = load_config()
    assert shipped["hybrid"]["enabled"] is False
    assert shipped["hybrid"]["artifact_dir"].endswith("hybrid/vocabfix")


def test_flag_off_leaves_the_report_unchanged() -> None:
    app = TEPApp.__new__(TEPApp)
    app.config = {}
    app.rca = None
    app.hybrid = None
    report = app._generate_report(stub_event())
    assert "diagnosis" not in report
