"""Golden renderer tests: three fixtures, table fidelity, one follow-up.

Fidelity is asserted twice per table cell: the stored *raw* value must be exactly
what the citation key resolves to, and the printed *cell* must be that value run
through the renderer's formatter. Model wording never enters a table, so this is
enough to prove no cell can be fabricated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.response import (
    FollowupAnswer,
    Intent,
    ResponseRouter,
    StoreAccess,
    TableID,
    build_report,
    lookup_answer,
    render_followup,
    render_report,
)
from llm.response.renderer import format_cell, fmt
from llm.response.schemas import Report

FIXTURES = ROOT / "tests" / "fixtures" / "response"
NAMES = ["classified", "unclassified", "normal_excursion"]


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(params=NAMES)
def fixture(request) -> dict:
    return load(request.param)


def _report_for(event: dict):
    report, result = build_report(event["report"], event)
    assert report is not None, [i.message for i in result.issues]
    return report


def test_report_renders_with_severity_and_prose(fixture):
    report = _report_for(fixture)
    doc = render_report(report, StoreAccess(fixture))
    assert doc.kind == "report"
    assert report.event_id in doc.markdown
    assert report.severity.level.value.capitalize() in doc.markdown
    assert report.summary.split(".")[0][:40] in doc.markdown
    assert "#" not in doc.markdown.split("\n")[0] or doc.markdown.startswith("## ")
    assert doc.ui["meta"]["severity"] == report.severity.level.value


def test_every_table_cell_traces_to_the_store(fixture):
    report = _report_for(fixture)
    store = StoreAccess(fixture)
    doc = render_report(report, store)
    assert doc.tables, "expected at least the detector table"

    checked = 0
    for table in doc.tables:
        assert len(table.rows) == len(table.keys) == len(table.raw)
        # the rendered cell must be the declared format applied to the stored value
        assert table.rows == table.expected_rows(), f"{table.id.value}: formatted rows drifted from raw"
        for key, raw, cell, spec in table.data_cells():
            assert raw == store.resolve(key), f"{table.id.value}:{key} raw mismatch"
            assert cell == format_cell(raw, spec), f"{table.id.value}:{key} cell mismatch"
            checked += 1
    assert checked >= 6


def test_detector_table_matches_store_numbers(fixture):
    report = _report_for(fixture)
    store = StoreAccess(fixture)
    doc = render_report(report, store)
    table = next(t for t in doc.tables if t.id is TableID.DETECTORS)
    row = next(r for r in table.rows if r[0].startswith("LSTM"))
    assert row[1] == f"{store.resolve('detector.lstm.score'):.4f}"
    assert row[2] == f"{store.resolve('detector.lstm.threshold'):.2f}"
    assert row[4] == ("ALARMED" if store.resolve("detector.lstm.alarmed") else "below threshold")
    # the ratio column is derived from the two stored numbers, not stored separately
    assert row[3] == f"{store.resolve('detector.lstm.ratio'):.2f}x"


def test_severity_is_computed_not_read_from_the_fixture(fixture):
    report = _report_for(fixture)
    assert report.severity.rule_version == "severity-crosscheck-v2"
    assert report.severity.components["authoritative_rule"].endswith("compute_deterministic_severity")
    assert report.severity.components["diagnosed_fault"] == report.root_cause.value
    assert "severity" not in fixture["report"], "fixtures must not carry a model-authored severity"


def test_synthetic_label_is_shown_when_applicable():
    event = load("classified")
    event["report"] = dict(event["report"], root_cause="SYNTHETIC_FAULT_21", hypotheses=[])
    report = _report_for(event)
    doc = render_report(report, StoreAccess(event))
    assert "(synthetic)" in doc.markdown
    assert doc.ui["meta"]["synthetic"] is True


def test_confidence_is_rendered_uncalibrated_until_a_calibrator_exists(fixture):
    report = _report_for(fixture)
    assert report.confidence is None, "no calibrator is fitted, so there is no value"
    doc = render_report(report, StoreAccess(fixture))
    assert "uncalibrated" in doc.markdown
    assert "Confidence:** 0." not in doc.markdown


def test_a_confidence_value_without_a_calibrator_is_rejected(fixture):
    report = _report_for(fixture)
    body = report.model_dump(exclude={"confidence", "confidence_calibrated",
                                      "confidence_source", "severity", "event_id",
                                      "table", "citations", "source"})
    with pytest.raises(ValueError):
        Report(**body, event_id="X", severity=report.severity,
               confidence=0.71, confidence_calibrated=False)


def test_a_calibrated_value_names_its_calibrator(fixture):
    report = _report_for(fixture)
    body = report.model_dump(exclude={"confidence", "confidence_calibrated",
                                      "confidence_source", "severity", "event_id",
                                      "table", "citations", "source"})
    calibrated = Report(**body, event_id="X", severity=report.severity,
                        confidence=0.71, confidence_calibrated=True,
                        confidence_source="isotonic-val")
    doc = render_report(calibrated, StoreAccess(fixture))
    assert "0.71 (isotonic-val)" in doc.markdown


def test_model_authored_confidence_is_a_schema_error():
    event = load("classified")
    event["report"] = dict(event["report"], confidence=0.71)
    report, result = build_report(event["report"], event)
    assert report is None, "confidence is not a model field"
    assert any(issue.code == "schema" for issue in result.issues)


def test_followup_renderer_uses_the_same_tables():
    event = load("classified")
    report = {"root_cause": "FAULT_15"}
    store = StoreAccess(event, report)
    answer = FollowupAnswer(
        intent=Intent.SENSOR_VALUES,
        answer="Two sensors dominate the excursion.",
        table=TableID.EVIDENCE,
        citations=["sensor.XMV_52.observed"] if "sensor.XMV_52.observed" in store.known_tags() else [],
        suggestions=["What are the detector scores?"],
    )
    doc = render_followup(answer, store)
    assert doc.kind == "followup"
    assert doc.tables and doc.tables[0].id is TableID.EVIDENCE
    table = doc.tables[0]
    assert table.rows == table.expected_rows()
    for key, raw, cell, spec in table.data_cells():
        assert raw == store.resolve(key)
        assert cell == format_cell(raw, spec)


def test_lookup_answer_renders_and_stays_valid():
    event = load("classified")
    report = {"root_cause": "FAULT_15"}
    router = ResponseRouter()
    hit = lookup_answer(event, report, "What are the detector scores?", router=router)
    assert hit is not None
    doc = render_followup(hit.answer, StoreAccess(event, report), {"fault": "FAULT_15"})
    assert "detector" in doc.markdown.lower() or "score" in doc.markdown.lower()
    assert doc.citations
    for key in doc.citations:
        StoreAccess(event, report).resolve(key)  # must not raise
