"""One validator failure case per rule, plus the positive controls.

Each test asserts the *specific* code fires, so a rule that stops working cannot
hide behind another one also failing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.response import ResponseRouter, StoreAccess, build_report, validate_followup
from llm.response.validators import number_supported
from llm.response.lookup_answers import suggestions_for
from llm.response.schemas import Intent, TableID

FIXTURES = ROOT / "tests" / "fixtures" / "response"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture()
def event() -> dict:
    return load("classified")


@pytest.fixture()
def report(event) -> dict:
    return {"root_cause": "FAULT_15", "affected_subsystem": "condenser_cooling_system"}


def valid_answer(**overrides) -> dict:
    payload = {
        "intent": Intent.DETECTOR_SCORES.value,
        "answer": "The CVA Q-statistic is above its threshold for this event.",
        "table": TableID.DETECTORS.value,
        "citations": ["detector.dynamic.score", "detector.dynamic.threshold"],
        "suggestions": suggestions_for(Intent.DETECTOR_SCORES),
        "clarifying_question": None,
    }
    payload.update(overrides)
    return payload


def codes(event, report, payload, **kwargs) -> list:
    _, result = validate_followup(payload, event, report, **kwargs)
    return result.codes()


def test_valid_answer_passes(event, report):
    answer, result = validate_followup(valid_answer(), event, report)
    assert result.ok, result.as_dict()
    assert answer.intent is Intent.DETECTOR_SCORES


def test_schema_rule(event, report):
    payload = valid_answer()
    payload.pop("intent")
    assert "schema" in codes(event, report, payload)


def test_answer_length_rule(event, report):
    payload = valid_answer(answer="One. Two. Three. Four. Five.")
    assert "answer_length" in codes(event, report, payload)


def test_too_many_suggestions_rule(event, report):
    payload = valid_answer(suggestions=["What are the detector scores?"] * 4)
    assert "too_many_suggestions" in codes(event, report, payload)


def test_duplicate_suggestion_rule(event, report):
    payload = valid_answer(suggestions=["What are the detector scores?", "what are the detector scores?"])
    found = codes(event, report, payload)
    assert "duplicate_suggestion" in found


def test_suggestion_not_answerable_rule(event, report):
    payload = valid_answer(suggestions=["Tell me a joke about the weather."])
    assert "suggestion_not_answerable" in codes(event, report, payload)


def test_table_not_allowed_rule(event, report):
    payload = valid_answer(intent=Intent.SEVERITY.value, table=TableID.EVIDENCE.value)
    assert "table_not_allowed" in codes(event, report, payload)


def test_unknown_citation_rule(event, report):
    payload = valid_answer(citations=["sensor.XMEAS_999.observed"])
    assert "unknown_citation" in codes(event, report, payload)


def test_unknown_tag_rule(event, report):
    payload = valid_answer(answer="XMV_99 is drifting away from its baseline.")
    assert "unknown_tag" in codes(event, report, payload)


def test_number_not_in_store_rule(event, report):
    payload = valid_answer(answer="The reconstruction error is 987.65, far above normal.")
    assert "number_not_in_store" in codes(event, report, payload)


def test_number_rounded_from_store_is_accepted(event, report):
    store = StoreAccess(event, report)
    score = store.resolve("detector.dynamic.score")
    payload = valid_answer(answer=f"The Q-statistic is {score:.1f}, above its limit.")
    assert "number_not_in_store" not in codes(event, report, payload)


def test_bare_integer_equal_to_a_stored_count_passes(event, report):
    # classified fixture: n_windows=2, top_sensors=5, timeline=6, sensors_total=52
    payload = valid_answer(answer="The event spans 2 windows of the alarm window stream.")
    assert "number_not_in_store" not in codes(event, report, payload)


def test_bare_integer_not_matching_any_count_is_rejected(event, report):
    # The Phase 2A hole: "17 windows" used to pass on a two-window event because
    # every integer 1-22 was accepted unconditionally.
    payload = valid_answer(answer="The event spans 17 windows of the alarm window stream.")
    assert "number_not_in_store" in codes(event, report, payload)


def test_sensor_total_is_still_quotable_as_a_count(event, report):
    payload = valid_answer(answer="The plant exposes 52 continuous process variables in total.")
    assert "number_not_in_store" not in codes(event, report, payload)


def test_tag_index_outside_a_tag_is_not_a_label(event, report):
    bare = valid_answer(answer="The plant records 41 measurement variables in this file.")
    assert "number_not_in_store" in codes(event, report, bare)
    inside = valid_answer(answer="XMEAS_12 is drifting away from its baseline for this event.")
    found = codes(event, report, inside)
    assert "number_not_in_store" not in found
    assert "unknown_tag" not in found


def test_fault_identifier_passes_without_a_store_value(event, report):
    payload = valid_answer(answer="FAULT_15 is the closest match to this lag signature.")
    assert "number_not_in_store" not in codes(event, report, payload)


def test_fault_identifier_list_passes(event, report):
    payload = valid_answer(answer="Faults 4 and 11 both remain plausible given the lag energy.")
    assert "number_not_in_store" not in codes(event, report, payload)


def test_fault_identifier_does_not_license_a_neighbouring_claim(event, report):
    # 15 is an identifier; 17 is a measurement claim about the window count.
    payload = valid_answer(answer="Fault 15 produced 17 alarmed windows in this event.")
    assert "number_not_in_store" in codes(event, report, payload)


def test_enumerated_list_markers_are_not_measurements(event, report):
    payload = valid_answer(answer="1. Compare the detector margins. 2. Rank the sensor contributions.")
    assert "number_not_in_store" not in codes(event, report, payload)


def test_number_supported_needs_a_label_or_a_value():
    assert number_supported(2.0, 0, [], label_integers={2})
    assert not number_supported(2.0, 0, [], label_integers={5})
    assert number_supported(0.6352, 4, [0.63521])
    assert not number_supported(0.6352, 4, [0.6359], label_integers={2})


def test_clarify_not_triggered_rule(event, report):
    payload = valid_answer(clarifying_question="Which subsystem should I inspect first?")
    assert "clarify_not_triggered" in codes(event, report, payload, question="What are the detector scores?")


def test_clarify_reasked_rule():
    event = load("unclassified")
    report = {"root_cause": "UNCLASSIFIED_OPEN_SET"}
    router = ResponseRouter()
    route = router.route(event, report, "What are the detector scores?", with_answer=False)
    assert route.clarification is not None, "fixture should fire the unclassified trigger"
    question = route.clarification.question
    payload = valid_answer(clarifying_question=question)
    found = codes(event, report, payload, router=router, asked=[question], question="What are the detector scores?")
    # The trigger still counts as fired when history suppresses it, and the
    # repeat is reported: history must not be able to fake a missing trigger.
    assert "clarify_not_triggered" not in found
    assert "clarify_reasked" in found
    # without history it is accepted
    _, result = validate_followup(payload, event, report, router=router, question="What are the detector scores?")
    assert "clarify_not_triggered" not in result.codes()
    assert "clarify_reasked" not in result.codes()


def test_clarify_reasked_is_reported_when_history_contains_it():
    event = load("unclassified")
    report = {"root_cause": "UNCLASSIFIED_OPEN_SET"}
    router = ResponseRouter()
    route = router.route(event, report, "What are the detector scores?", with_answer=False)
    payload = valid_answer(clarifying_question=route.clarification.question)
    _, result = validate_followup(
        payload, event, report, router=router, asked=[route.clarification.question], question="What are the detector scores?"
    )
    assert "clarify_reasked" in result.codes()


def test_severity_from_model_rule(event):
    raw = dict(event["report"])
    raw["severity"] = "critical"
    report, result = build_report(raw, event)
    assert "severity_from_model" in result.codes()
    assert report is not None  # the payload still parses; severity is computed regardless
    assert report.severity.components["authoritative_level"] == report.severity.level.value
    assert report.severity.components["authoritative_rule"].endswith("compute_deterministic_severity")


def test_report_from_fixture_is_valid_and_severity_matches_root_cause(event):
    report, result = build_report(event["report"], event)
    assert result.ok, result.as_dict()
    assert report.severity.components["diagnosed_fault"] == "FAULT_15"
    assert report.severity.kb_base.value == "critical"


def test_report_unknown_tag_in_evidence_bullet(event):
    raw = dict(event["report"])
    raw["evidence"] = ["XMV_99 climbed sharply during the event"]
    _, result = build_report(raw, event)
    assert "unknown_tag" in result.codes()
