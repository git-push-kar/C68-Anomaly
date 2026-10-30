"""The slim target: five model fields, everything else derived in code.

The point of these tests is the equivalence the slimming claim rests on. It is easy
to delete six fields from a target and quietly change what the report says; these
tests render a report from a *slim* answer plus the event and check that the
mitigation it shows is the KB's own text and that the response-layer validators
still accept the result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark.hydrate import (  # noqa: E402
    NORMAL_ID,
    NORMAL_MITIGATION,
    OPEN_SET_MITIGATION,
    SLIM_KEYS,
    hydrate,
    mitigation_for,
    subsystem_for,
    table_for,
)
from llm.benchmark.menu import UNCLASSIFIED  # noqa: E402
from llm.benchmark.targets import build_target  # noqa: E402
from llm.response import StoreAccess, TableID, build_report, render_report  # noqa: E402
from scripts.fault_knowledge import FAULT_KNOWLEDGE  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "response"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _slim(**overrides) -> dict:
    slim = {
        "fault_id": "FAULT_01",
        "verdict": "classified",
        "ranked_hypotheses": ["FAULT_01", "FAULT_06"],
        "physical_reasoning": "The A/C ratio moved before the reactor pressure did.",
        "narrative": "A feed ratio step change is the likely cause. Check the ratio controller.",
    }
    slim.update(overrides)
    return slim


def test_the_slim_schema_is_exactly_five_fields():
    assert SLIM_KEYS == ("fault_id", "verdict", "ranked_hypotheses",
                         "physical_reasoning", "narrative")


def test_derived_fields_come_from_the_kb_not_the_model():
    body = hydrate(_slim(), {"top_anomalous_sensors": []})
    kb = FAULT_KNOWLEDGE[1]
    assert body.affected_subsystem == kb["subsystem"]
    assert body.recommended_action == (kb.get("recommended_action") or kb.get("action")).strip()


def test_a_report_rendered_from_a_slim_answer_carries_the_kb_mitigation():
    """The equivalence the slimming has to keep: same KB text, still valid."""
    event = fixture("classified")
    body = hydrate(_slim(), event["evidence"])
    report, result = build_report(
        {**body.model_dump(), "event_id": event.get("event_id", "EVT-x")},
        event,
    )
    assert report is not None, [issue.message for issue in result.issues]
    doc = render_report(report, StoreAccess(event))
    action = FAULT_KNOWLEDGE[1].get("recommended_action") or FAULT_KNOWLEDGE[1].get("action")
    assert action.strip() in doc.markdown


@pytest.mark.parametrize("label", ["FAULT_01", "FAULT_05", "FAULT_13", "FAULT_20"])
def test_mitigation_matches_what_the_full_target_carried(label):
    record = {"fault": int(label.split("_")[-1]), "event_id": "EVT-x", "split": "train"}
    target = build_target(record)
    assert mitigation_for(target["fault_id"]) == target["mitigation"]


def test_the_abstention_mitigations_are_single_sourced():
    normal_record = {"fault": None, "event_id": "EVT-n", "label": NORMAL_ID}
    assert build_target(normal_record)["mitigation"] == NORMAL_MITIGATION
    open_record = {"fault": 17, "event_id": "EVT-o", "split": "train"}
    target = build_target(open_record)
    assert target["fault_id"] == UNCLASSIFIED
    assert target["mitigation"] == OPEN_SET_MITIGATION


def test_the_table_choice_is_one_row_per_verdict():
    assert table_for(NORMAL_ID) is TableID.NONE
    assert table_for(UNCLASSIFIED) is TableID.EVIDENCE
    assert table_for("FAULT_09") is TableID.EVIDENCE


def test_a_missing_field_is_an_error_not_a_default():
    with pytest.raises(ValueError):
        hydrate(_slim(narrative=""), {"top_anomalous_sensors": []})


def test_a_label_outside_the_enum_is_refused():
    with pytest.raises(ValueError):
        hydrate(_slim(fault_id="FAULT_99"), {"top_anomalous_sensors": []})


def test_the_root_cause_is_not_also_a_hypothesis():
    body = hydrate(_slim(ranked_hypotheses=["FAULT_01", "FAULT_06"]),
                   {"top_anomalous_sensors": []})
    assert [h.fault.value for h in body.hypotheses] == ["FAULT_06"]


def test_no_model_confidence_reaches_the_hypotheses():
    body = hydrate(_slim(), {"top_anomalous_sensors": []})
    assert all(h.confidence == 0.0 for h in body.hypotheses)


def test_evidence_bullets_are_built_from_the_event():
    event = fixture("classified")["evidence"]
    body = hydrate(_slim(), event)
    tags = [s.get("name") for s in event.get("top_anomalous_sensors") or []][:5]
    assert len(body.evidence) == len([t for t in tags if t])
    assert any(str(t).split("_")[0] in bullet for t, bullet in zip(tags, body.evidence))


def test_subsystem_for_an_abstention_is_unclassified():
    assert subsystem_for(UNCLASSIFIED) is None
    assert subsystem_for(NORMAL_ID) is None
    assert subsystem_for("FAULT_15") == FAULT_KNOWLEDGE[15]["subsystem"]
