"""Scaffolding rejection, the XMV direction scope and the teacher-only menu.

Three rules that exist to stop prose from describing the prompt rather than the
plant, or from asserting a flow direction the evidence never measured.
"""
from __future__ import annotations

import pytest

from llm.benchmark.filters import check_answer
from llm.benchmark.scaffolding import derived_phrases, scaffolding_hits
from llm.benchmark.teacher import teacher_menu
from llm.response.allowed_set import build_allowed_set


# -- derivation ----------------------------------------------------------------


def test_phrases_are_read_from_the_template_not_typed():
    phrases = {entry["phrase"] for entry in derived_phrases()}
    assert "decided diagnosis" in phrases
    assert "the disturbance" in phrases
    assert "physical_reasoning" in phrases
    # A quoted English word in the system prompt is not a field name.
    assert "flow" not in phrases


def test_menu_references_come_from_the_menu_block():
    kinds = {entry["phrase"]: entry["kind"] for entry in derived_phrases()}
    assert kinds.get("candidate menu") == "menu_reference"
    assert kinds.get("listed candidate") == "menu_reference"


# -- matching ------------------------------------------------------------------


def test_decided_diagnosis_anywhere_is_scaffolding():
    hits = scaffolding_hits("The decided diagnosis FAULT_04 is supported by the flow rise.")
    assert [hit["phrase"] for hit in hits] == ["decided diagnosis"]


def test_natural_disturbance_wording_is_not_scaffolding():
    assert scaffolding_hits("The disturbance is a sticking valve in the cooling loop.") == []


def test_label_used_as_a_label_is_scaffolding():
    hits = scaffolding_hits("affected area: feed system. The feed ratio changed.")
    assert any(hit["phrase"] == "affected area" for hit in hits)


def test_menu_reference_is_scaffolding():
    hits = scaffolding_hits("The candidate menu does not contain this signature.")
    assert any(hit["phrase"] == "candidate menu" for hit in hits)


def test_abstention_may_name_listed_candidates():
    text = "No listed candidate fits; FAULT_05 and FAULT_12 are ruled out."
    assert scaffolding_hits(text, abstain=False)
    assert scaffolding_hits(text, abstain=True) == []


# -- filter integration --------------------------------------------------------

EVIDENCE = {
    "detector_evidence": {"lstm_ae": {"score": 2.0, "threshold": 0.7, "alarmed": True}},
    "top_anomalous_sensors": [
        {"name": "XMV_45", "display_name": "A_C_Feed_Flow", "z_score": 12.0,
         "deviation_percent": 24.0, "trend": "increasing"},
        {"name": "XMEAS_7", "display_name": "Reactor_Pressure", "z_score": 8.0,
         "deviation_percent": 2.0, "trend": "increasing"},
    ],
    "temporal_sequence": [],
}


def _target(number: int, fault_id: str = "FAULT_07") -> dict:
    return {"fault_id": fault_id, "truth_fault_number": number,
            "ranked_hypotheses": [{"fault_id": fault_id}]}


def _answer(text: str) -> dict:
    return {"physical_reasoning": text, "narrative": "Check the valve on site."}


def test_scaffolding_is_a_rejection_code():
    allowed = build_allowed_set(EVIDENCE, extra_faults=[7])
    codes = check_answer(_answer("The decided diagnosis FAULT_07 fits the evidence."),
                         _target(7), EVIDENCE, allowed)
    assert any(code.startswith("teacher_scaffolding:") for code in codes)


def test_valve_position_direction_is_not_a_conflict():
    """The KB says XMV_45 decreases; the evidence says increasing; both are true."""
    allowed = build_allowed_set(EVIDENCE, extra_faults=[7])
    codes = check_answer(_answer("A_C_Feed_Flow [XMV_45] is increasing at onset."),
                         _target(7), EVIDENCE, allowed)
    assert not any(code.startswith("direction_conflict") for code in codes)


def test_measurement_direction_is_still_a_conflict():
    """Fault 13's KB entry says decrease, so an increase on its tag is a conflict.

    Note the vocabulary gap this test avoids: ``INCREASE_WORDS`` has "increase",
    "increases" and "higher" but not the gerund "increasing", so only some
    phrasings are caught. Extending it now would reject every fault-13 answer that
    correctly describes the rising pressure (76 of 118 accepted), which is a KB
    error, not a prose error -- see the fault-13 proposal in the report.
    """
    allowed = build_allowed_set(EVIDENCE, extra_faults=[13])
    target = _target(13, "FAULT_13")
    conflicting = check_answer(_answer("Reactor_Pressure [XMEAS_7] increases first."),
                               target, EVIDENCE, allowed)
    agreeing = check_answer(_answer("Reactor_Pressure [XMEAS_7] decreases first."),
                            target, EVIDENCE, allowed)
    assert any(code.startswith("direction_conflict") for code in conflicting)
    assert not any(code.startswith("direction_conflict") for code in agreeing)


# -- teacher menu --------------------------------------------------------------

MENU = [
    {"fault_id": "FAULT_07", "signature": "C header pressure loss — A_C_Feed_Flow decreases first"},
    {"fault_id": "FAULT_13", "signature": "Reaction kinetics slow drift — "
                                          "Reactor_Pressure decreases first"},
]


def test_teacher_menu_replaces_a_valve_direction_with_the_stored_trend():
    rewritten = teacher_menu(MENU, EVIDENCE)
    entry = next(item for item in rewritten if item["fault_id"] == "FAULT_07")
    assert "decreases first" not in entry["signature"]
    assert "trend in this event: increasing" in entry["signature"]
    assert "not a flow" in entry["signature"]


def test_teacher_menu_leaves_measurement_entries_alone():
    entry = teacher_menu(MENU, EVIDENCE)[1]
    assert entry["signature"] == MENU[1]["signature"]


def test_teacher_menu_is_teacher_only():
    """The published menu is unchanged; a student never sees the rewritten line."""
    from llm.benchmark.menu import menu_entry

    assert menu_entry(7)["signature"].endswith("decreases first")
