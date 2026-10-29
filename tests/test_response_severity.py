"""Severity tests.

The quoted level is the **authoritative** production rule
(`evidence.process_limits.compute_deterministic_severity`). The rule inside
`llm/response/severity.py` is a cross-check: these tests lock *its* shape
(monotone in detector margin, floored by the KB, capped for normal transients)
via ``components['proposed_level']``, and separately assert that the returned
level is always the authoritative one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.response import StoreAccess
from llm.response.schemas import SeverityLevel
from evidence.process_limits import compute_deterministic_severity
from llm.response.severity import (
    AUTHORITATIVE_RULE,
    PROPOSED_RULE,
    RULE_VERSION,
    assess,
    authoritative_level,
    severity_agreement,
)

ORDER = [SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL]


def event(
    fault_label=None,
    lstm=(0.9, 0.6),
    cva=(300.0, 240.0),
    z=4.0,
    n_windows=12,
    max_score=0.9,
    mean_score=None,
):
    l_score, l_th = lstm
    q_score, q_th = cva
    sensors = [
        {"name": f"XMEAS_{i}", "display_name": f"Sensor {i}", "current_value": 1.0, "baseline_value": 0.9,
         "deviation_percent": 11.1, "z_score": z}
        for i in (1, 2, 3)
    ]
    return {
        "event_id": "ANOM-TEST",
        "fault_label": fault_label,
        "n_windows": n_windows,
        "max_anomaly_score": max_score,
        "mean_anomaly_score": mean_score if mean_score is not None else max_score,
        "evidence": {
            "anomaly_score": max_score,
            "severity": "low",
            "top_anomalous_sensors": sensors,
            "detector_evidence": {
                "change_type": "dynamics",
                "triggered_by": "both",
                "lstm_ae": {"score": l_score, "threshold": l_th, "alarmed": l_score >= l_th},
                "dynamic": {"score": q_score, "threshold": q_th, "alarmed": q_score >= q_th},
            },
        },
    }


def test_rule_versions_name_their_roles():
    assert AUTHORITATIVE_RULE == "evidence.process_limits.compute_deterministic_severity"
    assert PROPOSED_RULE.endswith("proposed")
    assert RULE_VERSION == "severity-crosscheck-v2"


def test_authoritative_level_matches_the_production_function():
    ev = event()
    assert authoritative_level(ev).value == compute_deterministic_severity(ev["evidence"])
    assert assess(ev, fault="FAULT_15").level is authoritative_level(ev)


def test_cross_check_is_monotone_in_detector_margin():
    weak = assess(event(lstm=(0.61, 0.60), cva=(241.0, 240.0), z=2.1), fault="FAULT_13")
    strong = assess(event(lstm=(1.8, 0.60), cva=(600.0, 240.0), z=6.5), fault="FAULT_13")
    assert ORDER.index(SeverityLevel(strong.components["proposed_level"])) >= ORDER.index(
        SeverityLevel(weak.components["proposed_level"])
    )
    assert strong.score > weak.score


def test_kb_floor_applies_for_a_sustained_event():
    # weak detector evidence on a critical KB fault, but a long sustained event:
    # the cross-check's KB floor keeps it at the family's level.
    assessment = assess(event(lstm=(0.61, 0.60), cva=(241.0, 240.0), z=0.2), fault="FAULT_15")
    assert assessment.kb_base is SeverityLevel.CRITICAL
    assert ORDER.index(SeverityLevel(assessment.components["proposed_level"])) >= ORDER.index(
        SeverityLevel.HIGH
    )


def test_short_decaying_event_is_stepped_down_from_the_kb_floor():
    short = assess(event(n_windows=2, mean_score=0.2, max_score=0.9), fault="FAULT_15")
    assert short.components["floor"] == SeverityLevel.HIGH.value
    assert ORDER.index(SeverityLevel(short.components["proposed_level"])) < ORDER.index(
        SeverityLevel.CRITICAL
    )


def test_normal_transient_is_never_critical_in_the_cross_check():
    assessment = assess(event(lstm=(3.0, 0.6), cva=(900.0, 240.0), z=9.0), fault="NORMAL_TRANSIENT")
    assert ORDER.index(SeverityLevel(assessment.components["proposed_level"])) <= ORDER.index(
        SeverityLevel.MEDIUM
    )


def test_agreement_flag_is_recorded_and_never_displayed():
    ev = event()
    assessment = assess(ev, fault="FAULT_15")
    comp = assessment.components
    assert comp["authoritative_level"] == assessment.level.value
    assert comp["proposed_level"] in {level.value for level in ORDER}
    assert isinstance(comp["severity_agreement"], bool)
    assert comp["severity_gap"] == (
        ORDER.index(SeverityLevel(comp["proposed_level"])) - ORDER.index(assessment.level)
    )
    # the rationale quotes only the authoritative rule, never the cross-check
    assert "authoritative production rule" in assessment.rationale
    assert "proposed" not in assessment.rationale
    assert "cross-check" not in assessment.rationale


def test_agreement_helper_reports_a_signed_gap():
    assert severity_agreement(SeverityLevel.LOW, SeverityLevel.LOW) == {
        "agree": True, "authoritative": "low", "proposed": "low", "gap": 0
    }
    above = severity_agreement(SeverityLevel.MEDIUM, SeverityLevel.CRITICAL)
    assert above == {"agree": False, "authoritative": "medium", "proposed": "critical", "gap": 2}


def test_open_set_has_a_medium_base_and_no_kb_floor():
    assessment = assess(event(), fault="UNCLASSIFIED_OPEN_SET")
    assert assessment.kb_base is SeverityLevel.MEDIUM
    assert assessment.components["diagnosed_fault"] == "UNCLASSIFIED_OPEN_SET"


def test_store_exposes_the_computed_assessment():
    ev = event()
    report = {"root_cause": "FAULT_15"}
    assessment = assess(ev, report=report)
    store = StoreAccess(ev, report)
    assert store.resolve("severity.computed") == assessment.level.value
    assert store.resolve("severity.computed_score") == assessment.score
    assert store.resolve("severity.kb_base") == assessment.kb_base.value
    assert store.resolve("severity.rule") == RULE_VERSION
    # the stored value is reported separately, not overwritten
    assert store.resolve("severity.stored") == "low"


def test_severity_lookup_answers_with_the_computed_level_not_the_stored_one():
    from llm.response import lookup_answer
    from llm.response.validators import validate_followup

    ev = event()
    report = {"root_cause": "FAULT_15"}
    hit = lookup_answer(ev, report, "How serious is this?")
    assert hit is not None
    expected = assess(ev, report=report).level.value.upper()
    assert hit.answer.answer.startswith(f"Severity is {expected}")
    assert "from the older rule" in hit.answer.answer  # stored 'low' is disclosed, not mixed in
    _, result = validate_followup(hit.answer.model_dump(), ev, report, question="How serious is this?")
    assert result.ok, result.as_dict()


def test_missing_detector_blocks_do_not_crash():
    assessment = assess({"event_id": "ANOM-EMPTY", "evidence": {}}, fault="FAULT_01")
    assert assessment.level in ORDER
    assert assessment.components["max_ratio"] == 0.0
