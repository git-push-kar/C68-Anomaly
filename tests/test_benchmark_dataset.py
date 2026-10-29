"""Tests for the response-benchmark dataset pieces: prompt, menu, targets, filters.

These guard the properties that make the dataset trustworthy rather than the
numbers in it: no hint or rubric in the prompt, a stable open-set assignment, a
label that only the KB decides, and filters that reject the failure modes they
claim to.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.benchmark.filters import check_answer
from llm.benchmark.menu import CLASSIFIABLE_FAULTS, UNCLASSIFIED, build_menu, lookalikes
from llm.benchmark.prompt_v2 import (
    FORBIDDEN_PROMPT_FRAGMENTS,
    assert_prompt_clean,
    render_prompt,
    teacher_instruction,
)
from llm.benchmark.targets import (
    NORMAL_ID,
    build_target,
    eligible_for_open_set,
    menu_for_record,
    open_set_assignment,
)
from llm.response.allowed_set import build_allowed_set


def _event(fault: int = 15, run: int = 401, split: str = "test") -> dict:
    """A minimal-but-real event record: evidence shaped like the builder's output."""
    return {
        "event_id": f"EVT-{split}-f{fault:02d}-r{run:03d}",
        "fault": fault,
        "simulationRun": run,
        "split": split,
        "seed": 42,
        "source_file": "TEP_Faulty_Testing.csv",
        "injection_sample": 160,
        "event_open_sample": 240,
        "event": {
            "event_id": f"EVT-{split}-f{fault:02d}-r{run:03d}",
            "anomaly_score": 0.9123,
            "top_anomalous_sensors": [
                {"name": "XMV_52", "display_name": "Condenser_Cooling_Water_Flow",
                 "z_score": 2.41, "deviation_percent": -0.47, "trend": "flat"},
                {"name": "XMEAS_17", "display_name": "Stripper_Level",
                 "z_score": 1.88, "deviation_percent": 3.12, "trend": "rising"},
            ],
            "temporal_sequence": {"sequence": [
                {"sensor": "XMV_52", "display_name": "Condenser_Cooling_Water_Flow",
                 "relative_time_minutes": 0.0},
            ]},
            "detector_evidence": {
                "triggered_by": "both",
                "change_type": "mixed",
                "lstm_ae": {"score": 0.9312, "threshold": 0.687, "alarmed": True},
                "dynamic": {"score": 431.2, "threshold": 239.75, "alarmed": True,
                            "method": "cva"},
                "lag_attribution": {
                    "dominant_lag": 4, "slow_drift_ratio": 0.488,
                    "top_variables": [
                        {"variable": "XMV_52", "share": 0.09},
                        {"variable": "XMEAS_17", "share": 0.078},
                    ],
                },
            },
        },
    }


# -- prompt -------------------------------------------------------------------
def test_prompt_has_neither_subsystem_hint_nor_severity_rubric():
    prompt = render_prompt(_event()["event"], build_menu(), teacher_instruction(["FAULT_15"]))
    lowered = prompt.lower()
    for fragment in ("candidate affected subsystem", "severity"):
        assert fragment not in lowered


def test_assert_prompt_clean_rejects_a_leaky_prompt():
    with pytest.raises(ValueError):
        assert_prompt_clean("Candidate affected subsystem: cooling water", "test")


def test_prompt_forbids_every_listed_fragment():
    for fragment in FORBIDDEN_PROMPT_FRAGMENTS:
        with pytest.raises(ValueError):
            assert_prompt_clean(f"prefix {fragment} suffix", "test")


def test_prompt_carries_the_lag_attribution():
    prompt = render_prompt(_event()["event"], build_menu(), teacher_instruction(["FAULT_15"]))
    assert "DYNAMIC RESIDUAL ATTRIBUTION" in prompt
    assert "XMV_52: residual share 0.090" in prompt


# -- menu ---------------------------------------------------------------------
def test_menu_covers_classifiable_faults_and_always_offers_abstention():
    ids = [entry["fault_id"] for entry in build_menu()]
    assert ids[-1] == UNCLASSIFIED
    assert "FAULT_15" in ids
    for hidden in ("FAULT_16", "FAULT_20", "NORMAL_TRANSIENT"):
        assert hidden not in ids


def test_removed_variant_drops_exactly_one_entry():
    full = [e["fault_id"] for e in build_menu()]
    trimmed = [e["fault_id"] for e in build_menu(remove=[15])]
    assert set(full) - set(trimmed) == {"FAULT_15"}
    assert len(trimmed) == len(full) - 1


def test_open_set_assignment_is_stable_per_event_id():
    event_id = "EVT-test-f15-r401"
    first = open_set_assignment(event_id)
    assert all(open_set_assignment(event_id) == first for _ in range(5))
    assert 0.0 <= sum(open_set_assignment(f"EVT-x-{i}") for i in range(1000)) / 1000 <= 0.2


# -- targets ------------------------------------------------------------------
def test_classified_target_puts_the_truth_first_and_keeps_a_kb_lookalike():
    target = build_target(_event())
    assert target["fault_id"] == "FAULT_15"
    assert target["ranked_hypotheses"][0]["is_truth"] is True
    assert target["abstain"] is False
    assert target["truth_subsystem"]
    assert target["mitigation"]


def test_removed_variant_targets_the_open_set_without_the_truth():
    target = build_target(_event(), variant="removed")
    assert target["fault_id"] == UNCLASSIFIED
    assert target["abstain"] is True
    assert all(not h["is_truth"] for h in target["ranked_hypotheses"])
    assert target["menu_removed"] == ["FAULT_15"]
    assert target["clarifying_question"]


def test_open_set_fault_targets_unclassified_even_with_the_menu_present():
    target = build_target(_event(fault=17))
    assert target["fault_id"] == UNCLASSIFIED
    # The expected answer is the open-set label; the generating fault is still
    # recorded, so an audit can tell which fault produced the item.
    assert target["truth_fault"] == UNCLASSIFIED
    assert target["truth_fault_number"] == 17


def test_normal_target_is_no_fault_and_not_a_menu_entry():
    record = _event(fault=15)
    record["fault"] = None
    record["label"] = NORMAL_ID
    target = build_target(record)
    assert target["fault_id"] == NORMAL_ID
    assert target["verdict"] == "no_fault"
    assert target["abstain"] is True


def test_answer_keys_are_schema_shaped_and_never_include_severity():
    target = build_target(_event())
    assert "severity" not in json.dumps({k: v for k, v in target.items()
                                         if k != "primary_subsystem"}).lower()
    assert target["confidence"] is None, "confidence must not be invented"
    assert len(target["suggested_followups"]) <= 3


def test_synthetic_faults_are_flagged():
    target = build_target(_event(fault=21))
    assert target.get("synthetic") is True


def test_open_set_episodes_only_come_from_train_runs():
    assert eligible_for_open_set(_event(split="train"), "train") is True
    assert eligible_for_open_set(_event(split="test"), "test") is False
    assert eligible_for_open_set(_event(fault=16, split="train"), "train") is False


def test_menu_for_record_honours_the_variant():
    ids = [e["fault_id"] for e in menu_for_record(_event(), variant="removed")]
    assert "FAULT_15" not in ids


# -- filters ------------------------------------------------------------------
def _good_answer() -> dict:
    return {
        "physical_reasoning": (
            "The CVA residual energy concentrates on XMV_52, which is the "
            "condenser cooling water flow valve. The deviation appears while the "
            "condenser pressure is still near its baseline, so the valve is moving "
            "without a matching demand change. That pattern is consistent with a "
            "sticking valve rather than a flow measurement fault."
        ),
        "narrative": (
            "The condenser cooling water valve appears to be sticking: its "
            "residual contribution stays high while the controlled variable barely "
            "moves. Check the valve actuator and positioner first, then confirm the "
            "condenser pressure trend on site."
        ),
    }


def _check(answer, *, target=None, event=None):
    event = event or _event()
    target = target or build_target(_event())
    allowed = build_allowed_set(event["event"], extra_faults=[15])
    return check_answer(answer, target, event["event"], allowed)


def test_a_grounded_answer_is_accepted():
    assert _check(_good_answer()) == []


def test_missing_key_is_rejected():
    answer = dict(_good_answer(), fault_id="FAULT_15")
    assert any(code.startswith("missing_key") for code in _check(answer))


def test_narrative_of_one_sentence_is_accepted():
    # The response contract allows 1-4 sentences, so one sentence is compliant.
    answer = dict(_good_answer(), narrative="The condenser cooling water valve appears to be sticking.")
    assert _check(answer) == []


def test_narrative_of_five_sentences_is_rejected():
    answer = dict(_good_answer(), narrative="One thing. Two thing. Three thing. Four thing. Five thing.")
    assert any(code.startswith("narrative_length") for code in _check(answer))


def test_unknown_tag_is_rejected():
    answer = dict(_good_answer(),
                  narrative=_good_answer()["narrative"] + " Compare with XMV_99.")
    assert any(code.startswith("unknown_tag") for code in _check(answer))


def test_number_that_is_not_in_the_evidence_is_rejected():
    answer = dict(_good_answer(),
                  narrative=_good_answer()["narrative"] + " The flow fell by 87.3%.")
    assert any(code.startswith("number_not_in_store") for code in _check(answer))


def test_absolute_wording_is_rejected():
    answer = dict(_good_answer(),
                  narrative=_good_answer()["narrative"] + " This is definitely a sticking valve.")
    assert any(code.startswith("absolute_wording") for code in _check(answer))


def test_severity_claim_is_rejected():
    answer = dict(_good_answer(),
                  narrative=_good_answer()["narrative"] + " Severity is critical.")
    assert any(code.startswith("severity_claim") for code in _check(answer))


def test_label_change_is_rejected():
    answer = dict(_good_answer(), fault_id="FAULT_4")
    assert any(code.startswith("label_changed") for code in _check(answer))


def test_ungrounded_sensor_is_rejected():
    # XMEAS_36 is neither in this event's evidence nor in fault 15's KB entry.
    answer = dict(_good_answer(),
                  narrative=_good_answer()["narrative"] + " Watch XMEAS_36 as well.")
    codes = _check(answer)
    assert any(c.startswith(("sensor_not_grounded", "unknown_tag")) for c in codes)


def test_invalid_json_is_reported_as_such():
    allowed = build_allowed_set(_event()["event"], extra_faults=[15])
    codes = check_answer(None, build_target(_event()), _event()["event"], allowed,
                         parse_error="no JSON object in response")
    assert codes and codes[0].startswith("invalid_json")


def test_kb_lookalikes_are_present_for_a_differential():
    assert lookalikes(14), "fault 14 names its sibling in the KB differential"
