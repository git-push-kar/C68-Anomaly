"""Tests for the close-out-2 policies: KB cause mentions, direction flags,
the retry-then-drop rule, and the per-fault cap.

These are the rules that decide whether a teacher answer survives, so each test
states both directions: what is accepted, and the near-miss that must still be
rejected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark.filters import check_answer, direction_inconsistent
from llm.benchmark.kb_causes import (
    disturbance_phrase,
    is_cause_mention,
    unmeasured_cause_phrases,
)
from llm.benchmark.prompt_v2 import teacher_instruction
from llm.benchmark.targets import build_target
from llm.response.allowed_set import build_allowed_set
from scripts.build_benchmark_dataset import cap_open_set, teacher_call


def _event(fault: int = 5, run: int = 401, split: str = "test") -> dict:
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
            "anomaly_score": 0.91,
            "top_anomalous_sensors": [
                {"name": "XMV_52", "display_name": "Condenser_Cooling_Water_Flow",
                 "z_score": 2.41, "deviation_percent": 21.4, "trend": "increasing"},
                {"name": "XMEAS_22", "display_name": "Condenser_Cooling_Water_Outlet_Temperature",
                 "z_score": 1.88, "deviation_percent": -3.12, "trend": "decreasing"},
            ],
            "temporal_sequence": {"sequence": [
                {"sensor": "XMV_52", "display_name": "Condenser_Cooling_Water_Flow",
                 "relative_time_minutes": 0.0},
            ]},
            "detector_evidence": {
                "triggered_by": "both", "change_type": "mixed",
                "lstm_ae": {"score": 0.93, "threshold": 0.687, "alarmed": True},
                "dynamic": {"score": 431.2, "threshold": 239.75, "alarmed": True, "method": "cva"},
                "lag_attribution": {"dominant_lag": 4, "slow_drift_ratio": 0.48,
                                    "top_variables": [{"variable": "XMV_52", "share": 0.09}]},
            },
        },
    }


def _answer(text_extra: str = "", narrative_extra: str = "") -> dict:
    return {
        "physical_reasoning": (
            "The CVA residual share concentrates on XMV_52, the condenser cooling water "
            "flow valve, while the outlet temperature falls. The flow rises without a "
            "matching demand change, which points at the cooling-water side rather than "
            "at a flow measurement fault." + text_extra
        ),
        "narrative": (
            "The condenser cooling-water flow is rising and the outlet temperature is "
            "following it down. Check the valve actuator and the positioner on site, then "
            "confirm the cooling-water supply condition." + narrative_extra
        ),
    }


def _check(answer, fault: int = 5):
    event = _event(fault=fault)
    target = build_target(_event(fault=fault))
    allowed = build_allowed_set(event["event"], extra_faults=[fault])
    return check_answer(answer, target, event["event"], allowed)


# -- KB cause phrases ---------------------------------------------------------
def test_the_kb_name_is_an_unmeasured_cause():
    phrase = disturbance_phrase(5).lower()
    assert "cooling water inlet temperature" in phrase
    assert unmeasured_cause_phrases(5), "the KB name must be recognised as a cause"
    assert is_cause_mention("Condenser_Cooling_Water_Inlet_Temperature", 5) is True


def test_the_name_without_its_disturbance_tail_is_still_the_same_cause():
    assert is_cause_mention("condenser cooling water inlet temperature", 5) is True


def test_a_generic_word_is_not_a_cause_mention():
    for token in ("Condenser", "condenser temperature", "cooling water"):
        assert is_cause_mention(token, 5) is False


def test_an_invented_name_is_not_a_cause_mention():
    for token in ("Condenser_Pressure", "Condenser_Cooling_Water_Valve", "XMV_99"):
        assert is_cause_mention(token, 5) is False


def test_the_kb_phrase_is_accepted_in_prose():
    answer = _answer(text_extra=" The condenser cooling water inlet temperature is stepping.")
    assert _check(answer) == []


def test_an_invented_name_in_the_same_position_is_still_rejected():
    answer = _answer(text_extra=" The Condenser_Pressure is drifting.")
    codes = _check(answer)
    assert any(c.startswith("unknown_tag") for c in codes), codes


def test_a_measured_tag_the_event_never_showed_is_still_ungrounded():
    answer = _answer(text_extra=" Watch XMEAS_36 as well.")
    codes = _check(answer)
    assert any(c.startswith(("sensor_not_grounded", "unknown_tag")) for c in codes)


# -- direction flag (a flag, never a rejection) -------------------------------
def test_direction_flag_fires_when_a_sentence_contradicts_the_stored_sign():
    event = _event()
    allowed = build_allowed_set(event["event"], extra_faults=[5])
    answer = {"physical_reasoning": "Condenser_Cooling_Water_Outlet_Temperature is increasing.",
              "narrative": "Watch the trend."}
    flags = direction_inconsistent(answer, event["event"], allowed)
    assert flags, "outlet temperature is stored as falling (-3.12%)"
    assert flags[0]["sensor"] == "XMEAS_22"
    assert flags[0]["stored"] == "decrease"


def test_direction_flag_stays_quiet_when_the_sign_agrees():
    event = _event()
    allowed = build_allowed_set(event["event"], extra_faults=[5])
    answer = {"physical_reasoning": "Condenser_Cooling_Water_Flow is increasing.",
              "narrative": "The flow rises."}
    assert direction_inconsistent(answer, event["event"], allowed) == []


def test_direction_flag_skips_negated_sentences():
    event = _event()
    allowed = build_allowed_set(event["event"], extra_faults=[5])
    answer = {"physical_reasoning": "Condenser_Cooling_Water_Outlet_Temperature is no longer increasing.",
              "narrative": "The flow has stopped rising."}
    assert direction_inconsistent(answer, event["event"], allowed) == []


def test_a_climbing_level_below_baseline_is_not_a_contradiction():
    """trend and deviation answer different questions; the trend is the movement.

    A sensor sitting 5% below baseline while climbing is described correctly as
    "increasing". Comparing against the deviation sign flagged 29 correct sentences
    in the pilot, 0 of which the stored trend supported.
    """
    event = _event()
    event["event"]["top_anomalous_sensors"][1]["deviation_percent"] = -5.53
    event["event"]["top_anomalous_sensors"][1]["trend"] = "increasing"
    allowed = build_allowed_set(event["event"], extra_faults=[5])
    answer = {"physical_reasoning": "Condenser_Cooling_Water_Outlet_Temperature is increasing.",
              "narrative": "The trend is upward."}
    assert direction_inconsistent(answer, event["event"], allowed) == []


def test_direction_flag_reports_the_basis_it_used():
    event = _event()
    event["event"]["top_anomalous_sensors"][1]["trend"] = None
    allowed = build_allowed_set(event["event"], extra_faults=[5])
    answer = {"physical_reasoning": "Condenser_Cooling_Water_Outlet_Temperature is increasing.",
              "narrative": "Watch it."}
    flags = direction_inconsistent(answer, event["event"], allowed)
    assert flags and flags[0]["basis"] == "deviation"


def test_a_stable_sensor_makes_no_direction_claim():
    event = _event()
    event["event"]["top_anomalous_sensors"][1]["trend"] = "stable"
    allowed = build_allowed_set(event["event"], extra_faults=[5])
    answer = {"physical_reasoning": "Condenser_Cooling_Water_Outlet_Temperature is increasing.",
              "narrative": "Watch it."}
    assert direction_inconsistent(answer, event["event"], allowed) == []


def test_direction_flag_is_not_a_rejection():
    answer = _answer(text_extra=" Condenser_Cooling_Water_Outlet_Temperature is increasing.")
    assert _check(answer) == [], "the flag must not remove an otherwise grounded answer"


# -- retry then drop ----------------------------------------------------------
class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.model = "fake"
        self.seconds = 0.01
        self.prompt_chars = 10


class FakeTeacher:
    """Returns canned raw text, one per call, then repeats the last."""

    model = "fake"
    temperature = 0.2
    seed = 42

    def __init__(self, texts) -> None:
        self.texts = list(texts)
        self.calls = 0
        self.prompts = []

    def answer(self, evidence, menu, target, extra_instruction=None, seed=None):
        self.prompts.append(extra_instruction)
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        return parsed, _Response(text), None


def _item_and_record(fault: int = 5):
    record = _event(fault=fault)
    target = build_target(record)
    item = {
        "item_id": "EVT-test-f05-r401::menu",
        "event_id": record["event_id"],
        "fault": fault,
        "variant": "menu",
        "target": target,
        "prompt": "evidence",
    }
    return item, record


def test_a_rejected_answer_is_retried_once_and_can_be_accepted():
    good = json.dumps(_answer(text_extra=" The condenser cooling water inlet temperature is stepping."))
    bad = json.dumps(_answer(text_extra=" The Condenser_Pressure is drifting."))
    fake = FakeTeacher([bad, good])
    item, record = _item_and_record()
    result = teacher_call(item, record, fake, {"filter_sha256": "x"})
    assert fake.calls == 2
    assert result["attempts"] == 2
    assert result["accepted"] is True
    assert result["dropped"] is False
    assert fake.prompts[1], "the retry carries the rejection feedback"


def test_a_twice_rejected_item_is_dropped_and_logged():
    bad = json.dumps(_answer(text_extra=" The Condenser_Pressure is drifting."))
    fake = FakeTeacher([bad, bad])
    item, record = _item_and_record()
    result = teacher_call(item, record, fake, {"filter_sha256": "x"})
    assert fake.calls == 2
    assert result["accepted"] is False
    assert result["dropped"] is True
    assert any(code.startswith("retry:") for code in result["codes"])
    assert result["prose"] is None


def test_the_retry_is_judged_against_the_original_prompt():
    """The feedback names the offending token; quoting it back must not pass.

    87.3 is not in the evidence. It appears only in the rejection reason, so if the
    retry prompt counted as grounding, this would be accepted on the second try.
    """
    bad = json.dumps(_answer(narrative_extra=" The flow fell by 87.3%."))
    fake = FakeTeacher([bad, bad])
    item, record = _item_and_record()
    result = teacher_call(item, record, fake, {"filter_sha256": "x"})
    assert result["accepted"] is False
    assert any("87.3" in code for code in result["codes"])


# -- per-fault cap ------------------------------------------------------------
def _train_item(fault: int, index: int) -> dict:
    return {"item_id": f"it-{fault}-{index:03d}", "event_id": f"EVT-train-f{fault:02d}-r{index:03d}",
            "fault": fault, "split": "train",
            "target": {"fault_id": "UNCLASSIFIED_OPEN_SET"}}


def test_the_16_to_20_cap_is_an_equal_per_fault_budget():
    # 400 classified fixes the bound at 0.25*400/0.75 = 133, so 67 of the 200 go.
    classified = [{"item_id": f"c-{i}", "event_id": f"e-{i}", "fault": 1, "split": "train",
                   "target": {"fault_id": "FAULT_01"}} for i in range(400)]
    open_family = [_train_item(fault, i) for fault in (16, 17, 18, 19, 20) for i in range(40)]
    kept, report = cap_open_set(classified + open_family, seed=42)
    per = report["per_fault_16_20"]
    kept_counts = [row["kept"] for row in per.values()]
    assert max(kept_counts) - min(kept_counts) <= 1, kept_counts
    assert report["share_after"] <= 0.25
    assert len(kept) == len(classified) + sum(kept_counts)
