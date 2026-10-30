"""Follow-up answer filters: one test per rejection code, plus the accept case.

Every test builds the same context from a stored event fixture, so a rule is
measured on real evidence rather than on a hand-made store.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark.filters import store_numbers  # noqa: E402
from llm.followup.filters import (  # noqa: E402
    AnswerContext,
    check_answer,
    is_near_duplicate,
    overlap_ratio,
)
from llm.response.allowed_set import build_allowed_set  # noqa: E402
from llm.response.store_access import StoreAccess  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "response"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture()
def event() -> dict:
    return load("classified")


@pytest.fixture()
def report(event) -> dict:
    return event["report"]


@pytest.fixture()
def ctx(event, report) -> AnswerContext:
    store = StoreAccess(event, report)
    grounded = store_numbers(event["evidence"], report) + store.numeric_values()
    return AnswerContext(
        store=store,
        allowed=build_allowed_set(event, extra_faults=[15]),
        grounded_numbers=grounded,
        permitted_faults=list(range(1, 16)),
        event_id=event["event_id"],
        previous_answer=None,
    )


GOOD = "Reactor feed rate is steady at 42.31 against a baseline of 42.34, and the CVA Q-statistic is 248.3 against a limit of 239.8. The moving variables are listed with their baselines."


def test_grounded_answer_is_accepted(ctx):
    assert check_answer(GOOD, ctx) == []


def test_invented_number_is_rejected(ctx):
    codes = check_answer(GOOD + " The residual peaked at 12.345.", ctx)
    assert any(code.startswith("number_not_in_store") for code in codes)


def test_severity_claim_is_rejected(ctx):
    codes = check_answer(GOOD + " That makes this a high severity event.", ctx)
    assert any(code.startswith("severity_claim") for code in codes)


def test_confidence_number_is_rejected(ctx):
    codes = check_answer(GOOD + " I am 87% confident in that.", ctx)
    assert any(code.startswith("confidence_claim") for code in codes)


def test_absolute_wording_is_rejected(ctx):
    codes = check_answer(GOOD + " This definitely proves the valve failed.", ctx)
    assert any(code.startswith("absolute_wording") for code in codes)


def test_table_reference_is_rejected(ctx):
    codes = check_answer("The values are as shown in the table below.", ctx)
    assert any(code.startswith("table_reference") for code in codes)


def test_answer_must_not_be_a_question(ctx):
    codes = check_answer("Which view would you like me to expand?", ctx)
    assert "answer_is_question" in codes


def test_answer_must_be_one_to_four_sentences(ctx):
    long = GOOD + " One more. And another. And a fourth."
    assert any(code.startswith("answer_length") for code in check_answer(long, ctx))


def test_clarifying_question_must_not_be_repeated(ctx):
    ctx.clarifying_question = "Did you mean the sensor scores view or the detector scores view?"
    codes = check_answer(
        "I can answer either way. Did you mean the sensor scores view or the detector scores view?",
        ctx,
    )
    assert "answer_is_question" in codes
    assert any(code.startswith("clarify_text_repeated") for code in codes)


def test_open_set_answer_must_not_name_the_removed_fault(ctx):
    ctx.removed_fault = 15
    codes = check_answer(GOOD + " The record still points to FAULT_15.", ctx)
    assert any(code.startswith("names_removed_fault") for code in codes)


def test_removed_fault_can_be_denied_in_a_refusal(ctx):
    ctx.removed_fault = 15
    ctx.refusal = {
        "required": True,
        "reason": "contradicted",
        "premise": {"kind": "fault", "claim": "FAULT_07", "verdict": "contradicted",
                    "store_key": "kb.fault.07.name", "actual": "FAULT_15",
                    "reason": "the stored diagnosis differs"},
        "alternatives": [{"question": "What are the detector scores?", "intent": "detector_scores"}],
    }
    codes = check_answer(GOOD + " This is not fault 7; the stored diagnosis is different.", ctx)
    assert not any(code.startswith("names_removed_fault") for code in codes)


def test_valve_flow_mislabel_is_rejected(ctx):
    codes = check_answer(GOOD + " XMV_52 temperature is the value to watch.", ctx)
    assert any(code.startswith("valve_flow_mislabel") for code in codes)


def test_event_id_leak_is_rejected(ctx):
    codes = check_answer(GOOD + f" This came from {ctx.event_id}.", ctx)
    assert any(code.startswith("event_id_leak") for code in codes)


def test_refusal_must_decline(ctx):
    ctx.refusal = {
        "required": True,
        "reason": "key_missing",
        "premise": {"kind": "key", "claim": "vibration of XMEAS_12", "verdict": "missing",
                    "store_key": "sensor.XMEAS_12.vibration", "actual": None, "reason": "no such field"},
        "alternatives": [{"question": "What are the detector scores?", "intent": "detector_scores"}],
    }
    assert "refusal_missing" in check_answer(GOOD, ctx)


def test_refusal_must_offer_an_answerable_alternative(ctx):
    ctx.refusal = {
        "required": True,
        "reason": "key_missing",
        "premise": {"kind": "key", "claim": "vibration of XMEAS_12", "verdict": "missing",
                    "store_key": "sensor.XMEAS_12.vibration", "actual": None, "reason": "no such field"},
        "alternatives": [{"question": "What are the detector scores?", "intent": "detector_scores"}],
    }
    assert "refusal_no_alternative" in check_answer(
        "I cannot confirm that from this incident.", ctx
    )
    assert check_answer(
        "I cannot confirm that from this incident. I can show you the detector scores instead.",
        ctx,
    ) == []


def test_near_duplicate_of_the_previous_reply_is_rejected(ctx):
    ctx.previous_answer = GOOD
    codes = check_answer(GOOD, ctx)
    assert any(code.startswith("near_duplicate") for code in codes)


def test_a_different_answer_on_the_same_topic_passes(ctx):
    ctx.previous_answer = GOOD
    other = "The dynamic reading sits above its limit, and the loop settles slowly after the actuator frees."
    assert not any(code.startswith("near_duplicate") for code in check_answer(other, ctx))


def test_overlap_ratio_is_symmetric_enough_for_the_rule():
    a = "the loop settles once the actuator is freed and the reading returns"
    b = "the loop settles once the actuator is freed and the reading returns to normal"
    assert overlap_ratio(a, b) > 0.5
    assert is_near_duplicate(a, b) is True
    assert is_near_duplicate(a, "a completely different sentence about the stripper level") is False
