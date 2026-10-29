"""Router tests on LIVE-format questions.

Every question here is one an operator would type at the report screen. Nothing
comes from the distilled dataset (those prompts carry a subsystem hint and a KB
menu and would route for the wrong reason -- see docs/RESPONSE_SPEC.md §1).

The important test is the mirror: the router reproduces candidate lists without
editing `llm/inference.py`, so the two must agree on the intent the live path
actually picks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.inference import RCAInference
from llm.response import Intent, ResponseRouter

FIXTURES = ROOT / "tests" / "fixtures" / "response"

#: (question, intent the live dispatcher returns or None)
LIVE_CASES = [
    ("What are the detector scores?", "detector_scores"),
    ("What is the Q-statistic?", "detector_scores"),
    ("Why was this flagged?", "evidence_why"),
    ("Give me the evidence", "evidence_why"),
    ("What should I do?", "action"),
    ("How serious is this?", "severity"),
    ("Which sensors deviate most?", "sensors_inspect"),
    ("What happened?", "overview"),
    ("Which fault is this?", "overview"),
    ("Which sensors are used in this system?", "system_sensors"),
    ("Show me the sensor values against baseline", "sensor_values"),
    ("Explain the anomaly", "overview_fallback"),
    ("What is the score for the sensor anomaly?", "sensor_scores"),
    ("Is it the condenser or the stripper?", None),
]


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture()
def event() -> dict:
    return load("classified")


@pytest.fixture()
def report(event) -> dict:
    return event["report"]


@pytest.mark.parametrize("question,expected", LIVE_CASES)
def test_candidates_agree_with_the_live_dispatcher(event, report, question, expected):
    live = RCAInference._dispatch_with_intent(event, report, question)
    assert (live[0] if live else None) == expected, "fixture expectation drifted from llm/inference.py"
    candidates, _ = ResponseRouter().candidates(question)
    primary = candidates[0].value if candidates else None
    if expected is None:
        # The live path has no branch; the router may still hold candidates only
        # when a proposed intent is enabled, which it is not by default.
        assert primary is None
    else:
        assert primary == expected


def test_deterministic_answer_is_the_live_answer_verbatim(event, report):
    route = ResponseRouter().route(event, report, "What are the detector scores?")
    live = RCAInference._dispatch_with_intent(event, report, "What are the detector scores?")
    assert route.answerability == "deterministic"
    assert route.deterministic_answer == live[1]


def test_evidence_why_outranks_the_generic_why_fallback(event, report):
    candidates, _ = ResponseRouter().candidates("Why was this flagged?")
    assert candidates[0] is Intent.EVIDENCE_WHY
    assert Intent.OVERVIEW_FALLBACK in candidates


def test_system_sensors_is_not_also_a_sensor_values_question(event, report):
    candidates, _ = ResponseRouter().candidates("Which sensors are used in this system?")
    assert candidates == [Intent.SYSTEM_SENSORS]


def test_ambiguous_hypotheses_trigger_uses_the_configured_margin(event):
    report = {
        "root_cause": "FAULT_15",
        "hypotheses": [{"fault": "FAULT_15", "confidence": 0.52}, {"fault": "FAULT_14", "confidence": 0.46}],
    }
    route = ResponseRouter().route(event, report, "What happened?", with_answer=False)
    assert route.clarification is not None
    assert route.clarification.trigger == "ambiguous_hypotheses"
    assert set(route.clarification.options) == {"FAULT_15", "FAULT_14"}
    assert route.answerability == "clarify"

    # same margin, tighter threshold -> no clarification
    tight = ResponseRouter(hypothesis_margin=0.02).route(event, report, "What happened?", with_answer=False)
    assert tight.clarification is None

    # wide margin -> no clarification
    wide = {
        "root_cause": "FAULT_15",
        "hypotheses": [{"fault": "FAULT_15", "confidence": 0.71}, {"fault": "FAULT_14", "confidence": 0.22}],
    }
    assert ResponseRouter().route(event, wide, "What happened?", with_answer=False).clarification is None


def test_unclassified_report_triggers_a_clarification():
    event = load("unclassified")
    route = ResponseRouter().route(event, {"root_cause": "UNCLASSIFIED_OPEN_SET"}, "What are the detector scores?")
    assert route.clarification is not None
    assert route.clarification.trigger == "unclassified"


def test_ambiguous_question_trigger(event, report):
    route = ResponseRouter().route(event, report, "Explain the anomaly score", with_answer=False)
    assert route.clarification is not None
    assert route.clarification.trigger == "ambiguous_question"
    # an unambiguous question of the same family stays answerable
    assert ResponseRouter().route(event, report, "What are the detector scores?").clarification is None


def test_context_not_in_store_trigger(event):
    bare = {"event_id": "ANOM-BARE", "evidence": {}, "max_anomaly_score": 1.0}
    route = ResponseRouter().route(bare, {"root_cause": "FAULT_15"}, "What should I do?", with_answer=False)
    assert route.clarification is not None
    assert route.clarification.trigger == "context_not_in_store"


def test_asked_history_suppresses_the_same_trigger(event, report):
    router = ResponseRouter()
    first = router.route(event, report, "Explain the anomaly score", with_answer=False)
    assert first.clarification is not None
    again = router.route(
        event, report, "Explain the anomaly score", asked=[first.clarification.question], with_answer=False
    )
    assert again.clarification is None


def test_proposed_intents_are_behind_a_flag(event, report):
    question = "How is this different from fault 14?"
    base, _ = ResponseRouter().candidates(question)
    assert Intent.HYPOTHESIS_COMPARISON not in base
    proposed, _ = ResponseRouter(enable_proposed=True).candidates(question)
    assert proposed[0] is Intent.HYPOTHESIS_COMPARISON


def test_out_of_scope_refusal_only_when_enabled(event, report):
    question = "Tell me a joke about the weather."
    assert ResponseRouter().candidates(question)[0] == []
    router = ResponseRouter(enable_proposed=True)
    assert router.candidates(question)[0][0] is Intent.OUT_OF_SCOPE_REFUSAL
    assert router.route(event, report, question, with_answer=False).answerability == "refuse"


def test_clarification_reply_does_not_compete_with_a_real_question(event, report):
    router = ResponseRouter(enable_proposed=True)
    candidates, _ = router.candidates("What happened?")
    assert candidates[0] is Intent.OVERVIEW
    assert Intent.CLARIFICATION_REPLY not in candidates
    # an explicit "what do you mean" is a clarification reply
    reply, _ = router.candidates("What do you mean by that?")
    assert reply == [Intent.CLARIFICATION_REPLY]


def test_short_unroutable_stub_asks_for_clarification():
    candidates, _ = ResponseRouter(enable_proposed=True).candidates("and then?")
    assert candidates and candidates[0] is Intent.CLARIFICATION_REPLY
