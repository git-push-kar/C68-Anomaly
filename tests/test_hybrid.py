"""The hybrid's three contracts: the menu rule, the forced prefix, the fallback.

These are the parts that decide an answer's fault id, so they are tested against the cases
the protocol names rather than against their own implementation: an off-menu prediction, a
low-confidence on-menu prediction, a no-fault prediction, a shortfall threshold, a
counterfactual force, and a fallback that cannot contradict the diagnosis.

The classifier tests are skipped where LightGBM is not installed (the GPU environment does
not carry it), and they assert *determinism*, because the surrogate models in the report are
compared against this model and a rerun has to reproduce it exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.hybrid.generate import (  # noqa: E402
    contradictions,
    fallback_answer,
    forced_prefix,
    parse_slim,
    validate_slim,
)
from llm.hybrid.menu_rule import (  # noqa: E402
    NORMAL,
    UNCLASSIFIED,
    largest_threshold,
    menu_rule,
)

MENU = ["FAULT_01", "FAULT_05", "FAULT_13", "UNCLASSIFIED_OPEN_SET"]
ALLOWED = [f"FAULT_{index:02d}" for index in range(1, 16)] + [UNCLASSIFIED, NORMAL]


# --------------------------------------------------------------------------- #
# menu rule
# --------------------------------------------------------------------------- #
def test_on_menu_above_threshold_is_classified():
    decision = menu_rule("FAULT_13", 0.91, MENU, 0.5)
    assert decision["diagnosis"] == "FAULT_13"
    assert decision["verdict"] == "classified"
    assert decision["abstained"] is False


def test_off_menu_prediction_becomes_the_open_set_verdict():
    """The same prediction is a name on one item and an abstention on its twin."""
    on = menu_rule("FAULT_07", 0.99, MENU + ["FAULT_07"], 0.5)
    off = menu_rule("FAULT_07", 0.99, MENU, 0.5)
    assert on["diagnosis"] == "FAULT_07"
    assert off["diagnosis"] == UNCLASSIFIED
    assert off["verdict"] == "unclassified_open_set"
    assert off["reason"] == "not_on_menu"


def test_low_confidence_on_menu_abstains():
    decision = menu_rule("FAULT_13", 0.4, MENU, 0.5)
    assert decision["diagnosis"] == UNCLASSIFIED
    assert decision["reason"] == "below_threshold"
    assert decision["abstained"] is True


def test_normal_is_no_fault_and_never_an_abstention():
    decision = menu_rule(NORMAL, 0.2, MENU, 0.9)
    assert decision["diagnosis"] == NORMAL
    assert decision["verdict"] == "no_fault"
    assert decision["abstained"] is False


def test_missing_prediction_is_the_open_set_verdict():
    decision = menu_rule(None, None, MENU, 0.5)
    assert decision["diagnosis"] == UNCLASSIFIED
    assert decision["reason"] == "no_prediction"


# --------------------------------------------------------------------------- #
# threshold rule
# --------------------------------------------------------------------------- #
def test_largest_threshold_spends_but_does_not_exceed_the_budget():
    confidences = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    result = largest_threshold(confidences, budget=0.05)
    assert result["achieved_false_abstention"] <= 0.05
    assert result["abstained"] == sum(1 for value in confidences
                                      if value < result["threshold"])
    # The largest feasible threshold: one step higher and the budget would be spent twice.
    above = sorted(confidences)[result["abstained"]]
    assert above >= result["threshold"]


def test_threshold_rule_is_monotone_in_the_budget():
    confidences = [index / 100 for index in range(1, 101)]
    loose = largest_threshold(confidences, budget=0.10)
    tight = largest_threshold(confidences, budget=0.01)
    assert loose["threshold"] >= tight["threshold"]
    assert loose["achieved_false_abstention"] <= 0.10
    assert tight["achieved_false_abstention"] <= 0.01


def test_saturated_confidences_produce_a_threshold_that_abstains_on_nothing_extra():
    """A calibration that returns 1.0 everywhere makes the arm inert, not free."""
    result = largest_threshold([1.0] * 100, budget=0.05)
    assert result["threshold"] == 1.0
    assert result["abstained"] == 0


# --------------------------------------------------------------------------- #
# forced prefix
# --------------------------------------------------------------------------- #
def test_forced_prefix_opens_the_answer_with_the_diagnosis():
    prefix = forced_prefix("FAULT_10")
    assert prefix.startswith('{"fault_id": "FAULT_10"')
    assert prefix.endswith(", ")
    assert json.loads(prefix[:-2] + "}") == {"fault_id": "FAULT_10", "verdict": "classified"}


def test_forced_prefix_derives_the_verdict_from_the_diagnosis():
    assert (json.loads(forced_prefix(UNCLASSIFIED)[:-2] + "}")["verdict"]
            == "unclassified_open_set")
    assert json.loads(forced_prefix(NORMAL)[:-2] + "}")["verdict"] == "no_fault"


def test_a_continuation_round_trips_through_the_prefix():
    prefix = forced_prefix("FAULT_05")
    text = ('"ranked_hypotheses": ["FAULT_05", "FAULT_12"], '
            '"physical_reasoning": "XMEAS_22 rises first.", '
            '"narrative": "Diagnosis FAULT_05 fits the event."}')
    answer = parse_slim(text, prefix)
    assert answer["fault_id"] == "FAULT_05"
    assert validate_slim(answer, ALLOWED, "FAULT_05") == []


def test_a_model_that_rewrites_the_fault_id_is_reported():
    """The prefix is forced, but a model can still emit its own id; that is an enum error."""
    prefix = forced_prefix("FAULT_05")
    # A second copy of the key wins in JSON, which is exactly how a model can overrule the
    # prefix without touching the text that was forced.
    text = ('"fault_id": "FAULT_12", "ranked_hypotheses": ["FAULT_12"], '
            '"physical_reasoning": "x", "narrative": "y"}')
    answer = parse_slim(text, prefix)
    assert answer["fault_id"] == "FAULT_12"
    problems = validate_slim(answer, ALLOWED, "FAULT_05")
    assert any(problem.startswith("fault_id_not_forced") for problem in problems)


# --------------------------------------------------------------------------- #
# contradictions
# --------------------------------------------------------------------------- #
def _answer(diagnosis: str, reasoning: str, ranked=None) -> dict:
    return {"fault_id": diagnosis, "verdict": "classified",
            "ranked_hypotheses": ranked or [diagnosis],
            "physical_reasoning": reasoning, "narrative": "See the table."}


def test_faithful_prose_has_no_contradiction():
    answer = _answer("FAULT_13", "XMEAS_7 rises first, so the diagnosis is FAULT_13.")
    assert contradictions(answer, "FAULT_13") == []


def test_ranking_that_disagrees_is_recorded():
    answer = _answer("FAULT_13", "Steady.", ranked=["FAULT_12", "FAULT_13"])
    assert {found["code"] for found in contradictions(answer, "FAULT_13")} == {
        "ranked_top_mismatch"}


def test_prose_that_blames_another_fault_is_recorded():
    answer = _answer("FAULT_13", "The pattern is caused by FAULT_12.")
    codes = {found["code"] for found in contradictions(answer, "FAULT_13")}
    assert "names_other_fault_as_cause" in codes


def test_hedging_the_forced_diagnosis_is_recorded():
    for spelling, diagnosis in (("FAULT 5", "FAULT_05"), ("FAULT 13", "FAULT_13"),
                                ("FAULT_15", "FAULT_15")):
        answer = _answer(diagnosis, f"The evidence does not support {spelling}.")
        codes = {found["code"] for found in contradictions(answer, diagnosis)}
        assert "hedges_diagnosis" in codes, spelling


# --------------------------------------------------------------------------- #
# fallback
# --------------------------------------------------------------------------- #
def test_fallback_is_grounded_and_cannot_contradict_the_diagnosis():
    evidence = {"top_anomalous_sensors": [
        {"display_name": "Reactor_Pressure", "z_score": 18.1},
        {"name": "XMEAS_22", "z_score": 4.2},
    ]}
    answer = fallback_answer("FAULT_13", "classified", MENU, evidence)
    assert answer["fault_id"] == "FAULT_13"
    assert answer["ranked_hypotheses"][0] == "FAULT_13"
    assert "Reactor_Pressure" in answer["physical_reasoning"]
    assert contradictions(answer, "FAULT_13") == []
    assert validate_slim(answer, ALLOWED, "FAULT_13") == []


def test_fallback_ranking_stays_inside_the_menu():
    answer = fallback_answer("FAULT_05", "classified", ["FAULT_05"], {})
    assert answer["ranked_hypotheses"] == ["FAULT_05"]


# --------------------------------------------------------------------------- #
# classifier
# --------------------------------------------------------------------------- #
def test_feature_audit_finds_no_identifier_feature():
    pytest.importorskip("lightgbm")
    from scripts.evidence_classifier import feature_names, identifier_audit

    audit = identifier_audit(feature_names())
    assert audit["clean"] is True
    assert audit["offending_features"] == []


def test_classifier_training_is_deterministic():
    pytest.importorskip("lightgbm")
    import numpy as np

    import hybrid_classifier as hc

    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 8)).astype(np.float32)
    labels = ["FAULT_01", "FAULT_02", "FAULT_03"] * 20
    first, first_labels = hc.fit(features, labels)
    second, second_labels = hc.fit(features, labels)
    assert first_labels == second_labels
    left = first.predict_proba(features)
    right = second.predict_proba(features)
    assert np.allclose(left, right, atol=0.0, rtol=0.0)


def test_rows_are_deduplicated_per_event_for_training():
    """Both menu variants carry the same evidence; training must count the event once."""
    pytest.importorskip("lightgbm")
    import hybrid_classifier as hc

    items = [
        {"item_id": "a", "source_file": "f", "fault": 5, "simulationRun": 1,
         "variant": "menu", "prompt": "", "target": {}},
        {"item_id": "b", "source_file": "f", "fault": 5, "simulationRun": 1,
         "variant": "removed", "prompt": "", "target": {}},
    ]
    keys = {hc.event_key(item) for item in items}
    assert len(keys) == 1
    assert hc.true_label({"fault": 17}) == UNCLASSIFIED
    assert hc.true_label({"fault": None}) == NORMAL
