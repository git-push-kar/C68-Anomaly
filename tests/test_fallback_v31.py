"""Fallback v3.1 closes the sentence v3 cut, and the fixes are measurable, not asserted.

The abstention explainer listed its candidates as `FAULT_16 (initiating XMEAS_9)`. Those tags never
deviate — that is why the fault is named as absent — so v3's grounding guard dropped the clause and
left `"…, of the listed faults, so neither is supported by this page."` (PROTOCOL §23.3(i)).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fallback_v31 import (close_text, effective_verdict, signal_strength,  # noqa: E402
                                 structural_issues, trend_agrees)

TRUNCATED = ("No listed fault is asserted for this event (UNCLASSIFIED_OPEN_SET). Evidence: detector "
             "ratios are LSTM 1.20x and CVA 3.89x, the leading deviations are Stripper_Steam_Flow "
             "(XMEAS_19, z=0.37, stable), of the listed faults, so neither is supported by this page.")
CLOSED = ("No listed fault is asserted for this event (UNCLASSIFIED_OPEN_SET). Of the listed faults, "
          "FAULT_16 and FAULT_20 would each start at a tag that is not among these deviations, so "
          "neither is supported by this page.")


def test_the_truncated_sentence_is_flagged():
    issues = structural_issues(TRUNCATED)
    assert any("orphan conclusion" in issue for issue in issues)
    assert any("candidate opener" in issue for issue in issues)


def test_a_closed_sentence_is_not_flagged():
    assert structural_issues(CLOSED) == []


def test_close_text_drops_the_fragment_rather_than_printing_it():
    kept = close_text(TRUNCATED, {})
    assert "so neither is supported" not in kept


def test_unbalanced_parenthesis_is_flagged():
    assert any("unbalanced parentheses" in issue for issue in structural_issues("Flow (XMEAS_19, z=0.4."))


def test_non_fault_diagnosis_is_an_abstention():
    assert effective_verdict("UNCLASSIFIED_OPEN_SET", "classified") == "unclassified_open_set"
    assert effective_verdict("NORMAL_TRANSIENT", "classified") == "no_fault"
    assert effective_verdict("FAULT_04", "classified") == "classified"
    assert effective_verdict("FAULT_04", "no_fault") == "no_fault"


def test_trend_word_agrees_with_the_printed_deviation():
    assert trend_agrees({"trend": "increasing", "dev": "3.2"})
    assert not trend_agrees({"trend": "increasing", "dev": "-3.2"})
    assert trend_agrees({"trend": "decreasing", "dev": "-0.4"})
    assert not trend_agrees({"trend": "decreasing", "dev": "0.4"})


def test_low_signal_needs_both_a_small_z_and_a_small_ratio():
    small = {"sensors": [{"z": "0.4"}], "detectors": [{"ratio": "1.9"}]}
    assert signal_strength(small)["low"]
    assert not signal_strength({"sensors": [{"z": "0.4"}], "detectors": [{"ratio": "4.1"}]})["low"]
    assert not signal_strength({"sensors": [{"z": "2.4"}], "detectors": [{"ratio": "1.9"}]})["low"]
