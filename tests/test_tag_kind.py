"""The report-path tag-kind rule: what it must catch, and what it must not.

The rule exists because §17 claimed 21 of 242 answers mislabelled a tag's kind. Hand-reading those
21 in 3B-7 showed none of them is a mislabel — they were a 40-character window reading past the end
of a clause. So the tests here are as much about **not** firing as about firing: a rule that
rejects correct prose is worse than no rule, because the report path would then rewrite an answer
that was right.
"""
from __future__ import annotations

import pytest

from llm.hybrid.tag_kind import (
    MEASUREMENT_QUANTITIES,
    VALVE_QUANTITIES,
    axis,
    canonical,
    mask,
    rejects,
    summarise,
    violations,
)


# --------------------------------------------------------------------------- #
# the tag itself
# --------------------------------------------------------------------------- #
def test_canonical_accepts_every_spelling_in_the_corpus():
    assert canonical("XMV 5") == "XMV_05"
    assert canonical("xmv_51") == "XMV_51"
    assert canonical("XMEAS_7") == "XMEAS_07"
    assert canonical("A_Feed_Flow (XMV_44)") == "XMV_44"


def test_axis_is_the_only_distinction_that_is_policed():
    assert axis("XMV_44") == "valve_position"
    assert axis("XMEAS_7") == "measurement"
    # It says nothing about which measurement an XMEAS tag carries: that needs the tag table.
    assert not rejects("XMEAS_22 is a temperature measurement")


# --------------------------------------------------------------------------- #
# genuine confusions must be caught
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,tag", [
    ("XMV_44 is a temperature measurement.", "XMV_44"),
    ("The XMV_50 signal is a pressure reading.", "XMV_50"),
    ("XMV_52 behaves as a flow of 30 kscfm.", "XMV_52"),
    ("The level of XMV_49 is rising.", "XMV_49"),
    ("This indicates XMEAS_19 is a valve position.", "XMEAS_19"),
    ("The valve position of XMEAS_21 is fixed.", "XMEAS_21"),
])
def test_attributed_confusions_are_rejected(text, tag):
    found = violations(text)
    assert found, text
    assert tag in {entry["tag"] for entry in found}
    assert rejects(text)


def test_every_measurement_quantity_is_covered_for_a_valve():
    for quantity in MEASUREMENT_QUANTITIES:
        sample = f"XMV_45 is a {quantity} reading."
        assert rejects(sample), quantity


def test_every_valve_word_is_covered_for_a_measurement():
    for word in VALVE_QUANTITIES:
        sample = f"XMEAS_21 shows a {word} that is stuck."
        assert rejects(sample), word


# --------------------------------------------------------------------------- #
# correct prose must not be caught — the part that went wrong before
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    # a neighbouring sensor's name: the artefact class that produced 20 of the 21
    "The Stripper_Steam_Valve (XMV_50) and Stripper_Temperature (XMEAS_18) both increase.",
    "Stripper_Liquid_Product_Flow (XMV_49) falls while Stripper_Level (XMEAS_15) holds.",
    # a downstream consequence, not an attribution
    "The A/C feed valve (XMV_45) is increasing, causing pressure rises in the reactor.",
    # another variable's movement
    "Reactor_Cooling_Water_Flow [XMV_51] increases with the inlet temperature.",
    # the loop, not the tag: `for` is associative
    "Check the flow measurement for XMV_52 first.",
    # a compound name that contains a quantity word
    "The A/C feed flow valve (XMV_45) is opening more.",
    # the kind printed in the prompt, quoted back correctly
    "XMV_51 is a valve position and XMEAS_21 is a measurement.",
    # a negated attribution states the distinction correctly
    "XMV_45 is not a pressure reading; it is a valve position.",
    "The trend in XMV_51 reflects an inlet temperature step rather than a temperature of the valve.",
])
def test_correct_prose_is_not_rejected(text):
    assert not rejects(text), summarise(text)


def test_masking_removes_names_and_kinds_but_keeps_the_tag():
    masked = mask("Stripper_Steam_Valve (XMV_50, valve position, manipulated variable) increases")
    assert "XMV_50" in masked
    assert "valve position" not in masked.lower()
    assert "Stripper_Steam_Valve" not in masked
    assert "NAME" in masked


def test_a_bare_mention_without_a_copula_is_not_an_attribution():
    # The old 40-character window called this a mislabel; it is two variables in one sentence.
    assert not rejects("XMEAS_19 and Stripper_Temperature (XMEAS_18) rise together.")


def test_findings_carry_the_clause_so_a_reader_can_check_them():
    found = violations("The evidence is thin. XMV_44 is a pressure reading. Check the valve.")
    assert len(found) == 1
    assert found[0]["clause"].rstrip(".") == "XMV_44 is a pressure reading"
    assert found[0]["quantity"] == "pressure"
    assert "XMV_44" in found[0]["rule"]


def test_rejects_is_a_pure_predicate():
    text = "XMV_44 is a temperature."
    assert rejects(text) == rejects(text) == bool(violations(text))
