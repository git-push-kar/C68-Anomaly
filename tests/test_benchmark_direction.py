"""The fault-13 direction correction, and the gerund vocabulary that goes with it.

Two changes that only make sense together:

* the KB entry for fault 13 said Reactor_Pressure *decreases* first while the
  stored trend is increasing in 102 of 102 Protocol A train events where it is
  ranked, so the rule built on it rejected correct prose;
* the rule read ``increase``/``rise`` but not ``increasing``/``rising``, so a
  sentence asserting the same thing in the other grammatical form passed -- the
  rule was blind to about half the claims it was supposed to judge.

Each test states both directions: what is accepted, and the near-miss that must
still be rejected. A one-sided test would pass with a rule that accepts
everything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark import filters as F
from llm.benchmark.menu import signature_for
from llm.benchmark.provenance import MENU_REVISION, menu_sha256
from scripts.fault_knowledge import FAULT_KNOWLEDGE


def _initiating(fault: int) -> dict:
    entry = FAULT_KNOWLEDGE[fault]["initiating"]
    return entry[0] if isinstance(entry, list) else entry


# -- the data-derived KB correction ------------------------------------------
def test_fault_13_initiating_direction_is_increase():
    assert _initiating(13)["direction"] == "increase"
    assert _initiating(13)["tag"] == "XMEAS_7"


def test_the_menu_line_the_student_reads_says_increases():
    """The menu is the only place the claim reaches the model, so it is checked."""
    line = signature_for(13)
    assert "increases first" in line
    assert "decreases first" not in line


def test_no_other_menu_line_changed_with_it():
    """A one-entry correction must not rewrite another fault's signature."""
    assert signature_for(4).endswith("increases first")
    assert signature_for(1).endswith("decreases first")
    assert signature_for(14).endswith("decreases first")


# -- the rule agrees with the evidence now ------------------------------------
def test_increase_wording_for_fault_13_is_accepted():
    for text in ("XMEAS_7 rises within the first minutes",
                 "XMEAS_7 is increasing before anything else moves",
                 "Reactor_Pressure increases first"):
        assert F._direction_conflict(text, 13) is None, text


def test_decrease_wording_for_fault_13_is_rejected():
    for text in ("XMEAS_7 falls before anything else moves",
                 "XMEAS_7 is decreasing slowly",
                 "XMEAS_7 drops slowly"):
        assert F._direction_conflict(text, 13) is not None, text


def test_the_rule_reads_the_tag_not_the_display_name():
    """A known limit of the rejection rule, measured rather than assumed.

    ``_direction_conflict`` searches for the canonical *tag*. Prose that names
    only ``Reactor_Pressure`` is not seen by it, which is why the display-name
    audit exists as a separate advisory flag -- promoting that flag to a
    rejection is an owner decision, not something this rule already does.
    """
    assert F._direction_conflict("Reactor_Pressure is decreasing", 13) is None
    assert F._direction_conflict("Reactor_Pressure [XMEAS_7] is decreasing", 13) is not None


# -- the gerund vocabulary, in both directions -------------------------------
def test_gerunds_count_for_a_decrease_fault():
    """Fault 3's initiating measurement decreases, so a gerund up is a conflict."""
    assert F._direction_conflict("XMEAS_2 is decreasing", 3) is None
    assert F._direction_conflict("XMEAS_2 is increasing", 3) is not None
    assert F._direction_conflict("XMEAS_2 rising", 3) is not None
    assert F._direction_conflict("XMEAS_2 falling", 3) is None


def test_the_gerunds_are_in_the_vocabulary_not_matched_by_prefix():
    """An explicit list, so a future edit cannot silently lose one form."""
    assert "increasing" in F.INCREASE_WORDS and "rising" in F.INCREASE_WORDS
    assert "decreasing" in F.DECREASE_WORDS and "falling" in F.DECREASE_WORDS


def test_valve_positions_stay_out_of_scope():
    """Fault 14's initiating tag is a valve position; a flow direction is not its."""
    assert F._direction_conflict("XMV_51 is decreasing", 14) is None
    assert F._direction_conflict("XMV_51 is increasing", 14) is None


def test_a_stable_reading_is_not_a_conflict():
    assert F._direction_conflict("XMEAS_7 is stable at the setpoint", 13) is None


# -- the advisory display-name flag -------------------------------------------
_EVIDENCE = {
    "top_anomalous_sensors": [
        {"name": "XMEAS_7", "display_name": "Reactor_Pressure",
         "z_score": 1.2, "deviation_percent": -2.0, "trend": "increasing"},
        {"name": "XMV_51", "display_name": "Reactor_Cooling_Water_Flow",
         "z_score": 2.0, "deviation_percent": 8.0, "trend": "decreasing"},
    ],
}


def test_the_display_flag_catches_what_the_tag_rule_misses():
    prose = {"physical_reasoning": "Reactor_Pressure is decreasing through the window.",
             "narrative": "Check the reactor pressure."}
    assert F.direction_inconsistent_display(prose, _EVIDENCE), "display name not seen"
    assert F._direction_conflict("Reactor_Pressure is decreasing", 13) is None


def test_the_display_flag_accepts_the_agreeing_sentence():
    prose = {"physical_reasoning": "Reactor_Pressure is rising through the window.",
             "narrative": "Check the reactor pressure."}
    assert F.direction_inconsistent_display(prose, _EVIDENCE) == []


def test_the_display_flag_skips_valve_positions_and_negations():
    valve = {"physical_reasoning": "Reactor_Cooling_Water_Flow is increasing.",
             "narrative": "Check the valve."}
    assert F.direction_inconsistent_display(valve, _EVIDENCE) == []
    negated = {"physical_reasoning": "Reactor_Pressure is not decreasing.",
               "narrative": "Check the reactor pressure."}
    assert F.direction_inconsistent_display(negated, _EVIDENCE) == []


def test_the_display_flag_is_not_a_rejection():
    """Advisory by construction: check_answer's codes come from the tag rule only."""
    assert "direction_inconsistent_display" not in dir(F.check_answer)
    assert F.direction_inconsistent_display.__module__ == "llm.benchmark.filters"


# -- the menu revision label --------------------------------------------------
def test_the_revision_label_matches_the_published_text():
    """A forgotten bump is the failure this guards: r2 must mean this text."""
    recorded = json.loads((ROOT / "data/menu_revisions.json").read_text(encoding="utf-8"))
    assert recorded["current"] == MENU_REVISION
    assert recorded["revisions"][MENU_REVISION] == menu_sha256()
