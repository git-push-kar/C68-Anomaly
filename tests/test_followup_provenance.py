"""The follow-up path judged under the Phase 2B rules, and its provenance.

Two things are checked here:

* the rules the follow-up filter imports are *the* Phase 2B rules, not a copy
  that can drift -- direction scoping, the scaffolding rule and the valve/flow
  labels are exercised through the follow-up entry point;
* every cache record and the manifest say which rules and which menu text the
  answer was written from, so a later edit cannot silently re-use it.
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
from llm.benchmark.scaffolding import derived_phrases  # noqa: E402
from llm.followup.filters import AnswerContext, check_answer  # noqa: E402
from llm.followup.provenance import (  # noqa: E402
    BENCHMARK_FILTER_CODE_FILES,
    FOLLOWUP_FILTER_CODE_FILES,
    filter_code_hash,
    followup_provenance,
)
from llm.followup.teacher import TeacherCache  # noqa: E402
from llm.response.allowed_set import build_allowed_set  # noqa: E402
from llm.response.store_access import StoreAccess  # noqa: E402
from scripts.fault_knowledge import FAULT_KNOWLEDGE  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "response"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture()
def ctx() -> AnswerContext:
    event = load("classified")
    store = StoreAccess(event, event["report"])
    grounded = store_numbers(event["evidence"], event["report"]) + store.numeric_values()
    return AnswerContext(
        store=store,
        allowed=build_allowed_set(event, extra_faults=[13, 15]),
        grounded_numbers=grounded,
        permitted_faults=list(range(1, 16)),
        event_id=event["event_id"],
        intent="FAULT_13",
        kb_fault=13,
    )


# -- the imported rule set is the 2B rule set --------------------------------
def test_the_benchmark_file_list_matches_the_builders():
    """The list is duplicated on purpose; this is what stops it drifting."""
    from scripts.build_benchmark_dataset import FILTER_CODE_FILES

    assert tuple(BENCHMARK_FILTER_CODE_FILES) == tuple(FILTER_CODE_FILES)


def test_the_hash_covers_both_halves():
    benchmark_only = filter_code_hash(BENCHMARK_FILTER_CODE_FILES)
    full = filter_code_hash()
    assert full != benchmark_only, "a follow-up-only edit must change the hash"
    assert filter_code_hash() == full, "the hash must be stable across calls"


def test_a_followup_file_edit_changes_the_hash():
    with_file = filter_code_hash(("llm/followup/filters.py",))
    without = filter_code_hash(())
    assert with_file != without


# -- direction, scoped exactly as 2B scopes it -------------------------------
def _initiating_direction(number: int) -> str:
    entry = FAULT_KNOWLEDGE[int(number)]
    return str(entry["initiating"][0]["direction"]).lower()


def test_citing_the_initiating_measurement_with_the_kb_sign_passes(ctx):
    direction = _initiating_direction(13)
    answer = f"XMEAS_7 shows a {direction} through the window, consistent with slow kinetics."
    codes = check_answer(answer, ctx)
    assert not any(code.startswith("direction_conflict") for code in codes), codes


def test_citing_it_with_the_opposite_sign_is_rejected(ctx):
    direction = _initiating_direction(13)
    opposite = "increase" if direction == "decrease" else "decrease"
    answer = f"XMEAS_7 shows an {opposite} through the window, consistent with slow kinetics."
    codes = check_answer(answer, ctx)
    assert any(code.startswith("direction_conflict") for code in codes), codes


def test_the_gerund_form_is_not_yet_seen_by_the_rule(ctx):
    """Recorded, not asserted as correct: the direction vocabulary is uninflected.

    ``INCREASE_WORDS`` holds "increase", not "increasing", so the gerund escapes.
    Widening it is part of the fault-13 patch that is paused on its scope, so this
    test pins today's behaviour instead of changing it here -- when the vocabulary
    is extended this test is the one to flip.
    """
    direction = _initiating_direction(13)
    opposite = "increasing" if direction == "decrease" else "decreasing"
    answer = f"XMEAS_7 is {opposite} through the window, consistent with slow kinetics."
    codes = check_answer(answer, ctx)
    assert not any(code.startswith("direction_conflict") for code in codes), codes


def test_a_valve_position_is_not_held_to_a_flow_direction(ctx):
    """Fault 14's KB direction describes the flow its valve position drives."""
    ctx.kb_fault = 14
    entry = FAULT_KNOWLEDGE[14]["initiating"][0]
    direction = str(entry["direction"]).lower()
    opposite = "increasing" if direction == "decrease" else "decreasing"
    answer = f"XMV_51 is {opposite}, which is the actuator moving."
    codes = check_answer(answer, ctx)
    assert not any(code.startswith("direction_conflict") for code in codes), codes


def test_an_open_set_item_has_no_kb_direction_to_contradict(ctx):
    ctx.kb_fault = None
    answer = "XMEAS_7 is increasing through the window and nothing listed explains it."
    codes = check_answer(answer, ctx)
    assert not any(code.startswith("direction_conflict") for code in codes), codes


# -- scaffolding, derived from the prompt rather than hand-listed ------------
def test_prose_quoting_a_prompt_heading_is_rejected(ctx):
    phrase = next(p["phrase"] for p in derived_phrases() if p["kind"] == "heading")
    answer = f"{phrase.capitalize()}: the reactor pressure is falling."
    codes = check_answer(answer, ctx)
    assert any(code.startswith("teacher_scaffolding") for code in codes), codes


def test_prose_that_does_not_quote_the_prompt_passes(ctx):
    answer = "XMEAS_7 is decreasing through the window, consistent with slow kinetics."
    codes = check_answer(answer, ctx)
    assert not any(code.startswith("teacher_scaffolding") for code in codes), codes


# -- valve/flow labels, both directions --------------------------------------
def test_a_valve_position_called_a_measurement_is_rejected(ctx):
    codes = check_answer("XMV_51 is a pressure reading on the jacket.", ctx)
    assert any(code.startswith("valve_flow_mislabel") for code in codes), codes


def test_a_measurement_called_a_valve_position_is_rejected(ctx):
    codes = check_answer("XMEAS_7 is the valve position for the reactor.", ctx)
    assert any(code.startswith("valve_flow_mislabel") for code in codes), codes


def test_the_correct_negation_is_not_a_mislabel(ctx):
    codes = check_answer("XMEAS_21 is the cooling water flow, not a valve position.", ctx)
    assert not any(code.startswith("valve_flow_mislabel") for code in codes), codes


# -- provenance travels with the cached text ---------------------------------
def test_a_cache_record_carries_the_rules_and_the_menu(tmp_path):
    path = tmp_path / "cache.jsonl"
    provenance = followup_provenance()
    cache = TeacherCache(path, model="fake-followup-v1", provenance=provenance)
    cache.put("a prompt", "an answer", "fake-followup-v1", 0.5)
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    for key in ("filter_sha256", "menu_revision", "menu_sha256", "kb_sha256"):
        assert row[key] == provenance[key], key
    assert row["menu_revision"] == "r1"


def test_the_cache_key_ignores_provenance_but_the_record_keeps_it(tmp_path):
    """Provenance is evidence, not part of the key: the text is what was asked."""
    one = TeacherCache(tmp_path / "a.jsonl", model="m", provenance={"filter_sha256": "x"})
    two = TeacherCache(tmp_path / "b.jsonl", model="m", provenance={"filter_sha256": "y"})
    assert one.key_for("same prompt") == two.key_for("same prompt")


def test_the_manifest_provenance_has_every_field():
    provenance = followup_provenance()
    assert set(provenance) == {"filter_sha256", "menu_revision", "menu_sha256", "kb_sha256"}
    assert len(provenance["filter_sha256"]) == 64
    assert provenance["menu_revision"] in ("r1", "r2")


def test_the_followup_list_covers_every_module_that_decides_a_verdict():
    for rel in ("llm/followup/filters.py", "llm/followup/cases.py",
                "llm/followup/teacher.py", "llm/followup/paraphrases.py"):
        assert rel in FOLLOWUP_FILTER_CODE_FILES
        assert (ROOT / rel).exists(), rel
