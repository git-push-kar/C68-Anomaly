"""Follow-up SFT rows: one per turn, with the transcript as context.

The trainer learns one assistant answer per row, so a dialogue becomes several
rows and the earlier turns travel in the question. These tests pin the two things
that can silently go wrong: a rejected turn must not leave later turns in the
dataset, and a question must not name a fault the incident's context never had.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark import to_sft_followup as F  # noqa: E402

CONTEXT = ("INCIDENT INC-869302\n\nLEADING DIAGNOSIS - FAULT_01 (A/C feed ratio step change)\n"
           "Detector readings: LSTM 21.07 against a limit of 0.6870")


def _dialogue(**overrides) -> dict:
    dialogue = {
        "dialogue_id": "FD-train-00001",
        "fault": 1,
        "open_set": False,
        "context": CONTEXT,
        "turns": [
            {"question": "What are the detector scores?", "answer": "LSTM is 21.07.",
             "accepted": True},
            {"question": "Why does that matter?", "answer": "It is above the limit.",
             "accepted": True},
        ],
    }
    dialogue.update(overrides)
    return dialogue


def test_one_row_per_accepted_turn():
    rows = F.rows_for_dialogue(_dialogue())
    assert len(rows) == 2
    assert [row["item_id"] for row in rows] == ["FD-train-00001#0", "FD-train-00001#1"]


def test_the_answer_is_the_single_key_the_teacher_was_asked_for():
    row = F.rows_for_dialogue(_dialogue())[0]
    assert json.loads(row["answer"]) == {"answer": "LSTM is 21.07."}
    assert row["system"] == F.SYSTEM_PROMPT_FOLLOWUP


def test_the_earlier_turns_travel_in_the_question():
    rows = F.rows_for_dialogue(_dialogue())
    assert "CONVERSATION SO FAR" not in rows[0]["question"]
    assert "What are the detector scores?" in rows[1]["question"]
    assert "LSTM is 21.07." in rows[1]["question"]
    assert rows[1]["question"].rstrip().endswith("Why does that matter?")


def test_a_rejected_turn_ends_the_dialogue():
    dialogue = _dialogue()
    dialogue["turns"][0]["accepted"] = False
    assert F.rows_for_dialogue(dialogue) == []


def test_an_empty_answer_ends_the_dialogue():
    dialogue = _dialogue()
    dialogue["turns"][1]["answer"] = "   "
    assert len(F.rows_for_dialogue(dialogue)) == 1


def test_a_fault_outside_the_incident_is_refused():
    """The context names fault 1; a question naming fault 9 is not answerable."""
    dialogue = _dialogue()
    dialogue["turns"][0]["question"] = "Is this fault 9 again?"
    with pytest.raises(F.FollowupExportError):
        F.rows_for_dialogue(dialogue)


def test_an_open_set_dialogue_may_not_name_its_own_fault():
    dialogue = _dialogue(open_set=True)
    dialogue["turns"][0]["question"] = "So this is fault 1?"
    with pytest.raises(F.FollowupExportError):
        F.rows_for_dialogue(dialogue)


def test_severity_wording_is_refused_in_a_question():
    dialogue = _dialogue()
    dialogue["turns"][0]["question"] = "What severity is this?"
    with pytest.raises(F.FollowupExportError):
        F.rows_for_dialogue(dialogue)


def test_the_mix_is_two_report_rows_per_dialogue_row():
    report = [{"item_id": f"R{i}"} for i in range(10)]
    dialogues = [{"item_id": f"D0#{i}"} for i in range(4)]
    mixed = F.mix(report, dialogues, ratio=2, seed=1)
    assert mixed["report_rows"] == 10
    assert mixed["dialogue_rows"] == 4          # budget 10//2 = 5, but only 4 exist
    assert mixed["achieved_ratio"] == 2.5
    assert len(mixed["rows"]) == 14


def test_the_mix_keeps_whole_dialogues():
    report = [{"item_id": f"R{i}"} for i in range(6)]
    dialogues = [{"item_id": "A#0"}, {"item_id": "A#1"}, {"item_id": "B#0"}]
    mixed = F.mix(report, dialogues, ratio=2, seed=3)
    ids = {row["item_id"].split("#")[0] for row in mixed["rows"] if row["item_id"].startswith(("A", "B"))}
    for key in ids:
        expected = {row["item_id"] for row in dialogues if row["item_id"].startswith(key)}
        present = {row["item_id"] for row in mixed["rows"] if row["item_id"].startswith(key)}
        assert present == expected, key


def test_the_mix_reports_the_ratio_it_achieved():
    report = [{"item_id": f"R{i}"} for i in range(100)]
    dialogues = [{"item_id": f"D{i}#0"} for i in range(30)]
    mixed = F.mix(report, dialogues, ratio=2, seed=4)
    assert mixed["dialogue_rows"] == 30
    assert mixed["achieved_ratio"] == round(100 / 30, 3)


def _corpus(sizes: dict) -> tuple:
    """A dialogue corpus from {case: [rows per dialogue]}, one row per turn."""
    dialogues, case_of, n = [], {}, 0
    for case, shapes in sizes.items():
        for shape in shapes:
            key = f"D{n:03d}"
            for turn in range(shape):
                dialogues.append({"item_id": f"{key}#{turn}"})
            case_of[key] = case
            n += 1
    return dialogues, case_of


def test_each_case_is_filled_to_its_own_target():
    report = [{"item_id": f"R{i}"} for i in range(32)]      # budget 16
    dialogues, case_of = _corpus({"out_of_scope": [1] * 15, "valve_actuator": [1] * 30})
    mixed = F.mix(report, dialogues, ratio=2, seed=7, case_of=case_of,
                  targets={"out_of_scope": 6, "valve_actuator": 10}, tolerance=0)
    assert mixed["by_case"]["out_of_scope"]["rows_taken"] == 6
    assert mixed["by_case"]["valve_actuator"]["rows_taken"] == 10
    assert mixed["dialogue_rows"] == 16
    assert mixed["target_deviation"] == 0


def test_a_whole_dialogue_may_carry_a_case_past_its_target():
    """Whole dialogues are the unit, so a 3-turn dialogue lands where 2 would."""
    report = [{"item_id": f"R{i}"} for i in range(40)]      # budget 20
    dialogues, case_of = _corpus({"mechanism": [3] * 4, "causal_why": [1] * 20})
    mixed = F.mix(report, dialogues, ratio=2, seed=2, case_of=case_of,
                  targets={"mechanism": 4, "causal_why": 4}, tolerance=0)
    assert mixed["by_case"]["mechanism"]["rows_taken"] == 6      # 2 dialogues x 3
    assert mixed["by_case"]["mechanism"]["target"] == 4


def test_a_target_beyond_the_corpus_is_refused():
    report = [{"item_id": f"R{i}"} for i in range(40)]
    dialogues, case_of = _corpus({"template_sample": [1] * 3})
    with pytest.raises(ValueError, match="exceeds"):
        F.mix(report, dialogues, ratio=2, seed=1, case_of=case_of,
              targets={"template_sample": 10}, tolerance=0)


def test_the_total_is_balanced_back_to_the_slot_count():
    """1-turn dialogues and whole-dialogue rounding still hit the budget."""
    report = [{"item_id": f"R{i}"} for i in range(40)]      # budget 20
    dialogues, case_of = _corpus({"out_of_scope": [2] * 20, "mechanism": [1] * 40})
    mixed = F.mix(report, dialogues, ratio=2, seed=3, case_of=case_of,
                  targets={"out_of_scope": 8, "mechanism": 8}, tolerance=3)
    assert abs(mixed["dialogue_rows"] - 20) <= 3


def test_case_selection_is_seeded_not_file_ordered():
    report = [{"item_id": f"R{i}"} for i in range(40)]      # budget 20
    dialogues, case_of = _corpus({"mechanism": [1] * 40})
    first = F.mix(report, dialogues, ratio=2, seed=1, case_of=case_of,
                  targets={"mechanism": 20}, tolerance=0)["rows"]
    second = F.mix(report, dialogues, ratio=2, seed=2, case_of=case_of,
                   targets={"mechanism": 20}, tolerance=0)["rows"]
    ids_first = [r["item_id"] for r in first if r["item_id"].startswith("D")]
    ids_second = [r["item_id"] for r in second if r["item_id"].startswith("D")]
    assert len(ids_first) == len(ids_second) == 20
    assert ids_first != ids_second


def test_the_shipped_targets_add_up_to_the_dialogue_budget():
    assert F.TARGET_ROWS == 959
    assert sum(F.CASE_TARGETS.values()) == F.TARGET_ROWS


def test_without_a_case_map_the_old_seeded_order_is_kept():
    report = [{"item_id": f"R{i}"} for i in range(10)]
    dialogues = [{"item_id": f"D{i}#0"} for i in range(4)]
    mixed = F.mix(report, dialogues, ratio=2, seed=1)
    assert mixed["dialogue_rows"] == 4
    assert mixed["target_rows"] is None and "by_case" in mixed
