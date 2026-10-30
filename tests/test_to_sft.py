"""The SFT export must not re-introduce what the benchmark removed.

Each guard below exists because the trainer's default system prompt still carries
the fault->severity rubric: a row without a system would silently pick it up.
"""
from __future__ import annotations

import json

import pytest

from llm.benchmark import to_sft


def _item(**overrides):
    item = {
        "item_id": "EVT-train-f01-r001::menu",
        "prompt": "Evidence for EV-b05f8f4520a1 ...",
        "target": {
            "fault_id": "FAULT_01", "verdict": "classified", "abstain": False,
            "primary_subsystem": "feed_system",
            "ranked_hypotheses": [{"fault_id": "FAULT_01"}, {"fault_id": "FAULT_06"}],
            "mitigation": "Verify the A/C ratio controller.",
            "table_selection": "evidence",
            "suggested_followups": ["What are the sensor values?"],
            "clarifying_question": None,
        },
        "prose": {"physical_reasoning": "The ratio moved first.",
                  "narrative": "Check the ratio controller."},
    }
    item.update(overrides)
    return item


def test_row_has_the_three_trainer_fields():
    row = to_sft.to_row(_item())
    assert set(row) >= {"system", "question", "answer"}
    assert row["system"] == to_sft.SYSTEM_PROMPT_V2
    assert row["question"].startswith("Evidence for")


def test_answer_carries_no_severity_and_no_confidence():
    body = json.loads(to_sft.to_row(_item())["answer"])
    assert list(body) == list(to_sft.ANSWER_KEYS)
    assert "severity" not in body
    assert "confidence" not in body


def test_row_without_a_system_is_refused(monkeypatch):
    monkeypatch.setattr(to_sft, "SYSTEM_PROMPT_V2", "")
    with pytest.raises(to_sft.ExportGuardError):
        to_sft.to_row(_item())


def test_severity_wording_in_the_question_is_refused():
    with pytest.raises(to_sft.ExportGuardError):
        to_sft.to_row(_item(prompt="State the severity of this event."))


def test_subsystem_hint_in_the_question_is_refused():
    with pytest.raises(to_sft.ExportGuardError):
        to_sft.to_row(_item(prompt="Candidate affected subsystem: feed"))


def test_abstain_label_must_match_the_verdict():
    item = _item()
    item["target"] = {**item["target"], "abstain": True}  # still FAULT_01
    with pytest.raises(to_sft.ExportGuardError):
        to_sft.to_row(item)


def test_normal_excursion_may_abstain_without_a_subsystem():
    item = _item()
    item["target"] = {**item["target"], "fault_id": "NORMAL_TRANSIENT",
                      "abstain": True, "primary_subsystem": None,
                      "ranked_hypotheses": []}
    body = json.loads(to_sft.to_row(item)["answer"])
    assert body["abstain"] is True
    assert body["primary_subsystem"] is None
    assert body["ranked_hypotheses"] == []


def test_items_without_prose_are_not_exported():
    rows = to_sft.export([_item(), _item(item_id="b::menu", prose=None)])
    assert [row["item_id"] for row in rows] == ["EVT-train-f01-r001::menu"]


def test_missing_teacher_prose_is_refused():
    with pytest.raises(to_sft.ExportGuardError):
        to_sft.to_row(_item(prose={"physical_reasoning": "x"}))
