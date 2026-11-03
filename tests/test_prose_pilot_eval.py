"""The pilot evaluation must score the prompts it built, and must not accept a bad parse."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.prose_pilot_prompts import build_orders, contradiction, parse_prose, read_gates


def menu_lines(prompt: str):
    return sorted(line for line in prompt.splitlines()
                  if line.startswith("- FAULT_") or line.startswith("- UNCLASSIFIED"))


def test_orders_use_the_training_prompt_shape():
    rows = build_orders()
    assert len(rows) == 520
    assert all(row["order_only"] for row in rows)
    prompt = rows[0]["prompts"]["kb"]
    assert "SUPPLIED CLASSIFICATION" in prompt
    assert 'keys: "physical_reasoning", "narrative"' in prompt
    assert "fault_id must be a menu entry" not in prompt


def test_orders_permute_only_the_menu():
    row = build_orders()[0]
    assert menu_lines(row["prompts"]["kb"]) == menu_lines(row["prompts"]["shuffled-1"])
    assert menu_lines(row["prompts"]["kb"]) == menu_lines(row["prompts"]["shuffled-2"])
    assert row["prompts"]["kb"] != row["prompts"]["shuffled-1"]


def test_parse_prose_shapes():
    assert parse_prose('{"physical_reasoning": "a", "narrative": "b"}')["schema_valid"]
    assert not parse_prose("no json here")["schema_valid"]
    assert not parse_prose('{"physical_reasoning": "a"}')["schema_valid"]
    assert parse_prose('{"physical_reasoning": 1, "narrative": "b"}')["schema_valid"] is False
    assert parse_prose('{"physical_reasoning": "a", "narrative": ""}')["narrative"] == ""


def test_contradiction_flags_a_different_id_and_any_id_when_withheld():
    assert contradiction("see FAULT_03", "FAULT_08") == ["FAULT_03"]
    assert contradiction("see FAULT_08", "FAULT_08") == []
    assert contradiction("see FAULT_06", "UNCLASSIFIED_OPEN_SET") == ["FAULT_06"]
    assert contradiction("see FAULT_06", "NORMAL_TRANSIENT") == ["FAULT_06"]
    assert contradiction("no ids here", "NORMAL_TRANSIENT") == []


def test_read_gates_marks_a_cap_hit_and_an_empty_narrative():
    row = build_orders()[1]
    clean = read_gates(row["prompts"]["kb"],
                       '{"physical_reasoning": "A_Feed_Flow fell.", "narrative": "Feed lost."}',
                       120, row["gold_fault_id"])
    assert clean["schema_valid"] and clean["complete"]
    assert not clean["empty_narrative"] and not clean["contradiction"]
    capped = read_gates(row["prompts"]["kb"],
                        '{"physical_reasoning": "FAULT_04 is the cause.", "narrative": ""}',
                        512, row["gold_fault_id"])
    assert capped["schema_valid"] is True and capped["complete"] is False
    assert capped["empty_narrative"] is True
    assert capped["contradiction"] == ["FAULT_04"]
