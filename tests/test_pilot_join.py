"""The pilot evaluation must join generations to prompts by item_id (PROTOCOL 27.6)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.build_prose_dataset import load_jsonl  # noqa: E402
from scripts.pilot_join_check import EVAL_MODULES, positional_offences  # noqa: E402
from scripts.prose_pilot_prompts import EXPORT, build_orders  # noqa: E402
from scripts.prose_pilot_score import ORDERS, readings_for  # noqa: E402

BUILT = build_orders()
PROMPTS = {row["item_id"]: row["prompts"] for row in BUILT}


def test_every_generation_resolves_to_its_own_prompt():
    for order in ORDERS:
        for row in load_jsonl(EXPORT / ("raw_pilot_" + order + ".jsonl")):
            assert row["order"] == order
            assert order in PROMPTS.get(row["item_id"], {})


def test_permuted_raw_file_scores_identically(tmp_path):
    rows = load_jsonl(EXPORT / "raw_pilot_kb.jsonl")
    permuted = tmp_path / "permuted.jsonl"
    permuted.write_text(chr(10).join(json.dumps(row) for row in reversed(rows)) + chr(10),
                        encoding="utf-8")
    before = {reading["item_id"]: reading for reading in readings_for(rows, PROMPTS, "kb")}
    after = {reading["item_id"]: reading
             for reading in readings_for(load_jsonl(permuted), PROMPTS, "kb")}
    assert set(before) == set(after)
    assert all(before[item] == after[item] for item in before)


def test_followups_carry_a_dialogue_id_from_the_source():
    source = {str(turn["dialogue_id"]) for turn in
              load_jsonl(EXPORT / "prose_eval_followups_179.jsonl")}
    for row in load_jsonl(EXPORT / "raw_pilot_followups.jsonl"):
        assert str(row["dialogue_id"]) in source


def test_no_evaluation_module_joins_by_position():
    paths = [ROOT / "scripts" / name for name in EVAL_MODULES]
    assert positional_offences(paths) == []
