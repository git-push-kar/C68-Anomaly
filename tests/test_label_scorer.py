"""The label scorer must read the field the exporter writes.

A scorer that looks at the wrong token is the quietest possible failure: it still
produces a label and a probability, and they are meaningless. These tests pin the
prefix against the exporter's own JSON and check the probability arithmetic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark import label_scorer as L  # noqa: E402
from llm.benchmark.to_sft import ANSWER_KEYS  # noqa: E402
from llm.benchmark.menu import UNCLASSIFIED  # noqa: E402


def test_the_prefix_is_how_the_exported_answer_actually_starts():
    body = json.dumps({"fault_id": "FAULT_01", "verdict": "classified",
                       "ranked_hypotheses": [], "physical_reasoning": "x",
                       "narrative": "y"})
    assert body.startswith(L.ANSWER_PREFIX)
    assert list(json.loads(body)) == list(ANSWER_KEYS)


def test_fault_id_is_the_first_field_so_one_pass_is_enough():
    assert ANSWER_KEYS[0] == "fault_id"


def test_the_candidates_are_the_menu_and_include_abstention():
    labels = L.candidates()
    assert UNCLASSIFIED in labels
    assert labels[0] == "FAULT_01"
    assert len(labels) == 16  # faults 1-15 plus UNCLASSIFIED_OPEN_SET


def test_softmax_is_normalised_and_order_preserving():
    probs = L._softmax({"A": -1.0, "B": -3.0, "C": -7.0})
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert probs["A"] > probs["B"] > probs["C"]


def test_softmax_of_one_label_is_one():
    assert L._softmax({"A": -4.2}) == {"A": 1.0}


def test_the_prompt_uses_the_v2_system_prompt():
    from llm.benchmark.prompt_v2 import SYSTEM_PROMPT_V2

    prompt = L.prompt_for("What are the detector scores?")
    assert SYSTEM_PROMPT_V2 in prompt
    assert "CANDIDATE MENU" not in prompt, "the scorer prompt is built from the ×answer× side"


def test_the_explicit_mask_keeps_padding_out_and_stays_causal():
    """What makes a batched label scan comparable with an unbatched one.

    Two rows of different length, both left-padded to 4: row 0 has one real
    token, row 1 has three. The mask must let a query see only earlier keys and
    only real ones, whatever padding surrounds it.
    """
    torch = pytest.importorskip("torch")
    valid = torch.tensor([[0, 0, 0, 1], [1, 1, 1, 0]])
    mask = L._causal_mask(valid, q_len=4, kv_len=4, dtype=torch.float32)
    assert tuple(mask.shape) == (2, 1, 4, 4)
    blocked = torch.finfo(torch.float32).min
    # row 0: the only real token is at index 3, and it may see only itself
    assert mask[0, 0, 3, 3].item() == 0.0
    assert mask[0, 0, 3, 2].item() == blocked
    # padding is never attended, even by a later query
    assert mask[0, 0, 3, 0].item() == blocked
    # row 1: index 2 may see 0 and 1 and itself, never the trailing pad
    assert mask[1, 0, 2, 0].item() == 0.0
    assert mask[1, 0, 2, 3].item() == blocked
    # a step (q_len 1) sees every earlier real key and nothing masked
    step = L._causal_mask(torch.tensor([[1, 1, 1, 1, 1]]), q_len=1, kv_len=5,
                          dtype=torch.float32)
    assert tuple(step.shape) == (1, 1, 1, 5)
    assert step[0, 0, 0].tolist() == [0.0] * 5
