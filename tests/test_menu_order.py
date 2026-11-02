"""Menu order: the harness variant, and the invariance the hybrid claims (PROTOCOL §18.5).

Two separate claims are tested here, and they are not the same claim:

* the **shuffled-menu variant** really does move only the menu — same entries, same text
  outside the block — and does so reproducibly per item;
* the **classifier plus menu rule** cannot see the order at all. That is asserted directly,
  not argued: the feature extractor is run on a prompt and on its permutation, and the
  decision function is run over many orders of the same menu.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.benchmark.menu import build_menu
from llm.benchmark.menu_order import menu_ids, permute, reorder_menu
from llm.benchmark.prompt_v2 import render_prompt
from llm.hybrid.menu_rule import NORMAL, UNCLASSIFIED, menu_rule
from scripts.evidence_classifier import feature_names, prompt_features

ROOT = Path(__file__).resolve().parents[1]


def _prompt(menu=None, fault: int = 7) -> str:
    rows = [json.loads(line) for line in
            (ROOT / "outputs/benchmark_v2/dataset/val.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    row = next(r for r in rows if r.get("variant") == "menu" and r.get("fault") == fault)
    text = row["prompt"]
    if menu is not None:
        start = text.index("CANDIDATE MENU")
        head, tail = text[:start], text[start:]
        lines = tail.split("\n")
        end = 1
        while end < len(lines) and lines[end].startswith("- "):
            end += 1
        lines = lines[:1] + [f"- {entry['fault_id']}: {entry['signature']}" for entry in menu] + lines[end:]
        text = head + "\n".join(lines)
    return text


def test_reorder_moves_only_the_menu():
    prompt = _prompt()
    shuffled = reorder_menu(prompt, seed=42, item_id="EVT-val-f07-r301::menu")
    assert menu_ids(shuffled) != menu_ids(prompt), "a 16-entry menu should not survive a shuffle unchanged"
    assert sorted(menu_ids(shuffled)) == sorted(menu_ids(prompt))
    assert "CANDIDATE MENU" in shuffled
    # Everything outside the block is byte-identical.
    before, after = prompt.split("CANDIDATE MENU"), shuffled.split("CANDIDATE MENU")
    assert before[0] == after[0]
    assert before[1].split("\n")[-1] == after[1].split("\n")[-1]


def test_reorder_is_deterministic_per_item_and_varies_across_items():
    prompt = _prompt()
    first = reorder_menu(prompt, seed=42, item_id="EVT-val-f07-r301::menu")
    again = reorder_menu(prompt, seed=42, item_id="EVT-val-f07-r301::menu")
    other = reorder_menu(prompt, seed=42, item_id="EVT-val-f07-r302::menu")
    assert first == again
    assert first != other
    assert permute(["a", "b", "c", "d", "e"], 42, "x") == permute(["a", "b", "c", "d", "e"], 42, "x")


def test_a_prompt_without_a_menu_is_returned_unchanged():
    text = "An operator asks about the event. No menu here.\nAnswer briefly."
    assert reorder_menu(text, seed=1, item_id="x") == text
    assert menu_ids(text) == []


def test_classifier_features_cannot_see_the_menu():
    """The whole order-invariance claim for the classifier, in one assertion.

    The features are parsed from the evidence lines only, so permuting the menu leaves the
    feature vector — and therefore the prediction — bit-identical. A feature that read the menu
    would fail this immediately.
    """
    prompt = _prompt()
    baseline = prompt_features(prompt)
    for seed in (0, 1, 7, 42, 2026):
        shuffled = reorder_menu(prompt, seed=seed, item_id="EVT-val-f07-r301::menu")
        assert prompt_features(shuffled) == baseline
    assert not any("menu" in name.lower() for name in feature_names())


def test_a_permuted_menu_cannot_change_the_rule():
    menu = build_menu()
    ids = [entry["fault_id"] for entry in menu]
    orders = [permute(ids, seed=seed, item_id="EVT-x") for seed in range(12)]
    for label in ("FAULT_07", "FAULT_15", NORMAL, UNCLASSIFIED, "FAULT_03"):
        decisions = []
        for order in orders:
            assert sorted(order) == sorted(ids)
            decisions.append({key: value for key, value in
                              menu_rule(label, 0.5, order, 0.9792).items() if key != "confidence"})
        assert all(decision == decisions[0] for decision in decisions), label


def test_a_removed_entry_stays_removed_under_any_order():
    """An open-set episode's missing entry must not reappear because the list moved."""
    menu = build_menu(remove=[7])
    ids = [entry["fault_id"] for entry in menu]
    assert "FAULT_07" not in ids
    for seed in range(6):
        shuffled = permute(ids, seed=seed, item_id="EVT-y")
        decision = menu_rule("FAULT_07", 0.99, shuffled, 0.9792)
        assert decision["diagnosis"] == UNCLASSIFIED
        assert decision["reason"] == "not_on_menu"


def test_the_rendered_menu_keeps_its_kb_order_by_default():
    """`render_prompt` is untouched: KB order is still what every stored prompt carries."""
    prompt = _prompt()
    ids = menu_ids(prompt)
    assert ids == [entry["fault_id"] for entry in build_menu()]
    assert ids[-1] == UNCLASSIFIED
    rebuilt = render_prompt({"event_id": "EVT-train-f07-r301"}, build_menu())
    assert menu_ids(rebuilt) == [entry["fault_id"] for entry in build_menu()]
