"""Menu order as a controlled variable (PROTOCOL §18.5).

The candidate menu is a list, and a list has an order. A model that reads the menu can be
influenced by that order — a known failure mode for multiple-choice prompts — while the
classifier half of the hybrid cannot be, because the menu rule is a membership test and the
classifier never reads the menu at all. This module makes that difference testable:

* :func:`reorder_menu` rewrites the menu block of an already-rendered prompt into a seeded,
  per-item permutation, so adapter numbers can be reported in KB order and in shuffled order
  without re-rendering the evidence (the prompt text is what the decoder consumes).
* :func:`menu_ids` reads the entries back in order, for the check that only the order moved.

Determinism is per ``(seed, item_id)``: the same item shuffles the same way on every run, so a
shuffled run is reproducible and can be cached like any other.
"""
from __future__ import annotations

import random
import re
from typing import List, Sequence

from llm.benchmark.prompt_v2 import MENU_MARKER

_ENTRY = re.compile(r"^- (\S+): ")


def _block(prompt: str):
    """(start, end) line indices of the menu's entry lines, or ``None``."""
    lines = str(prompt or "").split("\n")
    start = None
    for index, line in enumerate(lines):
        if line.startswith(MENU_MARKER):
            start = index
            break
    if start is None:
        return None
    end = start + 1
    while end < len(lines) and lines[end].startswith("- "):
        end += 1
    return start, end


def menu_ids(prompt: str) -> List[str]:
    """The menu entries' ids, in the order they appear. Empty when there is no menu."""
    found = _block(prompt)
    if not found:
        return []
    start, end = found
    lines = str(prompt).split("\n")
    ids: List[str] = []
    for line in lines[start + 1:end]:
        match = _ENTRY.match(line)
        if match:
            ids.append(match.group(1))
    return ids


def permute(entries: Sequence[str], seed: int, item_id: str) -> List[str]:
    """A seeded permutation of ``entries``, fixed per ``(seed, item_id)``."""
    order = list(entries)
    random.Random(f"{seed}|{item_id}").shuffle(order)
    return order


def reorder_menu(prompt: str, seed: int, item_id: str) -> str:
    """Rewrite the menu block of one prompt in a seeded order.

    A prompt with no menu (a follow-up turn, for instance) is returned unchanged rather than
    raising: the harness should not have to know which rows carry a menu.
    """
    found = _block(prompt)
    if not found:
        return str(prompt or "")
    start, end = found
    lines = str(prompt).split("\n")
    entries = lines[start + 1:end]
    if len(entries) < 2:
        return "\n".join(lines)
    return "\n".join(lines[:start + 1] + permute(entries, seed, item_id) + lines[end:])


__all__ = ["menu_ids", "permute", "reorder_menu"]
