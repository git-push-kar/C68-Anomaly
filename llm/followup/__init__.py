"""Follow-up dialogue dataset (Phase 2C).

Everything here is about *follow-up* turns: questions an operator asks after the
automatic report, with the deterministic decisions (intent, table, citations,
clarification, refusal, suggestions) taken in code and only the natural answer
text left to a teacher model.

Layout::

    paraphrases.py  operator question bank, split into train / held-out halves
    cases.py        case taxonomy, mix, and the deterministic part of each turn
    filters.py      answer acceptance rules (grounding, refusal, repetition)
    teacher.py      prompt, response cache, fake teacher, optional Ollama hook
    leak.py         split/event overlap and question/context leak checks

Nothing in this package talks to a model by default: ``build_followup_dataset``
runs with the fake teacher unless ``--teacher ollama`` is passed explicitly.
"""
from __future__ import annotations

from llm.followup.cases import (
    CASE_TYPES,
    DEFAULT_MIX,
    FOLLOWUP_CASE_ORDER,
    ROUTING,
    TEMPLATE_CASE_TYPES,
)
from llm.followup.paraphrases import LABELS, heldout_questions, probe_questions, train_questions

__all__ = [
    "CASE_TYPES",
    "DEFAULT_MIX",
    "FOLLOWUP_CASE_ORDER",
    "LABELS",
    "ROUTING",
    "TEMPLATE_CASE_TYPES",
    "heldout_questions",
    "probe_questions",
    "train_questions",
]
