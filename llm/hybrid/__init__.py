"""Hybrid diagnosis: the classifier decides, the adapter explains.

The run-1 result that motivates this package is uncomfortable but clear: a tabular model on
the numbers in the prompt beats the adapter at naming the fault (0.9940 against 0.9199 on the
same 497 validation items), and the adapter's errors are the events where the prompt's
top-three sensor lines carry nothing. Naming a fault is a tabular decision; explaining it is
not. This package splits the two.

Nothing here trains the adapter, edits the prompts or the datasets, or touches a test
component. The pre-registration is PROTOCOL §16.
"""
from __future__ import annotations

from llm.hybrid.generate import (  # noqa: F401
    SLIM_KEYS,
    contradictions,
    continuation_prompt,
    fallback_answer,
    forced_prefix,
    parse_slim,
    validate_slim,
)
from llm.hybrid.menu_rule import (  # noqa: F401
    NORMAL,
    UNCLASSIFIED,
    VERDICTS,
    largest_threshold,
    menu_rule,
)

__all__ = [
    "SLIM_KEYS", "contradictions", "continuation_prompt", "fallback_answer",
    "forced_prefix", "parse_slim", "validate_slim",
    "NORMAL", "UNCLASSIFIED", "VERDICTS", "largest_threshold", "menu_rule",
]
