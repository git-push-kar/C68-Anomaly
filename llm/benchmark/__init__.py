"""Benchmark construction for the TEP RCA adapter (Phase 2B).

The pieces are deliberately small and deterministic except for one:
``teacher`` asks an LLM to write the two prose fields; everything else --
which fault, which subsystem, which hypothesis ranking, which mitigation,
whether to abstain, which table, which follow-ups -- comes from the knowledge
base and the router, so it cannot drift between runs and cannot be hallucinated.

    menu        KB-derived candidate menu and lookalikes
    prompt_v2   the prompt, with the subsystem hint and severity rubric removed
    targets     deterministic answer skeleton from fault id + KB
    filters     validation of teacher prose against store, tags and numbers
    teacher     Ollama/OpenAI-compatible client for the prose fields
"""
from llm.benchmark.menu import (  # noqa: F401
    CLASSIFIABLE_FAULTS,
    build_menu,
    lookalikes,
    menu_entry,
    signature_for,
)
from llm.benchmark.prompt_v2 import (  # noqa: F401
    FORBIDDEN_PROMPT_FRAGMENTS,
    SYSTEM_PROMPT_V2,
    assert_prompt_clean,
    render_prompt,
)
from llm.benchmark.targets import (  # noqa: F401
    open_set_assignment,
    build_target,
)

__all__ = [
    "CLASSIFIABLE_FAULTS",
    "build_menu",
    "lookalikes",
    "menu_entry",
    "signature_for",
    "FORBIDDEN_PROMPT_FRAGMENTS",
    "SYSTEM_PROMPT_V2",
    "assert_prompt_clean",
    "render_prompt",
    "open_set_assignment",
    "build_target",
]
