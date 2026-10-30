"""Which rules a follow-up answer was written and judged under.

The follow-up path reuses the Phase 2B rule set (``llm.benchmark.filters``):
direction scoping, the scaffolding rule and the valve/flow labels are one
definition, imported rather than copied. That means its provenance has to hash
*both* halves -- the 2B files that decide accept/reject and the follow-up files
that add cases, paraphrases and the dialogue-specific rules. Hashing only the
2B files would let an edit to ``llm/followup/filters.py`` change verdicts while
every cached answer still claimed to be judged under the current regime.

The menu fields come from :mod:`llm.benchmark.provenance` unchanged: a
follow-up answer about fault 13 depends on the same menu line the report task
does, so the revision label must be the same one.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

from llm.benchmark.provenance import menu_provenance

ROOT = Path(__file__).resolve().parents[2]

#: The Phase 2B files are imported here, so an edit to them is an edit to this
#: path's regime too. They are listed again rather than imported from
#: ``scripts/build_benchmark_dataset.py`` because a script is not a library; the
#: list is asserted equal to the script's in ``tests/test_followup_provenance.py``
#: so the two cannot drift apart silently.
BENCHMARK_FILTER_CODE_FILES = (
    "llm/benchmark/filters.py",
    "llm/benchmark/prompt_v2.py",
    "llm/response/allowed_set.py",
    "llm/response/validators.py",
)

FOLLOWUP_FILTER_CODE_FILES = (
    "llm/followup/filters.py",
    "llm/followup/cases.py",
    "llm/followup/teacher.py",
    "llm/followup/paraphrases.py",
)


def filter_code_hash(files=BENCHMARK_FILTER_CODE_FILES + FOLLOWUP_FILTER_CODE_FILES) -> str:
    """SHA-256 over every file whose contents can change a verdict."""
    digest = hashlib.sha256()
    for rel in files:
        path = ROOT / rel
        if path.exists():
            digest.update(rel.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def followup_provenance() -> Dict[str, str]:
    """What every follow-up cache record and the manifest carry."""
    record = {"filter_sha256": filter_code_hash()}
    record.update(menu_provenance())
    return record


__all__ = [
    "BENCHMARK_FILTER_CODE_FILES",
    "FOLLOWUP_FILTER_CODE_FILES",
    "filter_code_hash",
    "followup_provenance",
]
