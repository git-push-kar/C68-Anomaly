"""Pydantic contract for the response layer.

Mirrors `docs/RESPONSE_SPEC.md` and the *existing* report keys
(`llm/dataset.py:ANSWER_KEYS`), with two deliberate differences from today's
free-form dicts:

1. ``root_cause`` is a :class:`FaultID` from ``scripts/fault_ids.py``, not free
   text, so a diagnosis is scored on one label space and cannot drift into a
   synonym.
2. **Severity is not a model field.** ``Report.severity`` is a
   :class:`SeverityAssessment` that only :mod:`llm.response.severity` produces;
   the model-authored body has no ``severity`` key at all and
   ``extra="forbid"`` makes a model that emits one fail validation instead of
   silently echoing a rubric.

Limits enforced here (structure only; the failure *codes* live in
``validators.py``): answer 1-4 sentences, at most 3 suggestions, at most 1
clarifying question.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scripts.fault_ids import FaultID

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")


def count_sentences(text: str) -> int:
    """Sentence count that does not split on decimals or table rows."""
    if not isinstance(text, str) or not text.strip():
        return 0
    body = _DECIMAL.sub("·", text)
    body = "\n".join(line for line in body.splitlines() if not line.strip().startswith("|"))
    parts = [p for p in _SENTENCE_SPLIT.split(body.strip()) if p.strip()]
    return max(1, len(parts))


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TableID(str, Enum):
    """Table catalog. Each id maps to one renderer builder in ``renderer.py``."""

    NONE = "none"
    EVIDENCE = "evidence"
    HYPOTHESES = "hypotheses"
    DETECTORS = "detectors"
    TIMELINE = "timeline"
    COMPARISON = "comparison"
    ACTIONS = "actions"


class Intent(str, Enum):
    """The 11 live dispatcher intents plus the 3 proposed ones (§2.2 of the spec)."""

    SENSOR_VALUES = "sensor_values"
    EVIDENCE_WHY = "evidence_why"
    SYSTEM_SENSORS = "system_sensors"
    SENSOR_SCORES = "sensor_scores"
    DETECTOR_SEVERITY = "detector_severity"
    DETECTOR_SCORES = "detector_scores"
    SEVERITY = "severity"
    OVERVIEW = "overview"
    SENSORS_INSPECT = "sensors_inspect"
    ACTION = "action"
    OVERVIEW_FALLBACK = "overview_fallback"
    # proposed, implemented in router.py only when enable_proposed=True
    HYPOTHESIS_COMPARISON = "hypothesis_comparison"
    CLARIFICATION_REPLY = "clarification_reply"
    OUT_OF_SCOPE_REFUSAL = "out_of_scope_refusal"


#: Intents that exist in the live dispatcher today.
LIVE_INTENTS = frozenset(
    {
        Intent.SENSOR_VALUES,
        Intent.EVIDENCE_WHY,
        Intent.SYSTEM_SENSORS,
        Intent.SENSOR_SCORES,
        Intent.DETECTOR_SEVERITY,
        Intent.DETECTOR_SCORES,
        Intent.SEVERITY,
        Intent.OVERVIEW,
        Intent.SENSORS_INSPECT,
        Intent.ACTION,
        Intent.OVERVIEW_FALLBACK,
    }
)
#: Intents added by this package behind ``enable_proposed``.
PROPOSED_INTENTS = frozenset(set(Intent) - set(LIVE_INTENTS))


class SeverityAssessment(BaseModel):
    """Code-computed severity. Never authored by the model."""

    model_config = ConfigDict(extra="forbid")

    level: SeverityLevel
    score: float = Field(ge=0.0, le=1.0)
    kb_base: SeverityLevel
    rationale: str
    components: dict = Field(default_factory=dict)
    rule_version: str = "severity-rule-v1-proposed"


class Hypothesis(BaseModel):
    """One candidate cause. Numeric support lives in ``citations``."""

    model_config = ConfigDict(extra="forbid")

    fault: FaultID
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    supporting_tags: List[str] = Field(default_factory=list, max_length=5)

    @field_validator("rationale")
    @classmethod
    def _rationale_length(cls, value: str) -> str:
        if count_sentences(value) > 3:
            raise ValueError("hypothesis rationale may be at most 3 sentences")
        return value


class ReportBody(BaseModel):
    """The model-authored part of a report: everything except severity."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    root_cause: FaultID
    affected_subsystem: str
    evidence: List[str] = Field(default_factory=list, max_length=5)
    reasoning: str
    reasoning_steps: str
    recommended_action: str
    uncertainty: str
    confidence: float = Field(ge=0.0, le=1.0)
    hypotheses: List[Hypothesis] = Field(default_factory=list, max_length=3)

    @field_validator("summary")
    @classmethod
    def _summary_length(cls, value: str) -> str:
        if count_sentences(value) > 4:
            raise ValueError("summary may be at most 4 sentences")
        return value


class Report(ReportBody):
    """A complete report: model body + code-computed severity + provenance."""

    event_id: str
    severity: SeverityAssessment
    table: TableID = TableID.DETECTORS
    citations: List[str] = Field(default_factory=list, max_length=8)
    source: str = "model"  # "model" or "deterministic"

    @model_validator(mode="after")
    def _synthetic_is_labelled(self) -> "Report":
        # 21/22 must never appear unlabelled in headline-adjacent text.
        if self.severity is not None and self.root_cause.value.startswith("SYNTHETIC_"):
            self.source = "model"
        return self


class FollowupAnswer(BaseModel):
    """One follow-up answer, with its table, citations and follow-up options."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    answer: str
    table: TableID = TableID.NONE
    citations: List[str] = Field(default_factory=list, max_length=6)
    suggestions: List[str] = Field(default_factory=list, max_length=3)
    clarifying_question: Optional[str] = None

    @field_validator("answer")
    @classmethod
    def _answer_length(cls, value: str) -> str:
        n = count_sentences(value)
        if not 1 <= n <= 4:
            raise ValueError(f"answer must be 1-4 sentences, got {n}")
        return value

    @field_validator("clarifying_question")
    @classmethod
    def _one_question(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if value.count("?") > 1:
            raise ValueError("at most one clarifying question is allowed")
        return value

    @model_validator(mode="after")
    def _clarify_keeps_an_answer(self) -> "FollowupAnswer":
        if self.clarifying_question and not self.answer.strip():
            raise ValueError("a clarifying question must be accompanied by an answer")
        return self


def table_allowed_for(intent: Intent) -> frozenset:
    """Which table ids an intent may attach (validators enforce this)."""
    numeric = {
        Intent.SENSOR_VALUES: frozenset({TableID.EVIDENCE}),
        Intent.SENSOR_SCORES: frozenset({TableID.EVIDENCE}),
        Intent.EVIDENCE_WHY: frozenset({TableID.DETECTORS, TableID.EVIDENCE}),
        Intent.DETECTOR_SCORES: frozenset({TableID.DETECTORS}),
        Intent.DETECTOR_SEVERITY: frozenset({TableID.DETECTORS}),
    }
    if intent in numeric:
        return numeric[intent] | {TableID.NONE}
    by_intent = {
        Intent.SEVERITY: frozenset({TableID.DETECTORS, TableID.NONE}),
        Intent.SYSTEM_SENSORS: frozenset({TableID.EVIDENCE, TableID.NONE}),
        Intent.OVERVIEW: frozenset({TableID.HYPOTHESES, TableID.DETECTORS, TableID.NONE}),
        Intent.OVERVIEW_FALLBACK: frozenset({TableID.HYPOTHESES, TableID.NONE}),
        Intent.SENSORS_INSPECT: frozenset({TableID.EVIDENCE, TableID.NONE}),
        Intent.ACTION: frozenset({TableID.ACTIONS, TableID.NONE}),
        Intent.HYPOTHESIS_COMPARISON: frozenset({TableID.HYPOTHESES, TableID.COMPARISON, TableID.NONE}),
        Intent.CLARIFICATION_REPLY: frozenset({TableID.NONE}),
        Intent.OUT_OF_SCOPE_REFUSAL: frozenset({TableID.NONE}),
    }
    return by_intent.get(intent, frozenset({TableID.NONE}))
