"""Response layer for the TEP RCA benchmark (Phase 2A).

Contracts, not generation: this package turns a stored event plus a validated
model payload into a renderable, citation-checked answer.

    schemas        Pydantic Report / FollowupAnswer (severity is not a model field)
    severity       code-computed severity (PROPOSED rule, needs owner approval)
    allowed_set    one shared tag/label set for validators and the evaluator
    store_access   citation keys -> stored values (unknown key raises)
    renderer       markdown + UI blocks; every table cell is a citation key
    router         wrapper around the live dispatcher (11 intents + 3 proposed)
    validators     stable failure codes for every rule
    lookup_answers deterministic fact-lookup built on the live templates
"""
from llm.response.allowed_set import AllowedSet, build_allowed_set
from llm.response.lookup_answers import LookupHit, lookup_answer
from llm.response.renderer import RenderedDoc, render_followup, render_report
from llm.response.router import ResponseRouter, RouteResult
from llm.response.schemas import (
    FollowupAnswer,
    Hypothesis,
    Intent,
    Report,
    ReportBody,
    SeverityAssessment,
    SeverityLevel,
    TableID,
)
from llm.response.severity import assess as assess_severity
from llm.response.store_access import StoreAccess, UnknownKeyError
from llm.response.validators import ValidationResult, build_report, validate_followup

__all__ = [
    "AllowedSet",
    "build_allowed_set",
    "FollowupAnswer",
    "Hypothesis",
    "Intent",
    "LookupHit",
    "lookup_answer",
    "RenderedDoc",
    "render_followup",
    "render_report",
    "ResponseRouter",
    "Report",
    "ReportBody",
    "RouteResult",
    "SeverityAssessment",
    "SeverityLevel",
    "StoreAccess",
    "TableID",
    "UnknownKeyError",
    "ValidationResult",
    "assess_severity",
    "build_report",
    "validate_followup",
]
