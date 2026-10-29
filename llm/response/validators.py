"""Validation rules for reports and follow-up answers.

Each rule returns a stable ``code`` so a benchmark can count failure modes
instead of reading prose. Rules are deliberately independent: one answer can
fail several at once, and the first failure never masks the rest.

Rules
-----
``schema``                  the payload does not parse into the contract
``severity_from_model``     the model emitted ``severity`` (it must be computed)
``answer_length``           answer/summary outside the 1-4 sentence window
``too_many_suggestions``    more than 3 suggested follow-ups
``duplicate_suggestion``    the same suggestion twice (normalised)
``suggestion_not_answerable`` a suggestion that routes to no live intent
``table_not_allowed``       table id not permitted for this intent
``unknown_citation``        a citation key that does not resolve in the store
``unknown_tag``             a tag-shaped token that is not in the allowed set
``number_not_in_store``     a prose number that no store value rounds to
``clarify_not_triggered``   a clarifying question with no fired trigger
``clarify_reasked``         a clarifying question already asked in this thread
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from pydantic import ValidationError

from llm.response.allowed_set import AllowedSet, build_allowed_set
from llm.response.router import LOOKUP_INTENTS, ResponseRouter
from llm.response.schemas import (
    FollowupAnswer,
    Intent,
    Report,
    ReportBody,
    TableID,
    count_sentences,
    table_allowed_for,
)
from llm.response.severity import assess
from llm.response.store_access import StoreAccess, UnknownKeyError

#: Integers that are labels or counts, not measurements.
LABEL_INTEGERS = set(range(1, 23)) | {41, 52, 11}
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_TAG_NUMBER_RE = re.compile(r"(?:XMEAS|XMV)[ _-]?\d{1,2}", re.IGNORECASE)


@dataclass
class Issue:
    code: str
    message: str
    path: str = ""

    def as_dict(self) -> Dict:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass
class ValidationResult:
    issues: List[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def codes(self) -> List[str]:
        return sorted({issue.code for issue in self.issues})

    def add(self, code: str, message: str, path: str = "") -> None:
        self.issues.append(Issue(code=code, message=message, path=path))

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(f"{i.code}: {i.message}" for i in self.issues))

    def as_dict(self) -> Dict:
        return {"ok": self.ok, "codes": self.codes(), "issues": [i.as_dict() for i in self.issues]}


# ----------------------------------------------------------------------------
def numbers_in(text: str) -> List[tuple]:
    """(value, decimals) for every number in ``text``, ignoring tag numbers."""
    if not isinstance(text, str):
        return []
    cleaned = _TAG_NUMBER_RE.sub(" ", text)
    out = []
    for match in _NUMBER_RE.finditer(cleaned):
        token = match.group(0)
        decimals = len(token.split(".")[1]) if "." in token else 0
        try:
            out.append((float(token), decimals))
        except ValueError:
            continue
    return out


def number_supported(value: float, decimals: int, store_numbers: Sequence[float]) -> bool:
    """True when some store value rounds to the printed number."""
    if float(value).is_integer() and int(value) in LABEL_INTEGERS:
        return True
    for candidate in store_numbers:
        try:
            if round(float(candidate), decimals) == value:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _check_text(
    result: ValidationResult,
    text: str,
    path: str,
    allowed: AllowedSet,
    store_numbers: Sequence[float],
) -> None:
    for token in allowed.unknown_tag_like(text):
        result.add("unknown_tag", f"unresolvable tag-shaped token {token!r}", path)
    for value, decimals in numbers_in(text):
        if not number_supported(value, decimals, store_numbers):
            result.add("number_not_in_store", f"number {value} does not match any stored value", path)


def _check_citations(result: ValidationResult, keys: Sequence[str], store: StoreAccess, path: str) -> None:
    for key in keys or []:
        try:
            store.resolve(key)
        except UnknownKeyError as exc:
            result.add("unknown_citation", str(exc), path)


# ----------------------------------------------------------------------------
def build_report(
    raw: Dict,
    event: Dict,
    allowed: Optional[AllowedSet] = None,
    store: Optional[StoreAccess] = None,
    result: Optional[ValidationResult] = None,
    fault: Optional[object] = None,
) -> tuple:
    """Validate a model-authored report dict and attach code-computed severity.

    Returns ``(Report | None, ValidationResult)``. ``severity`` in ``raw`` is a
    failure, not something to overwrite silently.
    """
    result = result or ValidationResult()
    store = store or StoreAccess(event)
    raw = dict(raw or {})
    if allowed is None:
        # Names from the diagnosed cause and every hypothesis are part of the
        # answer's legitimate vocabulary (KB tags, not just this event's top 5).
        from llm.response.severity import coerce_fault

        cited = [raw.get("root_cause")] + [
            (h or {}).get("fault") for h in (raw.get("hypotheses") or []) if isinstance(h, dict)
        ]
        extra = []
        for item in cited:
            cited_fault = coerce_fault(item)
            if cited_fault is not None and cited_fault.value.startswith(("FAULT_", "SYNTHETIC_")):
                extra.append(int(cited_fault.value.split("_")[-1]))
        allowed = build_allowed_set(event, extra_faults=extra)
    # Report-level fields are not part of the model-authored body; take them out
    # before strict parsing so `extra="forbid"` only rejects genuinely unknown keys.
    report_level = {key: raw.pop(key) for key in ("table", "citations", "source", "event_id") if key in raw}

    if "severity" in raw:
        # Flag it, then drop it so the rest of the payload still parses: the
        # answer is still gradeable, and severity is computed regardless.
        result.add(
            "severity_from_model",
            "the model must not emit severity; it is computed by llm.response.severity",
            "severity",
        )
        raw.pop("severity", None)

    body = None
    try:
        body = ReportBody.model_validate(raw)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            code = "answer_length" if "sentences" in err.get("msg", "") else "schema"
            if "suggestions" in loc or "hypotheses" in loc:
                code = "schema"
            result.add(code, err.get("msg", "invalid"), loc)

    numbers = store.numeric_values()
    if body is not None:
        for path, text in (
            ("summary", body.summary),
            ("reasoning", body.reasoning),
            ("reasoning_steps", body.reasoning_steps),
            ("recommended_action", body.recommended_action),
            ("uncertainty", body.uncertainty),
        ):
            _check_text(result, text, path, allowed, numbers)
        for index, bullet in enumerate(body.evidence):
            _check_text(result, bullet, f"evidence[{index}]", allowed, numbers)
        for index, hypothesis in enumerate(body.hypotheses):
            _check_text(result, hypothesis.rationale, f"hypotheses[{index}].rationale", allowed, numbers)
            for tag in hypothesis.supporting_tags:
                if allowed.resolve_tag(tag) is None:
                    result.add("unknown_tag", f"hypothesis tag {tag!r} is not in the allowed set", f"hypotheses[{index}].supporting_tags")

    if report_level.get("table") is not None:
        try:
            TableID(report_level["table"])
        except ValueError:
            result.add("schema", f"unknown table id {report_level['table']!r}", "table")

    _check_citations(result, report_level.get("citations") or [], store, "citations")

    if body is None:
        return None, result

    assessment = assess(event, fault=fault if fault is not None else body.root_cause)
    report = Report(
        **body.model_dump(),
        event_id=str(event.get("event_id") or report_level.get("event_id") or ""),
        severity=assessment,
        table=TableID(report_level.get("table") or TableID.DETECTORS.value),
        citations=list(report_level.get("citations") or []),
        source=str(report_level.get("source") or "model"),
    )
    return report, result


# ----------------------------------------------------------------------------
def validate_followup(
    raw: Dict,
    event: Dict,
    report: Optional[Dict] = None,
    router: Optional[ResponseRouter] = None,
    asked: Optional[Sequence[str]] = None,
    allowed: Optional[AllowedSet] = None,
    store: Optional[StoreAccess] = None,
    question: str = "",
) -> tuple:
    """Validate a follow-up answer dict. Returns ``(FollowupAnswer | None, ValidationResult)``."""
    result = ValidationResult()
    allowed = allowed or build_allowed_set(event)
    store = store or StoreAccess(event, report)
    router = router or ResponseRouter()
    asked_set: Set[str] = set(asked or ())

    answer = None
    try:
        answer = FollowupAnswer.model_validate(raw or {})
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            msg = err.get("msg", "invalid")
            if "sentences" in msg:
                result.add("answer_length", msg, loc)
            elif "suggestions" in loc and ("at most" in msg or "max_length" in msg):
                result.add("too_many_suggestions", msg, loc)
            else:
                result.add("schema", msg, loc)

    if answer is None:
        return None, result

    if answer.table not in table_allowed_for(answer.intent):
        result.add(
            "table_not_allowed",
            f"table {answer.table.value!r} is not allowed for intent {answer.intent.value!r}",
            "table",
        )

    _check_citations(result, answer.citations, store, "citations")
    numbers = store.numeric_values()
    _check_text(result, answer.answer, "answer", allowed, numbers)

    seen = set()
    for index, suggestion in enumerate(answer.suggestions):
        normalised = " ".join(re.findall(r"[a-z0-9]+", suggestion.lower()))
        if normalised in seen:
            result.add("duplicate_suggestion", f"duplicate suggestion {suggestion!r}", f"suggestions[{index}]")
        seen.add(normalised)
        candidates, _ = router.candidates(suggestion)
        if not any(intent in LOOKUP_INTENTS for intent in candidates):
            result.add(
                "suggestion_not_answerable",
                f"suggestion {suggestion!r} does not route to an answerable live intent",
                f"suggestions[{index}]",
            )

    if answer.clarifying_question:
        # Trigger presence is a property of the event/answer, not of the thread:
        # ask the router with no history, then dedupe separately below.
        route = router.route(event, report or {}, question, with_answer=False)
        fired = route.clarification is not None
        if not fired:
            result.add(
                "clarify_not_triggered",
                "a clarifying question was asked but no trigger fired for this answer",
                "clarifying_question",
            )
        if answer.clarifying_question in asked_set:
            result.add("clarify_reasked", "this clarifying question was already asked", "clarifying_question")

    return answer, result


__all__ = [
    "Issue",
    "ValidationResult",
    "build_report",
    "validate_followup",
    "numbers_in",
    "number_supported",
    "LABEL_INTEGERS",
]
