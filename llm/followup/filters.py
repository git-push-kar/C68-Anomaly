"""Acceptance filters for follow-up answers.

Same policy as Phase 2B: every rule is a **rejection**, never a repair. A model
answer that fails is dropped and the code counted, because rewriting a claim is
how a fabrication becomes a fact.

Rules reused from ``llm/benchmark/filters.py`` (same constants, same number
helper, so a document that passes there passes here for the same reason):

``number_not_in_store``   a number that matches nothing the event holds
``unknown_tag``           a tag-shaped token that resolves to no sensor
``sensor_not_grounded``   a real tag that this event / diagnosis never mentions
``absolute_wording``      "definitely", "proves", ...
``severity_claim``        a severity level or priority assigned in prose
``fault_out_of_scope``    a fault number outside the permitted menu

Added for follow-up turns:

``empty_answer``          no text
``answer_length``         outside 1-4 sentences
``answer_is_question``    the answer asks instead of answering
``clarify_text_repeated`` the answer repeats the clarifying question
``near_duplicate``        >= the overlap threshold with the previous reply
``confidence_claim``      a numeric confidence
``refusal_missing``       a refusal that does not decline
``refusal_no_alternative``a refusal that offers nothing the store can answer
``names_removed_fault``   an open-set answer that names the removed fault
``table_reference``       "as shown in the table below" instead of numbers
``valve_flow_mislabel``   a manipulated variable described as a measurement
``direction_conflict``    the KB's initiating measurement cited with the wrong sign
``teacher_scaffolding``   prose that quotes the teacher's own prompt labels
``event_id_leak``         a benchmark/dataset id in the answer
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

from llm.benchmark.filters import (  # same rule set, one definition of valid
    ABSOLUTE_PATTERNS,
    SEVERITY_PATTERNS,
    _direction_conflict,
    _evidence_tags,
    _kb_tags,
    _tag_kind,
)
from llm.benchmark.scaffolding import scaffolding_hits
from llm.response.allowed_set import AllowedSet
from llm.response.schemas import count_sentences
from llm.response.store_access import StoreAccess
from llm.response.validators import fault_ids_in, number_supported, numbers_in
from scripts.fault_ids import SYNTHETIC_FAULTS
from scripts.tags import canonical_tag, iter_tags

#: A reply that repeats this much of the previous reply is a near-duplicate.
DEFAULT_OVERLAP_THRESHOLD = 0.6
OVERLAP_NGRAM = 4

#: A decline either states an inability/absence in the store, or states that the
#: thing asked about is *not there*. Both families are declines; the second one is
#: worded as an observation about the context, so it was missed by a list that
#: only held the first family, and "XMEAS_2 is not listed among the variables that
#: moved" was rejected as a refusal that does not decline.
DECLINE_MARKERS = (
    "cannot", "can't", "can not", "do not have", "don't have", "no record",
    "not stored", "not in the store", "not available", "no such",
    "does not include", "doesn't include", "the store does not",
    "the event store does not", "the stored diagnosis is", "what is stored is",
    "the record shows", "is not what the store", "no stored",
    # absence family
    "not listed", "not among", "not in this event", "not in the evidence",
    "does not mention", "doesn't mention", "does not name", "doesn't name",
    "no mention", "does not appear", "doesn't appear", "is not named",
    "cannot confirm", "can't confirm", "not possible to say", "no evidence",
)

#: The teacher's second way to decline: state that the *stored record* lacks the
#: thing asked for. Measured over the first full run: 43 of 50 rejected
#: out-of-scope answers said "the incident context does not provide ...", which
#: the inability family above did not contain. Scoped by subject so that a
#: physical sentence ("the feed does not contain enough reactant") is not read
#: as a decline.
ABSENCE_SUBJECTS = (
    "context", "store", "record", "incident", "evidence", "report", "log",
    "data", "dataset", "listing", "summary",
)
ABSENCE_VERBS = (
    "does not provide", "does not include", "does not contain", "does not record",
    "does not specify", "does not describe", "does not list", "does not give",
    "does not show", "does not track", "does not report", "does not hold",
)
#: Absence stated without a subject ("XMEAS_1 is not mentioned in the list").
BARE_ABSENCE = (
    "not mentioned", "not provided", "not reported", "not recorded",
    "not specified", "not listed", "no information", "no data",
)

CONFIDENCE_PATTERNS = (
    r"\b\d{1,3}\s?%\s?confiden",
    r"\bconfidence\s+(?:of|is|around|about)\s+\d",
    r"\b\d(?:\.\d+)?\s+confidence\b",
)

TABLE_REFERENCE_PATTERNS = (
    r"as shown in the table below",
    r"as shown below",
    r"see the table",
    r"in the table above",
)

#: Words that name a measurement. A manipulated variable (``XMV_*``) is a valve
#: position, so calling one a temperature/pressure/composition reading is a
#: labelling error the filters must catch, not a stylistic quibble.
MEASUREMENT_WORDS = ("temperature", "pressure", "level", "composition", "concentration")

#: And the other direction: an XMEAS reading is not a valve or an actuator.
VALVE_WORDS = ("valve position", "valve opening", "valve stem", "actuator position")

EVENT_ID_PATTERNS = (
    r"\bEVT-",
    r"\bNORM-",
    r"\bANOM-",
    r"\bNORMAL-",
)

#: Anchor words per answerable intent, used to check that a refusal offers
#: something the store can actually answer.
ALTERNATIVE_ANCHORS: Dict[str, Sequence[str]] = {
    "detector_scores": ("detector", "threshold", "lstm", "q-stat", "q stat"),
    "detector_severity": ("detector", "threshold", "metric"),
    "severity": ("severity", "alarm level", "risk level"),
    "sensors_inspect": ("variable", "variables", "sensor", "sensors"),
    "sensor_values": ("value", "values", "baseline"),
    "sensor_scores": ("score", "z-score", "trend"),
    "action": ("action", "recommended", "procedure", "mitigation"),
    "overview": ("cause", "subsystem", "diagnosis"),
    "overview_fallback": ("cause", "subsystem", "diagnosis"),
    "evidence_why": ("evidence", "alarm", "limit"),
    "system_sensors": ("monitored", "measurement", "variables"),
}


@dataclass
class AnswerContext:
    """Everything the filters need for one turn, resolved beforehand."""

    store: StoreAccess
    allowed: AllowedSet
    grounded_numbers: Sequence[float]
    permitted_faults: Sequence[int]
    event_id: str = ""
    removed_fault: Optional[int] = None
    previous_answer: Optional[str] = None
    clarifying_question: Optional[str] = None
    refusal: Optional[Dict] = None
    intent: Optional[str] = None
    #: The question this answer belongs to. Used only to tell a *leak* from an
    #: echo: a fault name the operator already used is not the answer leaking it.
    question: str = ""
    #: Tags the operator named in the question (a false premise). A refusal has
    #: to name the thing it denies, so those tags are legitimately grounded.
    extra_grounded_tags: Sequence[str] = ()
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD
    #: The fault whose menu line this item was written from, when the item is a
    #: classified one. ``None`` for an open-set item (the fault is removed, so
    #: there is no KB direction to hold the prose to) and for a normal one.
    kb_fault: Optional[int] = None


def ngrams(text: str, n: int = OVERLAP_NGRAM) -> Set[tuple]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {tuple(tokens[i:i + n]) for i in range(0, max(len(tokens) - n + 1, 0))}


def overlap_ratio(candidate: str, previous: str, n: int = OVERLAP_NGRAM) -> float:
    """Fraction of the candidate's n-grams that also appear in ``previous``."""
    cand = ngrams(candidate, n)
    if not cand:
        return 0.0
    return len(cand & ngrams(previous, n)) / float(len(cand))


def is_near_duplicate(candidate: str, previous: str, threshold: float = DEFAULT_OVERLAP_THRESHOLD) -> bool:
    if not previous:
        return False
    if " ".join(candidate.lower().split()) == " ".join(previous.lower().split()):
        return True
    return overlap_ratio(candidate, previous) >= threshold


def _normalise(text: str) -> str:
    return " ".join((text or "").lower().split())


def _declines(low: str) -> bool:
    """Does the answer decline, in either of the two ways English allows?

    Either it says it cannot (the inability family in ``DECLINE_MARKERS``), or
    it says the stored record does not hold what was asked for. The second form
    needs its subject checked: "the context does not provide" declines, while
    "the feed does not contain enough reactant" is a claim about the process.
    """
    if any(marker in low for marker in DECLINE_MARKERS):
        return True
    if any(phrase in low for phrase in BARE_ABSENCE):
        return True
    for verb in ABSENCE_VERBS:
        start = low.find(verb)
        while start != -1:
            before = low[max(0, start - 60):start]
            if any(subject in before for subject in ABSENCE_SUBJECTS):
                return True
            start = low.find(verb, start + 1)
    return False


def declines(text: str) -> bool:
    """Public view of the decline test, for reporting rather than filtering.

    The audit needs to know whether an answer declined without re-implementing the
    vocabulary in a script, where it would drift from the rule that rejects answers.
    """
    return _declines(_normalise(text))


def _mentions_alternative(text: str, refusal: Dict, store: StoreAccess) -> bool:
    low = _normalise(text)
    for alternative in refusal.get("alternatives") or []:
        for anchor in ALTERNATIVE_ANCHORS.get(str(alternative.get("intent")), ()):
            if anchor in low:
                return True
    for tag in store.known_tags():
        name = (store.display(tag) or tag).lower()
        if name and name.replace("_", " ") in low.replace("_", " "):
            return True
    return False


def _valve_mislabel(text: str) -> Optional[str]:
    """A tag described as the kind of variable it is not.

    The Phase 2B prompt labels every tag with its kind (``A_Feed_Flow [XMV_44,
    valve position, manipulated variable]``), so an answer that calls a valve
    position a pressure reading -- or a measurement a valve position -- has
    contradicted the context it was given, not merely worded it loosely.
    """
    for match in re.finditer(r"\b(XMV|XMEAS)[_ ]?(\d{1,2})\b", text or ""):
        tag = f"{match.group(1)}_{int(match.group(2)):02d}"
        window = (text or "")[match.end():match.end() + 40].lower()
        # "is not a pressure" states the distinction correctly; it is the
        # affirmative mislabel the rule is after.
        if any(neg in window for neg in ("not", "n't", "rather than", "isn't")):
            continue
        if _tag_kind(tag) == "valve_position":
            for word in MEASUREMENT_WORDS:
                if word in window:
                    return f"{match.group(0)} described as {word}"
        else:
            for word in VALVE_WORDS:
                if word in window:
                    return f"{match.group(0)} described as {word}"
    return None


def check_answer(text: str, ctx: AnswerContext) -> List[str]:
    """Return the rejection codes for one follow-up answer (empty = accept)."""
    body = (text or "").strip()
    if not body:
        return ["empty_answer"]

    codes: List[str] = []

    n_sentences = count_sentences(body)
    if not 1 <= n_sentences <= 4:
        codes.append(f"answer_length:{n_sentences}")

    if "?" in body:
        codes.append("answer_is_question")

    if ctx.clarifying_question:
        # The clarifying question is a field of its own; repeating it in the
        # answer is the "ask instead of answer" failure the schema forbids.
        if _normalise(ctx.clarifying_question) in _normalise(body):
            codes.append("clarify_text_repeated")

    if is_near_duplicate(body, ctx.previous_answer or "", ctx.overlap_threshold):
        codes.append(f"near_duplicate:{overlap_ratio(body, ctx.previous_answer or ''):.2f}")

    # -- grounding ------------------------------------------------------
    fabricated: List[str] = []
    ungrounded_names: List[str] = []
    for token in ctx.allowed.unknown_tag_like(body):
        if canonical_tag(token) is None:
            fabricated.append(token)
        else:
            ungrounded_names.append(token)
    if fabricated:
        codes.append("unknown_tag:" + ",".join(fabricated[:3]))

    cited = {canonical_tag(tag) for tag in iter_tags(body)}
    cited.discard(None)
    grounded: Set[str] = set(_evidence_tags(ctx.store.evidence))
    if ctx.intent is not None:
        number = None
        if str(ctx.intent).startswith(("FAULT_", "SYNTHETIC_")):
            number = int(str(ctx.intent).split("_")[-1])
        if number is not None:
            grounded |= _kb_tags(number)
    for hypothesis in (ctx.store.report.get("hypotheses") or []):
        value = str((hypothesis or {}).get("fault") or "")
        if value.startswith(("FAULT_", "SYNTHETIC_")):
            grounded |= _kb_tags(int(value.split("_")[-1]))
    grounded |= {canonical_tag(tag) for tag in ctx.extra_grounded_tags}
    grounded.discard(None)
    ungrounded = sorted(cited - grounded) + [t for t in ungrounded_names if canonical_tag(t) not in grounded]
    if ungrounded:
        codes.append("sensor_not_grounded:" + ",".join(dict.fromkeys(ungrounded))[:80])

    # -- the Phase 2B rules, applied here unchanged ----------------------
    # Direction scoping: only a *measurement* named as the KB's initiating
    # variable is held to the KB's sign, and a valve position is skipped because
    # a position and the flow it drives routinely move oppositely.
    conflict = _direction_conflict(body, ctx.kb_fault)
    if conflict:
        codes.append(f"direction_conflict:{conflict}")

    # Scaffolding: prose that quotes the teacher's own prompt labels is not an
    # observation about the plant. Derived from the prompt template, not a list.
    hits = scaffolding_hits(body, abstain=ctx.removed_fault is not None)
    if hits:
        codes.append("teacher_scaffolding:" + ",".join(hit["phrase"] for hit in hits[:3]))

    permitted = set(int(x) for x in ctx.permitted_faults)
    label_integers = set()
    for key in ("event.n_windows", "count.top_sensors", "count.timeline", "count.sensors_total"):
        try:
            label_integers.add(int(round(float(ctx.store.resolve(key)))))
        except Exception:
            continue
    for value, decimals in numbers_in(body):
        if number_supported(value, decimals, ctx.grounded_numbers, sorted(label_integers)):
            continue
        codes.append(f"number_not_in_store:{value}")
        break

    for number in fault_ids_in(body):
        if number in SYNTHETIC_FAULTS or not 1 <= number <= 20 or (permitted and number not in permitted):
            codes.append(f"fault_out_of_scope:{number}")
            break

    if ctx.removed_fault is not None:
        removed = int(ctx.removed_fault)
        named = removed in fault_ids_in(body)
        if not named:
            from scripts.fault_knowledge import FAULT_KNOWLEDGE

            name = str((FAULT_KNOWLEDGE.get(removed) or {}).get("name") or "")
            # Fault 15's KB name is a plain physical description ("condenser cooling
            # water valve sticking"), and an operator can ask exactly that. Echoing
            # the operator's own words is not a leak; naming the family as the
            # answer is.
            echoed = bool(name) and _normalise(name) in _normalise(ctx.question)
            if name and not echoed and _normalise(name) in _normalise(body):
                named = True
        if named:
            codes.append(f"names_removed_fault:{removed}")

    low = body.lower()
    for pattern in ABSOLUTE_PATTERNS:
        match = re.search(pattern, low)
        if match:
            codes.append(f"absolute_wording:{match.group(0)}")
            break
    for pattern in SEVERITY_PATTERNS:
        match = re.search(pattern, low)
        if match:
            codes.append(f"severity_claim:{match.group(0)}")
            break
    for pattern in CONFIDENCE_PATTERNS:
        match = re.search(pattern, low)
        if match:
            codes.append(f"confidence_claim:{match.group(0)}")
            break
    for pattern in TABLE_REFERENCE_PATTERNS:
        match = re.search(pattern, low)
        if match:
            codes.append(f"table_reference:{match.group(0)}")
            break

    mislabel = _valve_mislabel(body)
    if mislabel:
        codes.append(f"valve_flow_mislabel:{mislabel}")

    for pattern in EVENT_ID_PATTERNS:
        if re.search(pattern, body):
            codes.append(f"event_id_leak:{pattern}")
            break
    if ctx.event_id and _normalise(ctx.event_id) in _normalise(body):
        codes.append("event_id_leak:event_id")

    # -- refusal and clarification formats ------------------------------
    if ctx.refusal and ctx.refusal.get("required"):
        if not _declines(low):
            codes.append("refusal_missing")
        if not _mentions_alternative(body, ctx.refusal, ctx.store):
            codes.append("refusal_no_alternative")

    return codes


__all__ = [
    "ABSENCE_SUBJECTS",
    "ABSENCE_VERBS",
    "ALTERNATIVE_ANCHORS",
    "AnswerContext",
    "BARE_ABSENCE",
    "_declines",
    "CONFIDENCE_PATTERNS",
    "DEFAULT_OVERLAP_THRESHOLD",
    "DECLINE_MARKERS",
    "VALVE_WORDS",
    "check_answer",
    "is_near_duplicate",
    "ngrams",
    "overlap_ratio",
]
