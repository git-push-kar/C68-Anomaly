"""Acceptance filters for teacher prose.

The teacher writes two sentences-blocks; this module decides whether they are
usable. Every rule below is a *rejection*, not a repair: a filtered answer is
dropped and the reason counted, because rewriting a model's claim about the
plant is how a fabrication becomes a fact.

The rules reuse the response layer's own primitives (``numbers_in``,
``number_supported``, ``count_sentences``, ``AllowedSet``) rather than
re-implementing them, so a document that passes here passes the runtime
validator for the same reason -- one definition of "valid", not two.

Rules (code in brackets, as reported in the rejection log):

``missing_key``          the object does not have exactly the two prose keys
``empty_field``          one of them is blank
``narrative_length``     narrative outside 2-4 sentences
``reasoning_length``     reasoning outside 3-8 sentences
``label_changed``        the answer re-states a different fault id
``unknown_tag``          a tag-shaped token that resolves to no sensor
``sensor_not_grounded``  a cited sensor is in neither the evidence nor the KB
``number_not_in_store``  a number that matches no value in the event's evidence
``direction_conflict``   the KB's initiating variable is cited with the wrong sign
``absolute_wording``     "definitely", "proves", "guaranteed" and friends
``severity_claim``       a severity level or priority assigned in prose
``fault_out_of_scope``   a fault number outside 1-20, or a synthetic fault
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set

from llm.response.allowed_set import AllowedSet
from llm.response.schemas import count_sentences
from llm.response.validators import fault_ids_in, number_supported, numbers_in
from scripts.fault_ids import SYNTHETIC_FAULTS, fault_id_for
from scripts.fault_knowledge import FAULT_KNOWLEDGE
from scripts.tags import canonical_tag, iter_tags

TEACHER_ANSWER_KEYS = ("physical_reasoning", "narrative")

ABSOLUTE_PATTERNS = (
    r"\bdefinitely\b", r"\bcertainly\b", r"\bguarantee[ds]?\b", r"\bundoubtedly\b",
    r"\bwithout (?:a )?doubt\b", r"\bproves?\b", r"\bproven\b", r"\bconclusive(?:ly)?\b",
    r"\bunmistakabl[ey]\b", r"\b100\s?%\s?(?:sure|certain)\b",
)

SEVERITY_PATTERNS = (
    r"\bseverity\b", r"\bsev[- ]?[1-5]\b", r"\bpriority\s+(?:level\s+)?[1-5]\b",
    r"\bhigh[- ]risk\b", r"\brisk score\b", r"\bcriticality\b",
)

#: Words that assert an increase or a decrease. Used only to catch a *conflict*
#: with the KB's stated direction of the initiating variable.
INCREASE_WORDS = ("increase", "increases", "rise", "rises", "risen", "higher", "upward", "climb")
DECREASE_WORDS = ("decrease", "decreases", "fall", "falls", "fallen", "lower", "downward", "drop")

_NUMBER_RANGE = ("-.2f",)


#: Feed-component names (``Component_A`` .. ``Component_H`` and the streams).
#: They are shaped like tags but name a chemical species, so the tag rules must
#: not fire on them.
_COMPONENT_RE = re.compile(r"^component[_ -]?[a-h]$", re.IGNORECASE)
_STREAM_RE = re.compile(r"^(?:stream|feed)[_ -]?\d{1,2}$", re.IGNORECASE)


def _is_chemical_name(token: str) -> bool:
    return bool(_COMPONENT_RE.match(str(token).strip()) or _STREAM_RE.match(str(token).strip()))


def _sentences(text: str) -> int:
    return count_sentences(text)


def _kb_tags(number: Optional[int]) -> Set[str]:
    """Canonical tags the KB names for a fault (initiating + affected + equipment)."""
    if number is None:
        return set()
    entry = FAULT_KNOWLEDGE.get(int(number)) or {}
    text = " ".join(str(entry.get(key) or "") for key in
                    ("initiating", "affected_variables", "equipment", "reasoning", "mechanism"))
    tags = set()
    for tag in iter_tags(text):
        resolved = canonical_tag(tag)
        if resolved:
            tags.add(resolved)
    return tags


def _evidence_tags(evidence: Dict) -> Set[str]:
    tags: Set[str] = set()
    for sensor in (evidence.get("top_anomalous_sensors") or []):
        if isinstance(sensor, dict):
            for key in ("name", "display_name"):
                resolved = canonical_tag(sensor.get(key))
                if resolved:
                    tags.add(resolved)
    det = evidence.get("detector_evidence") or {}
    for item in ((det.get("lag_attribution") or {}).get("top_variables") or []):
        if isinstance(item, dict):
            resolved = canonical_tag(item.get("variable"))
            if resolved:
                tags.add(resolved)
    for item in (evidence.get("temporal_sequence") or []):
        if isinstance(item, dict):
            resolved = canonical_tag(item.get("sensor")) or canonical_tag(item.get("display_name"))
            if resolved:
                tags.add(resolved)
    return tags


def store_numbers(evidence: Dict, target: Dict) -> List[float]:
    """Every number in the evidence that prose may legitimately quote."""
    out: List[float] = []

    def walk(value: object) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            for token in re.findall(r"-?\d+(?:\.\d+)?", value):
                try:
                    out.append(float(token))
                except ValueError:
                    continue

    walk({k: v for k, v in evidence.items() if k != "event_id"})
    for hyp in target.get("ranked_hypotheses") or []:
        walk(hyp.get("rank"))
    return out


def _direction_conflict(text: str, number: Optional[int]) -> Optional[str]:
    """Flag a cited initiating variable described with the opposite sign."""
    if number is None:
        return None
    entry = FAULT_KNOWLEDGE.get(int(number)) or {}
    initiating = entry.get("initiating")
    if isinstance(initiating, list) and initiating:
        initiating = initiating[0]
    if not isinstance(initiating, dict):
        return None
    sensor = initiating.get("sensor")
    direction = str(initiating.get("direction") or "").lower()
    if not sensor or direction not in ("increase", "decrease"):
        return None
    tag = canonical_tag(sensor)
    if tag is None:
        return None
    body = (text or "").lower()
    for match in re.finditer(re.escape(tag.lower()), body):
        window = body[match.end():match.end() + 60]
        up = any(word in window for word in INCREASE_WORDS)
        down = any(word in window for word in DECREASE_WORDS)
        if up and not down and direction == "decrease":
            return f"{tag} described as increasing but the KB says decreasing"
        if down and not up and direction == "increase":
            return f"{tag} described as decreasing but the KB says increasing"
    return None


def check_answer(
    answer: Optional[Dict],
    target: Dict,
    evidence: Dict,
    allowed: AllowedSet,
    parse_error: Optional[str] = None,
    prompt: Optional[str] = None,
) -> List[str]:
    """Return the rejection codes for one teacher answer (empty means accept).

    ``prompt`` is the text the answer was written from. Numbers printed there but
    computed at render time -- the detector ratios, for instance -- are grounded
    by definition: a model that repeats a figure it was shown is quoting, not
    inventing. Leaving them out of the grounding set rejected correct quotations.
    """
    if parse_error:
        return [f"invalid_json:{parse_error}"[:120]]
    if not isinstance(answer, dict):
        return ["missing_key:not_an_object"]

    codes: List[str] = []
    keys = set(answer)
    if keys != set(TEACHER_ANSWER_KEYS):
        codes.append("missing_key:" + ",".join(sorted(keys ^ set(TEACHER_ANSWER_KEYS))) or "missing_key")
        if not set(TEACHER_ANSWER_KEYS).issubset(keys):
            return codes

    reasoning = str(answer.get("physical_reasoning") or "").strip()
    narrative = str(answer.get("narrative") or "").strip()
    if not reasoning or not narrative:
        codes.append("empty_field")
        return codes

    # The response contract allows 1-4 sentences for an answer; requiring two
    # would reject a compliant one-sentence narrative for no stated reason.
    n_narr = _sentences(narrative)
    if not 1 <= n_narr <= 4:
        codes.append(f"narrative_length:{n_narr}")
    n_reason = _sentences(reasoning)
    if not 3 <= n_reason <= 8:
        codes.append(f"reasoning_length:{n_reason}")

    whole = f"{reasoning}\n{narrative}"
    truth_id = str(target.get("fault_id") or "")
    claimed = allowed.resolve_fault(answer.get("fault_id")) if answer.get("fault_id") else None
    if answer.get("fault_id") and (claimed is None or claimed.value != truth_id):
        codes.append(f"label_changed:{answer.get('fault_id')}")

    # Two different failures hide behind a tag-shaped token, and they deserve
    # different names: an id that resolves to no sensor at all (``XMV_99``,
    # ``Condenser_Cooling_Water_Valve``) is a fabrication, while a real tag the
    # event never showed (``XMEAS_17`` on a fault whose evidence lists others) is
    # an ungrounded claim. ``AllowedSet.unknown_tag_like`` reports both, so the
    # split is made here by asking whether the token resolves to any canonical
    # tag at all.
    fabricated: List[str] = []
    ungrounded_names: List[str] = []
    for token in allowed.unknown_tag_like(whole):
        if _is_chemical_name(token):
            continue  # "Component_B" is a species, not a sensor id
        (fabricated if canonical_tag(token) is None else ungrounded_names).append(token)
    if fabricated:
        codes.append("unknown_tag:" + ",".join(fabricated[:3]))

    cited = set()
    for tag in iter_tags(whole):
        resolved = canonical_tag(tag)
        if resolved:
            cited.add(resolved)
    # ``truth_fault_number`` is the simulator's number, which survives the
    # UNCLASSIFIED_OPEN_SET relabelling of faults 16-20 -- without it their KB
    # tags would not count as grounding.
    kb_number = target.get("truth_fault_number")
    if kb_number is None:
        truth_number = target.get("truth_fault")
        if isinstance(truth_number, str) and truth_number.startswith(("FAULT_", "SYNTHETIC_")):
            kb_number = int(truth_number.split("_")[-1])
    grounded = _evidence_tags(evidence) | _kb_tags(kb_number)
    for extra in target.get("ranked_hypotheses") or []:
        value = str(extra.get("fault_id") or "")
        if value.startswith(("FAULT_", "SYNTHETIC_")):
            grounded |= _kb_tags(int(value.split("_")[-1]))
    ungrounded = sorted((cited - grounded) | {tag for tag in ungrounded_names
                                              if canonical_tag(tag) not in grounded})
    if ungrounded:
        codes.append("sensor_not_grounded:" + ",".join(ungrounded[:3]))

    permitted: Set[int] = set()
    for fault_number in fault_ids_in(whole):
        permitted.add(fault_number)
    if kb_number is not None:
        permitted.add(int(kb_number))
    for hyp in target.get("ranked_hypotheses") or []:
        value = str(hyp.get("fault_id") or "")
        if value.startswith(("FAULT_", "SYNTHETIC_")):
            permitted.add(int(value.split("_")[-1]))
    shown = store_numbers(evidence, target)
    if prompt:
        shown = shown + [value for value, _decimals in numbers_in(prompt)]
    # The evidence prints shares as fractions (0.702); "70.2% share" is the same
    # figure stated in percent, not a new one. Accepting the factor-100 reading is
    # a unit conversion, not a tolerance on precision - a truncated 206.82 is still
    # rejected.
    shown = shown + [value * 100.0 for value in shown] + [value / 100.0 for value in shown]
    for value, decimals in numbers_in(whole):
        if not number_supported(value, decimals, shown, sorted(permitted)):
            codes.append(f"number_not_in_store:{value}")
            break

    conflict = _direction_conflict(whole, kb_number)
    if conflict:
        codes.append(f"direction_conflict:{conflict}")

    low = whole.lower()
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

    for number in fault_ids_in(whole):
        if number in SYNTHETIC_FAULTS or not 1 <= number <= 20:
            codes.append(f"fault_out_of_scope:{number}")
            break

    return codes


__all__ = [
    "ABSOLUTE_PATTERNS",
    "SEVERITY_PATTERNS",
    "TEACHER_ANSWER_KEYS",
    "check_answer",
    "store_numbers",
]
