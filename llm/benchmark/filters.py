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
``names_removed_fault``  an abstention item that names the fault it abstains from
``teacher_scaffolding``   prose that quotes the teacher's own prompt labels
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set

from llm.benchmark.kb_causes import distinctive_signatures, is_cause_mention, normalise
from llm.benchmark.menu import UNCLASSIFIED
from llm.benchmark.scaffolding import scaffolding_hits
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
#:
#: The gerunds are present because "pressure is increasing" asserts exactly what
#: "pressure increases" asserts, and a rule that reads only the second rejected
#: one spelling and accepted the other. Before the gerunds were added the check
#: saw 5 of the fault-13 contradictions in this cache instead of 26.
INCREASE_WORDS = ("increase", "increases", "increasing", "rise", "rises", "rising", "risen",
                  "higher", "upward", "climb")
DECREASE_WORDS = ("decrease", "decreases", "decreasing", "fall", "falls", "falling", "fallen",
                  "lower", "downward", "drop")

_NUMBER_RANGE = ("-.2f",)


#: Feed-component names (``Component_A`` .. ``Component_H`` and the streams).
#: They are shaped like tags but name a chemical species, so the tag rules must
#: not fire on them.
_COMPONENT_RE = re.compile(r"^component[_ -]?[a-h]$", re.IGNORECASE)
_STREAM_RE = re.compile(r"^(?:stream|feed)[_ -]?\d{1,2}$", re.IGNORECASE)


def _is_chemical_name(token: str) -> bool:
    return bool(_COMPONENT_RE.match(str(token).strip()) or _STREAM_RE.match(str(token).strip()))


def names_removed_fault(text: str, target: Dict,
                        menu_ids: Optional[Sequence[str]] = None) -> Optional[str]:
    """The fault an abstention item must not name, if ``text`` names it.

    On an open-set episode the true fault's menu entry was removed; on faults
    16-20 it was never there. Either way the answer is UNCLASSIFIED_OPEN_SET, so
    prose that still identifies the removed fault -- by KB name, by signature
    phrase, or by number -- is answering the question that was deliberately made
    unanswerable, and the item would teach the model to recover the label from the
    text rather than from the evidence.

    Returns a short reason (``signature:...`` / ``number:...``) or ``None``.

    Only *distinctive* phrases count: faults 5 and 12 share a cause and differ in
    the disturbance tail, so the shared prefix is also a listed candidate's wording
    and ruling that candidate out is legitimate.
    """
    if str(target.get("fault_id") or "") != UNCLASSIFIED:
        return None
    number = target.get("truth_fault_number")
    if number is None:
        return None
    body = normalise(text)
    others = _menu_fault_numbers(menu_ids)
    for phrase in distinctive_signatures(int(number), others):
        if phrase and phrase in body:
            return f"signature:{phrase[:60]}"
    if re.search(rf"\bfault[_ -]?0?{int(number)}\b", str(text or ""), re.IGNORECASE):
        return f"number:{int(number)}"
    return None


def _menu_fault_numbers(menu_ids: Optional[Sequence[str]]) -> Set[int]:
    """KB numbers behind the fault_id strings a menu was built from."""
    numbers: Set[int] = set()
    for value in menu_ids or ():
        text = str(value or "")
        if text.startswith(("FAULT_", "SYNTHETIC_")):
            try:
                numbers.add(int(text.split("_")[-1]))
            except ValueError:
                continue
    return numbers


def _tag_kind(tag: str) -> str:
    """What an XMV/XMEAS tag measures. A valve position is not a flow."""
    return "valve_position" if str(tag).upper().startswith("XMV_") else "measurement"


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
    """Flag a cited initiating variable described with the opposite sign.

    Only measurements are checked. For faults whose KB ``initiating.sensor`` is an
    XMV, the tag is a valve *position* while the KB's direction describes the
    *flow* that position drives -- the two routinely move oppositely, and the
    stored trend disagreed with the KB in every fault 7, 14 and 15 training event
    (120/120, 120/120, 26/26) while agreeing with it in every fault 4 and 11
    event. Enforcing a flow direction on a position rejects correct prose, so a
    valve position is out of scope here and stays an advisory flag
    (``direction_inconsistent`` also skips it).

    The fault-13 entry this rule reads was itself corrected: it said
    ``decrease`` while the stored trend is increasing in 102/102 train events, so
    the rule was rejecting correct prose. The KB now says ``increase`` (owner
    decision, recorded in ``docs/PROTOCOL.md``) and the rule agrees with the
    evidence instead of the documentation.
    """
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
    if tag is None or str(tag).upper().startswith("XMV_"):
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
    menu_ids: Optional[Sequence[str]] = None,
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

    removal = names_removed_fault(whole, target, menu_ids)
    if removal:
        codes.append(f"names_removed_fault:{removal}")

    # Two different failures hide behind a tag-shaped token, and they deserve
    # different names: an id that resolves to no sensor at all (``XMV_99``,
    # ``Condenser_Cooling_Water_Valve``) is a fabrication, while a real tag the
    # event never showed (``XMEAS_17`` on a fault whose evidence lists others) is
    # an ungrounded claim. ``AllowedSet.unknown_tag_like`` reports both, so the
    # split is made here by asking whether the token resolves to any canonical
    # tag at all.
    # ``truth_fault_number`` is the simulator's number, which survives the
    # UNCLASSIFIED_OPEN_SET relabelling of faults 16-20 -- without it their KB
    # tags would not count as grounding. On an abstention item it licenses the
    # *sensors* only, never the cause phrasing: the menu entry was removed (or
    # never existed), so accepting the true fault's wording would let the answer
    # reconstruct exactly the label the item withholds.
    truth_number = target.get("truth_fault_number")
    if truth_number is None:
        raw_truth = target.get("truth_fault")
        if isinstance(raw_truth, str) and raw_truth.startswith(("FAULT_", "SYNTHETIC_")):
            truth_number = int(raw_truth.split("_")[-1])
    open_set = str(target.get("fault_id") or "") == UNCLASSIFIED
    kb_number = None if open_set else truth_number
    # A named cause is allowed for any fault the item's menu actually offered:
    # naming a listed candidate is legitimate differential reasoning, naming the
    # removed one is not.
    cause_numbers = _menu_fault_numbers(menu_ids)
    if not cause_numbers and not open_set and truth_number is not None:
        cause_numbers = {int(truth_number)}
    if open_set and truth_number is not None:
        cause_numbers.discard(int(truth_number))

    fabricated: List[str] = []
    ungrounded_names: List[str] = []
    for token in allowed.unknown_tag_like(whole):
        if _is_chemical_name(token):
            continue  # "Component_B" is a species, not a sensor id
        # A cause is usually not instrumented: naming "condenser cooling water
        # inlet temperature" is correct even though it has no tag. Only a menu
        # entry's own KB phrasing is allowed through, read from the KB.
        if any(is_cause_mention(token, number) for number in cause_numbers):
            continue
        (fabricated if canonical_tag(token) is None else ungrounded_names).append(token)
    if fabricated:
        codes.append("unknown_tag:" + ",".join(fabricated[:3]))

    cited = set()
    for tag in iter_tags(whole):
        resolved = canonical_tag(tag)
        if resolved:
            cited.add(resolved)
    grounded = _evidence_tags(evidence) | _kb_tags(truth_number)
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
    permitted |= cause_numbers
    if open_set and truth_number is not None:
        permitted.discard(int(truth_number))
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

    # Prose that refers to the teacher's own prompt ("the decided diagnosis ...",
    # "the affected area: ...") is only true inside that wrapper. The phrases are
    # read out of the template by ``llm.benchmark.scaffolding``, so a prompt edit
    # moves the rule with it.
    hits = scaffolding_hits(whole, abstain=str((target or {}).get("fault_id") or "") == UNCLASSIFIED)
    if hits:
        codes.append("teacher_scaffolding:" + ",".join(hit["phrase"] for hit in hits[:3]))

    return codes


#: Direction vocabulary for the consistency FLAG (never a rejection).
UP_WORDS = (
    "increase", "increases", "increasing", "increased", "rise", "rises", "rising", "rose",
    "higher", "upward", "climb", "climbs", "climbing", "grow", "grows", "growing",
)
DOWN_WORDS = (
    "decrease", "decreases", "decreasing", "decreased", "fall", "falls", "falling", "fell",
    "lower", "downward", "drop", "drops", "dropping", "dropped", "decline", "declines",
    "declining", "reduce", "reduces", "reduced", "reducing",
)
_NEGATION = re.compile(r"\b(?:no|not|without|never|neither|nor|n't)\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


#: The stored ``trend`` is the direction of *movement*; ``deviation_percent`` is
#: the offset from baseline. They disagree in 442 of 2,151 ranked sensors in this
#: dataset (a level still below baseline but climbing), so comparing a direction
#: word against the deviation sign flags correct prose. Prefer the trend.
_TREND_UP = frozenset({"increasing", "rising", "up"})
_TREND_DOWN = frozenset({"decreasing", "falling", "down"})


def _stored_directions(evidence: Dict) -> Dict[str, tuple]:
    """Direction the evidence records for each ranked sensor, and its basis."""
    directions: Dict[str, tuple] = {}
    for sensor in (evidence.get("top_anomalous_sensors") or []):
        if not isinstance(sensor, dict):
            continue
        tag = canonical_tag(sensor.get("name")) or canonical_tag(sensor.get("display_name"))
        if not tag:
            continue
        kind = _tag_kind(tag)
        trend = str(sensor.get("trend") or "").strip().lower()
        if trend in _TREND_UP:
            directions[tag] = (1, "trend", kind)
            continue
        if trend in _TREND_DOWN:
            directions[tag] = (-1, "trend", kind)
            continue
        if trend == "stable":
            continue  # no direction claim to contradict
        value = sensor.get("deviation_percent")
        if value is None:
            value = sensor.get("z_score", sensor.get("zScore"))
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0:
            directions[tag] = (1 if value > 0 else -1, "deviation", kind)
    return directions


def direction_inconsistent(answer: Optional[Dict], evidence: Dict, allowed: AllowedSet) -> List[Dict]:
    """Flag sentences whose direction word contradicts the stored sign.

    A flag, never a rejection: the sentence may be describing a *consequence* ("the
    flow is rising, so heat removal falls") in which only one sensor is named and
    the direction belongs to something else. Flagging keeps that visible for a
    human without throwing away a grounded answer.

    Only sentences naming exactly one ranked sensor are considered, and a sentence
    carrying a negation is skipped, because "no longer increasing" is not an
    increase.

    Valve positions (XMV tags) are skipped entirely. ``XMV_44`` is named
    "A_Feed_Flow" in the plant vocabulary and is a *valve position*, while
    ``XMEAS_1`` is the feed flow itself; a valve closing while the flow it admits
    rises under pressure is ordinary, so comparing the two directions flags
    correct prose. The prompt now labels every tag with its kind, and this check
    leaves valve sentences to the human.
    """
    if not isinstance(answer, dict):
        return []
    signs = _stored_directions(evidence)
    if not signs:
        return []
    text = f"{answer.get('physical_reasoning') or ''}\n{answer.get('narrative') or ''}"
    flags: List[Dict] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        body = sentence.strip()
        if not body or _NEGATION.search(body.lower()):
            continue
        tags = [tag for tag in allowed.tags_in(body) if tag in signs]
        if len(tags) != 1:
            continue
        low = body.lower()
        up = [w for w in UP_WORDS if re.search(rf"\b{re.escape(w)}\b", low)]
        down = [w for w in DOWN_WORDS if re.search(rf"\b{re.escape(w)}\b", low)]
        if bool(up) == bool(down):
            continue  # no direction word, or both directions in one clause
        stored, basis, kind = signs[tags[0]]
        if kind == "valve_position":
            continue
        if (up and stored < 0) or (down and stored > 0):
            flags.append({
                "sentence": body,
                "sensor": tags[0],
                "tag_kind": kind,
                "stored": "increase" if stored > 0 else "decrease",
                "basis": basis,
                "words": (up or down)[:3],
            })
    return flags


def direction_report(answer: Optional[Dict], evidence: Dict, allowed: AllowedSet) -> Dict:
    """Flags plus what was skipped, for the per-run report."""
    signs = _stored_directions(evidence)
    return {
        "flags": direction_inconsistent(answer, evidence, allowed),
        "valve_tags_skipped": sorted(tag for tag, (_s, _b, kind) in signs.items()
                                    if kind == "valve_position"),
    }


__all__ = [
    "ABSOLUTE_PATTERNS",
    "DOWN_WORDS",
    "SEVERITY_PATTERNS",
    "TEACHER_ANSWER_KEYS",
    "UP_WORDS",
    "check_answer",
    "direction_inconsistent",
    "direction_report",
    "names_removed_fault",
    "store_numbers",
]
