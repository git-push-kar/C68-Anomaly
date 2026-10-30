"""Case taxonomy and the deterministic half of every follow-up turn.

The rule this module implements is the one the phase brief states: **the model
never decides anything structural.** For each turn the code owns

* the intent (and therefore whether the turn is a template or a model answer),
* the table id and its parameters,
* the citations (store keys, all resolved before they are written),
* the clarifying-question decision (the router's own triggers),
* the refusal decision for an out-of-scope or false-premise question, taken from
  :class:`~llm.response.store_access.StoreAccess` (a key that is missing, or a
  value that is contradicted),
* the suggested follow-ups (router-verified, answerable, not duplicates),

and the teacher owns exactly one string: the 1-4 sentence ``answer``.

Case types
----------
``TEMPLATE_CASE_TYPES`` are answered by the lookup path and never sent to a
model; they appear only as the small controlled sample (default 5%).

The remaining case types are LLM-routed *cases*, not intents. Three of them have
no intent at all live (valve/actuator, free-form causal, mechanism): the live
dispatcher claims nothing and ``RCAInference.answer_followup`` generates the
prose. They are recorded with ``intent=None`` rather than being forced into a
template intent they do not have.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from llm.benchmark.prompt_v2 import tag_kind
from llm.benchmark.targets import NORMAL_ID
from llm.response.lookup_answers import (
    TABLE_FOR_INTENT,
    _citations_for,
    lookup_answer,
    suggestions_for,
)
from llm.response.router import LOOKUP_INTENTS, ResponseRouter
from llm.response.schemas import Intent, TableID, count_sentences
from llm.response.store_access import StoreAccess, UnknownKeyError
from llm.response.renderer import fmt
from scripts.fault_ids import FaultID, fault_id_for
from scripts.fault_knowledge import FAULT_KNOWLEDGE
from scripts.tags import canonical_tag, display_name

#: Live-dispatcher intents (the ones with a template that answers them).
LIVE_TEMPLATE_INTENTS: Tuple[str, ...] = (
    "sensor_values",
    "evidence_why",
    "system_sensors",
    "sensor_scores",
    "detector_severity",
    "detector_scores",
    "severity",
    "overview",
    "sensors_inspect",
    "action",
    "overview_fallback",
)

#: How each candidate intent is answered. ``template`` = lookup path only.
ROUTING: Dict[str, str] = {name: "template" for name in LIVE_TEMPLATE_INTENTS}
ROUTING.update(
    {
        "hypothesis_comparison": "llm",
        # The *decision* is code (store access); the wording is the model's.
        "out_of_scope_refusal": "llm",
        "clarification_reply": "llm",
    }
)

#: Case types that produce a teacher call. Everything else is lookup text.
CASE_TYPES: Dict[str, Dict[str, object]] = {
    "valve_actuator": {"routing": "llm", "intent": None, "answerability": "llm"},
    "causal_why": {"routing": "llm", "intent": None, "answerability": "llm"},
    "mechanism": {"routing": "llm", "intent": None, "answerability": "llm"},
    "hypothesis_comparison": {
        "routing": "llm",
        "intent": "hypothesis_comparison",
        "answerability": "llm",
    },
    "out_of_scope": {
        "routing": "llm",
        "intent": "out_of_scope_refusal",
        "answerability": "refuse",
    },
    "false_premise": {
        "routing": "llm",
        "intent": "out_of_scope_refusal",
        "answerability": "refuse",
    },
    "repeated_question": {"routing": "llm", "intent": None, "answerability": "llm"},
    "ambiguous_question": {"routing": "llm", "intent": None, "answerability": "clarify"},
    "clarification_reply": {
        "routing": "llm",
        "intent": "clarification_reply",
        "answerability": "llm",
    },
    "template_sample": {"routing": "template", "intent": None, "answerability": "deterministic"},
}

TEMPLATE_CASE_TYPES = frozenset({"template_sample"})

#: Order used in reports and tests, so a table reads the same everywhere.
FOLLOWUP_CASE_ORDER: Tuple[str, ...] = (
    "valve_actuator",
    "causal_why",
    "mechanism",
    "hypothesis_comparison",
    "out_of_scope",
    "false_premise",
    "repeated_question",
    "ambiguous_question",
    "clarification_reply",
    "template_sample",
)

#: Default mix per dialogue. The template sample is the 5% the spec requires.
DEFAULT_MIX: Dict[str, float] = {
    "valve_actuator": 0.22,
    "causal_why": 0.18,
    "mechanism": 0.12,
    "hypothesis_comparison": 0.12,
    "out_of_scope": 0.06,
    "false_premise": 0.08,
    "repeated_question": 0.06,
    "ambiguous_question": 0.06,
    "clarification_reply": 0.05,
    "template_sample": 0.05,
}

#: Turn kinds per primary case type: 2-3 turns, extended to 4 for some events.
PLANS: Dict[str, Tuple[str, ...]] = {
    "valve_actuator": ("valve_actuator", "valve_actuator"),
    "causal_why": ("causal_why", "mechanism"),
    "mechanism": ("mechanism", "causal_why"),
    "hypothesis_comparison": ("hypothesis_comparison", "valve_actuator"),
    "out_of_scope": ("out_of_scope", "valve_actuator"),
    "false_premise": ("false_premise", "valve_actuator"),
    "repeated_question": ("valve_actuator", "repeat", "valve_actuator"),
    "ambiguous_question": ("ambiguous_question", "valve_actuator"),
    "clarification_reply": ("ambiguous_question", "clarification_reply"),
    "template_sample": ("template", "template"),
}

#: Questions whose keywords collide across intents, so the router asks which
#: view was meant. Verified against the router at build time, never assumed.
AMBIGUOUS_QUESTIONS: Tuple[str, ...] = (
    "Explain the anomaly score",
    "What is the sensor score and the severity?",
    "Which sensors are used and which are deviating?",
    "What happened and what are the scores?",
    "Tell me the evidence and the severity.",
    "Which fault is this and what is the score?",
    "What is the sensor value and the severity?",
    "Explain the metrics and the threshold.",
)

#: Filled from the target's own hypotheses, so no fault outside the menu leaks in.
COMPARISON_TEMPLATES: Tuple[str, ...] = (
    "How do we distinguish this from {lookalike}?",
    "How is this different from {lookalike}?",
    "Is this {truth} or {lookalike}?",
    "What is the differential between {truth} and {lookalike}?",
    "Which lookalike fits better, {truth} or {lookalike}?",
)

#: Off-domain questions (nothing to do with the plant). Kept separate from the
#: in-domain "key is not stored" flavour, because the store check differs.
OFF_DOMAIN_QUESTIONS: Tuple[str, ...] = (
    "What is the weather in Houston?",
    "Who won the football last night?",
    "What is the capital of France?",
    "Convert 50 psi to bar.",
    "Book a meeting room for the shift handover.",
    "Translate valve stuck into German.",
)

#: Store fields that do not exist for any sensor: asking for one is a key miss.
MISSING_FIELDS: Tuple[str, ...] = ("vibration", "calibration", "corrosion")

#: Fallback questions offered by a refusal; each is routed before it is offered.
REFUSAL_ALTERNATIVES: Tuple[str, ...] = (
    "What are the detector scores?",
    "What is the severity?",
    "Which sensors should I inspect?",
    "What is the recommended action?",
)


# ---------------------------------------------------------------------------
# identifiers and context
# ---------------------------------------------------------------------------
def opaque_incident_id(event_id: str) -> str:
    """Opaque, stable label for the context.

    The benchmark event ids encode split, fault and run (``EVT-train-f15-r004``),
    which is exactly the leak the follow-up context must not carry. This hash is
    stable across rebuilds and carries none of that structure.
    """
    digest = hashlib.sha256(f"incident|{event_id}".encode("utf-8")).hexdigest()
    return f"INC-{digest[:6].upper()}"


def _kb(number: Optional[int]) -> Dict:
    if number is None:
        return {}
    return FAULT_KNOWLEDGE.get(int(number)) or {}


def label_for_fault(fault_id: Optional[str]) -> str:
    """``FAULT_15`` -> ``FAULT_15 (Condenser cooling water valve sticking)``."""
    if not fault_id:
        return "unclassified open-set disturbance"
    text = str(fault_id)
    if text.startswith(("FAULT_", "SYNTHETIC_")):
        number = int(text.split("_")[-1])
        name = _kb(number).get("name")
        return f"{text} ({name})" if name else text
    return text


def _sensor_lines(store: StoreAccess, limit: int = 5) -> List[str]:
    lines: List[str] = []
    for tag in store.known_tags()[:limit]:
        observed = _try(store, f"sensor.{tag}.observed")
        baseline = _try(store, f"sensor.{tag}.baseline")
        dev = _try(store, f"sensor.{tag}.deviation")
        z = _try(store, f"sensor.{tag}.z")
        trend = _try(store, f"sensor.{tag}.trend")
        name = display_name(tag) or tag
        if observed is None or baseline is None:
            continue
        # Every tag carries its kind, exactly as the report prompt does: an XMV
        # is a valve *position* and an XMEAS is the measurement, and two of them
        # share a display name (``A_C_Feed_Flow`` is both XMV_45 and XMEAS_4).
        # Without the label the teacher can describe a valve as a flow reading.
        lines.append(
            f"- {name} [{tag}, {tag_kind(tag)}]: observed {fmt(float(observed))}, baseline "
            f"{fmt(float(baseline))}, change {float(dev or 0):+.2f}%, z {float(z or 0):+.2f}, "
            f"trend {trend or 'unknown'}"
        )
    return lines or ["- (no single variable moved strongly)"]


def _try(store: StoreAccess, key: str) -> Optional[object]:
    try:
        return store.resolve(key)
    except UnknownKeyError:
        return None


def render_context(store: StoreAccess, target: Dict, *, open_set: bool) -> str:
    """The stored incident, as the model sees it.

    Every number comes from the event through a store key or the detector block;
    there is no event id, split name, run number or filename in the text.
    """
    lstm_score = float(_try(store, "detector.lstm.score") or 0.0)
    lstm_thr = float(_try(store, "detector.lstm.threshold") or 0.0)
    dyn_score = float(_try(store, "detector.dynamic.score") or 0.0)
    dyn_thr = float(_try(store, "detector.dynamic.threshold") or 0.0)
    lstm_alarmed = bool(_try(store, "detector.lstm.alarmed"))
    dyn_alarmed = bool(_try(store, "detector.dynamic.alarmed"))
    dynamic = store.detector.get("dynamic") if isinstance(store.detector.get("dynamic"), dict) else {}
    lag = dynamic.get("dominant_lag")
    drift = dynamic.get("slow_drift_ratio")

    lines = [
        f"INCIDENT {opaque_incident_id(str(store.event.get('event_id') or ''))}",
        "",
        "DETECTOR READINGS",
        f"- LSTM reconstruction error {lstm_score:.4f} against a limit of {lstm_thr:.4f} "
        f"({'above' if lstm_alarmed else 'below'} the limit)",
        f"- CVA Q-statistic {dyn_score:.1f} against a limit of {dyn_thr:.1f} "
        f"({'above' if dyn_alarmed else 'below'} the limit)",
    ]
    if lag is not None:
        lines.append(f"- dominant lag {lag} window(s), slow-drift ratio {float(drift or 0):.2f}")
    lines += ["", "VARIABLES THAT MOVED (largest first)"]
    lines += _sensor_lines(store)
    lines += ["", "LEADING DIAGNOSIS"]
    if open_set:
        lines.append(
            "- no listed fault family matches this signature; the closest candidates are "
            + ", ".join(label_for_fault(h.get("fault_id")) for h in (target.get("ranked_hypotheses") or [])[:2])
        )
    elif target.get("fault_id") == NORMAL_ID:
        lines.append(
            "- the excursion closed without a confirmed process fault "
            "(no fault-justified separation from normal operation)"
        )
    else:
        lines.append(f"- {label_for_fault(target.get('fault_id'))}")
        lines.append(f"- affected area: {str(target.get('primary_subsystem') or 'unknown').replace('_', ' ')}")
        others = [h for h in (target.get("ranked_hypotheses") or []) if not h.get("is_truth")]
        if others:
            lines.append(
                "- also considered: "
                + ", ".join(label_for_fault(h.get("fault_id")) for h in others[:2])
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# store checks: refusal and false premises
# ---------------------------------------------------------------------------
@dataclass
class RefusalDecision:
    """Why this question cannot be answered from the store, and what can."""

    required: bool
    reason: str  # key_missing | contradicted | off_domain
    premise: Dict
    alternatives: List[Dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "required": self.required,
            "reason": self.reason,
            "premise": dict(self.premise),
            "alternatives": [dict(a) for a in self.alternatives],
        }


def _answerable(router: ResponseRouter, event: Dict, report: Dict, question: str) -> Optional[Dict[str, str]]:
    """An alternative counts as answerable when the lookup path can answer it.

    The clarification triggers are deliberately not consulted: on an unclassified
    incident the router asks its "closest match or raw evidence" question about
    *every* turn, which says nothing about whether the detector readings exist.
    """
    # Answerability is a property of the LIVE path: the dispatcher claims the
    # question and answers it from evidence. The router's clarification triggers
    # are not consulted, because on an unclassified incident the router asks its
    # "closest match or raw evidence" question about every turn regardless of
    # whether the readings exist.
    intent = router.live_intent(event, report, question)
    if intent is None:
        return None
    return {"question": question, "intent": intent}


def refusal_alternatives(
    event: Dict, report: Dict, router: Optional[ResponseRouter] = None, limit: int = 2
) -> List[Dict[str, str]]:
    """Questions the store can actually answer, verified through the router."""
    router = router or ResponseRouter()
    out: List[Dict[str, str]] = []
    for question in REFUSAL_ALTERNATIVES:
        hit = _answerable(router, event, report, question)
        if hit and hit not in out:
            out.append(hit)
        if len(out) >= limit:
            break
    return out


def _store_has(store: StoreAccess, key: str) -> bool:
    return _try(store, key) is not None


def premise_check(
    event: Dict,
    store: StoreAccess,
    kind: str,
    *,
    claim: str,
    target: Dict,
    report: Dict,
) -> Dict:
    """Check one asserted premise against the store.

    Returns ``{"kind", "claim", "verdict", "store_key", "actual", "reason"}``.
    ``verdict`` is ``contradicted`` (the store says otherwise) or ``missing``
    (the store has no such key). A premise that comes back ``supported`` is a
    bug in the caller: the builder re-draws until the premise is false.
    """
    if kind == "sensor":
        tag = canonical_tag(claim) or claim
        key = f"sensor.{tag}.observed"
        actual = _try(store, key)
        verdict = "contradicted" if actual is not None else "missing"
        reason = "sensor is present in this event" if actual is not None else "sensor is not in this event"
        return {
            "kind": kind,
            "claim": claim,
            "verdict": verdict,
            "store_key": key,
            "actual": actual,
            "reason": reason,
        }
    if kind == "key":
        tag, _, field_name = claim.partition(":")
        key = f"sensor.{canonical_tag(tag) or tag}.{field_name or 'vibration'}"
        actual = _try(store, key)
        return {
            "kind": kind,
            "claim": f"{field_name} of {tag}",
            "verdict": "contradicted" if actual is not None else "missing",
            "store_key": key,
            "actual": actual,
            "reason": "the event store has no such field for this variable",
        }
    if kind == "fault":
        claimed = str(claim)
        truth = str(target.get("fault_id") or "")
        verdict = "contradicted"
        if claimed == truth:
            verdict = "supported"
        return {
            "kind": kind,
            "claim": claimed,
            "verdict": verdict,
            "store_key": "kb.fault.%s.name" % claimed.split("_")[-1] if "_" in claimed else "subsystem.candidate",
            "actual": truth,
            "reason": f"the stored diagnosis is {label_for_fault(truth)}",
        }
    if kind == "value":
        dynamic_score = _try(store, "detector.dynamic.score")
        claimed_value = float(claim)
        actual = float(dynamic_score or 0.0)
        verdict = "contradicted" if abs(actual - claimed_value) > 1e-6 else "supported"
        return {
            "kind": kind,
            "claim": claim,
            "verdict": verdict,
            "store_key": "detector.dynamic.score",
            "actual": actual,
            "reason": "the stored dynamic reading is different",
        }
    raise ValueError(f"unknown premise kind {kind!r}")


def refusal_for(
    event: Dict,
    store: StoreAccess,
    target: Dict,
    report: Dict,
    *,
    question: str,
    premise: Optional[Dict],
    router: Optional[ResponseRouter] = None,
) -> RefusalDecision:
    """The refusal decision for one turn, with the alternatives it may offer."""
    router = router or ResponseRouter()
    alternatives = refusal_alternatives(event, report, router)
    if premise is None:
        return RefusalDecision(
            required=True,
            reason="off_domain",
            premise={"kind": "domain", "claim": question, "verdict": "missing", "store_key": "", "actual": None,
                     "reason": "the question is about nothing the incident store holds"},
            alternatives=alternatives,
        )
    reason = "key_missing" if premise["verdict"] == "missing" else "contradicted"
    return RefusalDecision(required=True, reason=reason, premise=premise, alternatives=alternatives)


# ---------------------------------------------------------------------------
# clarification
# ---------------------------------------------------------------------------
def clarification_for(
    event: Dict,
    report: Dict,
    question: str,
    asked: Sequence[str],
    router: Optional[ResponseRouter] = None,
) -> Tuple[Optional[Dict], List[Intent]]:
    """The router's clarifying question for this turn, if a trigger fires."""
    router = router or ResponseRouter(enable_proposed=True)
    result = router.route(event, report, question, asked=list(asked), with_answer=False)
    if result.clarification is None:
        return None, list(result.candidates)
    return result.clarification.as_dict(), list(result.candidates)


def resolved_intent_for_option(option: str) -> Optional[str]:
    """Map a clarification option onto the intent it selects.

    ``unclassified`` offers "closest match" / "raw evidence"; an intent-collision
    clarification offers two intent names. Both cases stay in the allowed enum.
    """
    text = (option or "").strip().lower()
    if text in ("closest match", "closest", "first", "the first one"):
        return Intent.OVERVIEW.value
    if text in ("raw evidence", "evidence", "raw readings", "the second one"):
        return Intent.EVIDENCE_WHY.value
    try:
        return Intent(text).value
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# tables, citations, suggestions
# ---------------------------------------------------------------------------
def table_for(intent: Optional[str], target: Dict, store: StoreAccess) -> Tuple[str, Dict]:
    """Table id and its parameters for a turn."""
    if intent is None:
        return TableID.NONE.value, {}
    try:
        table = TABLE_FOR_INTENT[Intent(intent)]
    except (KeyError, ValueError):
        return TableID.NONE.value, {}
    params: Dict[str, object] = {}
    if table == TableID.EVIDENCE:
        params["tags"] = store.known_tags()[:5]
    if table == TableID.HYPOTHESES:
        params["hypotheses"] = [
            h.get("fault_id") for h in (target.get("ranked_hypotheses") or [])
        ][:3]
    return table.value, params


def citations_for(
    intent: Optional[str],
    event: Dict,
    report: Dict,
    store: StoreAccess,
    target: Dict,
    *,
    extra_faults: Sequence[str] = (),
) -> List[str]:
    """Citation keys for a turn; every one is resolved before it is kept."""
    keys: List[str] = []
    if intent is None:
        # Free-form cases: the detector readings and the top variables are what
        # an answer may quote, so those are the keys it is scored against.
        keys += [
            "detector.lstm.score",
            "detector.lstm.threshold",
            "detector.dynamic.score",
            "detector.dynamic.threshold",
        ]
        for tag in store.known_tags()[:1]:
            keys += [f"sensor.{tag}.observed", f"sensor.{tag}.baseline"]
    else:
        keys += _citations_for(Intent(intent), store, report)
    for fault_id in extra_faults:
        number = str(fault_id).split("_")[-1]
        if str(fault_id).startswith(("FAULT_", "SYNTHETIC_")):
            keys.append(f"kb.fault.{number}.name")
    resolved: List[str] = []
    for key in dict.fromkeys(keys):
        if _try(store, key) is not None:
            resolved.append(key)
    return resolved[:6]


def suggestions_for_turn(
    intent: Optional[str],
    event: Dict,
    report: Dict,
    question: str,
    router: Optional[ResponseRouter] = None,
    limit: int = 3,
) -> List[str]:
    """Follow-up suggestions that the live router really answers, deduplicated."""
    router = router or ResponseRouter()
    pool: List[str] = []
    if intent is not None:
        try:
            pool += suggestions_for(Intent(intent))
        except ValueError:
            pass
    pool += list(REFUSAL_ALTERNATIVES)
    seen = {" ".join((question or "").lower().split())}
    out: List[str] = []
    for suggestion in pool:
        key = " ".join(suggestion.lower().split())
        if key in seen:
            continue
        hit = _answerable(router, event, report, suggestion)
        if hit is None:
            continue
        out.append(suggestion)
        seen.add(key)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def _hash_int(seed: str, modulo: int) -> int:
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16) % max(modulo, 1)


def choose_case_type(mix: Dict[str, float], key: str) -> str:
    """Deterministic weighted draw; the same key always yields the same case."""
    names = [name for name in FOLLOWUP_CASE_ORDER if mix.get(name, 0.0) > 0]
    total = sum(float(mix[name]) for name in names)
    if not names or total <= 0:
        raise ValueError("empty case mix")
    point = (_hash_int(f"case|{key}", 1_000_000) / 1_000_000.0) * total
    running = 0.0
    for name in names:
        running += float(mix[name])
        if point < running:
            return name
    return names[-1]


def plan_for(case_type: str, key: str) -> List[str]:
    """Turn kinds for one dialogue (2-3 turns, extended to 4 on some keys)."""
    plan = list(PLANS[case_type])
    if _hash_int(f"extra|{key}", 4) == 0:
        filler = "mechanism" if "causal_why" not in plan else "valve_actuator"
        if len(plan) < 4 and filler not in plan[-1:]:
            plan.append(filler)
    return plan[:4]


def template_labels(case_type: str, key: str) -> List[str]:
    """Template intents used by a template-sample dialogue (two distinct ones)."""
    if case_type != "template_sample":
        return []
    first = _hash_int(f"tpl-a|{key}", len(LIVE_TEMPLATE_INTENTS))
    second = _hash_int(f"tpl-b|{key}", len(LIVE_TEMPLATE_INTENTS))
    if second == first:
        second = (second + 1) % len(LIVE_TEMPLATE_INTENTS)
    return [LIVE_TEMPLATE_INTENTS[first], LIVE_TEMPLATE_INTENTS[second]]


def pick_template_turn(event: Dict, report: Dict, question: str) -> Optional[Dict]:
    """The lookup answer for a template question, or ``None`` if not routed."""
    hit = lookup_answer(event, report, question)
    if hit is None:
        return None
    return {
        "answer": hit.answer.answer,
        "intent": hit.answer.intent.value,
        "table": hit.answer.table.value,
        "citations": list(hit.citations),
        "suggestions": list(hit.answer.suggestions),
        "detail": hit.detail,
    }


def hypothesis_pair(target: Dict) -> Tuple[str, Optional[str], Optional[str]]:
    """(truth label, lookalike label, lookalike fault id) for a comparison."""
    truth = str(target.get("fault_id") or "")
    others = [h.get("fault_id") for h in (target.get("ranked_hypotheses") or []) if not h.get("is_truth")]
    lookalike = str(others[0]) if others else None
    return truth, lookalike, lookalike


def comparison_question(templates: Sequence[str], target: Dict, index: int, *, open_set: bool) -> str:
    """Fill a comparison template from the target's own hypotheses.

    An open-set item has no permitted fault id to name (its true entry was
    removed from the menu), so only the templates that speak about "this" and
    the lookalike are usable there.
    """
    truth, lookalike, _ = hypothesis_pair(target)
    lookalike_label = label_for_fault(lookalike) if lookalike else "the closest candidate"
    template = templates[index % len(templates)]
    truth_is_nameable = (not open_set) and str(truth).startswith(("FAULT_", "SYNTHETIC_"))
    if "{truth}" in template and not truth_is_nameable:
        template = next((item for item in templates if "{truth}" not in item), templates[0])
    if "{truth}" in template:
        return template.format(truth=label_for_fault(truth), lookalike=lookalike_label)
    return template.format(lookalike=lookalike_label)


def premise_sentence(premise: Dict) -> str:
    """Human-readable statement of a checked premise (context for the teacher)."""
    kind = premise.get("kind")
    if kind == "sensor":
        return f"the question asserts {premise.get('claim')} is the earliest moving variable"
    if kind == "key":
        return f"the question asks for {premise.get('claim')}, which the store does not hold"
    if kind == "fault":
        return f"the question asserts {label_for_fault(premise.get('claim'))}"
    if kind == "value":
        return f"the question asserts the dynamic reading is {premise.get('claim')}"
    return "the question is outside what the incident store holds"


def sentence_count(text: str) -> int:
    return count_sentences(text)


__all__ = [
    "AMBIGUOUS_QUESTIONS",
    "CASE_TYPES",
    "COMPARISON_TEMPLATES",
    "DEFAULT_MIX",
    "FOLLOWUP_CASE_ORDER",
    "LIVE_TEMPLATE_INTENTS",
    "MISSING_FIELDS",
    "OFF_DOMAIN_QUESTIONS",
    "PLANS",
    "REFUSAL_ALTERNATIVES",
    "ROUTING",
    "RefusalDecision",
    "TEMPLATE_CASE_TYPES",
    "choose_case_type",
    "citations_for",
    "clarification_for",
    "comparison_question",
    "hypothesis_pair",
    "label_for_fault",
    "opaque_incident_id",
    "pick_template_turn",
    "plan_for",
    "premise_check",
    "premise_sentence",
    "refusal_alternatives",
    "refusal_for",
    "render_context",
    "resolved_intent_for_option",
    "suggestions_for_turn",
    "table_for",
    "template_labels",
]
