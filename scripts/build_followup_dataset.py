#!/usr/bin/env python
"""Build the follow-up dialogue dataset from Phase 2B events.

One dialogue per event, of 2-4 turns, where the *structural* part of every turn
-- intent, table, citations, clarifying question, refusal decision, suggestions --
is computed here and only the natural ``answer`` text is left to a teacher. With
``--teacher fake`` (the default) nothing talks to a model at all.

Splits are the Phase 2B splits: train dialogues come from train events, val from
val, test from test. Faults 3 and 9 are excluded from train (their events exist
but the detector barely sees them). Every question and context is leak-checked
and every split pair is proven run-disjoint before a single teacher call.

Usage::

    python scripts/build_followup_dataset.py --teacher fake                # default, no model
    python scripts/build_followup_dataset.py --splits train --limit 40     # sample
    python scripts/build_followup_dataset.py --teacher ollama --model qwen3:30b   # real run

The real run is costed in the manifest (``cost_estimate``) and is never started
by the tests or by the default invocation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_benchmark_dataset import (  # noqa: E402  (shared record loading)
    evidence_of,
    load_records,
    record_key_of,
)
from llm.benchmark.filters import (  # noqa: E402
    ABSOLUTE_PATTERNS,
    SEVERITY_PATTERNS,
    names_removed_fault,
    store_numbers,
)
from llm.benchmark.menu import CLASSIFIABLE_FAULTS, UNCLASSIFIED  # noqa: E402
from llm.benchmark.targets import (  # noqa: E402
    NORMAL_ID,
    build_target,
    eligible_for_open_set,
    open_set_assignment,
)
from llm.followup import cases  # noqa: E402
from llm.followup.cases import (  # noqa: E402
    AMBIGUOUS_QUESTIONS,
    CASE_TYPES,
    COMPARISON_TEMPLATES,
    DEFAULT_MIX,
    FOLLOWUP_CASE_ORDER,
    MISSING_FIELDS,
    OFF_DOMAIN_QUESTIONS,
)
from llm.followup.filters import AnswerContext, check_answer  # noqa: E402
from llm.followup.leak import (  # noqa: E402
    assert_clean,
    check_event_overlap,
    check_paraphrase_disjointness,
)
from llm.followup.provenance import followup_provenance  # noqa: E402
from llm.followup.paraphrases import (  # noqa: E402
    FALSE_PREMISE_TEMPLATES,
    LABELS,
    normalise,
    pick_question,
    questions_for_split,
)
from llm.followup.teacher import (  # noqa: E402
    FakeTeacher,
    TeacherCache,
    TeacherPayload,
    build_prompt,
    sha256_text,
)
from llm.response.allowed_set import build_allowed_set  # noqa: E402
from llm.response.lookup_answers import lookup_answer  # noqa: E402
from llm.response.renderer import fmt  # noqa: E402
from llm.response.router import ResponseRouter  # noqa: E402
from llm.response.schemas import Intent  # noqa: E402
from llm.response.store_access import StoreAccess, UnknownKeyError  # noqa: E402
from llm.response.validators import numbers_in  # noqa: E402
from scripts.fault_ids import fault_id_for  # noqa: E402
from scripts.fault_knowledge import FAULT_KNOWLEDGE  # noqa: E402
from scripts.tags import iter_tags  # noqa: E402

logger = logging.getLogger("build_followup")

EVENTS = ROOT / "outputs" / "benchmark_v2" / "events"
OUT = ROOT / "outputs" / "benchmark_v2" / "followup"
SPLITS_FILE = ROOT / "data" / "splits_v2.json"

#: Measured Phase 2B teacher latency, used only for the cost estimate.
SECONDS_PER_CALL = 10.9
#: Phase 2B pilot rejection rate, used to cost one retry per rejection.
PILOT_REJECTION_RATE = 0.12
#: Report items in the Phase 2B train split, from the phase brief. Used only
#: when the Phase 2B manifest is absent: the built dataset has since published
#: its own count, which is what the projection prefers.
PHASE_2B_TRAIN_REPORT_ITEMS = 1939


def phase2b_train_report_items() -> int:
    """The train item count the report task actually ships, when it is readable."""
    path = ROOT / "outputs" / "benchmark_v2" / "dataset" / "manifest.json"
    try:
        counts = json.loads(path.read_text(encoding="utf-8")).get("counts") or {}
        return int((counts.get("train") or {}).get("items") or PHASE_2B_TRAIN_REPORT_ITEMS)
    except (OSError, ValueError, TypeError):
        return PHASE_2B_TRAIN_REPORT_ITEMS
#: Train faults whose detector signature is too weak to learn from (brief §2).
TRAIN_EXCLUDED_FAULTS = (3, 9)
#: Added to the shared filters for template-sample lookups only. The KB's
#: remediation text opens with "CRITICAL SAFETY INTERLOCK: ...", which is a
#: severity claim in all but name; a 5% format sample must not teach it.
TEMPLATE_BLOCKED = SEVERITY_PATTERNS + ABSOLUTE_PATTERNS + (r"\bcritical\b",)


# ---------------------------------------------------------------------------
# per-dialogue state
# ---------------------------------------------------------------------------
@dataclass
class DialogueState:
    record: Dict
    split: str
    variant: str
    target: Dict
    event: Dict
    report: Dict
    store: StoreAccess
    router: ResponseRouter
    allowed: object
    permitted_faults: List[int]
    removed_fault: Optional[int]
    grounded_numbers: List[float]
    key: str
    open_set: bool
    normal: bool
    asked: List[str]
    history: List[Dict[str, str]]
    turn_records: List[Dict]
    labels: Dict[str, str]


def build_report(target: Dict, evidence: Dict) -> Dict:
    """The stored report shape the lookup path and the router expect."""
    return {
        "root_cause": target.get("fault_id"),
        "affected_subsystem": target.get("primary_subsystem") or evidence.get("candidate_subsystem"),
        "hypotheses": [
            {"fault": h.get("fault_id"), "confidence": None}
            for h in (target.get("ranked_hypotheses") or [])
        ],
        "recommended_action": target.get("mitigation"),
    }


def numeric_strings(store: StoreAccess) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """Pre-formatted numbers the fake teacher may quote, plus display labels.

    The strings are exactly what :func:`llm.followup.cases.render_context` prints,
    so a fake answer quoting one is quoting the prompt, which the filters accept
    by construction.
    """
    values: Dict[str, str] = {}
    labels: Dict[str, str] = {}
    for name, key, spec in (
        ("lstm", "detector.lstm.score", "{:.4f}"),
        ("dynamic", "detector.dynamic.score", "{:.1f}"),
    ):
        try:
            values[name] = spec.format(float(store.resolve(key)))
        except UnknownKeyError:
            continue
    tags = store.known_tags()
    if tags:
        tag = tags[0]
        values["sensor"] = store.display(tag) or tag
        labels["first_sensor"] = values["sensor"]
        for field in ("observed", "baseline"):
            try:
                values[field] = fmt(float(store.resolve(f"sensor.{tag}.{field}")))
            except UnknownKeyError:
                continue
    valves = [tag for tag in tags if tag.startswith("XMV_")]
    if valves:
        labels["valve"] = store.display(valves[0]) or valves[0]
    return list(values.values()), values, labels


def _menu_faults(open_set: bool, removed: Optional[int]) -> List[int]:
    faults = list(CLASSIFIABLE_FAULTS)
    if open_set and removed in faults:
        faults.remove(int(removed))
    return faults


# ---------------------------------------------------------------------------
# question selection with routing verification
# ---------------------------------------------------------------------------
def pick_routed_question(
    label: str,
    state: DialogueState,
    accept: Callable[[str], bool],
    *,
    salt: str = "",
) -> Tuple[str, int, int]:
    """First question in the pool (from a hashed start) that routes as intended.

    The bank is measured against the router, not trusted: a paraphrase the live
    dispatcher answers by template cannot be used for an LLM turn, and vice
    versa. ``skipped`` counts how many paraphrases were tried and rejected.
    """
    pool = questions_for_split(label, state.split)
    start, _ = pick_question(label, f"{state.key}{salt}", state.split)
    start = pool.index(start)
    skipped = 0
    for offset in range(len(pool)):
        question = pool[(start + offset) % len(pool)]
        # An open-set dialogue's target is "none of the listed candidates", so the
        # operator's question must not name the removed family either: the bank is
        # written per *case type*, and its wording sometimes carries one fault's
        # KB cause ("is the condenser cooling water valve sticking?"), which for a
        # fault-15 open-set item would put the answer in the prompt. The same
        # signature rule that rejects an answer names the question here.
        if state.removed_fault is not None and names_removed_fault(
            question,
            {"fault_id": UNCLASSIFIED, "truth_fault_number": int(state.removed_fault)},
        ):
            skipped += 1
            continue
        if accept(question):
            return question, (start + offset) % len(pool), skipped
        skipped += 1
    raise RuntimeError(f"{label}: no paraphrase in the {state.split} pool routes as intended")


def template_accept(label: str, state: DialogueState) -> Callable[[str], bool]:
    def _accept(question: str) -> bool:
        live = state.router.live_intent(state.event, state.report, question)
        if live != label:
            return False
        hit = lookup_answer(state.event, state.report, question, router=state.router)
        if hit is None:
            return False
        clarification, _ = cases.clarification_for(
            state.event, state.report, question, [], router=state.router
        )
        return clarification is None

    return _accept


def llm_accept(state: DialogueState, *, require_comparison: bool = False) -> Callable[[str], bool]:
    def _accept(question: str) -> bool:
        live = state.router.live_intent(state.event, state.report, question)
        if require_comparison:
            bench = cases.clarification_for(state.event, state.report, question, [], router=state.router)[1]
            return any(intent is Intent.HYPOTHESIS_COMPARISON for intent in bench)
        return live is None

    return _accept


# ---------------------------------------------------------------------------
# turn construction
# ---------------------------------------------------------------------------
def _premise_for(kind: str, state: DialogueState, index: int) -> Tuple[str, Dict]:
    """Build a false premise that the store actually contradicts or misses."""
    store = state.store
    if kind == "sensor":
        # A variable that is neither ranked nor in the onset sequence: one that
        # appears there really did move, so asserting it did not would be a true
        # premise dressed up as a false one.
        present = set(store.known_tags())
        present |= {str(item.get("sensor")) for item in store.timeline if item.get("sensor")}
        candidates = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(42, 53)]
        unknown = [tag for tag in candidates if tag not in present]
        template = FALSE_PREMISE_TEMPLATES["sensor"][index % len(FALSE_PREMISE_TEMPLATES["sensor"])]
        for offset in range(len(unknown)):
            tag = unknown[(index + offset) % len(unknown)]
            premise = cases.premise_check(
                state.event, store, "sensor", claim=tag, target=state.target, report=state.report
            )
            if premise["verdict"] != "supported":
                return template.format(tag=tag), premise
        raise RuntimeError("no sensor outside this event to use as a false premise")
    if kind == "fault":
        truth = str(state.target.get("fault_id") or "")
        numbers = [n for n in state.permitted_faults if fault_id_for(n).value != truth]
        # An open-set item removed exactly one fault from the menu, and its target
        # is UNCLASSIFIED_OPEN_SET -- so ``truth`` above does not name it, and the
        # claim pool would happily quote the removed fault's KB name back to the
        # model ("is this condenser cooling water valve sticking?") in the very
        # prompt whose target is "none of the listed candidates". That is the leak
        # the open-set scan reports as an operator echo, so the removed fault is
        # excluded here, at the point the claim is chosen.
        if state.removed_fault is not None:
            numbers = [n for n in numbers if int(n) != int(state.removed_fault)]
        if not numbers:
            raise RuntimeError("no permitted alternative fault for a false premise")
        template = FALSE_PREMISE_TEMPLATES["fault"][index % len(FALSE_PREMISE_TEMPLATES["fault"])]
        claim = fault_id_for(numbers[index % len(numbers)]).value
        premise = cases.premise_check(
            state.event, store, "fault", claim=claim, target=state.target, report=state.report
        )
        return template.format(fault=cases.label_for_fault(claim)), premise
    if kind == "value":
        try:
            actual = float(store.resolve("detector.dynamic.score"))
        except UnknownKeyError:
            actual = 0.0
        claimed = round(actual * 0.61 + 7.0, 1)
        if abs(claimed - actual) < 1e-6:
            claimed = round(actual + 11.0, 1)
        template = FALSE_PREMISE_TEMPLATES["value"][index % len(FALSE_PREMISE_TEMPLATES["value"])]
        premise = cases.premise_check(
            state.event, store, "value", claim=str(claimed), target=state.target, report=state.report
        )
        return template.format(value=f"{claimed}"), premise
    raise ValueError(f"unknown premise kind {kind!r}")


def _key_premise(state: DialogueState, index: int) -> Tuple[str, Dict, str]:
    """An in-domain question about a field the store does not hold."""
    tags = state.store.known_tags() or ["XMEAS_1"]
    tag = tags[index % len(tags)]
    field_name = MISSING_FIELDS[index % len(MISSING_FIELDS)]
    template = FALSE_PREMISE_TEMPLATES["key"][index % len(FALSE_PREMISE_TEMPLATES["key"])]
    premise = cases.premise_check(
        state.event,
        state.store,
        "key",
        claim=f"{tag}:{field_name}",
        target=state.target,
        report=state.report,
    )
    return template.format(tag=tag), premise, "in_domain"


def build_turn(kind: str, state: DialogueState, index: int, *, repeat_index: Optional[int] = None) -> Dict:
    """Deterministic fields for one turn, plus the payload the answer is written from."""
    case = CASE_TYPES.get(kind)
    if case is None:  # "template" and "repeat" are turn kinds, not case types
        case = {"routing": "template" if kind == "template" else "llm", "intent": None}
    intent: Optional[str] = case.get("intent") if isinstance(case.get("intent"), str) else None
    answerability = str(case.get("answerability") or "llm")
    route_extra: Dict[str, object] = {}
    premise: Optional[Dict] = None
    refusal: Optional[Dict] = None
    clarification: Optional[Dict] = None
    candidates: List[str] = []
    skipped = 0
    resolved_intent: Optional[str] = None
    repeat_of: Optional[int] = None
    question = ""
    paraphrase = ""

    if kind == "template":
        labels = cases.template_labels("template_sample", state.key)
        label = labels[index % len(labels)] if labels else "overview"
        hit = None
        excluded: List[str] = []
        for offset in range(len(cases.LIVE_TEMPLATE_INTENTS)):
            candidate_label = cases.LIVE_TEMPLATE_INTENTS[
                (cases.LIVE_TEMPLATE_INTENTS.index(label) + offset) % len(cases.LIVE_TEMPLATE_INTENTS)
            ]
            try:
                candidate_question, position, skipped = pick_routed_question(
                    candidate_label, state, template_accept(candidate_label, state), salt=f"|t{index}"
                )
            except RuntimeError:
                excluded.append(f"{candidate_label}:no_paraphrase")
                continue
            candidate_hit = lookup_answer(state.event, state.report, candidate_question, router=state.router)
            if candidate_hit is None:
                excluded.append(f"{candidate_label}:no_lookup")
                continue
            # The template sample trains on lookup text, so that text has to obey
            # the same rules the teacher prose does (plus the KB's "CRITICAL"
            # opening, see TEMPLATE_BLOCKED); such a label is skipped, not taught.
            blob = f"{candidate_hit.answer.answer} {candidate_hit.detail}".lower()
            blocked = next((pattern for pattern in TEMPLATE_BLOCKED if re.search(pattern, blob)), None)
            if blocked is not None:
                excluded.append(f"{candidate_label}:" + ("severity_claim" if blocked in SEVERITY_PATTERNS or "critical" in blocked else "absolute_wording"))
                continue
            label, question, hit = candidate_label, candidate_question, candidate_hit
            break
        if hit is None:
            raise RuntimeError(f"no template label has a clean lookup answer ({excluded})")
        paraphrase = f"{label}#{position}"
        intent = hit.answer.intent.value
        resolved_intent = intent
        route_extra = {"live_intent": intent, "answerability": "deterministic"}
        turn = {
            "kind": kind,
            "case_type": "template",
            "question": question,
            "paraphrase": paraphrase,
            "paraphrases_skipped": skipped,
            "template_excluded": excluded,
            "intent": intent,
            "resolved_intent": intent,
            "answerability": "deterministic",
            "table": hit.answer.table.value,
            "table_params": {"tags": state.store.known_tags()[:5]} if hit.answer.table.value == "evidence" else {},
            "citations": list(hit.citations),
            "clarifying_question": None,
            "clarification_trigger": None,
            "suggestions": list(hit.answer.suggestions),
            "refusal": None,
            "premise": None,
            "answer_source": "lookup",
            "answer": hit.answer.answer,
            "detail": hit.detail,
            "repeat_of": None,
        }
        return turn

    elif kind == "repeat":
        if repeat_index is None:
            raise RuntimeError("a repeat turn needs the index of the turn it repeats")
        source = state.turn_records[repeat_index]
        question = source["question"]
        paraphrase = source["paraphrase"]
        intent = source["intent"]
        resolved_intent = source.get("resolved_intent")
        repeat_of = repeat_index
        route_extra = {"live_intent": state.router.live_intent(state.event, state.report, question)}
        answerability = "llm"
    elif kind in ("valve_actuator", "causal_why", "mechanism"):
        question, position, skipped = pick_routed_question(
            kind, state, llm_accept(state), salt=f"|t{index}"
        )
        paraphrase = f"{kind}#{position}"
        route_extra = {"live_intent": None, "answerability": "llm"}
        answerability = "llm"
    elif kind == "hypothesis_comparison":
        # Slot-filled from this item's own hypotheses, so an open-set item never
        # names the family that was removed from its menu.
        question = ""
        position = 0
        accept = llm_accept(state, require_comparison=True)
        for offset in range(len(COMPARISON_TEMPLATES)):
            candidate = cases.comparison_question(
                COMPARISON_TEMPLATES, state.target, index + offset, open_set=state.open_set
            )
            if accept(candidate):
                question, position = candidate, (index + offset) % len(COMPARISON_TEMPLATES)
                break
            skipped += 1
        if not question:
            raise RuntimeError("no comparison paraphrase routes to the comparison intent")
        paraphrase = f"hypothesis_comparison#{position}"
        intent = Intent.HYPOTHESIS_COMPARISON.value
        route_extra = {"live_intent": state.router.live_intent(state.event, state.report, question),
                       "answerability": "llm"}
        answerability = "llm"
    elif kind == "out_of_scope":
        if index % 3 == 2:
            question = OFF_DOMAIN_QUESTIONS[index % len(OFF_DOMAIN_QUESTIONS)]
            premise = None
            paraphrase = "off_domain"
            route_extra = {"live_intent": state.router.live_intent(state.event, state.report, question)}
        else:
            question, premise, flavour = _key_premise(state, index)
            paraphrase = f"key_missing:{flavour}"
            route_extra = {"live_intent": state.router.live_intent(state.event, state.report, question)}
        intent = Intent.OUT_OF_SCOPE_REFUSAL.value
        refusal = cases.refusal_for(
            state.event, state.store, state.target, state.report,
            question=question, premise=premise, router=state.router,
        ).as_dict()
        answerability = "refuse"
    elif kind == "false_premise":
        flavour = ("sensor", "fault", "value")[index % 3]
        question, premise = _premise_for(flavour, state, index)
        paraphrase = f"false_premise:{flavour}"
        intent = Intent.OUT_OF_SCOPE_REFUSAL.value
        refusal = cases.refusal_for(
            state.event, state.store, state.target, state.report,
            question=question, premise=premise, router=state.router,
        ).as_dict()
        route_extra = {"live_intent": state.router.live_intent(state.event, state.report, question)}
        answerability = "refuse"
    elif kind == "ambiguous_question":
        question = ""
        for offset in range(len(AMBIGUOUS_QUESTIONS)):
            candidate = AMBIGUOUS_QUESTIONS[(index + offset) % len(AMBIGUOUS_QUESTIONS)]
            request, cand = cases.clarification_for(
                state.event, state.report, candidate, state.asked, router=state.router
            )
            if request is not None:
                question, clarification, candidates = candidate, request, [c.value for c in cand]
                break
        if not question:
            raise RuntimeError("no ambiguous question triggered a clarification for this event")
        paraphrase = "ambiguous_question"
        intent = candidates[0] if candidates else None
        answerability = "clarify"
        route_extra = {"live_intent": state.router.live_intent(state.event, state.report, question)}
    elif kind == "clarification_reply":
        pending = next(
            (turn for turn in reversed(state.turn_records) if turn.get("clarifying_question")), None
        )
        if pending is None:
            raise RuntimeError("a clarification reply needs a pending clarifying question")
        options = (pending.get("clarification_options") or [])
        option = options[index % len(options)] if options else "raw evidence"
        question = f"I mean the {option} side, please." if option not in ("closest match", "raw evidence") else f"{option.capitalize()}, please."
        options_check = [option]
        resolved_intent = cases.resolved_intent_for_option(str(option))
        intent = Intent.CLARIFICATION_REPLY.value
        paraphrase = f"clarification_reply:{option}"
        answerability = "llm"
        route_extra = {"live_intent": None, "reply_option": option, "reply_options": options_check}
        if resolved_intent is None:
            resolved_intent = Intent.OVERVIEW.value
    else:
        raise ValueError(f"unknown turn kind {kind!r}")

    if repeat_of is not None:
        source_turn = state.turn_records[repeat_of]
        intent = source_turn["intent"]
        resolved_intent = source_turn.get("resolved_intent")

    # -- clarification decision ------------------------------------------
    # The router's own triggers decide; the builder only records the decision.
    # A repeat and a reply to a clarification must never re-ask, so a trigger
    # that fires there is recorded as suppressed rather than attached.
    suppressed: Optional[str] = None
    if kind in ("repeat", "clarification_reply"):
        request, _cands = cases.clarification_for(
            state.event, state.report, question, state.asked, router=state.router
        )
        if request is not None:
            suppressed = str(request.get("trigger"))
    elif clarification is None:
        request, cands = cases.clarification_for(
            state.event, state.report, question, state.asked, router=state.router
        )
        if request is not None:
            clarification = request
            candidates = [c.value for c in cands]
            if intent is None and candidates:
                intent = candidates[0]
            if answerability != "refuse":
                answerability = "clarify"

    # -- deterministic table / citations / suggestions -------------------
    table_intent = resolved_intent or intent
    table, table_params = cases.table_for(table_intent, state.target, state.store)
    if answerability == "refuse":
        table, table_params = "none", {}
    if clarification is not None:
        table, table_params = "none", {}
    if intent == "clarification_reply":
        # The schema allows only `none` for a clarification reply: the resolved
        # view is prose, and its numbers were already shown in the earlier turn.
        table, table_params = "none", {}
    extra_faults: List[str] = []
    truth = str(state.target.get("fault_id") or "")
    if truth.startswith(("FAULT_", "SYNTHETIC_")):
        extra_faults.append(truth)
    for hypothesis in state.target.get("ranked_hypotheses") or []:
        value = str(hypothesis.get("fault_id") or "")
        if value.startswith(("FAULT_", "SYNTHETIC_")):
            extra_faults.append(value)
    citations = cases.citations_for(
        table_intent, state.event, state.report, state.store, state.target, extra_faults=extra_faults
    )
    suggestions = cases.suggestions_for_turn(
        table_intent, state.event, state.report, question, router=state.router
    )
    if refusal is not None:
        # A refusal's citations are the keys of what it offers instead; the key
        # it denies does not resolve, which is exactly why it can be denied.
        alternatives = refusal.get("alternatives") or []
        suggestions = [a["question"] for a in alternatives][:3]
        if alternatives:
            citations = cases.citations_for(
                alternatives[0].get("intent"), state.event, state.report, state.store, state.target
            )

    turn = {
        "kind": kind,
        "case_type": kind,
        "question": question,
        "paraphrase": paraphrase,
        "paraphrases_skipped": skipped,
        "intent": intent,
        "resolved_intent": resolved_intent,
        "answerability": answerability,
        "table": table,
        "table_params": table_params,
        "citations": citations,
        "clarifying_question": (clarification or {}).get("question"),
        "clarification_trigger": (clarification or {}).get("trigger"),
        "clarification_options": (clarification or {}).get("options") or [],
        "clarification_suppressed": suppressed,
        "clarification_candidates": candidates,
        "suggestions": suggestions,
        "refusal": refusal,
        "premise": premise,
        "answer_source": "teacher",
        "answer": None,
        "detail": None,
        "repeat_of": repeat_of,
        "route": route_extra,
    }
    if clarification is not None:
        state.asked.append(str(clarification.get("question") or ""))
    return turn


# ---------------------------------------------------------------------------
# answer generation
# ---------------------------------------------------------------------------
def teacher_payload(state: DialogueState, turn: Dict, extra_prompt_note: str = "") -> Tuple[TeacherPayload, str]:
    numbers, values, labels = numeric_strings(state.store)
    if "first_sensor" not in labels and state.store.known_tags():
        tag = state.store.known_tags()[0]
        labels["first_sensor"] = state.store.display(tag) or tag
    reply_option = str((turn.get("route") or {}).get("reply_option") or "")
    if reply_option:
        labels["reply_option"] = reply_option.replace("_", " ")
    premise = turn.get("premise") or {}
    payload = TeacherPayload(
        context=cases.render_context(state.store, state.target, open_set=state.open_set),
        question=turn["question"],
        intent=turn.get("intent"),
        table=turn.get("table") or "none",
        citations=list(turn.get("citations") or []),
        clarifying_question=turn.get("clarifying_question"),
        refusal=turn.get("refusal"),
        premise=premise,
        previous_turns=[dict(t) for t in state.history[-4:]],
        numbers=numbers,
        values=values,
        labels=labels,
        open_set=state.open_set,
        repeat=turn.get("repeat_of") is not None,
        turn_index=len(state.turn_records),
        removed_fault_label=(
            cases.label_for_fault(f"FAULT_{state.removed_fault:02d}") if state.removed_fault else None
        ),
    )
    prompt = build_prompt(payload)
    if extra_prompt_note:
        prompt += "\n\n" + extra_prompt_note
    return payload, prompt


def answer_context(state: DialogueState, turn: Dict, prompt: str) -> AnswerContext:
    evidence = evidence_of(state.record)
    premise = turn.get("premise") or {}
    extra_tags: List[str] = []
    if premise.get("kind") == "sensor":
        extra_tags.append(str(premise.get("claim") or ""))
    elif premise.get("kind") == "key":
        extra_tags.append(str(premise.get("claim") or "").split(":")[0])
    grounded = store_numbers(evidence, state.target) + state.store.numeric_values()
    grounded += [value for value, _decimals in numbers_in(prompt)]
    grounded += [value * 100.0 for value in list(grounded)] + [value / 100.0 for value in list(grounded)]
    truth = str(state.target.get("fault_id") or "")
    return AnswerContext(
        store=state.store,
        allowed=state.allowed,
        grounded_numbers=grounded,
        permitted_faults=state.permitted_faults,
        event_id=str(state.record.get("event_id") or ""),
        removed_fault=state.removed_fault,
        previous_answer=(state.history[-1]["assistant"] if state.history else None),
        clarifying_question=turn.get("clarifying_question"),
        refusal=turn.get("refusal"),
        intent=truth or None,
        question=str(turn.get("question") or ""),
        extra_grounded_tags=[tag for tag in extra_tags if tag],
        # Direction is only enforced against a *classified* item's KB line: an
        # open-set item has had its menu entry removed, so there is no line the
        # prose could contradict.
        kb_fault=(int(str(truth).split("_")[-1])
                  if state.removed_fault is None
                  and str(truth).startswith(("FAULT_", "SYNTHETIC_"))
                  else None),
    )


def fill_answer(state: DialogueState, turn: Dict, teacher, cache: Optional[TeacherCache]) -> Dict:
    """One turn's answer: cache, teacher, filters, one retry."""
    payload, prompt = teacher_payload(state, turn)
    text = cache.get(prompt) if cache is not None else None
    used_cache = text is not None
    seconds = 0.0
    model = getattr(teacher, "model", "unknown")
    if text is None:
        answer, response, error = teacher.answer(payload, prompt)
        model = response.model
        seconds = response.seconds
        text = response.text
        if error:
            text = ""
    codes: List[str] = []
    answer_text = ""
    attempts = 0
    while attempts < 2:
        attempts += 1
        from llm.followup.teacher import extract_answer

        answer_text, error = extract_answer(text or "")
        if error:
            codes = [f"invalid_json:{error}"[:120]]
        else:
            codes = check_answer(answer_text or "", answer_context(state, turn, prompt))
        if not codes:
            break
        if attempts == 1:
            note = (
                "PREVIOUS ATTEMPT REJECTED for: "
                + ", ".join(codes)
                + ". Write a different answer that fixes exactly that."
            )
            payload, retry_prompt = teacher_payload(state, turn, extra_prompt_note=note)
            cached = cache.get(retry_prompt) if cache is not None else None
            if cached is not None:
                text = cached
            else:
                _answer, response, error = teacher.answer(payload, retry_prompt)
                text = response.text
                seconds += response.seconds
                if cache is not None:
                    cache.put(retry_prompt, text, response.model, response.seconds)
            prompt = retry_prompt
    if cache is not None and not used_cache:
        cache.put(prompt, text, model, seconds)

    turn.update(
        {
            "answer": answer_text if not codes else None,
            "rejected_answer": None if not codes else (answer_text or "")[:600],
            "filter_codes": codes,
            "accepted": not codes,
            "answer_attempts": attempts,
            "teacher_model": model,
            "teacher_seconds": round(seconds, 3),
            "prompt_sha256": sha256_text(prompt),
            "prompt_cache_hit": used_cache,
        }
    )
    return turn


# ---------------------------------------------------------------------------
# dialogue assembly
# ---------------------------------------------------------------------------
def condition_case(case_type: str, *, open_set: bool, normal: bool, has_lookalike: bool,
                   unclassified: bool = False) -> Tuple[str, Optional[str]]:
    """Fall back to a case the event can carry, recording the substitution.

    ``unclassified`` is the *target* being ``UNCLASSIFIED_OPEN_SET``, which is
    true for faults 16-20 even in the ``menu`` variant -- they have no menu entry
    to remove, so the variant flag alone does not detect them. Such an item has
    the ``unclassified`` clarification trigger on for every question, so no
    paraphrase can route cleanly to a template and a template sample cannot be
    built. Drawing one produced a hard failure rather than a substitution.
    """
    if case_type == "template_sample" and (open_set or normal or unclassified):
        return "valve_actuator", case_type
    if case_type == "hypothesis_comparison" and not has_lookalike:
        return "valve_actuator", case_type
    if case_type == "false_premise" and normal:
        return "valve_actuator", case_type
    return case_type, None


def build_dialogue(record: Dict, split: str, variant: str, case_type: str, teacher,
                   cache: Optional[TeacherCache], *, mix: Dict[str, float], seed: int,
                   index: int) -> Dict:
    evidence = evidence_of(record)
    target = build_target(record, variant=variant, split=split)
    raw_event = record.get("event") if isinstance(record.get("event"), dict) else {}
    if "evidence" not in raw_event:
        raw_event = {"evidence": evidence, "event_id": record.get("event_id"),
                     "max_anomaly_score": record.get("max_anomaly_score"),
                     "n_windows": record.get("n_windows")}
    report = build_report(target, evidence)
    store = StoreAccess(raw_event, report)
    router = ResponseRouter(enable_proposed=True)
    open_set = variant == "removed"
    normal = str(target.get("fault_id")) == NORMAL_ID
    removed_fault = int(record.get("fault")) if open_set and record.get("fault") is not None else None
    permitted = _menu_faults(open_set, removed_fault)
    extra_faults = [n for n in (record.get("fault") if record.get("fault") is not None else 0,) if n] \
        + [int(str(h.get("fault_id")).split("_")[-1]) for h in (target.get("ranked_hypotheses") or [])
           if str(h.get("fault_id", "")).startswith(("FAULT_", "SYNTHETIC_"))]
    allowed = build_allowed_set(raw_event, extra_faults=extra_faults)
    has_lookalike = any(h.get("fault_id") for h in (target.get("ranked_hypotheses") or []) if not h.get("is_truth"))
    final_case, substituted = condition_case(
        case_type, open_set=open_set, normal=normal, has_lookalike=has_lookalike,
        unclassified=str(target.get("fault_id")) == UNCLASSIFIED,
    )
    key = f"{record_key_of(record)}|{final_case}"
    event_key = record_key_of(record)

    state = DialogueState(
        record=record,
        split=split,
        variant=variant,
        target=target,
        event=raw_event,
        report=report,
        store=store,
        router=router,
        allowed=allowed,
        permitted_faults=permitted,
        removed_fault=removed_fault,
        grounded_numbers=[],
        key=key,
        open_set=open_set,
        normal=normal,
        asked=[],
        history=[],
        turn_records=[],
        labels={},
    )
    context = cases.render_context(store, target, open_set=open_set)
    assert_clean(
        context, permitted_faults=permitted, event_id=str(record.get("event_id") or ""),
        allow_opaque=True, where=f"context[{record.get('event_id')}]",
    )

    plan = cases.plan_for(final_case, key)
    turns: List[Dict] = []
    template_labels = cases.template_labels(final_case, key)
    for position, kind in enumerate(plan):
        repeat_index = None
        if kind == "repeat":
            repeat_index = next(
                (i for i in range(len(turns) - 1, -1, -1) if turns[i]["answer_source"] == "teacher"),
                None,
            )
        turn = build_turn(kind, state, position, repeat_index=repeat_index)
        assert_clean(
            turn["question"], permitted_faults=permitted,
            event_id=str(record.get("event_id") or ""), where=f"question[{record.get('event_id')}:{position}]",
        )
        if turn["answer_source"] == "lookup":
            turn["accepted"] = True
            turn["filter_codes"] = []
        else:
            turn = fill_answer(state, turn, teacher, cache)
        turns.append(turn)
        state.turn_records = turns
        state.history.append({"user": turn["question"], "assistant": turn.get("answer") or ""})

    return {
        "dialogue_id": f"FD-{split}-{index:05d}",
        "event_key": event_key,
        "event_sha256": sha256_text(json.dumps(record, sort_keys=True, default=str)),
        "split": split,
        "variant": variant,
        "case_type": final_case,
        "requested_case_type": case_type,
        "case_substituted": substituted,
        "open_set": open_set,
        "normal": normal,
        "fault": record.get("fault"),
        "keys": {
            "source_file": record.get("source_file"),
            "simulationRun": record.get("simulationRun"),
            "fault": record.get("fault"),
        },
        "context": context,
        "template_labels": template_labels,
        "turns": turns,
        "mix": mix,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def variant_for(record: Dict, split: str, rate: int) -> str:
    fault = record.get("fault")
    if fault is None:
        return "menu"
    if split == "train" and not eligible_for_open_set(record, split):
        return "menu"
    if int(fault) not in CLASSIFIABLE_FAULTS:
        return "menu"
    digest = hashlib.sha256(f"followup|{record.get('event_id')}".encode("utf-8")).hexdigest()
    return "removed" if int(digest[:8], 16) % 100 < rate else "menu"


def _family(case_type: str) -> str:
    return case_type


def thin_by_fault_family(rows: Sequence[Dict], per_dialogue: int, seed: int) -> List[Dict]:
    """One dialogue per ``per_dialogue`` report items, seeded and stratified.

    The mix is a training decision: the report task is the headline metric, so
    follow-up dialogues are paired with report items rather than outnumbering
    them. Thinning per fault family (rather than globally) keeps every fault's
    share of the follow-up set equal to its share of the report set, which a
    global sample would only approximate -- the large families would dominate.

    Selection is by hash order, not by file order: the voices are as likely to
    come from run 401 as from run 1.
    """
    if per_dialogue <= 1:
        return list(rows)
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        fault = row.get("fault")
        buckets["NORMAL" if fault is None else f"F{int(fault):02d}"].append(row)
    kept: List[Dict] = []
    for family, items in buckets.items():
        ordered = sorted(
            items,
            key=lambda r: hashlib.sha256(
                f"{seed}|{family}|{r.get('event_id')}".encode("utf-8")
            ).hexdigest(),
        )
        kept.extend(ordered[::per_dialogue])
    kept.sort(key=lambda r: str(r.get("event_id")))
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default=str(EVENTS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="max dialogues per split")
    ap.add_argument("--report-items-per-dialogue", type=int, default=1,
                    help="pair one dialogue with every Nth report item (2 = the 2:1 mix)")
    ap.add_argument("--teacher", choices=("fake", "ollama", "none"), default="fake",
                    help="fake (default, no model), ollama (real run), none (deterministic fields only)")
    ap.add_argument("--fake-profile", default="good", help="fake teacher failure profile")
    ap.add_argument("--model", default="qwen3:30b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--mix", default=None, help="JSON file or inline JSON overriding DEFAULT_MIX")
    ap.add_argument("--open-set-rate", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default=None, help="teacher cache path (default <out>/teacher_cache.jsonl)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--audit", type=int, default=100)
    ap.add_argument("--progress-file", default=None,
                    help="append a one-line progress note every --progress-every teacher calls")
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)-8s %(message)s")

    events_dir = Path(args.events)
    if not events_dir.exists():
        raise SystemExit(
            f"events not found: {events_dir}\n"
            "The Phase 2B event set is the input to this builder. Regenerate it with\n"
            "  python scripts/build_benchmark_events.py\n"
            "and check that outputs/anomaly_detector/model.pt matches the architecture in\n"
            "outputs/anomaly_detector/config.json first (see docs/reports/phase_2c_report.md §2)."
        )

    mix = dict(DEFAULT_MIX)
    if args.mix:
        raw = Path(args.mix).read_text(encoding="utf-8") if Path(args.mix).exists() else args.mix
        override = json.loads(raw)
        mix.update({name: float(value) for name, value in override.items() if name in mix})
        total = sum(mix.values())
        if total <= 0:
            raise SystemExit("case mix sums to zero")
        mix = {name: value / total for name, value in mix.items()}

    records = load_records(events_dir)
    overlap = check_event_overlap({split: records.get(split, []) for split in args.splits}, SPLITS_FILE)
    disjointness = check_paraphrase_disjointness()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = None
    provenance = followup_provenance()
    if not args.no_cache:
        cache = TeacherCache(
            Path(args.cache) if args.cache else out / "teacher_cache.jsonl",
            model=args.model if args.teacher == "ollama" else "fake-followup-v1",
            seed=args.seed,
            provenance=provenance,
        )
    if args.teacher == "fake":
        teacher = FakeTeacher(profile=args.fake_profile)
    elif args.teacher == "ollama":
        from llm.followup.teacher import OllamaFollowupTeacher

        teacher = OllamaFollowupTeacher(model=args.model, host=args.host, seed=args.seed)
        logger.warning("REAL teacher requested: %s at %s (cost estimate in the manifest)", args.model, args.host)
    else:
        teacher = None

    dialogues: Dict[str, List[Dict]] = defaultdict(list)
    counts: Dict[str, Counter] = defaultdict(Counter)
    skipped: Counter = Counter()
    started = time.time()
    progress = Path(args.progress_file) if args.progress_file else None
    next_milestone = max(int(args.progress_every), 1)

    for split in args.splits:
        rows = sorted(records.get(split, []), key=lambda r: str(r.get("event_id")))
        available = len(rows)
        rows = thin_by_fault_family(rows, args.report_items_per_dialogue, args.seed)
        counts[split]["events_available"] = available
        built = 0
        for record in rows:
            if args.limit is not None and built >= args.limit:
                break
            if record.get("oracle") and split != "train":
                skipped["oracle_in_eval"] += 1
                continue
            fault = record.get("fault")
            if split == "train" and fault is not None and int(fault) in TRAIN_EXCLUDED_FAULTS:
                skipped["fault_3_9_excluded_from_train"] += 1
                continue
            case_type = cases.choose_case_type(mix, f"{record_key_of(record)}|{args.seed}")
            variant = variant_for(record, split, args.open_set_rate)
            dialogue = build_dialogue(
                record, split, variant, case_type, teacher, cache,
                mix=mix, seed=args.seed, index=built,
            )
            dialogues[split].append(dialogue)
            counts[split][dialogue["case_type"]] += 1
            counts[split]["open_set"] += int(dialogue["open_set"])
            counts[split]["normal"] += int(dialogue["normal"])
            counts[split]["dialogues"] += 1
            counts[split]["turns"] += len(dialogue["turns"])
            counts[split]["teacher_calls"] += sum(
                1 for turn in dialogue["turns"] if turn["answer_source"] == "teacher"
            )
            counts[split]["accepted"] += sum(1 for turn in dialogue["turns"] if turn.get("accepted"))
            counts[split]["rejected"] += sum(1 for turn in dialogue["turns"] if not turn.get("accepted"))
            if dialogue["case_substituted"]:
                counts[split]["case_substituted"] += 1
            built += 1
            if built % 100 == 0:
                logger.info("%s: %d dialogues, %.0fs", split, built, time.time() - started)
            # A multi-hour teacher run is watched from outside, so the call count
            # -- not the dialogue count -- is what the progress note reports.
            if progress is not None and counts[split]["teacher_calls"] >= next_milestone:
                stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
                note = (f"{stamp} followup {split}: {counts[split]['teacher_calls']} teacher calls, "
                        f"{counts[split]['dialogues']} dialogues, {counts[split]['accepted']} accepted, "
                        f"{counts[split]['rejected']} rejected, {time.time() - started:.0f}s elapsed\n")
                progress.parent.mkdir(parents=True, exist_ok=True)
                with open(progress, "a", encoding="utf-8") as handle:
                    handle.write(note)
                logger.info("progress: %s", note.strip())
                while next_milestone <= counts[split]["teacher_calls"]:
                    next_milestone += max(int(args.progress_every), 1)
        logger.info("%s: %d dialogues done (%.0fs)", split, built, time.time() - started)

    # -- write ------------------------------------------------------------
    hashes: Dict[str, str] = {}
    for split, rows in dialogues.items():
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[path.name] = digest

    teacher_turns = [turn for rows in dialogues.values() for row in rows for turn in row["turns"]
                     if turn["answer_source"] == "teacher"]
    accepted = sum(1 for turn in teacher_turns if turn.get("accepted"))
    rejection_codes = Counter(
        code.split(":")[0] for turn in teacher_turns for code in (turn.get("filter_codes") or [])
    )
    seconds = sum(float(turn.get("teacher_seconds") or 0.0) for turn in teacher_turns)
    prompts = {turn.get("prompt_sha256") for turn in teacher_turns if turn.get("prompt_sha256")}

    # -- audit sheet ------------------------------------------------------
    audit_path = out / "audit_sample.csv"
    write_audit(audit_path, dialogues, limit=args.audit, seed=args.seed)
    hashes[audit_path.name] = hashlib.sha256(audit_path.read_bytes()).hexdigest()

    # -- mix proposal -----------------------------------------------------
    train = dialogues.get("train", [])
    train_turns = sum(len(row["turns"]) for row in train)
    train_teacher = sum(1 for row in train for turn in row["turns"] if turn["answer_source"] == "teacher")
    train_template = train_turns - train_teacher
    routing = _load_measured_llm_share()
    items = phase2b_train_report_items()
    followup_ratio = {
        "phase_2b_train_report_items": items,
        "followup_dialogues_train": len(train),
        # The builder's unit is the event, and two report items can share one, so
        # the achieved ratio is reported next to the requested one rather than
        # assumed to equal it.
        "effective_report_items_per_dialogue": round(items / max(len(train), 1), 2),
        "followup_turns_train": train_turns,
        "llm_routed_turns_train": train_teacher,
        "template_turns_train": train_template,
        "llm_routed_share_of_turns": round(train_teacher / max(train_turns, 1), 4),
        "measured_llm_share_of_operator_questions": routing,
        "paired_report_items_per_dialogue": args.report_items_per_dialogue,
        "recommended_report_to_followup_dialogues": (
            f"{args.report_items_per_dialogue}:1" if args.report_items_per_dialogue > 1 else "1:1"
        ),
        "recommended_note": (
            f"Every dialogue is paired with {args.report_items_per_dialogue} report items, and "
            f"each dialogue carries {round(train_teacher / max(len(train), 1), 2)} model answers, so "
            "the report task stays the headline metric in every epoch rather than being "
            "outnumbered by dialogue turns. The template sample is 5% by weight and is "
            "scored on the lookup path, so it anchors format without teaching prose."
        ),
    }

    # -- cost -------------------------------------------------------------
    calls = len(prompts) if teacher_turns else 0
    retry_calls = int(round(calls * PILOT_REJECTION_RATE))
    cost = {
        "teacher_turns": len(teacher_turns),
        "distinct_prompts": calls,
        "seconds_per_call": SECONDS_PER_CALL,
        "serial_hours": round(calls * SECONDS_PER_CALL / 3600.0, 2),
        "with_one_retry_per_rejection": {
            "extra_calls": retry_calls,
            "serial_hours": round((calls + retry_calls) * SECONDS_PER_CALL / 3600.0, 2),
        },
        "measured_rejection_rate": PILOT_REJECTION_RATE,
        "measured_seconds_total": round(seconds, 2),
        "teacher": args.teacher,
    }
    projection = _project_cost(dialogues, mix, args.splits, args.report_items_per_dialogue)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "events_dir": str(events_dir),
        "teacher": args.teacher,
        "teacher_model": getattr(teacher, "model", None),
        "seed": args.seed,
        "mix": mix,
        "counts": {split: dict(counter) for split, counter in counts.items()},
        "cases": list(FOLLOWUP_CASE_ORDER),
        "skipped": dict(skipped),
        "acceptance": {
            "teacher_turns": len(teacher_turns),
            "accepted": accepted,
            "rejected": len(teacher_turns) - accepted,
            "rejection_rate": round((len(teacher_turns) - accepted) / max(len(teacher_turns), 1), 4),
            "rejection_codes": dict(rejection_codes),
            "mean_seconds_per_call": round(seconds / max(len(teacher_turns), 1), 3),
            "cache_hits": sum(1 for turn in teacher_turns if turn.get("prompt_cache_hit")),
        },
        "provenance": dict(provenance),
        "report_items_per_dialogue": args.report_items_per_dialogue,
        "split_integrity": overlap,
        "paraphrase_disjointness": disjointness,
        "leak_check": {"checked": "every context and question", "violations": 0},
        "files_sha256": hashes,
        "cost_estimate": cost,
        "full_run_projection": projection,
        "mix_proposal": followup_ratio,
        "train_faults_excluded": list(TRAIN_EXCLUDED_FAULTS),
        "temperature": 0.2,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("wrote %s", out / "manifest.json")
    logger.info("counts: %s", json.dumps({k: dict(v) for k, v in counts.items()}))
    return 0


def _load_measured_llm_share() -> Dict:
    """The routing table from inspection (a), read back for the mix proposal.

    Two numbers matter and they are not the same thing:

    * ``template_bank_model_share`` -- how often a question that *has* a
      template still escapes it (the dispatcher is keyword-based, so it does).
    * ``free_form_bank_model_share`` -- how often a question with no template
      reaches the model, which is the population this dataset trains.

    Neither is the operator traffic share: that needs production logs. The bank
    is balanced (50 per label), so a pooled average over it would only restate
    the bank's own composition.
    """
    path = ROOT / "outputs" / "evaluation" / "followup_routing.json"
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    families = {row["family"]: row for row in data.get("families", [])}
    if not families:
        return {"available": False, "reason": "routing artefact predates the family split"}
    return {
        "available": True,
        "per_family": {
            name: {"n": row["n"], "model": row["model"], "model_share": row["model_share"]}
            for name, row in families.items()
        },
        "template_bank_model_share": families.get("live_template", {}).get("model_share"),
        "free_form_bank_model_share": families.get("live_free_form", {}).get("model_share"),
        "note": "balanced 50-per-label bank; not a traffic share",
    }


def _project_cost(dialogues: Dict[str, List[Dict]], mix: Dict[str, float], splits: Sequence[str],
                  report_items_per_dialogue: int = 1) -> Dict:
    """Estimate the full-size run from the sample that was built.

    ``report_items_per_dialogue`` is the pairing actually used: two report items
    per dialogue means half as many dialogues (and half the teacher calls) as the
    1:1 framing, so a projection that ignored it would double the cost.
    """
    built = sum(len(rows) for rows in dialogues.values())
    turns = sum(len(row["turns"]) for rows in dialogues.values() for row in rows)
    teacher = sum(1 for rows in dialogues.values() for row in rows for turn in row["turns"]
                  if turn["answer_source"] == "teacher")
    if built == 0:
        return {}
    per_dialogue = teacher / built
    k = max(int(report_items_per_dialogue), 1)
    items = phase2b_train_report_items()
    train_dialogues = -(-items // k)
    calls = int(round(train_dialogues * per_dialogue))
    retry = int(round(calls * PILOT_REJECTION_RATE))
    return {
        "sample_dialogues": built,
        "sample_turns": turns,
        "sample_teacher_turns": teacher,
        "teacher_turns_per_dialogue": round(per_dialogue, 2),
        "phase_2b_train_report_items": items,
        "report_items_per_dialogue": k,
        "full_train_dialogues": train_dialogues,
        "full_train_calls": calls,
        "full_train_serial_hours": round(calls * SECONDS_PER_CALL / 3600.0, 2),
        "full_train_with_retry_hours": round((calls + retry) * SECONDS_PER_CALL / 3600.0, 2),
    }


def write_audit(path: Path, dialogues: Dict[str, List[Dict]], *, limit: int, seed: int) -> None:
    """Audit sheet design: stratified by case type, one row per turn.

    The scoring columns are left blank on purpose: this sheet is for a human
    reviewer, and a number filled in by the builder would be a claim about
    quality that nothing measured.
    """
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for rows in dialogues.values():
        for row in rows:
            for turn in row["turns"]:
                if not turn.get("answer"):
                    continue
                buckets[row["case_type"]].append(
                    {
                        "dialogue_id": row["dialogue_id"],
                        "case_type": row["case_type"],
                        "split": row["split"],
                        "turn": turn["kind"],
                        "question": turn["question"],
                        "answer": turn.get("answer") or turn.get("rejected_answer") or "",
                        "answer_source": turn["answer_source"],
                        "accepted": turn.get("accepted"),
                        "intent": turn.get("intent") or "",
                        # A free-form turn has no intent, so it cannot be scored on
                        # intent agreement; it is judged on grounding alone. Keeping
                        # the two populations in one sheet with this column explicit
                        # is what stops a pooled score from mixing them.
                        "scoring_path": ("intent_format" if turn.get("intent")
                                         else "free_form_grounding"),
                        "table": turn.get("table") or "",
                        "citations": ";".join(turn.get("citations") or []),
                        "clarifying_question": turn.get("clarifying_question") or "",
                        "refusal_reason": ((turn.get("refusal") or {}).get("reason") or ""),
                    }
                )
    for bucket in buckets.values():
        rng.shuffle(bucket)
    rows: List[Dict] = []
    order = sorted(buckets, key=lambda name: FOLLOWUP_CASE_ORDER.index(name) if name in FOLLOWUP_CASE_ORDER else 99)
    index = 0
    while len(rows) < limit:
        added = False
        for name in order:
            bucket = buckets[name]
            if index < len(bucket):
                rows.append(bucket[index])
                added = True
                if len(rows) >= limit:
                    break
        if not added:
            break
        index += 1
    columns = [
        "dialogue_id", "case_type", "split", "turn", "question", "answer", "answer_source",
        "accepted", "intent", "scoring_path", "table", "citations", "clarifying_question",
        "refusal_reason",
        "correct", "grounded", "refusal_appropriate", "not_repetitive", "useful", "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            row.update({name: "" for name in
                        ("correct", "grounded", "refusal_appropriate", "not_repetitive", "useful", "notes")})
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
