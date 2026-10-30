"""Deterministic answer targets: everything except the prose.

The teacher is asked for two strings (physical_reasoning, narrative). Every
structured field of the target is derived here from the simulator fault id and
the knowledge base, so:

* the label can never be re-decided by the teacher (a filter rejects a changed
  fault_id rather than accepting a plausible one);
* the ranking, mitigation, table choice and follow-ups are identical for every
  event of the same fault and cannot drift between runs;
* a KB edit propagates to the dataset instead of leaving stale text behind.

Open-set construction (task 5): a seeded 12% of TRAIN events get their true
fault's menu entry removed and a target of UNCLASSIFIED_OPEN_SET with the KB's
nearest candidates. The assignment is a hash of the event id, so it is stable
and identical for the same event across Protocol B folds. Evaluation items for
held-out faults are always produced in BOTH variants.

Confidence is deliberately absent. A number next to a hypothesis would have to
come from somewhere, and the honest source is the model's own token probability
at inference time, calibrated on validation. Fabricating one here would make the
field look measured when it is not; the proposal is in the phase 2B report.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

from llm.benchmark.menu import (
    CLASSIFIABLE_FAULTS,
    UNCLASSIFIED,
    build_menu,
    lookalikes,
    nearest_candidates,
)
from llm.response.lookup_answers import suggestions_for
from llm.response.router import LOOKUP_INTENTS, ResponseRouter
from llm.response.schemas import Intent, TableID
from scripts.fault_ids import SYNTHETIC_FAULTS, fault_id_for
from scripts.fault_knowledge import FAULT_KNOWLEDGE

#: Share of TRAIN events whose true menu entry is removed (the brief allows
#: 10-15%; 12% leaves ~260 episodes in a 2,168-event train split).
OPEN_SET_RATE_PERCENT = 12

NORMAL_ID = "NORMAL_TRANSIENT"
#: Deterministic question used to interrogate the router's clarification
#: triggers. It is a real operator question, not a synthetic probe.
TRIGGER_QUESTION = "What is the likely root cause of this event?"


def open_set_assignment(event_id: str, rate: int = OPEN_SET_RATE_PERCENT) -> bool:
    """Stable per-event decision, so a fold cannot reshuffle the open-set set."""
    digest = hashlib.sha256(str(event_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100 < rate


def _kb(number: Optional[int]) -> Dict:
    if number is None:
        return {}
    return FAULT_KNOWLEDGE.get(int(number)) or {}


def _followups(intents: Sequence[Intent], router: ResponseRouter, limit: int = 3) -> List[str]:
    """Suggested follow-ups that the *live* router actually answers."""
    out: List[str] = []
    seen = set()
    for intent in intents:
        for question in suggestions_for(intent):
            key = " ".join(question.lower().split())
            if key in seen:
                continue
            candidates, _ = router.candidates(question)
            if not any(c in LOOKUP_INTENTS for c in candidates):
                continue
            out.append(question)
            seen.add(key)
            if len(out) >= limit:
                return out
    return out


def _clarifying_question(target_id: str, candidates: Sequence[str]) -> Optional[str]:
    """Route the canonical question and keep the router's own clarification."""
    if target_id != UNCLASSIFIED:
        return None
    router = ResponseRouter()
    event = {"evidence": {"detector_evidence": {"triggered_by": "both"}}}
    report = {"root_cause": UNCLASSIFIED}
    result = router.route(event, report, TRIGGER_QUESTION)
    if result.clarification is not None and result.clarification.trigger == "unclassified":
        return result.clarification.question
    return f"Which of these is closest: {', '.join(candidates)}?" if candidates else None


def build_target(
    record: Dict,
    *,
    variant: str = "menu",
    split: Optional[str] = None,
) -> Dict:
    """Build the deterministic target for one event record.

    ``variant``:
      ``menu``     the true fault is in the menu (normal item)
      ``removed``  the true fault's entry has been removed (open-set item)
    """
    fault = record.get("fault")
    is_normal = fault is None or record.get("label") == NORMAL_ID
    event_id = str(record.get("event_id") or "")
    router = ResponseRouter()

    if is_normal:
        target = {
            "event_id": event_id,
            "label_source": "fault_free_run",
            "fault_id": NORMAL_ID,
            "verdict": "no_fault",
            "abstain": True,
            "primary_subsystem": None,
            "ranked_hypotheses": [],
            "mitigation": "No control action: the excursion closed without a confirmed process fault. "
                          "Verify on site if the same loop repeats.",
            "table_selection": TableID.NONE.value,
            "suggested_followups": _followups(
                (Intent.EVIDENCE_WHY, Intent.DETECTOR_SCORES, Intent.ACTION), router
            ),
            "clarifying_question": None,
            "truth_fault": None,
            "truth_fault_number": None,
            "truth_subsystem": None,
            "menu_removed": [],
            "confidence": None,
            "confidence_note": "not emitted; see phase 2B report for the proposal",
        }
        return target

    number = int(fault)
    truth_id = fault_id_for(number).value
    entry = _kb(number)
    subsystem = entry.get("subsystem")
    mitigation = str(entry.get("recommended_action") or entry.get("action") or "").strip()

    if variant == "removed" or number not in CLASSIFIABLE_FAULTS:
        # Only a real menu entry can be removed: for faults 16-20 the true fault
        # was never on the menu, so an open-set item removes nothing. Those faults
        # are unanswerable by construction, so their target is UNCLASSIFIED in
        # both variants -- a ``menu`` variant that expected FAULT_17 would demand
        # an answer the menu does not offer.
        removed = [number] if number in CLASSIFIABLE_FAULTS else []
        target_id = UNCLASSIFIED
        verdict = "unclassified_open_set"
        ranked = [
            {"rank": i + 1, "fault_id": candidate, "basis": "kb_nearest_candidate", "is_truth": False}
            for i, candidate in enumerate(nearest_candidates(number))
        ]
        mitigation_text = (
            "No listed family explains this signature. Keep the unit on manual watch, "
            "compare the deviating loop against the nearest candidates above, and confirm on site."
        )
        table = TableID.EVIDENCE.value
        abstain = True
    else:
        removed = []
        target_id = truth_id
        verdict = "classified"
        ranked = [{"rank": 1, "fault_id": truth_id, "basis": "kb_truth", "is_truth": True}]
        for i, lookalike in enumerate(lookalikes(number)):
            ranked.append({
                "rank": 2 + i,
                "fault_id": fault_id_for(lookalike).value,
                "basis": "kb_differential",
                "is_truth": False,
            })
        mitigation_text = mitigation
        table = TableID.EVIDENCE.value
        abstain = False

    # Synthetic faults never appear in a headline item; flag rather than drop so
    # an accidental inclusion is visible downstream.
    synthetic = number in SYNTHETIC_FAULTS
    candidates = [r["fault_id"] for r in ranked if not r["is_truth"]]

    return {
        "event_id": event_id,
        "label_source": "simulator_fault_id",
        "fault_id": target_id,
        "verdict": verdict,
        "abstain": abstain,
        "primary_subsystem": subsystem if not abstain else None,
        "ranked_hypotheses": ranked,
        "mitigation": mitigation_text,
        "table_selection": table,
        "suggested_followups": _followups(
            (Intent.EVIDENCE_WHY, Intent.DETECTOR_SCORES, Intent.ACTION), router
        ),
        "clarifying_question": _clarifying_question(target_id, candidates),
        # ``truth_fault`` is the expected *answer* (so a fault-17 item expects
        # UNCLASSIFIED_OPEN_SET); the simulator's number is kept separately,
        # because otherwise the generating fault is unrecoverable in an audit.
        "truth_fault": target_id,
        "truth_fault_number": number,
        "truth_subsystem": subsystem,
        "menu_removed": [fault_id_for(n).value for n in removed],
        "synthetic": synthetic,
        "split": split or record.get("split"),
        "confidence": None,
        "confidence_note": "not emitted; see phase 2B report for the proposal",
    }


def menu_for_record(record: Dict, *, variant: str = "menu") -> List[Dict[str, str]]:
    """The menu an event is shown, honouring the open-set variant."""
    fault = record.get("fault")
    if variant == "removed" and fault is not None:
        return build_menu(remove=[int(fault)])
    return build_menu()


def eligible_for_open_set(record: Dict, split: str) -> bool:
    """Open-set training episodes come from the train split only, faults 1-15."""
    fault = record.get("fault")
    if fault is None or split != "train":
        return False
    return int(fault) in CLASSIFIABLE_FAULTS


__all__ = [
    "NORMAL_ID",
    "OPEN_SET_RATE_PERCENT",
    "TRIGGER_QUESTION",
    "build_target",
    "eligible_for_open_set",
    "menu_for_record",
    "open_set_assignment",
]
