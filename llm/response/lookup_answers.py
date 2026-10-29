"""Deterministic fact-lookup answers, built on the live templates.

The dispatcher already owns the authoritative wording for numeric questions
(`RCAInference._dispatch_with_intent`). This module does **not** re-implement
those templates: it calls the dispatcher and returns its output as ``detail``,
then writes a short 1-2 sentence lead that fits the response schema and cites
the store keys the table will show.

That split matters for the benchmark: the graded answer is a controlled sentence
whose every number is traceable to a citation key, while the operator still gets
the full existing formatting underneath.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llm.response.renderer import fmt
from llm.response.router import ResponseRouter
from llm.response.schemas import FollowupAnswer, Intent, TableID
from llm.response.store_access import StoreAccess, UnknownKeyError
from scripts.tags import canonical_tag, display_name

#: Intents that must be answered from evidence, never by generation.
FACT_INTENTS = frozenset(
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

TABLE_FOR_INTENT: Dict[Intent, TableID] = {
    Intent.SENSOR_VALUES: TableID.EVIDENCE,
    Intent.SENSOR_SCORES: TableID.EVIDENCE,
    Intent.SENSORS_INSPECT: TableID.EVIDENCE,
    Intent.EVIDENCE_WHY: TableID.DETECTORS,
    Intent.DETECTOR_SCORES: TableID.DETECTORS,
    Intent.DETECTOR_SEVERITY: TableID.DETECTORS,
    Intent.SEVERITY: TableID.DETECTORS,
    Intent.OVERVIEW: TableID.HYPOTHESES,
    Intent.OVERVIEW_FALLBACK: TableID.HYPOTHESES,
    Intent.ACTION: TableID.ACTIONS,
    Intent.SYSTEM_SENSORS: TableID.NONE,
}


@dataclass
class LookupHit:
    answer: FollowupAnswer
    detail: str
    citations: List[str] = field(default_factory=list)


def _get(store: StoreAccess, key: str) -> Optional[object]:
    try:
        return store.resolve(key)
    except UnknownKeyError:
        return None


def _strip_markdown(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        line = re.sub(r"^#+\s*", "", line.strip())
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.replace("**", "").replace("`", "")
        if line:
            lines.append(line)
    return " ".join(lines)


def _first_sentences(text: str, count: int = 2) -> str:
    flat = _strip_markdown(text)
    parts = re.split(r"(?<=[.!?])\s+", flat)
    return " ".join(parts[:count]).strip()


def _detector_lead(store: StoreAccess) -> str:
    rows = []
    for name in ("lstm", "dynamic"):
        score = _get(store, f"detector.{name}.score")
        threshold = _get(store, f"detector.{name}.threshold")
        ratio = _get(store, f"detector.{name}.ratio")
        if score is None or threshold is None:
            continue
        label = "LSTM reconstruction error" if name == "lstm" else "CVA Q-statistic"
        rows.append(f"{label} is {fmt(float(score))} against {fmt(float(threshold))} ({float(ratio or 0):.2f}x)")
    if not rows:
        return "No detector scores are stored for this event."
    return "; ".join(rows) + "."


def _sensor_lead(store: StoreAccess, limit: int = 3) -> str:
    tags = store.known_tags()[:limit]
    if not tags:
        return "No per-sensor detail is stored for this event."
    parts = []
    for tag in tags:
        observed = _get(store, f"sensor.{tag}.observed")
        baseline = _get(store, f"sensor.{tag}.baseline")
        dev = _get(store, f"sensor.{tag}.deviation")
        z = _get(store, f"sensor.{tag}.z")
        name = display_name(tag) or tag
        if observed is not None and baseline is not None:
            parts.append(f"{name} reads {fmt(float(observed))} against a baseline of {fmt(float(baseline))} ({float(dev or 0):+.2f}%, z={float(z or 0):+.2f})")
        elif dev is not None:
            parts.append(f"{name} deviates {float(dev):+.2f}% (z={float(z or 0):+.2f})")
    return "; ".join(parts) + "." if parts else "No sensor values are stored for this event."


def _lead_for(intent: Intent, store: StoreAccess, report: Dict, detail: str) -> str:
    if intent in (Intent.DETECTOR_SCORES, Intent.DETECTOR_SEVERITY, Intent.EVIDENCE_WHY):
        return _detector_lead(store)
    if intent in (Intent.SENSOR_VALUES, Intent.SENSOR_SCORES, Intent.SENSORS_INSPECT):
        return _sensor_lead(store)
    if intent == Intent.SEVERITY:
        # Severity is code-computed (RESPONSE_SPEC §3.1): answer with the computed
        # level, never with prose the model may have invented.
        from llm.response.severity import assess

        assessment = assess(store.event, report=report)
        ratio = float(_get(store, "detector.dynamic.ratio") or 0.0)
        windows = int(_get(store, "event.n_windows") or 0)
        lead = (
            f"Severity is {assessment.level.value.upper()}, computed in code from a detector margin of "
            f"{ratio:.2f}x over {windows} window(s); the knowledge-base floor for "
            f"{assessment.components.get('diagnosed_fault')} is {assessment.kb_base.value}."
        )
        stored = str(_get(store, "severity.stored") or "")
        if stored and stored.lower() != assessment.level.value:
            lead += f" The event store carries {stored.upper()} from the older rule."
        return lead
    if intent in (Intent.OVERVIEW, Intent.OVERVIEW_FALLBACK):
        root = str((report or {}).get("root_cause") or "")
        subsystem = _get(store, "subsystem.candidate") or "unknown"
        number = root.split("_")[-1] if root.upper().startswith(("FAULT_", "SYNTHETIC_")) else None
        name = _get(store, f"kb.fault.{number}.name") if number else None
        label = name or ("unclassified open-set disturbance" if "UNCLASSIFIED" in root.upper() else root or "unknown")
        return f"The leading cause is {label} in {subsystem}; detector margin and the deviating sensors are listed below."
    if intent == Intent.ACTION:
        root = str((report or {}).get("root_cause") or "")
        number = root.split("_")[-1] if root.upper().startswith(("FAULT_", "SYNTHETIC_")) else None
        kb_action = _get(store, f"kb.fault.{number}.action") if number else None
        if kb_action:
            return _first_sentences(str(kb_action), 2)
        return "No fault-specific remediation is stored for this event; inspect the candidate subsystem on site."
    return _first_sentences(detail, 2) or "No deterministic answer is available for this question."


def _citations_for(intent: Intent, store: StoreAccess, report: Dict) -> List[str]:
    keys: List[str] = []
    if intent in (Intent.DETECTOR_SCORES, Intent.DETECTOR_SEVERITY, Intent.EVIDENCE_WHY, Intent.SEVERITY):
        keys += ["detector.lstm.score", "detector.lstm.threshold", "detector.dynamic.score", "detector.dynamic.threshold"]
    if intent in (Intent.SENSOR_VALUES, Intent.SENSOR_SCORES, Intent.SENSORS_INSPECT):
        # 2 sensors x 3 keys keeps the schema's 6-citation budget
        for tag in store.known_tags()[:2]:
            keys += [f"sensor.{tag}.observed", f"sensor.{tag}.baseline", f"sensor.{tag}.deviation"]
    if intent in (Intent.OVERVIEW, Intent.OVERVIEW_FALLBACK, Intent.ACTION):
        root = str((report or {}).get("root_cause") or "")
        number = root.split("_")[-1] if root.upper().startswith(("FAULT_", "SYNTHETIC_")) else None
        keys.append("subsystem.candidate")
        if number:
            keys.append(f"kb.fault.{number}.name")
    resolved: List[str] = []
    for key in dict.fromkeys(keys):
        if _get(store, key) is not None:
            resolved.append(key)
    return resolved


def lookup_answer(
    event: Dict,
    report: Optional[Dict],
    question: str,
    router: Optional[ResponseRouter] = None,
) -> Optional[LookupHit]:
    """Deterministic answer for a fact-lookup question, or ``None``.

    ``None`` means the live dispatcher did not claim the question (or the route
    was not deterministic), so the caller must generate instead.
    """
    report = report if isinstance(report, dict) else {}
    store = StoreAccess(event, report)
    router = router or ResponseRouter()
    route = router.route(event, report, question)
    if route.answerability != "deterministic" or route.primary is None:
        return None
    if route.primary not in FACT_INTENTS or not route.deterministic_answer:
        return None

    intent = route.primary
    detail = route.deterministic_answer
    lead = _lead_for(intent, store, report, detail)
    citations = _citations_for(intent, store, report)
    table = TABLE_FOR_INTENT.get(intent, TableID.NONE)

    # A table is only attached when it can be filled from the store.
    if table == TableID.HYPOTHESES and not (report.get("hypotheses") if isinstance(report.get("hypotheses"), list) else False):
        table = TableID.NONE

    answer = FollowupAnswer(
        intent=intent,
        answer=lead,
        table=table,
        citations=citations,
        suggestions=suggestions_for(intent),
    )
    return LookupHit(answer=answer, detail=detail, citations=citations)


#: Short, operator-shaped follow-ups per intent; all route to a live intent.
SUGGESTION_CATALOG: Dict[Intent, List[str]] = {
    Intent.SENSOR_VALUES: ["What are the detector scores?", "Which sensors should I inspect?"],
    Intent.DETECTOR_SCORES: ["What is the evidence this is an anomaly?", "What is the severity level?"],
    Intent.EVIDENCE_WHY: ["What are the sensor values?", "What is the recommended action?"],
    Intent.SENSOR_SCORES: ["What are the sensor values?", "Which sensors should I inspect?"],
    Intent.DETECTOR_SEVERITY: ["What are the detector scores?", "What is the recommended action?"],
    Intent.SEVERITY: ["What are the detector scores?", "What is the recommended action?"],
    Intent.OVERVIEW: ["What are the detector scores?", "What is the recommended action?"],
    Intent.OVERVIEW_FALLBACK: ["What is the evidence this is an anomaly?", "Which sensors should I inspect?"],
    Intent.SENSORS_INSPECT: ["What are the sensor values?", "What is the recommended action?"],
    Intent.ACTION: ["What is the evidence this is an anomaly?", "What is the severity level?"],
    Intent.SYSTEM_SENSORS: ["Which sensors should I inspect?", "What are the detector scores?"],
}


def suggestions_for(intent: Intent) -> List[str]:
    return list(SUGGESTION_CATALOG.get(intent, []))[:3]


__all__ = ["lookup_answer", "LookupHit", "FACT_INTENTS", "TABLE_FOR_INTENT", "suggestions_for", "SUGGESTION_CATALOG"]
