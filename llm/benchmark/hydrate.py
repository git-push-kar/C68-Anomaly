"""Turn a slim model answer into a full report, at render time.

The student emits five fields: ``fault_id``, ``verdict``, ``ranked_hypotheses``,
``physical_reasoning``, ``narrative``. Everything else a report shows -- the
subsystem, the mitigation, the table choice, the suggested follow-ups, the
clarifying question -- is **code**, derived here from the knowledge base and the
live router.

Why it moved: those fields were in the training target, which meant the model was
being taught to recite strings that the deployment path already knows. Every
recited field is a place the answer can disagree with the KB, costs answer tokens
on every example, and has to be validated on both sides. Deriving them removes
the disagreement entirely and shrinks what the model has to generate.

The equivalence this keeps: for a classified event the mitigation is the KB's
``recommended_action``, computed with the same expression ``targets.build_target``
uses, so a report rendered from a slim answer shows the same mitigation text the
full target carried. ``tests/test_slim_target.py`` asserts exactly that over the
whole training set rather than on a sample.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from llm.benchmark.menu import UNCLASSIFIED, build_menu
from llm.response.schemas import ReportBody, TableID
from scripts.fault_ids import FaultID, fault_id_for
from scripts.fault_knowledge import FAULT_KNOWLEDGE
from scripts.tags import canonical_tag, display_name

#: The fields the student generates, in order. Nothing else is model text.
SLIM_KEYS = ("fault_id", "verdict", "ranked_hypotheses",
             "physical_reasoning", "narrative")

#: A fault-free excursion. Defined here because both the target builder and the
#: render path need it, and two constants with the same value is one too many.
NORMAL_ID = "NORMAL_TRANSIENT"

#: Fixed wording for the two abstention verdicts. They are not fault-specific, so
#: they are constants rather than a question for the model.
OPEN_SET_MITIGATION = (
    "No listed family explains this signature. Keep the unit on manual watch, "
    "compare the deviating loop against the nearest candidates above, and confirm on site."
)
NORMAL_MITIGATION = (
    "No control action: the excursion closed without a confirmed process fault. "
    "Verify on site if the same loop repeats."
)
OPEN_SET_UNCERTAINTY = (
    "The evidence matches no menu entry; the ranking names the closest families "
    "only, and an on-site check is what settles it."
)
NORMAL_UNCERTAINTY = (
    "The excursion closed without a confirmed fault, so no cause is asserted."
)


def fault_number(fault_id: object) -> Optional[int]:
    text = str(fault_id or "")
    if text.startswith(("FAULT_", "SYNTHETIC_")):
        try:
            return int(text.split("_")[-1])
        except ValueError:
            return None
    return None


def _kb(number: Optional[int]) -> Dict:
    return (FAULT_KNOWLEDGE.get(int(number)) or {}) if number is not None else {}


def mitigation_for(fault_id: object) -> str:
    """The mitigation a report shows, derived from the label and the KB.

    Kept as one function so the benchmark target and the render path cannot drift:
    both call this, so "the same text as before" is true by construction rather
    than by a copied string.
    """
    text = str(fault_id or "")
    if text == NORMAL_ID:
        return NORMAL_MITIGATION
    if text == UNCLASSIFIED or not text.startswith(("FAULT_", "SYNTHETIC_")):
        return OPEN_SET_MITIGATION
    entry = _kb(fault_number(text))
    return str(entry.get("recommended_action") or entry.get("action") or "").strip()


def subsystem_for(fault_id: object) -> Optional[str]:
    text = str(fault_id or "")
    if text in (NORMAL_ID, UNCLASSIFIED):
        return None
    return _kb(fault_number(text)).get("subsystem")


def table_for(fault_id: object) -> TableID:
    """Which table a report attaches. Deterministic, one row per verdict."""
    if str(fault_id or "") == NORMAL_ID:
        return TableID.NONE
    return TableID.EVIDENCE


def clarifying_question_for(fault_id: object) -> Optional[str]:
    """The unclassified item's one clarifying question, from the menu, not the model."""
    if str(fault_id or "") != UNCLASSIFIED:
        return None
    candidates = [entry["fault_id"] for entry in build_menu() if entry["fault_id"] != UNCLASSIFIED]
    return f"Which of these is closest: {', '.join(candidates[:2])}?" if candidates else None


def evidence_bullets(event: Dict, limit: int = 5) -> List[str]:
    """One line per ranked sensor, from the event, so the bullets are store-derived."""
    bullets: List[str] = []
    for sensor in (event.get("top_anomalous_sensors") or [])[:limit]:
        if not isinstance(sensor, dict):
            continue
        tag = canonical_tag(sensor.get("name")) or canonical_tag(sensor.get("display_name"))
        name = display_name(tag) or str(sensor.get("display_name") or sensor.get("name") or tag)
        dev = sensor.get("deviation_percent")
        trend = sensor.get("trend")
        piece = f"{name} [{tag}]" if tag else name
        if dev is not None:
            try:
                piece += f": {float(dev):+.2f}% deviation"
            except (TypeError, ValueError):
                piece += f": {dev}"
        if trend:
            piece += f", trend {trend}"
        bullets.append(piece)
    return bullets


def hydrate(slim: Dict, event: Dict) -> ReportBody:
    """The full report body for one slim answer.

    Raises ``ValueError`` when a field the renderer cannot invent is missing: a
    silent default here would put a made-up label into a report.
    """
    if not isinstance(slim, dict):
        raise ValueError("slim answer is not an object")
    missing = [key for key in ("fault_id", "verdict", "narrative", "physical_reasoning")
               if not str(slim.get(key) or "").strip()]
    if missing:
        raise ValueError(f"slim answer missing {missing}")
    label = str(slim["fault_id"]).strip()
    try:
        root = FaultID(label)
    except ValueError as exc:  # an id outside the enum is a schema failure, not a label
        raise ValueError(f"fault_id {label!r} is not a FaultID") from exc

    ranked = [str(item).strip() if not isinstance(item, dict) else str(item.get("fault_id") or "").strip()
              for item in (slim.get("ranked_hypotheses") or [])]
    hypotheses = []
    for value in ranked[:3]:
        if not value or value == label:
            continue  # the root cause is not also a hypothesis about itself
        try:
            fault = FaultID(value)
        except ValueError:
            continue
        number = fault_number(value)
        entry = _kb(number)
        hypotheses.append({
            "fault": fault,
            "confidence": 0.0,   # no model confidence: see the confidence contract
            "rationale": str(entry.get("name") or value),
            "supporting_tags": [],
        })

    return ReportBody(
        summary=str(slim["narrative"]).strip(),
        root_cause=root,
        affected_subsystem=str(subsystem_for(label) or "unclassified"),
        evidence=evidence_bullets(event),
        reasoning=str(slim["physical_reasoning"]).strip(),
        reasoning_steps=str(slim["physical_reasoning"]).strip(),
        recommended_action=mitigation_for(label),
        uncertainty=(NORMAL_UNCERTAINTY if label == NORMAL_ID
                     else OPEN_SET_UNCERTAINTY if label == UNCLASSIFIED
                     else "The diagnosis rests on the detector evidence; the ranking names the "
                          "differential, and an on-site check is what confirms it."),
        hypotheses=hypotheses,
    )


__all__ = [
    "SLIM_KEYS",
    "NORMAL_ID",
    "OPEN_SET_MITIGATION",
    "NORMAL_MITIGATION",
    "clarifying_question_for",
    "evidence_bullets",
    "fault_number",
    "hydrate",
    "mitigation_for",
    "subsystem_for",
    "table_for",
]
