"""Severity: one authoritative rule, one cross-check.

**The authoritative value is** ``evidence.process_limits.compute_deterministic_severity``,
the rule that already runs in the production pipeline and wrote the ``severity``
column of every stored event. :func:`assess` returns that level; it is the only
level the renderer, the API and the benchmark may quote.

The rule implemented in this module is kept as a **cross-check** and is exposed
as ``components['proposed_level']`` plus ``components['severity_agreement']``.
It is never displayed. Recording both is what lets us measure whether the
proposed rule would change a decision before it is allowed to change one.

### The proposed (non-authoritative) rule

Severity is still a host-side judgement, not a model skill (RESPONSE_SPEC
§3.1): the adapter must never emit it, and no benchmark row may score it as a
prediction. The cross-check derives it from three kinds of evidence plus the
knowledge base:

* **detector margin** — how far the LSTM and CVA scores sit above their own
  thresholds (``ratio`` = score / threshold);
* **persistence** — how many windows the event lasted and whether it held its
  peak (``mean_score / max_score``), so a single-window blip is not escalated;
* **sensor magnitude** — the largest |z| among the event's top sensors;
* **KB base level** — the knowledge base's own severity for the diagnosed
  fault family, which acts as a floor calibrated by fault physics.

The result is a weighted score mapped to a level, floored by the KB level
(one step down when the event is short or decaying) and escalated by one step
when both detectors are alarmed with sustained, large-magnitude evidence.

Thresholds below are **proposals**, chosen to be monotone and interpretable, and
have not been fitted to any data — in particular not to Phase 1 test numbers.
They are recorded in ``docs/reports/phase_2a_report.md`` for approval.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from evidence.process_limits import compute_deterministic_severity
from llm.response.schemas import SeverityAssessment, SeverityLevel
from scripts.fault_ids import FaultID, OPEN_SET_FAULTS, SYNTHETIC_FAULTS, fault_id_for, kb_entry

#: The rule whose level is quoted everywhere.
AUTHORITATIVE_RULE = "evidence.process_limits.compute_deterministic_severity"
#: The rule kept only as a cross-check (never rendered, never scored).
PROPOSED_RULE = "severity-rule-v1-proposed"
RULE_VERSION = "severity-crosscheck-v2"

#: Score -> level cut points (proposed).
LEVEL_CUTS: Tuple[Tuple[float, SeverityLevel], ...] = (
    (0.78, SeverityLevel.CRITICAL),
    (0.55, SeverityLevel.HIGH),
    (0.30, SeverityLevel.MEDIUM),
    (0.0, SeverityLevel.LOW),
)
#: Component weights (sum to 1).
WEIGHTS = {"kb_base": 0.35, "detector": 0.30, "sensor": 0.20, "persistence": 0.15}
#: KB level -> numeric base.
KB_BASE_SCORE = {
    SeverityLevel.LOW: 0.25,
    SeverityLevel.MEDIUM: 0.50,
    SeverityLevel.HIGH: 0.75,
    SeverityLevel.CRITICAL: 1.00,
}
_ORDER = [SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def severity_agreement(authoritative: SeverityLevel, proposed: SeverityLevel) -> Dict[str, object]:
    """Compare the authoritative level with the cross-check level.

    ``gap`` is the signed number of levels the cross-check sits above the
    authoritative value, so a systematic bias is visible as a sign, not just a
    rate.
    """
    return {
        "agree": authoritative == proposed,
        "authoritative": authoritative.value,
        "proposed": proposed.value,
        "gap": _ORDER.index(proposed) - _ORDER.index(authoritative),
    }


def authoritative_level(event: Dict) -> SeverityLevel:
    """The production rule's level for this event."""
    evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}
    raw = str(compute_deterministic_severity(evidence) or "medium").lower()
    try:
        return SeverityLevel(raw)
    except ValueError:
        return SeverityLevel.MEDIUM


def _level_for(score: float) -> SeverityLevel:
    for cut, level in LEVEL_CUTS:
        if score >= cut:
            return level
    return SeverityLevel.LOW


def _shift(level: SeverityLevel, steps: int) -> SeverityLevel:
    index = _ORDER.index(level) + steps
    return _ORDER[max(0, min(len(_ORDER) - 1, index))]


def _kb_base_level(fault: FaultID) -> SeverityLevel:
    """KB severity for the diagnosed fault, with explicit open-set/normal rules."""
    if fault == FaultID.UNCLASSIFIED_OPEN_SET:
        return SeverityLevel.MEDIUM
    if fault == FaultID.NORMAL_TRANSIENT:
        return SeverityLevel.LOW
    number = _fault_number(fault)
    if number is None:
        return SeverityLevel.MEDIUM
    entry = kb_entry(number) or {}
    raw = str(entry.get("severity", "medium")).lower()
    try:
        return SeverityLevel(raw)
    except ValueError:
        return SeverityLevel.MEDIUM


def _fault_number(fault: FaultID) -> Optional[int]:
    if not fault.value.startswith(("FAULT_", "SYNTHETIC_")):
        return None
    try:
        return int(fault.value.split("_")[-1])
    except ValueError:
        return None


def coerce_fault(value: Any) -> Optional[FaultID]:
    """Best-effort FaultID from an id string, a fault number or a KB-style text."""
    if isinstance(value, FaultID):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        try:
            return fault_id_for(value)
        except ValueError:
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    upper = text.upper()
    if upper in {member.value for member in FaultID}:
        return FaultID(upper)
    import re

    match = re.search(r"\b(\d{1,2})\b", text)
    if match:
        try:
            return fault_id_for(int(match.group(1)))
        except ValueError:
            pass
    lowered = text.lower()
    for number in list(range(1, 21)) + list(SYNTHETIC_FAULTS):
        entry = kb_entry(number) or {}
        name = str(entry.get("name") or "").lower()
        if name and (name in lowered or lowered in name):
            return fault_id_for(number)
    if "unclassified" in lowered or "unknown disturbance" in lowered:
        return FaultID.UNCLASSIFIED_OPEN_SET
    return None


def detector_components(event: Dict) -> Dict[str, float]:
    """Ratios and booleans for both detectors, tolerant of older event shapes."""
    evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}
    det = evidence.get("detector_evidence") if isinstance(evidence.get("detector_evidence"), dict) else {}
    lstm = det.get("lstm_ae") if isinstance(det.get("lstm_ae"), dict) else {}
    dyn = det.get("dynamic") if isinstance(det.get("dynamic"), dict) else {}

    def ratio(block: Dict) -> float:
        score = _as_float(block.get("score"))
        threshold = _as_float(block.get("threshold"))
        return score / threshold if threshold > 0 else 0.0

    l_ratio = ratio(lstm)
    d_ratio = ratio(dyn)
    return {
        "lstm_ratio": l_ratio,
        "dynamic_ratio": d_ratio,
        "max_ratio": max(l_ratio, d_ratio),
        "lstm_alarmed": bool(lstm.get("alarmed") or lstm.get("is_anomalous") or l_ratio >= 1.0),
        "dynamic_alarmed": bool(dyn.get("alarmed") or dyn.get("is_anomalous") or d_ratio >= 1.0),
    }


def assess(
    event: Dict,
    fault: Optional[Any] = None,
    report: Optional[Dict] = None,
) -> SeverityAssessment:
    """Compute the severity assessment for one event.

    ``fault`` wins over ``report.root_cause``, which wins over ``event.fault_label``.
    With no fault at all the event is treated as a normal transient, which caps
    the level at medium.
    """
    event = event if isinstance(event, dict) else {}
    evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}

    resolved = coerce_fault(fault)
    if resolved is None and isinstance(report, dict):
        resolved = coerce_fault(report.get("root_cause"))
    if resolved is None:
        resolved = coerce_fault(event.get("fault_label"))
    if resolved is None:
        resolved = FaultID.NORMAL_TRANSIENT

    comp = detector_components(event)
    max_ratio = comp["max_ratio"]
    detector_score = _clamp((max_ratio - 1.0) / 1.5)

    sensors = evidence.get("top_anomalous_sensors") or []
    max_z = 0.0
    for sensor in sensors if isinstance(sensors, list) else []:
        if isinstance(sensor, dict):
            max_z = max(max_z, abs(_as_float(sensor.get("z_score", sensor.get("zScore")))))
    sensor_score = _clamp((max_z - 2.0) / 4.0)

    n_windows = int(_as_float(event.get("n_windows"), 0.0))
    max_anom = _as_float(event.get("max_anomaly_score", evidence.get("anomaly_score")))
    mean_anom = _as_float(event.get("mean_anomaly_score"), max_anom)
    sustain = (mean_anom / max_anom) if max_anom > 0 else 0.0
    sustain_score = 1.0 if sustain >= 0.90 else (0.6 if sustain >= 0.75 else 0.3)
    persistence_score = _clamp(0.6 * _clamp(n_windows / 20.0) + 0.4 * sustain_score)

    kb_base = _kb_base_level(resolved)
    kb_score = KB_BASE_SCORE[kb_base]

    score = (
        WEIGHTS["kb_base"] * kb_score
        + WEIGHTS["detector"] * detector_score
        + WEIGHTS["sensor"] * sensor_score
        + WEIGHTS["persistence"] * persistence_score
    )
    level = _level_for(score)

    # KB floor: the fault family's own severity, stepped down one level for a
    # short or decaying event so a transient does not inherit a critical label.
    short_event = n_windows < 5 or sustain < 0.70
    floor = _shift(kb_base, -1) if short_event else kb_base
    level = max(level, floor, key=_ORDER.index)

    # Escalation: both detectors alarmed, sustained, and large sensor excursion.
    escalated = False
    if comp["lstm_alarmed"] and comp["dynamic_alarmed"] and persistence_score >= 0.6 and sensor_score >= 0.5:
        level = _shift(level, 1)
        escalated = True

    # A normal transient is never critical: nothing fault-justified was diagnosed.
    if resolved == FaultID.NORMAL_TRANSIENT:
        level = min(level, SeverityLevel.MEDIUM, key=_ORDER.index)

    # The cross-check stops here; the quoted level is the production rule's.
    proposed = level
    authoritative = authoritative_level(event)
    agreement = severity_agreement(authoritative, proposed)

    # Plain words here: a dotted module path in operator-facing prose reads as
    # noise (and trips the tag check if an answer ever reuses this sentence).
    rationale = (
        f"{authoritative.value.capitalize()} severity from the authoritative production rule; "
        f"detector margin {max_ratio:.2f}x, "
        f"{n_windows} window(s) at {sustain * 100:.0f}% of peak, max |z| {max_z:.1f}; "
        f"KB base for {resolved.value} is {kb_base.value}."
    )

    return SeverityAssessment(
        level=authoritative,
        score=round(float(min(score, 1.0)), 4),
        kb_base=kb_base,
        rationale=rationale,
        components={
            "authoritative_level": authoritative.value,
            "proposed_level": proposed.value,
            "severity_agreement": bool(agreement["agree"]),
            "severity_gap": int(agreement["gap"]),
            "authoritative_rule": AUTHORITATIVE_RULE,
            "proposed_rule": PROPOSED_RULE,
            "kb_base_level": kb_base.value,
            "kb_base_score": round(kb_score, 4),
            "detector_score": round(detector_score, 4),
            "sensor_score": round(sensor_score, 4),
            "persistence_score": round(persistence_score, 4),
            "max_ratio": round(max_ratio, 4),
            "lstm_ratio": round(comp["lstm_ratio"], 4),
            "dynamic_ratio": round(comp["dynamic_ratio"], 4),
            "max_abs_z": round(max_z, 4),
            "n_windows": n_windows,
            "sustain": round(sustain, 4),
            "floor": floor.value,
            "escalated": escalated,
            "diagnosed_fault": resolved.value,
        },
        rule_version=RULE_VERSION,
    )


def table_rows(assessment: SeverityAssessment) -> Dict[str, str]:
    """Level, score and the two strongest inputs — for a renderer cell set."""
    comp = assessment.components
    return {
        "level": assessment.level.value,
        "score": f"{assessment.score:.2f}",
        "detector_margin": f"{comp.get('max_ratio', 0):.2f}x",
        "persistence": f"{comp.get('n_windows', 0)} windows, sustain {comp.get('sustain', 0):.2f}",
        "kb_base": assessment.kb_base.value,
    }


def cap_for_open_set(assessment: SeverityAssessment) -> SeverityAssessment:
    """Proposed cap: an open-set disturbance carries no KB physics, so it cannot
    exceed high.

    Applied to the ``proposed_level`` component only. The level itself stays the
    authoritative rule's: capping it here would quietly replace the production
    decision, which is exactly what this module is no longer allowed to do.
    """
    comp = dict(assessment.components)
    if str(comp.get("diagnosed_fault", "")).startswith("UNCLASSIFIED"):
        proposed = SeverityLevel(str(comp.get("proposed_level", "high")))
        if _ORDER.index(proposed) > _ORDER.index(SeverityLevel.HIGH):
            comp["proposed_level"] = SeverityLevel.HIGH.value
            comp["proposed_capped"] = True
    return assessment.model_copy(update={"components": comp})


__all__ = [
    "assess",
    "authoritative_level",
    "severity_agreement",
    "coerce_fault",
    "detector_components",
    "table_rows",
    "cap_for_open_set",
    "AUTHORITATIVE_RULE",
    "PROPOSED_RULE",
    "RULE_VERSION",
    "LEVEL_CUTS",
    "WEIGHTS",
    "OPEN_SET_FAULTS",
]
