"""Prompt v2: evidence + KB candidate menu, and nothing that gives the answer away.

Removed relative to the v0 prompt (`llm/dataset.py:30-49` and
`format_evidence_question`):

* the **severity rubric** that enumerated fault numbers and mapped them to
  levels, which let the model read severity off the fault identity instead of
  the evidence;
* the **"Candidate affected subsystem"** line, which handed over a strong hint
  toward the answer before any reasoning happened;
* the **"Assessed severity level"** header line, for the same reason: severity
  is computed in code when a report is rendered, never predicted here.

Added: an explicit candidate menu. Every classifiable fault gets one line, plus
UNCLASSIFIED_OPEN_SET, so "which fault is this" is a choice among stated
options rather than an open-ended generation -- and an open-set episode is
constructed by removing exactly one line.

`assert_prompt_clean` is the script check: it runs over every generated prompt,
not just a sample, and fails the build if a forbidden fragment appears.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Optional, Sequence

from llm.benchmark.menu import UNCLASSIFIED
from scripts.fault_knowledge import FAULT_KNOWLEDGE

#: Case-insensitive fragments that must never appear in a v2 prompt.
#: "severity" is on the list because severity is a host-side computation: if the
#: word reaches the model, so does the temptation to answer with it.
FORBIDDEN_PROMPT_FRAGMENTS = (
    "candidate affected subsystem",
    "candidate_subsystem",
    "severity",
    "severity rubric",
    "fault 15-type",
    "for fault 15",
)

#: Tokens that identify an event, its split, or its run. Every one of them is a
#: label the model could memorise instead of reading the evidence.
LEAK_PATTERNS = (
    r"\bevt[-_ ]?\d",          # EVT-train-f04-r123
    r"\bevt\b",
    r"\bnorm(?:al)?_transient\b",
    r"\bnorm\b",
    r"\bsimulationrun\b",
    r"\br\d{3}\b",              # r123
    r"\btrain\b",
    r"\bvalidation\b",
    r"\bval\b",
    r"\btest\b",
)
#: A fault reference in the evidence section (the menu is allowed to name faults).
FAULT_REF_PATTERN = r"\bfault[_ ]?\d{1,2}\b"
MENU_MARKER = "CANDIDATE MENU"

SYSTEM_PROMPT_V2 = (
    "You are a control-room diagnostic assistant for the Tennessee Eastman "
    "Process (TEP). You are given structured evidence from an anomaly detector "
    "and a numbered menu of candidate faults. Decide which menu entry the "
    "evidence supports, or abstain.\n\n"
    "RULES\n"
    "1. The answer MUST be one of the fault_id values in the menu, or "
    f"{UNCLASSIFIED}. Never invent an id and never answer with a fault that is "
    "not on the menu.\n"
    "2. If the evidence matches no menu entry, answer "
    f"{UNCLASSIFIED} and name the closest entries in your reasoning. Abstaining "
    "is a correct answer, not a failure.\n"
    "3. Ground every physical claim in the given sensor deviations, the onset "
    "order, and the detector statistics. Do not introduce sensors, numbers or "
    "mechanisms that are not in the evidence.\n"
    "4. Never claim certainty. Correlation and temporal order are supporting "
    "evidence, not proof; recommend on-site verification.\n"
    "5. Write two fields only: 'physical_reasoning' (why the evidence implies "
    "the chosen mechanism, step by step) and 'narrative' (2-4 sentences for an "
    "operator: what is happening and what to check first).\n\n"
    "OUTPUT: a single JSON object with exactly the keys "
    '{"fault_id", "physical_reasoning", "narrative"}. No markdown fences, no '
    "prose outside the JSON."
)

ANSWER_KEYS_V2 = ("fault_id", "physical_reasoning", "narrative")


def _fmt(value: object, decimals: int = 2) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _detector_lines(evidence: Dict) -> List[str]:
    det = evidence.get("detector_evidence") or {}
    if not isinstance(det, dict) or not det:
        return ["Detector evidence: (none recorded for this event)"]
    lines: List[str] = []
    lstm = det.get("lstm_ae") or {}
    if isinstance(lstm, dict) and "score" in lstm:
        score = float(lstm.get("score") or 0.0)
        thr = float(lstm.get("threshold") or 0.0)
        ratio = score / thr if thr > 0 else 0.0
        lines.append(
            f"- LSTM autoencoder reconstruction error {score:.4f} "
            f"(threshold {thr:.4f}, ratio {ratio:.2f}x, alarmed {bool(lstm.get('alarmed'))})"
        )
    dyn = det.get("dynamic") or {}
    if isinstance(dyn, dict) and "score" in dyn:
        score = float(dyn.get("score") or 0.0)
        thr = float(dyn.get("threshold") or 0.0)
        ratio = score / thr if thr > 0 else 0.0
        lines.append(
            f"- CVA Q-statistic {score:.1f} (threshold {thr:.1f}, ratio {ratio:.2f}x, "
            f"alarmed {bool(dyn.get('alarmed'))})"
        )
    trig = det.get("triggered_by")
    change = det.get("change_type")
    if trig:
        lines.append(f"- Fused trigger: {trig} ({change} change)")
    return lines


def _sensor_lines(evidence: Dict) -> List[str]:
    sensors = evidence.get("top_anomalous_sensors") or []
    if not isinstance(sensors, list) or not sensors:
        return ["- (no strong single-sensor deviation)"]
    lines: List[str] = []
    for s in sensors:
        if not isinstance(s, dict):
            continue
        name = str(s.get("display_name") or s.get("name") or "")
        z = s.get("z_score", s.get("zScore"))
        dev = s.get("deviation_percent")
        trend = s.get("trend")
        line = f"- {name}: z={_fmt(z)}, deviation {_fmt(dev)}%"
        if trend:
            line += f", trend {trend}"
        lines.append(line)
    return lines


def _lag_lines(evidence: Dict) -> List[str]:
    """CVA residual attribution: the evidence that localizes dynamics faults.

    A sticking valve under closed-loop control barely moves a level, so the
    LSTM error ranking is nearly flat for those faults. The CVA residual energy
    still concentrates on the affected variables, so it belongs in the prompt.
    """
    det = evidence.get("detector_evidence") or {}
    lag = det.get("lag_attribution") or {} if isinstance(det, dict) else {}
    tops = [v for v in (lag.get("top_variables") or []) if isinstance(v, dict)]
    if not tops:
        return ["- (no dynamic residual attribution recorded)"]
    lines = [
        f"- {v.get('variable')}: residual share {_fmt(v.get('share'), 3)}" for v in tops
    ]
    if lag.get("dominant_lag") is not None:
        lines.append(
            f"Dominant lag {lag['dominant_lag']} window(s), slow-drift ratio "
            f"{_fmt(lag.get('slow_drift_ratio'), 2)}"
        )
    return lines


def _sequence_lines(evidence: Dict) -> List[str]:
    raw = evidence.get("temporal_sequence")
    if isinstance(raw, dict):
        seq = [x for x in (raw.get("sequence") or []) if isinstance(x, dict)]
    elif isinstance(raw, list):
        seq = [x for x in raw if isinstance(x, dict)]
    else:
        seq = []
    if not seq:
        return ["(no clear onset ordering)"]
    lines = []
    for i, item in enumerate(seq, start=1):
        name = item.get("display_name") or item.get("name") or "sensor"
        dt = item.get("relative_time_minutes", 0.0)
        lines.append(f"{i}. {name} at +{_fmt(dt, 1)} min")
    return lines


def _menu_block(menu: Sequence[Dict[str, str]]) -> List[str]:
    lines = ["CANDIDATE MENU (answer with one of these fault_id values):"]
    for entry in menu:
        lines.append(f"- {entry['fault_id']}: {entry['signature']}")
    return lines


def opaque_event_id(raw: object) -> str:
    """A stable id that reveals nothing about the event it names.

    The internal id is ``EVT-<split>-f<fault>-r<run>``: split, fault number and
    run number, all in one token. A prompt carrying it hands the model the label
    and the split, so the prompt gets a hash instead. The salt is constant, so
    the same event always renders the same id and a cache stays valid.
    """
    digest = hashlib.sha256(f"prompt-event-id-v1|{raw}".encode("utf-8")).hexdigest()
    return f"EV-{digest[:12]}"


def render_prompt(
    evidence: Dict,
    menu: Sequence[Dict[str, str]],
    instruction: Optional[str] = None,
    include_schema: bool = True,
    event_id: Optional[str] = None,
) -> str:
    """Render one event + menu into the v2 user prompt.

    Whatever id is passed is replaced by its opaque hash, so a caller cannot leak
    the internal one by accident.
    """
    raw_id = event_id if event_id is not None else evidence.get("event_id", "UNKNOWN")
    lines: List[str] = [
        "A new anomaly has been detected in the Tennessee Eastman Process.",
        "",
        f"Event: {opaque_event_id(raw_id)}",
        f"Overall anomaly score: {_fmt(evidence.get('anomaly_score'), 3)}",
        "",
        "DETECTORS",
        *_detector_lines(evidence),
        "",
        "TOP DEVIATING SENSORS",
        *_sensor_lines(evidence),
        "",
        "DYNAMIC RESIDUAL ATTRIBUTION (CVA, high-lag share per variable)",
        *_lag_lines(evidence),
        "",
        "ONSET ORDER",
        *_sequence_lines(evidence),
        "",
        *_menu_block(menu),
        "",
    ]
    if instruction:
        lines += [instruction, ""]
    if include_schema:
        lines += [
            "Return one JSON object with exactly these keys: "
            + ", ".join(f'"{key}"' for key in ANSWER_KEYS_V2)
            + ". fault_id must be a menu entry.",
        ]
    return "\n".join(lines)


def _head(text: str) -> str:
    """The part of the prompt before the menu: everything the model may read as
    evidence. Fault names and ids are legal in the menu and illegal here."""
    index = text.find(MENU_MARKER)
    return text if index < 0 else text[:index]


def prompt_leaks(text: str) -> List[str]:
    """Every leak found in one prompt, as human-readable findings."""
    body = text or ""
    findings: List[str] = [
        f"token:{match.group(0)!r}"
        for pattern in LEAK_PATTERNS
        for match in [re.search(pattern, body, re.IGNORECASE)]
        if match
    ]
    head = _head(body)
    for match in re.finditer(FAULT_REF_PATTERN, head, re.IGNORECASE):
        findings.append(f"fault_reference:{match.group(0)!r}")
    normalised = " ".join(head.lower().split())
    for number, entry in FAULT_KNOWLEDGE.items():
        name = " ".join(str(entry.get("name") or "").lower().split())
        # Short names ("valve sticking") collide with ordinary words; a full KB
        # name is a phrase, so a phrase match is evidence of a real leak.
        if len(name) >= 12 and name in normalised:
            findings.append(f"fault_name:{number}")
    return findings


def assert_prompt_clean(text: str, where: str = "prompt") -> None:
    """Raise if a forbidden fragment or a leak appears in one prompt."""
    lowered = (text or "").lower()
    hits = [frag for frag in FORBIDDEN_PROMPT_FRAGMENTS if frag in lowered]
    if hits:
        raise ValueError(f"{where}: forbidden prompt fragment(s) {hits}")
    leaks = prompt_leaks(text)
    if leaks:
        raise ValueError(f"{where}: prompt leak(s) {leaks}")


def teacher_instruction(menu_ids: Sequence[str]) -> str:
    """The concrete ask appended to the prompt, naming the legal answers."""
    return (
        "Choose the single best entry from the menu above, or "
        f"{UNCLASSIFIED} if none fits. Legal answers: "
        + ", ".join(menu_ids)
        + "."
    )


def schema_example() -> str:
    return json.dumps({key: "<text>" for key in ANSWER_KEYS_V2}, indent=2)


__all__ = [
    "ANSWER_KEYS_V2",
    "FORBIDDEN_PROMPT_FRAGMENTS",
    "LEAK_PATTERNS",
    "SYSTEM_PROMPT_V2",
    "assert_prompt_clean",
    "opaque_event_id",
    "prompt_leaks",
    "render_prompt",
    "schema_example",
    "teacher_instruction",
]
