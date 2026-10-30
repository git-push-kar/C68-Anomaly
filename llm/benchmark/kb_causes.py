"""The disturbance a fault names is a cause, and a cause is usually unmeasured.

Every fault's KB ``name`` describes *what happened to the plant* -- "Condenser
cooling water inlet temperature step", "Reaction kinetics slow drift" -- not a
sensor. In this process the cooling-water inlet temperature is not instrumented at
all: the evidence can only show its effects (flow, pressure, the valve's residual
share). A teacher that writes "the condenser cooling water inlet temperature is
stepping" is naming the cause correctly, and the sensor filter used to reject it
as a fabricated tag, because that name does not resolve to an XMEAS/XMV tag.

This module is the one place that decides which prose is a legitimate cause
mention. It reads the phrases out of the knowledge base, so a KB edit moves the
allowed phrasing with it; there is no list of sensor-shaped words here, and no
pattern that would accept an arbitrary plant name.

What is accepted, and only this:

* the KB name itself, after normalising separators and case;
* the KB name with a trailing disturbance-type tail dropped ("... temperature
  **step**"), which is the same cause named less fully -- accepted only when what
  remains is at least three words, so "condenser" alone cannot pass;
* a KB ``initiating`` sensor *when that sensor is not a measured tag* (general
  case; every fault in this KB happens to name a measured one).

Anything else stays a fabricated tag. "Condenser_Pressure" and
"Condenser_Cooling_Water_Valve" are not the KB's phrase, and are still rejected.
"""
from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional

from scripts.fault_knowledge import FAULT_KNOWLEDGE
from scripts.tags import canonical_tag

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

#: A cause phrase shorter than this many words is too generic to accept: it would
#: let "condenser" or "reactor" pass as a cause.
MIN_PHRASE_WORDS = 3


def normalise(text: object) -> str:
    return _NON_ALNUM.sub(" ", str(text or "").lower()).strip()


def _initiating(fault: int) -> Dict:
    raw = (FAULT_KNOWLEDGE.get(fault) or {}).get("initiating")
    if not raw:
        return {}
    if isinstance(raw, list):
        return raw[0] if raw and isinstance(raw[0], dict) else {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    return parsed[0] if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) else {}


def disturbance_phrase(fault: int) -> str:
    """The KB's own wording for what happened, for the teacher prompt."""
    entry = FAULT_KNOWLEDGE.get(int(fault)) or {}
    return str(entry.get("name") or "").strip()


def unmeasured_cause_phrases(fault: int) -> List[str]:
    """Normalised phrases from the KB that name an unmeasured cause."""
    phrases: List[str] = []
    name = normalise(disturbance_phrase(fault))
    if name and canonical_tag(disturbance_phrase(fault)) is None:
        phrases.append(name)
    sensor = _initiating(int(fault)).get("sensor")
    if sensor and canonical_tag(sensor) is None:
        phrases.append(normalise(sensor))
    return [p for p in dict.fromkeys(phrases) if p]


def is_cause_mention(token: object, fault: Optional[int]) -> bool:
    """True when ``token`` is this fault's own KB cause phrasing.

    Prefix acceptance exists because a teacher often writes the cause without its
    disturbance tail ("condenser cooling water inlet temperature" for "... step").
    It is a whole-word prefix of one specific KB phrase, three words minimum, so
    it cannot be satisfied by a name the KB does not contain.
    """
    if fault is None:
        return False
    candidate = normalise(token)
    if not candidate:
        return False
    for phrase in unmeasured_cause_phrases(int(fault)):
        if candidate == phrase:
            return True
        if len(candidate.split()) >= MIN_PHRASE_WORDS and phrase.startswith(candidate + " "):
            return True
    return False


def audit() -> List[Dict]:
    """Per-fault count of KB phrases that name something unmeasured."""
    rows: List[Dict] = []
    for fault in sorted(FAULT_KNOWLEDGE):
        entry = FAULT_KNOWLEDGE.get(fault) or {}
        name = str(entry.get("name") or "")
        sensor = str(_initiating(int(fault)).get("sensor") or "")
        rows.append({
            "fault": int(fault),
            "name": name,
            "name_measured": canonical_tag(name) is not None,
            "initiating_sensor": sensor,
            "initiating_measured": (canonical_tag(sensor) is not None) if sensor else None,
            "unmeasured_phrases": unmeasured_cause_phrases(int(fault)),
        })
    return rows


__all__ = [
    "MIN_PHRASE_WORDS",
    "audit",
    "disturbance_phrase",
    "is_cause_mention",
    "normalise",
    "unmeasured_cause_phrases",
]
