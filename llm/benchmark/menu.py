"""Candidate menu derived from the knowledge base.

The menu is the model's closed set: one line per classifiable fault, plus
UNCLASSIFIED_OPEN_SET which is always available. Faults 16-20 are the open-set
family (their KB entries exist but naming one is wrong by construction), and
NORMAL_TRANSIENT is not a fault at all, so neither is a menu entry.

Everything here is computed from ``scripts.fault_knowledge`` -- no prose is
invented, so a KB edit propagates instead of silently diverging. The one-line
signature is the fault name plus its initiating variable and direction, which is
the part of the mechanism the evidence can actually show.
"""
from __future__ import annotations

import ast
import re
from typing import Dict, Iterable, List, Optional

from scripts.fault_ids import SYNTHETIC_FAULTS, fault_id_for
from scripts.fault_knowledge import FAULT_KNOWLEDGE

#: Faults whose identity is a legitimate answer: 1-15. 16-20 are open-set by
#: protocol, 21/22 are synthetic and never appear in a headline item.
CLASSIFIABLE_FAULTS: List[int] = [n for n in range(1, 16) if n not in SYNTHETIC_FAULTS]

UNCLASSIFIED = "UNCLASSIFIED_OPEN_SET"
_FAULT_REF = re.compile(r"FAULT\s+(\d{1,2})", re.IGNORECASE)


def _initiating(fault: int) -> Dict:
    """Parse the KB's ``initiating`` field (a Python-literal list of dicts)."""
    raw = (FAULT_KNOWLEDGE.get(fault) or {}).get("initiating")
    if not raw:
        return {}
    if isinstance(raw, list):
        return raw[0] if raw and isinstance(raw[0], dict) else {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return parsed[0]
    return {}


def signature_for(fault: int) -> str:
    """One deterministic line: what changes first, and in which direction."""
    entry = FAULT_KNOWLEDGE.get(fault) or {}
    name = str(entry.get("name") or f"Fault {fault}")
    init = _initiating(fault)
    sensor = init.get("sensor")
    direction = init.get("direction")
    if sensor and direction:
        return f"{name} — {sensor} {direction}s first"
    if sensor:
        return f"{name} — {sensor} deviates first"
    return name


def menu_entry(fault: int) -> Dict[str, str]:
    entry = FAULT_KNOWLEDGE.get(fault) or {}
    return {
        "fault_id": fault_id_for(int(fault)).value,
        "signature": signature_for(int(fault)),
    }


def lookalikes(fault: int, limit: int = 2) -> List[int]:
    """Fault ids the KB itself names as the differential for ``fault``.

    Deterministic: the KB's ``differential_diagnosis`` text references the
    competing faults by number ("CRITICAL DIFFERENTIAL VS FAULT 5: ..."), so the
    list is read out of the KB rather than guessed from similarity.
    """
    entry = FAULT_KNOWLEDGE.get(fault) or {}
    text = " ".join(str(entry.get(key) or "") for key in ("differential_diagnosis", "reasoning"))
    seen: List[int] = []
    for match in _FAULT_REF.finditer(text):
        number = int(match.group(1))
        if number == int(fault) or number in seen:
            continue
        if number not in CLASSIFIABLE_FAULTS:
            continue
        seen.append(number)
    return seen[:limit]


def _fault_number(value) -> Optional[int]:
    """Accept a KB number or a benchmark id (``FAULT_15``); ``None`` otherwise.

    Labels that are not menu entries at all (``UNCLASSIFIED_OPEN_SET``,
    ``NORMAL_TRANSIENT``) resolve to ``None`` and are ignored by ``build_menu``:
    there is nothing to remove for them, because they were never in the menu.
    """
    if isinstance(value, str):
        text = value.strip().upper()
        if text.startswith(("FAULT_", "SYNTHETIC_")):
            try:
                return int(text.split("_")[-1])
            except ValueError:
                return None
        try:
            return int(text)
        except ValueError:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_menu(remove: Optional[Iterable] = None) -> List[Dict[str, str]]:
    """The full menu, minus ``remove``, with UNCLASSIFIED_OPEN_SET appended.

    Removing the true fault's entry is how an open-set episode is constructed:
    the remaining entries are still the only legal answers.
    """
    drop = {n for n in (_fault_number(value) for value in (remove or [])) if n is not None}
    entries = [menu_entry(n) for n in CLASSIFIABLE_FAULTS if n not in drop]
    entries.append({
        "fault_id": UNCLASSIFIED,
        "signature": "a disturbance that matches no listed fault family; abstain and say why",
    })
    return entries


def nearest_candidates(fault: int, k: int = 2) -> List[str]:
    """Deterministic 'nearest' list for an open-set episode: KB lookalikes first.

    The fault itself is never a candidate. Without that exclusion the fill loop
    could pad the list with the removed entry, and the teacher prompt would then
    name the very fault the item asks it to abstain from -- number and KB name
    both.
    """
    candidates = [fault_id_for(n).value for n in lookalikes(fault, limit=k)]
    for number in CLASSIFIABLE_FAULTS:
        if len(candidates) >= k:
            break
        if int(number) == int(fault):
            continue
        value = fault_id_for(number).value
        if value not in candidates:
            candidates.append(value)
    return candidates[:k]

__all__ = [
    "CLASSIFIABLE_FAULTS",
    "UNCLASSIFIED",
    "build_menu",
    "lookalikes",
    "menu_entry",
    "nearest_candidates",
    "signature_for",
]
