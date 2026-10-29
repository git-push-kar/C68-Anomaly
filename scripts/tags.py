"""Canonical TEP variable tags and alias resolution.

The convention here is the repo's own, not a new one:

* ``evidence/process_relationships.py`` (``SENSOR_NAMES``) defines the 52
  canonical column names: ``XMEAS_1`` .. ``XMEAS_41`` for the measurement
  variables and ``XMV_42`` .. ``XMV_52`` for the manipulated variables, using
  the *global* 1..52 index for ``XMV``.
* ``scripts/fault_knowledge.py`` writes the 1-based manipulated-variable alias
  next to the canonical tag, e.g. ``"XMV_52 (Condenser Cooling Water Flow valve
  / XMV 11)"`` or ``"XMV_45 / XMEAS_4"``.
* ``data/raw/**/*.csv`` names the columns after the *alias* family:
  ``xmeas_1`` .. ``xmeas_41`` and ``xmv_1`` .. ``xmv_11``.

So three spellings of the same variable are in circulation:

===================  ============================  ================
canonical            alias (fault_knowledge form)  raw CSV column
===================  ============================  ================
``XMV_52``           ``XMV 11``                    ``xmv_11``
``XMV_45``           ``XMV 4``                     ``xmv_4``
``XMV_42``           ``XMV 1``                     ``xmv_1``
``XMEAS_4``          ``XMEAS 4``                   ``xmeas_4``
===================  ============================  ================

``XMV_52`` is therefore a valid tag and is the canonical form of ``XMV 11``.
The manipulated alias is 1-based (1..11); ``XMV_INDEX_OFFSET`` (41) converts it
to the canonical global index.  The repo already uses that relation: fault 2
lists ``XMV_42 (...) / XMV 1`` while ``process_relationships`` maps
``XMV_42 -> D_Feed_Flow``, and fault 15 lists ``XMV_52 (...) / XMV 11`` while
``process_relationships`` maps ``XMV_52 -> Condenser_Cooling_Water_Flow``.

Anything that is not a resolvable tag returns ``None`` -- callers must treat an
unresolvable tag as an error rather than silently inventing a canonical form.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from evidence.process_relationships import (
    SENSOR_NAMES,
    sensor_display_name,
    subsystem_for_sensor,
)

#: 1-based manipulated-variable alias -> global index offset (XMV <n> -> XMV_(n+41)).
XMV_INDEX_OFFSET = 41
#: Canonical XMEAS range is 1..41.
MEASUREMENT_COUNT = 41
#: Canonical XMV range is 42..52.
MANIPULATED_FIRST = 42
MANIPULATED_LAST = 52
#: Highest 1-based manipulated-variable alias (XMV 1..XMV 11).
MANIPULATED_ALIAS_MAX = MANIPULATED_LAST - XMV_INDEX_OFFSET

_TAG_RE = re.compile(r"(?<![A-Za-z0-9])(XMEAS|XMV)[ _-]?(\d{1,2})(?![0-9])", re.IGNORECASE)

_DISPLAY_TO_CANONICAL: Dict[str, str] = {v: k for k, v in SENSOR_NAMES.items()}


def _canonicalize_family(family: str, number: int) -> Optional[str]:
    if family == "XMEAS":
        if 1 <= number <= MEASUREMENT_COUNT:
            return f"XMEAS_{number}"
        return None
    # XMV: a value inside the canonical global range is already canonical;
    # anything in 1..11 is the 1-based alias and must be offset.
    if MANIPULATED_FIRST <= number <= MANIPULATED_LAST:
        return f"XMV_{number}"
    if 1 <= number <= MANIPULATED_ALIAS_MAX:
        return f"XMV_{number + XMV_INDEX_OFFSET}"
    return None


def canonical_tag(text: object) -> Optional[str]:
    """Return the canonical tag named by ``text``, or ``None``.

    Accepts every spelling used in this repo: ``XMV_52``, ``XMV 11``,
    ``xmv_11``, ``XMEAS_4``, ``XMEAS 4``, and composite KB strings such as
    ``"XMV_52 (Condenser Cooling Water Flow valve / XMV 11)"`` -- the first
    resolvable tag in the string wins, which is the canonical one in the KB.
    """
    if not isinstance(text, str):
        return None
    match = _TAG_RE.search(text)
    if match is None:
        return None
    return _canonicalize_family(match.group(1).upper(), int(match.group(2)))


def iter_tags(text: object) -> List[str]:
    """Return every canonical tag in ``text``, de-duplicated, in order.

    Composite KB strings can name the same variable twice
    (``"XMV_45 (...) / XMV 4"``); both aliases collapse to one canonical tag.
    """
    if not isinstance(text, str):
        return []
    found: List[str] = []
    for family, number in _TAG_RE.findall(text):
        canonical = _canonicalize_family(family.upper(), int(number))
        if canonical is not None and canonical not in found:
            found.append(canonical)
    return found


def is_canonical(tag: object) -> bool:
    """True when ``tag`` is already in canonical form."""
    return isinstance(tag, str) and tag in SENSOR_NAMES


def display_name(tag: object) -> Optional[str]:
    """English process name for a tag, e.g. ``XMV_52 -> Condenser_Cooling_Water_Flow``."""
    canonical = tag if is_canonical(tag) else canonical_tag(tag)
    if canonical is None:
        return None
    return sensor_display_name(canonical)


def subsystem(tag: object) -> Optional[str]:
    """Subsystem from the repo taxonomy, e.g. ``XMV_52 -> condenser_cooling_system``."""
    name = display_name(tag)
    if name is None:
        return None
    return subsystem_for_sensor(name)


def manipulated_index(tag: object) -> Optional[int]:
    """1-based manipulated-variable alias for an ``XMV`` tag (``XMV_52 -> 11``)."""
    canonical = tag if is_canonical(tag) else canonical_tag(tag)
    if canonical is None or not canonical.startswith("XMV_"):
        return None
    return int(canonical.split("_", 1)[1]) - XMV_INDEX_OFFSET


def fault_tags(fault_id: int) -> List[str]:
    """Canonical tags the fault knowledge base associates with ``fault_id``.

    Reads the same fields the KB writes: ``valves``, ``initiating[].tag`` and
    ``initiating[].sensor`` (a display name).  Unknown faults return ``[]``.
    """
    from scripts.fault_knowledge import FAULT_KNOWLEDGE  # local import: avoids a cycle

    entry = FAULT_KNOWLEDGE.get(int(fault_id))
    if not entry:
        return []

    tags: List[str] = []

    def _add(canonical: Optional[str]) -> None:
        if canonical is not None and canonical not in tags:
            tags.append(canonical)

    for valve in entry.get("valves") or []:
        for canonical in iter_tags(valve):
            _add(canonical)
    for item in entry.get("initiating") or []:
        for canonical in iter_tags(item.get("tag")):
            _add(canonical)
        name = item.get("sensor")
        if isinstance(name, str):
            _add(_DISPLAY_TO_CANONICAL.get(name))
    return tags


def describe(tag: object) -> Optional[Dict[str, object]]:
    """Convenience bundle for report tables: canonical tag, name, subsystem."""
    canonical = tag if is_canonical(tag) else canonical_tag(tag)
    if canonical is None:
        return None
    return {
        "tag": canonical,
        "display_name": sensor_display_name(canonical),
        "subsystem": subsystem(canonical),
        "manipulated_index": manipulated_index(canonical),
    }
