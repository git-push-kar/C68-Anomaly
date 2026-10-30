"""Shared tag- and label-aware allowed set for the response layer.

One object is used by the validators *and* (later) by the evaluator, so an
answer that is accepted here is scored on exactly the same names. Resolution
rules come from `scripts/tags.py`, which is the repo's single place for tag
spelling; this module only assembles the set for one event.

Accepted spellings for a variable, all of which resolve to one canonical tag:

* canonical       ``XMV_52``
* 1-based alias   ``XMV 11``, ``XMV11``, ``xmv_11``      (offset +41)
* display name    ``Condenser_Cooling_Water_Flow``, ``condenser cooling water flow``

Accepted spellings for a fault label: benchmark id (``FAULT_15``), the KB number
(``15``, ``fault 15``), the KB name (``Condenser cooling water valve sticking``)
and the open-set label (``UNCLASSIFIED_OPEN_SET``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Optional

from evidence.process_relationships import SUBSYSTEM_GROUPS
from scripts.fault_ids import (
    FaultID,
    SYNTHETIC_FAULTS,
    fault_id_for,
    is_synthetic,
    kb_entry,
)
from scripts.tags import canonical_tag, display_name, fault_tags, iter_tags, is_canonical

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

_SCHEMA_VOCABULARY_CACHE: Optional[FrozenSet[str]] = None


def schema_vocabulary() -> FrozenSet[str]:
    """Identifiers that name the *answer format*, not a variable in the plant.

    A token such as ``overview_fallback`` or ``ranked_hypotheses`` is shaped like
    a tag (one underscore, letters on both sides), so ``unknown_tag_like`` used to
    report it as a fabricated sensor and reject an answer for naming its own
    schema. The vocabulary here is derived from the schema module -- the intent
    and table and severity enums plus every pydantic field name -- rather than
    typed out, so a new enum member or field cannot silently become a "fabricated
    tag". Sensor claims are unaffected: nothing in this set is a tag.
    """
    global _SCHEMA_VOCABULARY_CACHE
    if _SCHEMA_VOCABULARY_CACHE is not None:
        return _SCHEMA_VOCABULARY_CACHE
    names: set = set()
    try:
        from pydantic import BaseModel

        from llm.response import schemas
    except Exception:  # pragma: no cover - schema module is always importable here
        _SCHEMA_VOCABULARY_CACHE = frozenset()
        return _SCHEMA_VOCABULARY_CACHE
    for attr in dir(schemas):
        obj = getattr(schemas, attr)
        if isinstance(obj, type) and issubclass(obj, Enum):
            for member in obj:
                names.add(str(member.value).lower())
                names.add(member.name.lower())
        elif isinstance(obj, type) and issubclass(obj, BaseModel):
            names.update(str(name).lower() for name in obj.model_fields)
        elif isinstance(obj, tuple) and obj and all(isinstance(item, str) for item in obj):
            # module-level tuples of field or intent names (e.g. ANSWER_KEYS)
            names.update(item.lower() for item in obj)
    for group in (getattr(schemas, "LIVE_INTENTS", ()), getattr(schemas, "PROPOSED_INTENTS", ())):
        for member in group:
            names.add(str(getattr(member, "value", member)).lower())
    # The slim SFT answer contract (fault_id, verdict, ranked_hypotheses, ...) is
    # declared with the export, not with the pydantic models, so it is read from
    # there rather than restated here.
    try:
        from llm.benchmark import to_sft

        for group in (getattr(to_sft, "ANSWER_KEYS", ()), getattr(to_sft, "DERIVED_AT_RENDER", ())):
            names.update(str(item).lower() for item in group)
    except Exception:  # pragma: no cover - export module is optional at import time
        pass
    _SCHEMA_VOCABULARY_CACHE = frozenset(names)
    return _SCHEMA_VOCABULARY_CACHE


def normalize_name(text: object) -> str:
    """Lowercase, punctuation-free form used for display-name matching."""
    if not isinstance(text, str):
        return ""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


@dataclass(frozen=True)
class AllowedSet:
    """Every variable and label an answer about one event may name."""

    canonical_tags: frozenset = frozenset()
    _tag_lookup: Dict[str, str] = field(default_factory=dict)
    _fault_lookup: Dict[str, FaultID] = field(default_factory=dict)
    faults: tuple = ()
    #: Subsystem / equipment identifiers that look tag-shaped but are not tags.
    other_names: frozenset = frozenset()

    # -- variables ------------------------------------------------------
    def resolve_tag(self, text: object) -> Optional[str]:
        """Canonical tag for any accepted spelling, else ``None``."""
        if isinstance(text, str) and is_canonical(text):
            return text if text in self.canonical_tags else None
        canonical = canonical_tag(text)
        if canonical is not None:
            return canonical if canonical in self.canonical_tags else None
        return self._tag_lookup.get(normalize_name(text))

    def tags_in(self, text: object) -> List[str]:
        """Canonical tags mentioned anywhere in ``text``."""
        found: List[str] = []
        if not isinstance(text, str):
            return found
        for tag in iter_tags(text):
            if tag in self.canonical_tags and tag not in found:
                found.append(tag)
        low = normalize_name(text)
        for name, tag in self._tag_lookup.items():
            if tag not in found and name and name in low:
                found.append(tag)
        return found

    def unknown_tag_like(self, text: object) -> List[str]:
        """Tag-shaped tokens that do not resolve (``XMV_99``, ``XMV 12``)."""
        if not isinstance(text, str):
            return []
        bad: List[str] = []
        for match in re.finditer(r"(?<![A-Za-z0-9])(XMEAS|XMV)[ _-]?(\d{1,2})(?![0-9])", text, re.IGNORECASE):
            token = match.group(0).strip()
            if self.resolve_tag(token) is None and token not in bad:
                bad.append(token)
        # display-name shaped tokens that look like a sensor but are not allowed
        vocabulary = schema_vocabulary()
        for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b", text):
            token = match.group(1)
            if self.resolve_tag(token) is not None:
                continue
            if token.lower().startswith(("fault_", "xmv", "xmeas")):
                continue
            if token.lower() in vocabulary:
                continue
            if normalize_name(token) in self.other_names:
                continue
            if token not in bad:
                bad.append(token)
        return bad

    # -- faults ---------------------------------------------------------
    def resolve_fault(self, text: object) -> Optional[FaultID]:
        if isinstance(text, FaultID):
            return text if text in self.faults else None
        if isinstance(text, int) and not isinstance(text, bool):
            text = str(text)
        if not isinstance(text, str):
            return None
        key = normalize_name(text)
        if key in self._fault_lookup:
            return self._fault_lookup[key]
        numbers = re.findall(r"\d{1,2}", text)
        if len(numbers) == 1:
            candidate = self._fault_lookup.get(numbers[0])
            if candidate is not None:
                return candidate
        return None

    def is_allowed_label(self, fault: FaultID) -> bool:
        return fault in self.faults

    def label_for(self, fault: FaultID) -> str:
        """Human display for a label: KB name, or the open-set/normal wording."""
        if fault == FaultID.UNCLASSIFIED_OPEN_SET:
            return "unclassified open-set disturbance"
        if fault == FaultID.NORMAL_TRANSIENT:
            return "no fault-justified excursion"
        entry = kb_entry(int(fault.value.split("_")[-1])) if fault.value.startswith(("FAULT_", "SYNTHETIC_")) else None
        return str((entry or {}).get("name") or fault.value)

    def is_synthetic(self, fault: FaultID) -> bool:
        if not fault.value.startswith(("FAULT_", "SYNTHETIC_")):
            return False
        return is_synthetic(int(fault.value.split("_")[-1]))


def _fault_lookup_keys(number: int) -> Iterable[str]:
    fault_id = fault_id_for(number)
    keys = {normalize_name(fault_id.value), str(number)}
    entry = kb_entry(number) or {}
    if entry.get("name"):
        keys.add(normalize_name(entry["name"]))
    if fault_id == FaultID.UNCLASSIFIED_OPEN_SET:
        keys.add(normalize_name("unclassified open set"))
        keys.add(normalize_name("unclassified_open_set"))
    return keys


def build_allowed_set(event: Optional[Dict] = None, extra_faults: Optional[Iterable[int]] = None) -> AllowedSet:
    """Allowed set for one event: its sensor tags, its candidate faults, all labels.

    Sensor tags come from the event's evidence (top sensors, temporal sequence,
    candidate subsystem sensors); labels cover the full benchmark space so a
    hypothesis may name any fault, not only the one the event was generated from.
    """
    event = event if isinstance(event, dict) else {}
    evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else event

    tags: List[str] = []
    for sensor in (evidence.get("top_anomalous_sensors") or []):
        if isinstance(sensor, dict):
            for key in ("name", "display_name"):
                resolved = canonical_tag(sensor.get(key))
                if resolved and resolved not in tags:
                    tags.append(resolved)
    for item in (evidence.get("temporal_sequence") or []):
        if isinstance(item, dict):
            resolved = canonical_tag(item.get("sensor")) or canonical_tag(item.get("display_name"))
            if resolved and resolved not in tags:
                tags.append(resolved)
    # A hypothesis may legitimately name KB variables this event did not rank;
    # the label space is the whole benchmark, so its KB tags are allowed too.
    for extra in extra_faults or []:
        for tag in fault_tags(int(extra)):
            if tag not in tags:
                tags.append(tag)

    tag_lookup: Dict[str, str] = {}
    for tag in tags:
        name = display_name(tag)
        if name:
            tag_lookup[normalize_name(name)] = tag
            tag_lookup[normalize_name(name.replace("_", " "))] = tag

    fault_lookup: Dict[str, FaultID] = {}
    numbers = list(range(1, 21)) + list(SYNTHETIC_FAULTS)
    for number in numbers:
        for key in _fault_lookup_keys(number):
            if key:
                fault_lookup.setdefault(key, fault_id_for(number))
    fault_lookup[normalize_name("NORMAL_TRANSIENT")] = FaultID.NORMAL_TRANSIENT
    for extra in extra_faults or []:
        for key in _fault_lookup_keys(int(extra)):
            fault_lookup.setdefault(key, fault_id_for(int(extra)))

    labels = [fault_id_for(n) for n in numbers] + [FaultID.UNCLASSIFIED_OPEN_SET, FaultID.NORMAL_TRANSIENT]

    other = {normalize_name(name) for name in SUBSYSTEM_GROUPS}
    for entry in (kb_entry(n) or {} for n in numbers):
        if entry.get("subsystem"):
            other.add(normalize_name(entry["subsystem"]))
        for item in entry.get("equipment") or []:
            other.add(normalize_name(item))
        for item in entry.get("equipment") or []:
            # equipment entries often embed a tag gloss in parentheses
            other.add(normalize_name(re.sub(r"\(.*?\)", " ", str(item))))
    other.discard("")

    return AllowedSet(
        canonical_tags=frozenset(tags),
        _tag_lookup=tag_lookup,
        _fault_lookup=fault_lookup,
        faults=tuple(dict.fromkeys(labels)),
        other_names=frozenset(other),
    )
