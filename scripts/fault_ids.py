"""Benchmark label space for TEP faults, derived from the knowledge base.

Every member is derived from ``scripts/fault_knowledge.py`` -- this module only
names and classifies, it never invents a fault.

Label space
-----------
``FAULT_01`` .. ``FAULT_15``
    Real, individually diagnosable faults. The benchmark expects the adapter to
    name the fault.
``UNCLASSIFIED_OPEN_SET``
    The label for faults 16-20. Their KB names are literally "Unknown
    disturbance in ...", so demanding the specific number would score a
    distinction the KB does not make. They stay *in* the benchmark, but the
    accepted answer is the open-set label.
``NORMAL_TRANSIENT``
    Fault-free / no-fault-justified events (used by the detector side, not a
    fault number).
``SYNTHETIC_FAULT_21`` / ``SYNTHETIC_FAULT_22``
    Exist only in generated data. They are never headline-eligible and must be
    labelled ``synthetic`` wherever they appear.

Flags carried alongside, not folded into the id:

* ``UNOBSERVABLE_FAULTS = {3, 9}`` -- weak fault signatures; see
  ``docs/PROTOCOL.md``. They remain in the split but are reported separately.
* ``OPEN_SET_FAULTS = {16, 17, 18, 19, 20}`` -- map to ``UNCLASSIFIED_OPEN_SET``.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from scripts.fault_knowledge import FAULT_KNOWLEDGE

#: Faults with a real simulation run and an individually diagnosable signature.
HEADLINE_FAULTS: List[int] = list(range(1, 21))
#: Faults whose accepted label is the open-set label.
OPEN_SET_FAULTS: List[int] = [16, 17, 18, 19, 20]
#: Faults flagged as weak / hard to observe; reported separately.
UNOBSERVABLE_FAULTS: List[int] = [3, 9]
#: Faults that exist only as synthetic data.
SYNTHETIC_FAULTS: List[int] = [21, 22]


class FaultID(str, Enum):
    """Label space for a single benchmark item."""

    FAULT_01 = "FAULT_01"
    FAULT_02 = "FAULT_02"
    FAULT_03 = "FAULT_03"
    FAULT_04 = "FAULT_04"
    FAULT_05 = "FAULT_05"
    FAULT_06 = "FAULT_06"
    FAULT_07 = "FAULT_07"
    FAULT_08 = "FAULT_08"
    FAULT_09 = "FAULT_09"
    FAULT_10 = "FAULT_10"
    FAULT_11 = "FAULT_11"
    FAULT_12 = "FAULT_12"
    FAULT_13 = "FAULT_13"
    FAULT_14 = "FAULT_14"
    FAULT_15 = "FAULT_15"
    FAULT_16 = "FAULT_16"
    FAULT_17 = "FAULT_17"
    FAULT_18 = "FAULT_18"
    FAULT_19 = "FAULT_19"
    FAULT_20 = "FAULT_20"
    UNCLASSIFIED_OPEN_SET = "UNCLASSIFIED_OPEN_SET"
    NORMAL_TRANSIENT = "NORMAL_TRANSIENT"
    SYNTHETIC_FAULT_21 = "SYNTHETIC_FAULT_21"
    SYNTHETIC_FAULT_22 = "SYNTHETIC_FAULT_22"


def fault_id_for(fault_number: int) -> FaultID:
    """Map a fault number to its benchmark id.

    Faults 16-20 collapse to ``UNCLASSIFIED_OPEN_SET``; 21/22 become the
    synthetic members. An unknown number raises instead of guessing.
    """
    number = int(fault_number)
    if number in OPEN_SET_FAULTS:
        return FaultID.UNCLASSIFIED_OPEN_SET
    if number in SYNTHETIC_FAULTS:
        return FaultID(f"SYNTHETIC_FAULT_{number:02d}")
    if number in HEADLINE_FAULTS:
        return FaultID(f"FAULT_{number:02d}")
    raise ValueError(f"unknown fault number: {fault_number}")


def kb_entry(fault_number: int) -> Optional[Dict]:
    """KB entry for a fault number, or ``None`` when the KB does not define it."""
    return FAULT_KNOWLEDGE.get(int(fault_number))


def is_synthetic(fault_number: int) -> bool:
    return int(fault_number) in SYNTHETIC_FAULTS


def headline_eligible(fault_number: int) -> bool:
    """True only for real faults: 21/22 are synthetic and never headline."""
    return int(fault_number) in HEADLINE_FAULTS


def is_unobservable(fault_number: int) -> bool:
    return int(fault_number) in UNOBSERVABLE_FAULTS


def fault_table() -> List[Dict[str, object]]:
    """Number -> id -> KB entry view over the whole space, for reports and tests."""
    table: List[Dict[str, object]] = []
    for number in HEADLINE_FAULTS + SYNTHETIC_FAULTS:
        entry = kb_entry(number) or {}
        table.append(
            {
                "faultNumber": number,
                "id": fault_id_for(number).value,
                "kb_name": entry.get("name"),
                "kb_subsystem": entry.get("subsystem"),
                "open_set": number in OPEN_SET_FAULTS,
                "unobservable": number in UNOBSERVABLE_FAULTS,
                "synthetic": number in SYNTHETIC_FAULTS,
                "headline_eligible": number in HEADLINE_FAULTS,
                "evidence_label": "synthetic" if number in SYNTHETIC_FAULTS else None,
            }
        )
    return table


def validate() -> List[str]:
    """Return a list of problems with the derivation; empty means consistent."""
    problems: List[str] = []
    missing = [n for n in HEADLINE_FAULTS + SYNTHETIC_FAULTS if n not in FAULT_KNOWLEDGE]
    if missing:
        problems.append(f"KB has no entry for fault(s) {missing}")
    extra = sorted(set(FAULT_KNOWLEDGE) - set(HEADLINE_FAULTS) - set(SYNTHETIC_FAULTS))
    if extra:
        problems.append(f"KB defines fault(s) outside the label space: {extra}")
    for number in OPEN_SET_FAULTS:
        name = str((kb_entry(number) or {}).get("name", ""))
        if not name.lower().startswith("unknown disturbance"):
            problems.append(f"fault {number} is treated as open-set but is named {name!r}")
    for number in SYNTHETIC_FAULTS:
        if headline_eligible(number):
            problems.append(f"synthetic fault {number} is headline-eligible")
    for number in HEADLINE_FAULTS:
        if not kb_entry(number):
            continue
        try:
            fault_id_for(number)
        except ValueError as exc:  # pragma: no cover - defensive
            problems.append(str(exc))
    return problems
