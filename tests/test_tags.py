"""Tests for scripts/tags.py and a tag-validity audit over the whole KB."""
from __future__ import annotations

import re

import pytest

from scripts.fault_knowledge import FAULT_KNOWLEDGE
from scripts.tags import (
    XMV_INDEX_OFFSET,
    canonical_tag,
    describe,
    display_name,
    fault_tags,
    is_canonical,
    iter_tags,
    manipulated_index,
    subsystem,
)

RAW_TAG = re.compile(r"(?<![A-Za-z0-9])(XMEAS|XMV)[ _-]?(\d{1,2})(?![0-9])", re.IGNORECASE)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("XMV_52", "XMV_52"),
        ("XMV 11", "XMV_52"),
        ("XMV11", "XMV_52"),
        ("xmv_11", "XMV_52"),
        ("XMV_45", "XMV_45"),
        ("XMV 4", "XMV_45"),
        ("xmv_4", "XMV_45"),
        ("XMV_42", "XMV_42"),
        ("XMV 1", "XMV_42"),
        ("XMEAS_4", "XMEAS_4"),
        ("XMEAS 4", "XMEAS_4"),
        ("xmeas_4", "XMEAS_4"),
    ],
)
def test_canonical_forms(text, expected):
    assert canonical_tag(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "XMV_52 (Condenser Cooling Water Flow valve / XMV 11)",
        "XMV_52 / XMEAS_22",
    ],
)
def test_composite_kb_strings_resolve_to_first_canonical(text):
    assert canonical_tag(text) == "XMV_52"


def test_alias_offset_is_the_documented_one():
    # XMV 11 -> XMV_52, XMV 1 -> XMV_42
    assert XMV_INDEX_OFFSET == 41
    assert manipulated_index("XMV_52") == 11
    assert manipulated_index("XMV 1") == 1
    assert manipulated_index("XMEAS_4") is None


@pytest.mark.parametrize("text", ["XMV_99", "XMV 12", "XMEAS_42", "XMEAS 0", "nonsense", "", None, 42])
def test_unresolvable_returns_none(text):
    assert canonical_tag(text) is None


def test_iter_tags_dedupes_composite_self_alias():
    assert iter_tags("XMV_45 (A and C Feed Flow valve / XMV 4)") == ["XMV_45"]
    assert iter_tags("XMV_45 / XMEAS_4") == ["XMV_45", "XMEAS_4"]
    assert iter_tags("no tags here") == []


def test_describe_and_lookups():
    assert display_name("XMV 11") == "Condenser_Cooling_Water_Flow"
    assert subsystem("XMV 11") == "condenser_cooling_system"
    assert is_canonical("XMV_52") is True
    assert is_canonical("XMV 11") is False
    assert describe("XMV 11") == {
        "tag": "XMV_52",
        "display_name": "Condenser_Cooling_Water_Flow",
        "subsystem": "condenser_cooling_system",
        "manipulated_index": 11,
    }


@pytest.mark.parametrize(
    "fault_id,expected",
    [
        (1, ["XMV_45", "XMEAS_4"]),
        (2, ["XMV_47", "XMEAS_24"]),
        (4, ["XMV_51", "XMEAS_21"]),
        (5, ["XMV_52", "XMEAS_22"]),
        (14, ["XMV_51", "XMEAS_21"]),
        (15, ["XMV_52", "XMEAS_22"]),
    ],
)
def test_fault_tags(fault_id, expected):
    assert fault_tags(fault_id) == expected


def test_fault_tags_unknown_fault_is_empty():
    assert fault_tags(999) == []


def test_kb_tag_validity_audit():
    """Every tag string in the KB must resolve, and never ambiguously.

    A string is *invalid* when it contains a tag-shaped token that cannot be
    resolved (e.g. XMV 12). It is *ambiguous* when one string names two
    different canonical tags of the same family (e.g. XMV_45 ... XMV 7), which
    would mean the KB contradicts itself.
    """
    invalid = []
    ambiguous = []
    referenced = set()
    print("\nKB tag audit (faults 1-22):")
    for fault_id in sorted(FAULT_KNOWLEDGE):
        entry = FAULT_KNOWLEDGE[fault_id]
        strings = [("valves", s) for s in entry.get("valves") or []]
        strings += [("initiating.tag", i["tag"]) for i in entry.get("initiating") or [] if i.get("tag")]
        fault_tags_resolved = fault_tags(fault_id)
        referenced.update(fault_tags_resolved)
        for field, text in strings:
            tokens = [f"{family.upper()}_{number}" for family, number in RAW_TAG.findall(text)]
            resolved = [canonical_tag(token) for token in tokens]
            missing = [token for token, value in zip(tokens, resolved) if value is None]
            if missing:
                invalid.append((fault_id, field, text, missing))
            by_family = {}
            for value in resolved:
                if value:
                    by_family.setdefault(value[:4], set()).add(value)
            clashes = [sorted(values) for values in by_family.values() if len(values) > 1]
            if clashes:
                ambiguous.append((fault_id, field, text, clashes))
        print(f"  fault {fault_id:2d}: {fault_tags_resolved or '(no tags)'}")

    print(f"\n  distinct canonical tags referenced: {len(referenced)} -> {sorted(referenced)}")
    print(f"  invalid: {len(invalid)}  ambiguous: {len(ambiguous)}")
    assert not invalid, f"invalid KB tags: {invalid}"
    assert not ambiguous, f"ambiguous KB tags: {ambiguous}"
