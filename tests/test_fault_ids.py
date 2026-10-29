"""Tests for scripts/fault_ids.py."""
from __future__ import annotations

import pytest

from scripts.fault_ids import (
    HEADLINE_FAULTS,
    OPEN_SET_FAULTS,
    SYNTHETIC_FAULTS,
    UNOBSERVABLE_FAULTS,
    FaultID,
    fault_id_for,
    fault_table,
    headline_eligible,
    is_synthetic,
    is_unobservable,
    kb_entry,
    validate,
)
from scripts.fault_knowledge import FAULT_KNOWLEDGE


def test_label_space_covers_the_kb():
    assert validate() == []
    assert sorted(FAULT_KNOWLEDGE) == sorted(HEADLINE_FAULTS + SYNTHETIC_FAULTS)


@pytest.mark.parametrize("number", range(1, 16))
def test_individually_diagnosable_faults_keep_their_id(number):
    assert fault_id_for(number) == FaultID(f"FAULT_{number:02d}")


@pytest.mark.parametrize("number", OPEN_SET_FAULTS)
def test_unknown_disturbance_faults_collapse_to_open_set(number):
    assert fault_id_for(number) == FaultID.UNCLASSIFIED_OPEN_SET
    assert kb_entry(number)["name"].lower().startswith("unknown disturbance")


@pytest.mark.parametrize("number", SYNTHETIC_FAULTS)
def test_synthetic_faults_are_non_headline(number):
    assert fault_id_for(number) == FaultID(f"SYNTHETIC_FAULT_{number:02d}")
    assert is_synthetic(number) is True
    assert headline_eligible(number) is False


def test_unknown_number_raises():
    with pytest.raises(ValueError):
        fault_id_for(99)
    with pytest.raises(ValueError):
        fault_id_for(0)


def test_unobservable_flags():
    assert sorted(UNOBSERVABLE_FAULTS) == [3, 9]
    for number in UNOBSERVABLE_FAULTS:
        assert is_unobservable(number) is True
        assert headline_eligible(number) is True  # flagged, not dropped
    assert is_unobservable(4) is False


def test_fault_table_is_number_to_id_to_kb():
    table = fault_table()
    assert len(table) == 22
    for row in table:
        number = row["faultNumber"]
        assert row["id"] == fault_id_for(number).value
        assert row["kb_name"] == FAULT_KNOWLEDGE[number]["name"]
        assert row["kb_subsystem"] == FAULT_KNOWLEDGE[number]["subsystem"]
        if number in SYNTHETIC_FAULTS:
            assert row["synthetic"] is True
            assert row["headline_eligible"] is False
            assert row["evidence_label"] == "synthetic"
        else:
            assert row["synthetic"] is False
            assert row["headline_eligible"] is True
            assert row["evidence_label"] is None
