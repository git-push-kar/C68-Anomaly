"""The re-specified artefact checks (PROTOCOL 28) on a hand-labelled case set."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.pilot_rechecks import (anchor_status, calibration, contradiction_flags,  # noqa: E402
                                    numeric_flags, run_case, tag_role_flags)
from scripts.recheck_cases import CASES, FAULT_STORE  # noqa: E402


def test_forty_cases_with_known_labels():
    assert len(CASES) == 40
    wrong = [case for case in CASES if run_case(case) != bool(case["expect"])]
    assert wrong == []


def test_calibration_is_exact_on_every_check():
    report = calibration()
    assert set(report) == {"contradiction", "tag_role", "anchor", "numeric"}
    for name, entry in report.items():
        assert entry["precision"] == 1.0, (name, entry)
        assert entry["recall"] == 1.0, (name, entry)


def test_known_bad_bindings_are_flagged():
    class_flag = tag_role_flags("The Condenser_Cooling_Water_Outlet_Temperature (XMEAS_17) rises.")
    assert [flag["kind"] for flag in class_flag] == ["class_mismatch"]
    name_flag = tag_role_flags("The condenser valve (XMV_47) is stuck.")
    assert [flag["kind"] for flag in name_flag] == ["name_does_not_match_tag"]


def test_correct_bindings_are_not_flagged():
    for text in ("A_Feed_Stream1 (XMEAS_1) decreases.",
                 "Stripper_Steam_Valve (XMV_50) opens.",
                 "Reactor_Temperature (XMEAS_9) rises."):
        assert tag_role_flags(text) == []


def test_lstm_inequality_is_a_direction_failure():
    flags = numeric_flags("The LSTM reconstruction error is 0.6971, below the threshold 0.6870.",
                          FAULT_STORE)
    assert [flag["kind"] for flag in flags] == ["wrong_inequality_direction"]
    assert numeric_flags("The LSTM reconstruction error is 0.6971, above the threshold 0.6870.",
                         FAULT_STORE) == []


def test_rejecting_a_differential_is_not_a_contradiction():
    assert contradiction_flags("This pattern contradicts FAULT_06 (A feed loss), which would show "
                               "A_Feed_Flow decreasing first.", "FAULT_01", "classified") == []
    assert contradiction_flags("FAULT_07 is ruled out by the separator pressure.",
                               "FAULT_01", "classified") == []
    assert contradiction_flags("The event is caused by FAULT_07.", "FAULT_01",
                               "classified") != []


def test_verdict_claims_are_compared_to_the_supplied_verdict():
    assert contradiction_flags("The verdict is no_fault.", "FAULT_06", "classified") != []
    assert contradiction_flags("The verdict is classified.", "FAULT_06", "classified") == []


def test_anchor_tolerance_is_one_hundredth():
    assert anchor_status("XMEAS_1 at z=18.11.", "XMEAS_1", 18.12)[0] == "exact"
    assert anchor_status("XMEAS_1 at z=18.10.", "XMEAS_1", 18.12)[0] == "contradicting_value"
    assert anchor_status("XMEAS_1 at z=18.12.", "XMEAS_1", 18.12)[0] == "exact"
    assert anchor_status("A deviation of 222.71%.", "XMEAS_1", 18.12)[0] == "sensor_absent"
    assert anchor_status("XMEAS_1 with a 222.71% deviation.", "XMEAS_1", 18.12)[0] == "z_omitted"
