"""Tests for data/splits_v2.json: both protocols pass, and tampering is caught.

Each negative test mutates a deep copy of the committed document and asserts the
validator reports the specific violation, so the checker cannot silently pass a
broken split. The committed file itself is never modified.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.build_splits_v2 import build, render
from scripts.check_splits_v2 import REPO_ROOT, check_document, expand_split, key_digest

SPLITS_PATH = REPO_ROOT / "data" / "splits_v2.json"


@pytest.fixture()
def doc() -> dict:
    return json.loads(SPLITS_PATH.read_text(encoding="utf-8"))


def failures(doc: dict) -> str:
    return "\n".join(check_document(doc))


def test_committed_file_is_clean(doc):
    assert check_document(doc) == []


def test_committed_file_matches_the_builder(doc):
    assert render(build()) == SPLITS_PATH.read_text(encoding="utf-8"), (
        "data/splits_v2.json is stale; run python scripts/build_splits_v2.py"
    )


def test_key_contract_and_source(doc):
    assert doc["unique_key"] == ["source_file", "faultNumber", "simulationRun"]
    assert doc["source"]["file"] == "data/raw/faults/TEP_Faulty_Testing.csv"
    assert doc["source"]["samplesPerRun"] == 960
    assert doc["source"]["faultOnsetIndex"] == 160


def test_protocol_a_is_run_disjoint(doc):
    splits = doc["protocols"]["A_seen_faults_run_disjoint"]["splits"]
    assert splits["train"]["faults"][0]["simulationRunRange"] == [1, 300]
    assert splits["val"]["faults"][0]["simulationRunRange"] == [301, 400]
    assert splits["test"]["faults"][0]["simulationRunRange"] == [401, 500]
    for name in ("train", "val", "test"):
        assert key_digest(expand_split(splits[name])) == splits[name]["sha256"][name]


def test_protocol_b_fault_disjointness(doc):
    folds = doc["protocols"]["B_leave_faults_out"]["folds"]
    fold1 = folds["fold_1_cooling_loop_holdout"]
    fold2 = folds["fold_2_valve_sticking_family"]
    assert fold1["held_out_faults"] == [4, 11, 15]
    assert fold2["held_out_faults"] == [14, 15]
    for fold in (fold1, fold2):
        held = {e["faultNumber"] for e in fold["splits"]["test"]["faults"]}
        seen = {e["faultNumber"] for e in fold["splits"]["train"]["faults"]}
        assert held == set(fold["held_out_faults"])
        assert held.isdisjoint(seen)


def test_protocol_b_test_seen_component(doc):
    """test_seen = runs 401-500 of the kept faults; test stays the unseen component."""
    folds = doc["protocols"]["B_leave_faults_out"]["folds"]
    expected = {
        "fold_1_cooling_loop_holdout": {"faults": 17, "keys": 1700},
        "fold_2_valve_sticking_family": {"faults": 18, "keys": 1800},
    }
    for name, want in expected.items():
        splits = folds[name]["splits"]
        train_faults = {e["faultNumber"] for e in splits["train"]["faults"]}
        seen_entries = splits["test_seen"]["faults"]
        assert len(seen_entries) == want["faults"]
        assert sum(e["runCount"] for e in seen_entries) == want["keys"]
        for entry in seen_entries:
            assert entry["simulationRunRange"] == [401, 500]
            assert entry["faultNumber"] in train_faults  # subset of train faults
        # run-disjoint from train and val
        train_runs = {(e["faultNumber"], r) for e in splits["train"]["faults"] for r in range(1, 301)}
        seen_runs = {(e["faultNumber"], r) for e in seen_entries for r in range(401, 501)}
        assert train_runs.isdisjoint(seen_runs)
        # the unseen component is unchanged
        assert {e["faultNumber"] for e in splits["test"]["faults"]} == set(folds[name]["held_out_faults"])


def test_synthetic_faults_are_absent_from_every_split(doc):
    for label, splits, _ in [
        (name, p["splits"], {})
        for name, p in doc["protocols"].items()
        if "splits" in p
    ] + [
        (f"{name}/{fold_name}", fold["splits"], {})
        for name, p in doc["protocols"].items()
        for fold_name, fold in (p.get("folds") or {}).items()
    ]:
        for block in splits.values():
            faults = {entry["faultNumber"] for entry in block["faults"]}
            assert faults.isdisjoint({21, 22}), f"{label} contains a synthetic fault"


# ----------------------------------------------------------------------------
# negative tests: each mutation must be rejected
# ----------------------------------------------------------------------------


def test_rejects_train_test_run_overlap(doc):
    bad = copy.deepcopy(doc)
    bad["protocols"]["A_seen_faults_run_disjoint"]["splits"]["train"]["faults"].append(
        {"faultNumber": 1, "simulationRunRange": [401, 500], "runCount": 100}
    )
    assert "both train and test" in failures(bad)


def test_rejects_train_val_run_overlap(doc):
    bad = copy.deepcopy(doc)
    bad["protocols"]["A_seen_faults_run_disjoint"]["splits"]["train"]["faults"].append(
        {"faultNumber": 2, "simulationRunRange": [301, 400], "runCount": 100}
    )
    assert "both train and val" in failures(bad)


def test_rejects_duplicate_split_keys(doc):
    bad = copy.deepcopy(doc)
    bad["protocols"]["A_seen_faults_run_disjoint"]["splits"]["val"]["faults"].append(
        {"faultNumber": 3, "simulationRunRange": [301, 400], "runCount": 100}
    )
    assert "duplicate keys" in failures(bad)


def test_rejects_seen_fault_run_in_both_train_and_test_seen(doc):
    bad = copy.deepcopy(doc)
    bad["protocols"]["B_leave_faults_out"]["folds"]["fold_1_cooling_loop_holdout"]["splits"]["test_seen"][
        "faults"
    ].append({"faultNumber": 1, "simulationRunRange": [1, 300], "runCount": 300})
    text = failures(bad)
    assert "both train and test_seen" in text


def test_rejects_held_out_fault_inside_test_seen(doc):
    bad = copy.deepcopy(doc)
    bad["protocols"]["B_leave_faults_out"]["folds"]["fold_1_cooling_loop_holdout"]["splits"]["test_seen"][
        "faults"
    ].append({"faultNumber": 4, "simulationRunRange": [401, 500], "runCount": 100})
    text = failures(bad)
    # held-out fault 4 is neither a train fault nor allowed in a seen component
    assert "test_seen contains faults not present in train: [4]" in text
    assert "held-out test faults [4] also appear in train/val/test_seen" in text


def test_rejects_fault_leak_into_fold_train(doc):
    bad = copy.deepcopy(doc)
    bad["protocols"]["B_leave_faults_out"]["folds"]["fold_1_cooling_loop_holdout"]["splits"]["train"][
        "faults"
    ].append({"faultNumber": 4, "simulationRunRange": [1, 300], "runCount": 300})
    text = failures(bad)
    assert "fault_disjoint_across_splits" in text
    assert "[4]" in text


def test_rejects_hash_drift(doc):
    bad = copy.deepcopy(doc)
    bad["protocols"]["A_seen_faults_run_disjoint"]["splits"]["test"]["faults"][0][
        "simulationRunRange"
    ] = [401, 450]
    text = failures(bad)
    assert "sha256 drift" in text


def test_rejects_synthetic_fault_in_headline_split(doc):
    bad = copy.deepcopy(doc)
    bad["protocols"]["A_seen_faults_run_disjoint"]["splits"]["test"]["faults"].append(
        {"faultNumber": 21, "simulationRunRange": [1, 500], "runCount": 500}
    )
    assert "synthetic-only fault 21" in failures(bad)


def test_rejects_wrong_key_contract(doc):
    bad = copy.deepcopy(doc)
    bad["unique_key"] = ["split", "faultNumber", "simulationRun"]
    assert "unique_key" in failures(bad)


def test_rejects_run_outside_bounds(doc):
    bad = copy.deepcopy(doc)
    bad["protocols"]["A_seen_faults_run_disjoint"]["splits"]["train"]["faults"][0][
        "simulationRunRange"
    ] = [1, 900]
    assert "outside [1, 500]" in failures(bad)
