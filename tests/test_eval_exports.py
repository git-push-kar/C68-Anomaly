"""The evaluation view must not carry the answer, and the folds must not leak.

Two claims are worth testing rather than asserting in prose:

* an evaluation row is prompt + label + component only -- no prose, no target
  dict, nothing a scorer could accidentally read as a reference answer;
* a Protocol B fold's training file contains no held-out fault, neither as the
  label nor inside the ranking.

The dataset on disk is a build artifact, so the file-based tests skip when it is
absent instead of failing in a checkout that has not built it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark import to_sft  # noqa: E402
from scripts import build_eval_exports as E  # noqa: E402

DATASET = ROOT / "outputs" / "benchmark_v2" / "dataset"
HAVE_DATA = (DATASET / "val.jsonl").exists()
HAVE_SFT = (ROOT / "outputs" / "benchmark_v2" / "sft" / "train.jsonl").exists()


def test_the_component_map_covers_both_protocols():
    comps = E.components()
    for name in ("A/val", "A/test", "fold_1/train", "fold_1/val", "fold_1/test",
                 "fold_1/test_seen", "fold_2/train", "fold_2/val", "fold_2/test",
                 "fold_2/test_seen"):
        assert name in comps, name
        assert comps[name], name


def test_a_held_out_fault_is_in_the_fold_test_component_for_every_run():
    """The held-out component spans all 500 runs: there is nothing to train on."""
    comps = E.components()
    held = comps["fold_1/test"]
    for run in (1, 150, 300, 400, 500):
        assert f"TEP_Faulty_Testing.csv|4|{run}" in held
    # and no kept fault appears there
    assert not any(key.split("|")[1] == "1" for key in held)


def test_fold_faults_are_read_from_the_split_file():
    assert to_sft.fold_faults("fold_1_cooling_loop_holdout") == [4, 11, 15]
    assert to_sft.fold_faults("fold_2_valve_sticking_family") == [14, 15]


def test_the_fault_number_helper_handles_every_label_shape():
    assert to_sft._fault_number("FAULT_07") == 7
    assert to_sft._fault_number("SYNTHETIC_FAULT_21") == 21
    assert to_sft._fault_number("UNCLASSIFIED_OPEN_SET") is None
    assert to_sft._fault_number(None) is None


@pytest.mark.skipif(not HAVE_DATA, reason="dataset not built in this checkout")
def test_an_eval_row_carries_no_answer_text():
    allowed = {"item_id", "system", "question", "label", "variant", "component",
               "unobservable"}
    with (DATASET / "val_eval.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert rows
    for row in rows[:200]:
        assert set(row) <= allowed, set(row) - allowed
        assert "prose" not in row and "target" not in row
        assert row["system"] and row["question"] and row["label"]


@pytest.mark.skipif(not HAVE_DATA, reason="dataset not built in this checkout")
def test_protocol_a_eval_rows_match_the_dataset_item_count():
    summary = json.loads((DATASET / "eval_export_summary.json").read_text(encoding="utf-8"))
    for split in ("val", "test"):
        assert summary[split]["matches_dataset"], split
        assert summary[split]["protocol_a_rows"] == summary[split]["dataset_items"]


@pytest.mark.skipif(not HAVE_SFT, reason="SFT export not built in this checkout")
@pytest.mark.parametrize("fold,held", [("fold_1_cooling_loop_holdout", {4, 11, 15}),
                                       ("fold_2_valve_sticking_family", {14, 15})])
def test_a_fold_train_file_names_no_held_out_fault(fold, held):
    path = ROOT / "outputs" / "benchmark_v2" / "sft" / f"train_{fold}.jsonl"
    if not path.exists():
        pytest.skip("fold export not built")
    for line in path.read_text(encoding="utf-8").splitlines():
        body = json.loads(json.loads(line)["answer"])
        label = body["fault_id"]
        if label.startswith("FAULT_"):
            assert int(label.split("_")[-1]) not in held
        for value in body["ranked_hypotheses"]:
            assert int(str(value).split("_")[-1]) not in held, value


@pytest.mark.skipif(not HAVE_SFT, reason="SFT export not built in this checkout")
def test_the_fold_files_are_smaller_than_the_full_train_file():
    base = ROOT / "outputs" / "benchmark_v2" / "sft" / "train.jsonl"
    if not base.exists():
        pytest.skip("train export not built")
    full = len(base.read_text(encoding="utf-8").splitlines())
    for fold in to_sft.FOLDS:
        path = ROOT / "outputs" / "benchmark_v2" / "sft" / f"train_{fold}.jsonl"
        if not path.exists():
            continue
        assert len(path.read_text(encoding="utf-8").splitlines()) < full
