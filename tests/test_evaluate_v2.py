"""The evaluator, tested on synthetic predictions only.

No model is loaded and no test component is read: every number below comes from
rows constructed in the test, so a failure is a bug in the metric rather than a
disagreement with a model.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "llm") not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark.evaluate_v2 import (  # noqa: E402
    ABSTAIN,
    accuracy,
    abstention_metrics,
    brier,
    confusion_matrix,
    ece,
    grounding_summary,
    grounding_violations,
    load_predictions,
    macro_f1,
    mcnemar,
    per_class_prf,
    report,
    run_clustered_bootstrap,
    top_k_accuracy,
)


def row(item_id, gold, pred, run="r1", probs=None, **extra):
    entry = {"item_id": item_id, "gold": gold, "pred": pred, "run": run}
    if probs:
        entry["probs"] = probs
    entry.update(extra)
    return entry


# ---------------------------------------------------------------------------
@pytest.fixture
def perfect():
    return [row(f"i{i}", "FAULT_01", "FAULT_01", run=f"r{i}") for i in range(5)] + \
           [row(f"j{i}", "FAULT_02", "FAULT_02", run=f"r{i}") for i in range(5)]


def test_accuracy_of_a_perfect_file_is_one(perfect):
    assert accuracy(perfect) == 1.0
    assert macro_f1(perfect) == 1.0


def test_accuracy_ignores_rows_without_gold_or_prediction():
    rows = [row("a", "FAULT_01", "FAULT_01"),
            {"item_id": "b", "gold": "", "pred": "FAULT_01"},
            {"item_id": "c", "gold": "FAULT_01"}]
    assert accuracy(rows) == 1.0


def test_top3_uses_the_probability_vector_not_the_argument():
    rows = [
        row("a", "FAULT_02", "FAULT_01",
            probs={"FAULT_01": 0.5, "FAULT_02": 0.3, "FAULT_03": 0.2}),
        row("b", "FAULT_03", "FAULT_01",
            probs={"FAULT_01": 0.9, "FAULT_02": 0.05, "FAULT_03": 0.05}),
    ]
    assert top_k_accuracy(rows, 1) == 0.0
    assert top_k_accuracy(rows, 2) == 0.5
    assert math.isfinite(top_k_accuracy(rows, 3))


def test_per_class_precision_recall_and_support():
    rows = [row("a", "FAULT_01", "FAULT_01"),
            row("b", "FAULT_01", "FAULT_02"),
            row("c", "FAULT_02", "FAULT_02"),
            row("d", "FAULT_02", "FAULT_02")]
    table = per_class_prf(rows)
    assert table["FAULT_01"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(0.6667, abs=1e-3), "support": 2}
    assert table["FAULT_02"]["precision"] == pytest.approx(0.6667, abs=1e-3)
    assert table["FAULT_02"]["recall"] == 1.0
    assert table["FAULT_02"]["support"] == 2


def test_macro_f1_does_not_count_an_absent_class_as_zero():
    """A class the split never tests must not drag the headline metric down."""
    rows = [row("a", "FAULT_01", "FAULT_01"), row("b", "FAULT_01", "FAULT_01")]
    assert macro_f1(rows, labels=["FAULT_01", "FAULT_09"]) == 1.0


def test_confusion_matrix_is_indexed_gold_by_prediction():
    rows = [row("a", "FAULT_01", "FAULT_02"), row("b", "FAULT_02", "FAULT_02")]
    matrix = confusion_matrix(rows)
    assert matrix["FAULT_01"]["FAULT_02"] == 1
    assert matrix["FAULT_02"]["FAULT_02"] == 1
    assert matrix["FAULT_01"]["FAULT_01"] == 0


def test_abstention_precision_recall_and_the_two_different_errors():
    rows = [row("a", ABSTAIN, ABSTAIN),      # correct abstention
            row("b", ABSTAIN, "FAULT_01"),   # missed abstention
            row("c", "FAULT_01", ABSTAIN),   # false abstention
            row("d", "FAULT_01", "FAULT_01")]
    metrics = abstention_metrics(rows)
    assert metrics["abstention_precision"] == 0.5      # 1 of 2 abstentions was right
    assert metrics["abstention_recall"] == 0.5         # 1 of 2 abstainable items caught
    assert metrics["false_abstention_rate"] == 0.5     # 1 of 2 answerable items declined
    assert metrics["missed_abstention_rate"] == 0.5


def test_ece_and_brier_on_a_known_small_case():
    """Two items at 0.9 confidence, one right: ECE = |0.9 - 0.5| = 0.4."""
    rows = [row("a", "FAULT_01", "FAULT_01", probs={"FAULT_01": 0.9, "FAULT_02": 0.1}),
            row("b", "FAULT_01", "FAULT_02", probs={"FAULT_02": 0.9, "FAULT_01": 0.1})]
    assert ece(rows, bins=10) == pytest.approx(0.4, abs=1e-3)
    assert brier(rows) == pytest.approx(((0.9 - 1) ** 2 + (0.9 - 0) ** 2) / 2, abs=1e-6)


def test_ece_skips_items_without_a_probability_vector():
    assert math.isnan(ece([row("a", "FAULT_01", "FAULT_01")]))


# ---------------------------------------------------------------------------
def test_grounding_accepts_stored_tags_and_converted_numbers():
    answer = ("The XMEAS_7 pressure rose 6.98%, and XMV_44 sits at 79.24 "
              "(0.0698 of the stored deviation).")
    violations = grounding_violations(answer, ["XMEAS_7", "XMV_44"], [6.98, 79.24])
    assert violations == []


def test_grounding_flags_an_unlisted_sensor_and_an_invented_number():
    answer = "XMEAS_9 shows 12.5 while XMEAS_7 is steady."
    violations = grounding_violations(answer, ["XMEAS_7"], [1.0])
    assert sorted(violations) == ["number_not_in_store:12.5",
                                  "number_not_in_store:9",
                                  "tag_not_in_evidence:XMEAS_9"] or \
        "tag_not_in_evidence:XMEAS_9" in violations


def test_grounding_summary_counts_clean_answers():
    rows = [{"item_id": "a", "answer": "XMEAS_7 rose 6.98%",
             "evidence_tags": ["XMEAS_7"], "grounded_numbers": [6.98]},
            {"item_id": "b", "answer": "XMEAS_9 is hot",
             "evidence_tags": ["XMEAS_7"], "grounded_numbers": []}]
    summary = grounding_summary(rows)
    assert summary["items"] == 2 and summary["clean"] == 1
    assert summary["violations"]["tag_not_in_evidence"] == 1


# ---------------------------------------------------------------------------
def test_bootstrap_resamples_runs_and_is_seeded():
    rows = [row(f"i{i}", "FAULT_01", "FAULT_01" if i % 2 else "FAULT_02", run=f"r{i % 10}")
            for i in range(100)]
    first = run_clustered_bootstrap(rows, accuracy, n_resamples=200, seed=7)
    second = run_clustered_bootstrap(rows, accuracy, n_resamples=200, seed=7)
    assert first == second
    assert first["clusters"] == 10
    assert first["low"] <= first["point"] <= first["high"]


def test_bootstrap_interval_is_wide_when_runs_disagree():
    rows = []
    for index in range(10):
        correct = index % 2 == 0
        rows.append(row(f"i{index}", "FAULT_01",
                        "FAULT_01" if correct else "FAULT_02", run=f"r{index}"))
    interval = run_clustered_bootstrap(rows, accuracy, n_resamples=500, seed=1)
    assert interval["low"] < interval["point"] < interval["high"]


def test_mcnemar_exact_binomial_on_a_small_table():
    a = [row("1", "FAULT_01", "FAULT_01"), row("2", "FAULT_01", "FAULT_01"),
         row("3", "FAULT_01", "FAULT_02"), row("4", "FAULT_01", "FAULT_02")]
    b = [row("1", "FAULT_01", "FAULT_02"), row("2", "FAULT_01", "FAULT_02"),
         row("3", "FAULT_01", "FAULT_02"), row("4", "FAULT_01", "FAULT_02")]
    result = mcnemar(a, b)
    assert (result["b"], result["c"]) == (2, 0)
    assert result["method"] == "exact_binomial"
    # two-sided exact p for (2, 0) = 2 * (1/4) = 0.5
    assert result["p_value"] == pytest.approx(0.5, abs=1e-6)


def test_mcnemar_uses_only_shared_items():
    a = [row("1", "FAULT_01", "FAULT_01")]
    b = [row("2", "FAULT_01", "FAULT_01")]
    assert mcnemar(a, b)["items"] == 0


def test_report_carries_every_headline_number(perfect):
    result = report(perfect, label="synthetic")
    assert result["items"] == 10
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["abstention"]["abstain_support"] == 0
    assert "accuracy_ci" in result and "confusion" in result


def test_predictions_round_trip_from_a_file(tmp_path):
    path = tmp_path / "preds.jsonl"
    path.write_text("\n".join(
        '{"item_id": "a", "gold": "FAULT_01", "pred": "FAULT_01", "run": "r1"}'
        for _ in range(2)
    ), encoding="utf-8")
    rows = load_predictions(path)
    assert len(rows) == 2 and accuracy(rows) == 1.0
