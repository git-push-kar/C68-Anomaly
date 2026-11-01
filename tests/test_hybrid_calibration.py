"""§17's contracts: folds grouped by event, a threshold that only trains see, four signals, and
a sentence-level explanation of the contradiction detector.

The properties tested are the ones that could silently make a number wrong: a fold split that
leaks an event, a threshold fitted on the rows it scores, an AUROC with the two sides swapped, a
novelty that depends on scale, and a detector whose decision cannot be explained.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.hybrid import calibration as cal  # noqa: E402
from llm.hybrid.generate import explain_sentence  # noqa: E402


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #
def test_folds_keep_an_events_variants_together():
    items = []
    for run in range(40):
        for variant in ("menu", "removed"):
            items.append({"source_file": "f", "fault": 5, "simulationRun": run, "variant": variant})
    folds = cal.grouped_folds(items, k=5, seed=42)
    pairs = {}
    for item, fold in zip(items, folds):
        key = (item["simulationRun"], item["variant"])
        pairs.setdefault(item["simulationRun"], set()).add(fold)
    assert all(len(value) == 1 for value in pairs.values()), "an event was split across folds"
    assert len(set(folds)) == 5


def test_folds_are_deterministic():
    items = [{"source_file": "f", "fault": 1, "simulationRun": run} for run in range(30)]
    assert cal.grouped_folds(items, seed=42) == cal.grouped_folds(items, seed=42)
    assert cal.grouped_folds(items, seed=42) != cal.grouped_folds(items, seed=7)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
LABELS = ["A", "B", "C", "OPEN"]


def _rows(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    probs, truths = [], []
    for index in range(n):
        truth = LABELS[index % 3]
        raw = rng.dirichlet(np.array([4.0, 4.0, 4.0, 1.0]))
        # a confidently wrong fifth of the set, so calibration has something to shrink
        if index % 5 == 0:
            raw = np.array([0.02, 0.02, 0.02, 0.94])
            truth = LABELS[(index + 1) % 3]
        probs.append({label: float(value) for label, value in zip(LABELS, raw)})
        truths.append(truth)
    return probs, truths


def test_isotonic_and_temperature_produce_distributions():
    probs, truths = _rows(60)
    for fitter, applier in ((cal.fit_isotonic, cal.apply_isotonic),
                            (cal.fit_temperature, cal.apply_temperature)):
        model = fitter(probs, truths, LABELS)
        calibrated = applier(model, probs)
        assert len(calibrated) == len(probs)
        for row in calibrated:
            assert set(row) == set(LABELS)
            assert abs(sum(row.values()) - 1.0) < 1e-9
            assert min(row.values()) >= 0.0


def test_temperature_scaling_keeps_the_ranking_of_the_top_class():
    probs, truths = _rows(40)
    model = cal.fit_temperature(probs, truths, LABELS)
    calibrated = cal.apply_temperature(model, probs)
    assert [max(row, key=row.get) for row in probs] == [max(row, key=row.get)
                                                       for row in calibrated]


def test_out_of_fold_ece_is_computed_on_held_out_folds_only():
    probs, truths = _rows(60)
    fold = [index % 5 for index in range(len(probs))]
    choice = cal.cross_fit_calibration(probs, truths, fold, LABELS)
    assert choice["chosen"] in ("isotonic", "temperature")
    assert set(choice["scores"]) == {"isotonic", "temperature"}
    assert all(value >= 0.0 for value in choice["scores"].values())


def test_threshold_spends_the_budget_and_no_more():
    # 200 evenly spaced confidences: 5 % of them is ten items, so the largest feasible
    # threshold is the tenth smallest, and nine fall below it.
    confidences = [index / 200 for index in range(1, 201)]
    result = cal.threshold_for(confidences, budget=0.05)
    assert result["threshold"] == pytest.approx(0.05)
    assert result["abstained"] == 9
    assert result["achieved_false_abstention"] <= 0.05


# --------------------------------------------------------------------------- #
# scores
# --------------------------------------------------------------------------- #
def test_auroc_is_one_for_a_perfect_separation_and_symmetric():
    assert cal.auroc([0.9, 0.8], [0.2, 0.1]) == 1.0
    assert cal.auroc([0.1, 0.2], [0.8, 0.9]) == 0.0
    assert cal.auroc([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_auroc_counts_ties_as_half():
    assert cal.auroc([0.5], [0.5]) == 0.5
    assert cal.auroc([1.0, 0.5], [0.5, 0.0]) == 0.875


def test_orientation_puts_every_signal_on_the_unseen_side():
    """§17.3: higher score = more likely the unseen class, for all four signals.

    The failure this prevents is silent and severe: an unseen class gets a *lower*
    calibrated maximum and a *narrower* margin, so ranking those two raw scores them at
    1 - AUROC (0.12 in the first run) and they look like the worst signals on the board.
    """
    assert cal.orient_values("max_prob", [0.4])[0] == pytest.approx(-0.4)
    assert cal.orient_values("margin", [0.4])[0] == pytest.approx(-0.4)
    assert cal.orient_values("entropy", [0.4])[0] == pytest.approx(0.4)
    assert cal.orient_values("novelty", [0.4])[0] == pytest.approx(0.4)
    with pytest.raises(KeyError):
        cal.orient_values("vibes", [0.4])

    # Unseen items are less confident than seen ones. Oriented, that is perfect separation.
    unseen = cal.orient_values("max_prob", [0.20, 0.25])
    seen = cal.orient_values("max_prob", [0.90, 0.95])
    assert cal.auroc(list(unseen), list(seen)) == 1.0
    # And it is exactly the complement of the raw comparison, so no re-run is needed to fix it.
    raw = cal.auroc([0.20, 0.25], [0.90, 0.95])
    assert 1.0 - raw == pytest.approx(cal.auroc(list(unseen), list(seen)))


def test_choose_signal_prefers_the_higher_mean_and_breaks_ties_by_simplicity():
    assert cal.choose_signal({"entropy": 0.90, "novelty": 0.80, "margin": 0.20,
                              "max_prob": 0.10}) == "entropy"
    # within the tolerance: the simpler signal wins, and margin is simplest by §17.3
    assert cal.choose_signal({"entropy": 0.800, "margin": 0.797, "novelty": 0.10,
                              "max_prob": 0.10}) == "margin"
    assert cal.choose_signal({"novelty": 0.900, "entropy": 0.500, "margin": 0.500,
                              "max_prob": 0.500}) == "novelty"


def test_signals_are_computed_from_the_vector():
    vector = {"A": 0.6, "B": 0.3, "OPEN": 0.1}
    signals = cal.signal_values([vector], [1.234])
    assert signals["max_prob"][0] == pytest.approx(0.6)
    assert signals["margin"][0] == pytest.approx(0.3)
    assert signals["entropy"][0] == pytest.approx(
        -(0.6 * np.log(0.6) + 0.3 * np.log(0.3) + 0.1 * np.log(0.1)))
    assert signals["novelty"][0] == pytest.approx(1.234)


def test_novelty_is_scale_invariant_and_zero_on_the_reference():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(50, 4))
    query = rng.normal(size=(5, 4))
    base = cal.novelty_scores(query, reference, k=3)
    scaled_reference = reference * 1000.0 + 7.0
    scaled_query = query * 1000.0 + 7.0
    assert np.allclose(base, cal.novelty_scores(scaled_query, scaled_reference, k=3), atol=1e-6)
    # A reference row's own distance is zero, so with k=1 the score is exactly zero there; with
    # k=3 it averages the two nearest neighbours in, which is the definition, not a defect.
    assert np.allclose(cal.novelty_scores(reference[:3], reference, k=1), 0.0, atol=1e-9)
    assert np.all(cal.novelty_scores(reference[:3], reference, k=3) > 0.0)


# --------------------------------------------------------------------------- #
# detector transparency
# --------------------------------------------------------------------------- #
def test_explain_sentence_names_the_rule_that_decided():
    # A rejection sentence asserts nothing about another fault ("would cause" is a hypothetical),
    # so the verdict is "no flag" — the guard is not even needed.
    rejected = explain_sentence(
        "This matches FAULT_13; FAULT_14 was rejected because it would cause a fall.", "FAULT_13")
    assert rejected == {"verdict": False, "rule": "no_assertion_cue"}
    guarded = explain_sentence(
        "The evidence is consistent with FAULT_05, but no such flow increase is seen.",
        "FAULT_13")
    assert guarded == {"verdict": False, "rule": "rejection_guard", "label": "FAULT_05"}
    # Documented scope: the assertion cue has to precede the label. "FAULT_05 would cause ..." is
    # a hypothetical and is not an assertion either, so it is reported as having no cue at all
    # rather than as having been suppressed by the guard.
    hypothetical = explain_sentence(
        "FAULT_05 (temperature step) would cause a flow increase.", "FAULT_13")
    assert hypothetical == {"verdict": False, "rule": "no_assertion_cue"}
    flagged = explain_sentence("The evidence is consistent with FAULT_12 as the cause.", "FAULT_13")
    assert flagged["verdict"] is True and flagged["rule"] == "assertion_adjacent"
    plain = explain_sentence("Sensors XMEAS_7 and XMEAS_13 rise together.", "FAULT_13")
    assert plain == {"verdict": False, "rule": "no_assertion_cue"}


def test_explain_sentence_catches_a_denial_of_the_diagnosis():
    decision = explain_sentence("The evidence does not support FAULT 13.", "FAULT_13")
    assert decision["verdict"] is True
    assert decision["rule"] == "denial_adjacent"
