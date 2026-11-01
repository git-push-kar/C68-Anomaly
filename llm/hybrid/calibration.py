"""Out-of-fold calibration, abstention thresholds and uncertainty signals (PROTOCOL §17).

Everything fitted in this module is fitted on **train events only**, out of fold, so that a
validation row is never used to choose a number that a validation row is then scored with. The
§16 path is untouched: `scripts/hybrid_classifier.py` keeps its own in-sample fitting exactly as
it was, because §16's artifacts are a baseline and re-deriving them would destroy the comparison.

The four signals exist because a classifier cannot say "I have never seen this". Each is a scalar
that rises when an item is unlike what the model was trained on, and §17.3 fixes in advance how
one of them is chosen: highest mean AUROC across the leave-one-fault-out surrogate, ties to the
simpler signal. Nothing here tunes a threshold per signal or per fault.

Novelty is the only signal that needs state from training (a reference set), and it says so by
taking that reference set as an argument rather than reading it from disk.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from llm.benchmark import evaluate_v2 as ev
from llm.hybrid.menu_rule import largest_threshold

SIGNALS = ("max_prob", "margin", "entropy", "novelty")
#: Fixed simplicity order for ties (§17.3): no fitted component first.
SIMPLICITY = ("margin", "entropy", "max_prob", "novelty")
TIE_TOLERANCE = 0.005

#: §17.3 fixes one orientation for all four signals: *higher = more likely the unseen class*.
#: Entropy and novelty already run that way. Calibrated confidence runs the other way -- a
#: class the model never saw gets a *lower* maximum probability and a *narrower* margin -- so
#: ranking those two without negating them scores them at 1 - AUROC and quietly rules them out.
#: The sign lives here, once, instead of in each caller.
ORIENTATION = {"max_prob": -1.0, "margin": -1.0, "entropy": 1.0, "novelty": 1.0}


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #
def event_key(item: Dict) -> str:
    """The event's identity, used only for grouping folds."""
    return "|".join(str(item.get(name)) for name in
                    ("source_file", "fault", "simulationRun"))


def grouped_folds(items: Sequence[Dict], k: int = 5, seed: int = 42) -> List[int]:
    """Fold index per item, with every event's variants in one fold.

    Grouping by event is not a refinement: an episode's two menu variants carry the same
    evidence, so splitting them across folds would leak an event into its own out-of-fold
    prediction and make the OOF estimate optimistic in exactly the direction that matters.
    """
    import random

    keys = [event_key(item) for item in items]
    unique = sorted(set(keys))
    random.Random(seed).shuffle(unique)
    assignment = {key: index % k for index, key in enumerate(unique)}
    return [assignment[key] for key in keys]


# --------------------------------------------------------------------------- #
# calibrators (same definitions as §16, different fitting data)
# --------------------------------------------------------------------------- #
def _vector(probs: Dict[str, float], labels: Sequence[str]) -> np.ndarray:
    return np.asarray([float(probs.get(label, 0.0)) for label in labels], dtype=np.float64)


def _softmax(matrix: np.ndarray) -> np.ndarray:
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def fit_isotonic(probs: Sequence[Dict], truths: Sequence[str], labels: Sequence[str]) -> Dict:
    from sklearn.isotonic import IsotonicRegression

    matrix = np.asarray([_vector(row, labels) for row in probs])
    curves = {}
    for index, label in enumerate(labels):
        target = np.asarray([1.0 if truth == label else 0.0 for truth in truths])
        if target.sum() == 0:
            curves[label] = {"kind": "constant", "value": 0.0}
            continue
        if target.sum() == len(target):
            curves[label] = {"kind": "constant", "value": 1.0}
            continue
        curve = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        curve.fit(matrix[:, index], target)
        curves[label] = {"kind": "isotonic",
                         "x": [float(value) for value in curve.X_thresholds_],
                         "y": [float(value) for value in curve.y_thresholds_]}
    return {"method": "isotonic", "labels": list(labels), "curves": curves}


def apply_isotonic(model: Dict, probs: Sequence[Dict]) -> List[Dict[str, float]]:
    labels = model["labels"]
    out = []
    for row in probs:
        vector = []
        for label in labels:
            curve = model["curves"][label]
            value = float(row.get(label, 0.0))
            if curve["kind"] == "constant":
                vector.append(curve["value"])
            else:
                vector.append(float(np.interp(value, curve["x"], curve["y"])))
        total = sum(vector) or 1.0
        out.append({label: value / total for label, value in zip(labels, vector)})
    return out


def fit_temperature(probs: Sequence[Dict], truths: Sequence[str], labels: Sequence[str]) -> Dict:
    from scipy.optimize import minimize_scalar

    matrix = np.asarray([_vector(row, labels) for row in probs])
    logits = np.log(np.clip(matrix, 1e-12, 1.0))

    def objective(temperature: float) -> float:
        scaled = _softmax(logits / max(temperature, 1e-3))
        rows = [{label: float(value) for label, value in zip(labels, row)} for row in scaled]
        return -sum(math.log(max(float(row.get(truth, 0.0)), 1e-12))
                    for row, truth in zip(rows, truths)) / max(len(rows), 1)

    result = minimize_scalar(objective, bounds=(0.05, 20.0), method="bounded")
    return {"method": "temperature", "labels": list(labels),
            "temperature": float(result.x), "nll": float(result.fun)}


def apply_temperature(model: Dict, probs: Sequence[Dict]) -> List[Dict[str, float]]:
    labels = model["labels"]
    matrix = np.asarray([_vector(row, labels) for row in probs])
    logits = np.log(np.clip(matrix, 1e-12, 1.0)) / max(model["temperature"], 1e-3)
    scaled = _softmax(logits)
    return [{label: float(value) for label, value in zip(labels, row)} for row in scaled]


FITTERS = {"isotonic": fit_isotonic, "temperature": fit_temperature}
APPLIERS = {"isotonic": apply_isotonic, "temperature": apply_temperature}


def apply_calibrator(model: Dict, probs: Sequence[Dict]) -> List[Dict[str, float]]:
    if model.get("method") == "identity":
        return [dict(row) for row in probs]
    return APPLIERS[model["method"]](model, probs)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def ece(probs: Sequence[Dict], truths: Sequence[str]) -> float:
    rows = [{"item_id": "x", "gold": truth, "pred": max(row, key=row.get), "probs": row}
            for row, truth in zip(probs, truths)]
    return float(ev.ece(rows)) if rows else float("nan")


def brier(probs: Sequence[Dict], truths: Sequence[str]) -> float:
    rows = [{"item_id": "x", "gold": truth, "pred": max(row, key=row.get), "probs": row}
            for row, truth in zip(probs, truths)]
    return float(ev.brier(rows)) if rows else float("nan")


def auroc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Rank AUROC, ties half. Returns NaN when one side is empty."""
    pos = [float(value) for value in positive]
    neg = [float(value) for value in negative]
    if not pos or not neg:
        return float("nan")
    combined = sorted([(value, 1) for value in pos] + [(value, 0) for value in neg])
    ranks: List[float] = []
    index = 0
    while index < len(combined):
        end = index
        while end + 1 < len(combined) and combined[end + 1][0] == combined[index][0]:
            end += 1
        average = (index + end) / 2.0 + 1.0  # 1-based average rank
        ranks.extend([average] * (end - index + 1))
        index = end + 1
    positive_rank = sum(rank for rank, (_value, flag) in zip(ranks, combined) if flag == 1)
    return (positive_rank - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #
def entropy_of(vector: Dict[str, float]) -> float:
    return float(-sum(p * math.log(p) for p in vector.values() if p > 0.0))


def top_two(vector: Dict[str, float]) -> Tuple[float, float]:
    ordered = sorted(vector.values(), reverse=True)
    while len(ordered) < 2:
        ordered.append(0.0)
    return ordered[0], ordered[1]


def novelty_scores(features: np.ndarray, reference: np.ndarray, k: int = 10) -> np.ndarray:
    """Mean distance to the ``k`` nearest reference rows, in standardised space.

    Standardisation uses the reference set's own mean and standard deviation, so a single
    constant column cannot dominate by scale. Nothing is fitted per item.
    """
    mean = reference.mean(axis=0, keepdims=True)
    scale = reference.std(axis=0, keepdims=True) + 1e-9
    ref = (reference - mean) / scale
    query = (features - mean) / scale
    k = max(1, min(int(k), len(ref)))
    # Chunked so a 1,900 x 1,900 distance matrix is never materialised.
    out = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), 128):
        block = query[start:start + 128]
        distances = np.sqrt(((block[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2))
        nearest = np.partition(distances, k - 1, axis=1)[:, :k]
        out[start:start + 128] = nearest.mean(axis=1)
    return out


def orient_values(name: str, values: Sequence[float]) -> np.ndarray:
    """A signal flipped to §17.3's orientation: higher score = more likely unseen."""
    if name not in ORIENTATION:
        raise KeyError(f"unknown signal {name!r}")
    return np.asarray(values, dtype=np.float64) * ORIENTATION[name]


def signal_values(probs: Sequence[Dict], novelty: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
    """All four signals from calibrated vectors (and novelty when it is supplied)."""
    max_prob = np.asarray([top_two(row)[0] for row in probs], dtype=np.float64)
    margin = np.asarray([top_two(row)[0] - top_two(row)[1] for row in probs], dtype=np.float64)
    entropy = np.asarray([entropy_of(row) for row in probs], dtype=np.float64)
    out = {"max_prob": max_prob, "margin": margin, "entropy": entropy}
    if novelty is not None:
        out["novelty"] = np.asarray(novelty, dtype=np.float64)
    return out


def choose_signal(aurocs: Dict[str, float]) -> str:
    """Highest mean AUROC; ties to the simpler signal (§17.3)."""
    best = max(aurocs.values())
    tied = [name for name, value in aurocs.items() if best - value <= TIE_TOLERANCE]
    for name in SIMPLICITY:
        if name in tied:
            return name
    return tied[0]


def threshold_for(confidences: Iterable[float], budget: float = 0.05) -> Dict:
    return largest_threshold(list(confidences), budget=budget)


# --------------------------------------------------------------------------- #
# cross-fitting
# --------------------------------------------------------------------------- #
def cross_fit(features: np.ndarray, labels: Sequence[str], items: Sequence[Dict],
              fit_predict: Callable[[np.ndarray, Sequence[str], np.ndarray], List[Dict[str, float]]],
              k: int = 5, seed: int = 42) -> Tuple[List[Dict[str, float]], List[int]]:
    """Out-of-fold probabilities for every train item, one model per fold."""
    fold = grouped_folds(items, k=k, seed=seed)
    index = np.arange(len(items))
    out: List[Optional[Dict[str, float]]] = [None] * len(items)
    for number in range(k):
        held = index[np.asarray(fold) == number]
        kept = index[np.asarray(fold) != number]
        if len(held) == 0:
            continue
        produced = fit_predict(features[kept], [labels[i] for i in kept], features[held])
        for position, row in zip(held, produced):
            out[int(position)] = row
    return [row or {} for row in out], fold


def cross_fit_calibration(probs: Sequence[Dict], truths: Sequence[str], fold: Sequence[int],
                          labels: Sequence[str], k: int = 5) -> Dict[str, Dict]:
    """Nested choice of calibration method, scored only on held-out OOF folds.

    Each method is fitted on four of the five OOF folds and scored on the fifth; the reported
    score is the mean over the five held-out folds, so the method is chosen by data the calibrator
    has not seen. The returned calibrators are refitted on all OOF probabilities, which is the
    fit the validation rows are then scored with — a different set of rows from any that chose it.
    """
    scores: Dict[str, float] = {}
    for method, fitter in FITTERS.items():
        folds = []
        for number in sorted(set(fold)):
            held = [i for i, f in enumerate(fold) if f == number]
            kept = [i for i, f in enumerate(fold) if f != number]
            if not held or not kept:
                continue
            model = fitter([probs[i] for i in kept], [truths[i] for i in kept], labels)
            calibrated = APPLIERS[method](model, [probs[i] for i in held])
            folds.append(ece(calibrated, [truths[i] for i in held]))
        scores[method] = float(np.mean(folds)) if folds else float("inf")
    chosen = min(scores, key=scores.get)
    fitted = FITTERS[chosen](probs, truths, labels)
    fitted["oof_ece"] = scores
    fitted["chosen_by"] = "out-of-fold ECE"
    return {"chosen": chosen, "scores": scores, "calibrator": fitted}
