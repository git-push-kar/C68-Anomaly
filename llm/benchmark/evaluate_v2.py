"""Metrics for the closed-set evaluation, computed from prediction files.

The evaluation of Protocol A asks one question per item -- which menu entry is
this? -- so most of what is reported is a confusion matrix and the numbers derived
from it. Everything here takes plain dictionaries, never a model: a prediction file
is the boundary, which is what lets the metrics be tested on synthetic rows and
re-computed from a saved file without a GPU.

Canonical row shape (one JSON object per line)::

    {"item_id": "A-test-000123", "gold": "FAULT_05", "pred": "FAULT_13",
     "run": "TEP_Faulty_Testing.csv|5|401", "probs": {"FAULT_01": 0.02, ...},
     "answer": "...", "evidence_tags": ["XMEAS_7"], "grounded_numbers": [1.5]}

``probs``, ``answer``, ``evidence_tags`` and ``grounded_numbers`` are optional;
the metrics that need them skip the rows that do not have them rather than
counting a missing field as a wrong answer.

Usage::

    python -m llm.benchmark.evaluate_v2 --predictions preds.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark.menu import UNCLASSIFIED  # noqa: E402

#: The abstention label. It is a menu entry like any other for scoring purposes;
#: what makes it special is that a *false* abstention is a different mistake from a
#: confusion between two faults, so it is reported separately.
ABSTAIN = UNCLASSIFIED

TAG_PATTERN = re.compile(r"\bX(?:MEAS|MV)_\d{1,2}\b")
#: A number, but not the run number inside a tag: ``XMEAS_7`` ends in a digit
#: that is part of the name, and counting it as a figure would flag every answer
#: that cites a sensor.
NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_predictions(path: Path) -> List[Dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _valid(rows: Iterable[Dict]) -> List[Dict]:
    """Rows with a gold label and a prediction; anything else is not scored."""
    return [row for row in rows
            if row.get("gold") and row.get("pred") and str(row["gold"]).strip()]


def labels_in(rows: Sequence[Dict]) -> List[str]:
    """Every label that appears as a truth, plus the abstention label."""
    labels = {str(row["gold"]) for row in rows} | {str(row["pred"]) for row in rows}
    labels.add(ABSTAIN)
    return sorted(labels)


# ---------------------------------------------------------------------------
# closed-set metrics
# ---------------------------------------------------------------------------
def accuracy(rows: Sequence[Dict]) -> float:
    rows = _valid(rows)
    if not rows:
        return float("nan")
    return sum(1 for row in rows if row["pred"] == row["gold"]) / len(rows)


def top_k_accuracy(rows: Sequence[Dict], k: int = 3) -> float:
    """Fraction of items whose gold label is among the k most likely entries."""
    scored = [row for row in _valid(rows) if row.get("probs")]
    if not scored:
        return float("nan")
    hits = 0
    for row in scored:
        ranked = sorted(row["probs"], key=row["probs"].get, reverse=True)[:k]
        hits += int(row["gold"] in ranked)
    return hits / len(scored)


def confusion_matrix(rows: Sequence[Dict],
                     labels: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, int]]:
    rows = _valid(rows)
    labels = list(labels) if labels else labels_in(rows)
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for row in rows:
        if row["gold"] in matrix:
            matrix[row["gold"]][row["pred"]] = matrix[row["gold"]].get(row["pred"], 0) + 1
    return matrix


def per_class_prf(rows: Sequence[Dict],
                  labels: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    """Precision, recall and F1 per gold class, over the classes with support."""
    rows = _valid(rows)
    labels = list(labels) if labels else labels_in(rows)
    out: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for row in rows if row["gold"] == label and row["pred"] == label)
        fp = sum(1 for row in rows if row["gold"] != label and row["pred"] == label)
        fn = sum(1 for row in rows if row["gold"] == label and row["pred"] != label)
        support = tp + fn
        if support == 0 and fp == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        out[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                      "f1": round(f1, 4), "support": support}
    return out


def macro_f1(rows: Sequence[Dict], labels: Optional[Sequence[str]] = None) -> float:
    """Unweighted mean F1 over the classes present -- the checkpoint rule.

    A class with no support is not a zero: it is a class the split does not test,
    and averaging it in would make the metric depend on the menu's length.
    """
    per_class = per_class_prf(rows, labels)
    scored = [entry["f1"] for entry in per_class.values() if entry["support"] > 0]
    if not scored:
        return float("nan")
    return sum(scored) / len(scored)


def abstention_metrics(rows: Sequence[Dict]) -> Dict[str, float]:
    """How the abstention label behaves, as a class in its own right.

    ``false_abstention_rate`` is the share of *answerable* items the model
    declines; ``missed_abstention_rate`` is the share of unclassifiable items it
    answers as if it knew. Two different errors, so two different numbers.
    """
    rows = _valid(rows)
    if not rows:
        return {}
    abstained = [row for row in rows if row["pred"] == ABSTAIN]
    should = [row for row in rows if row["gold"] == ABSTAIN]
    true_abstain = [row for row in rows if row["pred"] == ABSTAIN and row["gold"] == ABSTAIN]
    answerable = [row for row in rows if row["gold"] != ABSTAIN]
    precision = len(true_abstain) / len(abstained) if abstained else 0.0
    recall = len(true_abstain) / len(should) if should else 0.0
    return {
        "abstention_rate": round(len(abstained) / len(rows), 4),
        "abstention_precision": round(precision, 4),
        "abstention_recall": round(recall, 4),
        "false_abstention_rate": round(
            sum(1 for row in answerable if row["pred"] == ABSTAIN) / len(answerable), 4
        ) if answerable else None,
        "missed_abstention_rate": round(
            sum(1 for row in should if row["pred"] != ABSTAIN) / len(should), 4
        ) if should else None,
        "abstain_support": len(should),
    }


def ece(rows: Sequence[Dict], bins: int = 10) -> float:
    """Expected calibration error over the confidence of the *chosen* label.

    Confidence is the probability the model gave its own answer, so an item with
    no probability vector cannot contribute and is skipped.
    """
    scored = []
    for row in _valid(rows):
        probs = row.get("probs") or {}
        if not probs:
            continue
        top = max(probs.values())
        scored.append((float(top), 1.0 if row["pred"] == row["gold"] else 0.0))
    if not scored:
        return float("nan")
    total = len(scored)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [pair for pair in scored
                  if (pair[0] > low or (index == 0 and pair[0] >= low)) and pair[0] <= high]
        if not bucket:
            continue
        confidence = sum(pair[0] for pair in bucket) / len(bucket)
        hit_rate = sum(pair[1] for pair in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(confidence - hit_rate)
    return round(error, 4)


def brier(rows: Sequence[Dict]) -> float:
    """Brier score for the one-vs-rest event "the chosen label is right"."""
    scored = []
    for row in _valid(rows):
        probs = row.get("probs") or {}
        if not probs:
            continue
        top = float(max(probs.values()))
        scored.append((top - (1.0 if row["pred"] == row["gold"] else 0.0)) ** 2)
    if not scored:
        return float("nan")
    return round(sum(scored) / len(scored), 4)


# ---------------------------------------------------------------------------
# grounding
# ---------------------------------------------------------------------------
def grounding_violations(answer: str, evidence_tags: Sequence[str],
                         grounded_numbers: Sequence[float],
                         number_tolerance: float = 1e-6) -> List[str]:
    """The tags and numbers in an answer that the event does not support.

    A number counts as grounded if it matches a stored value exactly or after the
    percent conversion the store uses (a 6.98 % deviation may be written ``6.98``
    or ``0.0698``); a tag counts as grounded if it names a sensor in the evidence.
    """
    violations: List[str] = []
    allowed_tags = {str(tag).strip().upper() for tag in evidence_tags}
    for tag in TAG_PATTERN.findall(answer or ""):
        if tag.upper() not in allowed_tags:
            violations.append(f"tag_not_in_evidence:{tag}")
    stored = [float(value) for value in grounded_numbers]
    scaled = stored + [value * 100.0 for value in stored] + [value / 100.0 for value in stored]
    for match in NUMBER_PATTERN.findall(answer or ""):
        value = float(match)
        if any(abs(value - candidate) <= number_tolerance for candidate in scaled):
            continue
        violations.append(f"number_not_in_store:{match}")
    return violations


def grounding_summary(rows: Sequence[Dict]) -> Dict[str, object]:
    scored = [row for row in rows if row.get("answer") is not None]
    if not scored:
        return {"items": 0}
    clean = 0
    counts: Counter = Counter()
    for row in scored:
        found = grounding_violations(
            str(row.get("answer") or ""),
            row.get("evidence_tags") or [],
            row.get("grounded_numbers") or [],
        )
        if not found:
            clean += 1
        for code in found:
            counts[code.split(":")[0]] += 1
    return {"items": len(scored), "clean": clean,
            "clean_rate": round(clean / len(scored), 4),
            "violations": dict(sorted(counts.items()))}


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------
def run_clustered_bootstrap(rows: Sequence[Dict], metric: Callable[[Sequence[Dict]], float],
                            n_resamples: int = 1000, seed: int = 42,
                            cluster_key: str = "run") -> Dict[str, object]:
    """Percentile interval from resampling *runs*, not items.

    Items from one run share an event, a fault and a set of sensor traces, so they
    are not independent: resampling items would report an interval narrower than
    the data can support. Runs are the unit that was actually drawn.
    """
    rows = _valid(rows)
    if not rows:
        return {"point": float("nan"), "low": None, "high": None, "clusters": 0}
    clusters: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        clusters[str(row.get(cluster_key) or row["item_id"])].append(row)
    keys = sorted(clusters)
    point = metric(rows)
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(n_resamples):
        drawn: List[Dict] = []
        for _ in range(len(keys)):
            drawn.extend(clusters[keys[rng.randrange(len(keys))]])
        value = metric(drawn)
        if value == value:  # not NaN
            samples.append(value)
    samples.sort()
    if not samples:
        return {"point": point, "low": None, "high": None, "clusters": len(keys)}
    low = samples[int(0.025 * (len(samples) - 1))]
    high = samples[int(0.975 * (len(samples) - 1))]
    return {"point": round(point, 4), "low": round(low, 4), "high": round(high, 4),
            "clusters": len(keys), "resamples": len(samples)}


def mcnemar(rows_a: Sequence[Dict], rows_b: Sequence[Dict]) -> Dict[str, object]:
    """Paired comparison of two prediction files on the items they share.

    Only the discordant pairs carry information: ``b`` is where A was right and B
    wrong. With few discordant pairs the exact binomial is used, because the chi-
    square approximation is exactly wrong in the regime a benchmark this size sits
    in.
    """
    by_id_b = {row["item_id"]: row for row in _valid(rows_b)}
    pairs = [(row, by_id_b[row["item_id"]]) for row in _valid(rows_a)
             if row["item_id"] in by_id_b and by_id_b[row["item_id"]]["gold"] == row["gold"]]
    b = sum(1 for a, other in pairs if a["pred"] == a["gold"] and other["pred"] != other["gold"])
    c = sum(1 for a, other in pairs if a["pred"] != a["gold"] and other["pred"] == other["gold"])
    n = b + c
    if n == 0:
        return {"items": len(pairs), "b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0,
                "method": "none"}
    if n <= 25:
        tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
        p_value = min(1.0, 2 * tail)
        method = "exact_binomial"
    else:
        statistic = (abs(b - c) - 1) ** 2 / n
        p_value = math.erfc(math.sqrt(statistic / 2.0))
        method = "chi2_continuity"
    return {"items": len(pairs), "b": b, "c": c,
            "statistic": round((abs(b - c) - 1) ** 2 / n, 4) if n > 25 else float(abs(b - c)),
            "p_value": round(p_value, 6), "method": method}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def report(rows: Sequence[Dict], label: str = "") -> Dict[str, object]:
    rows = _valid(rows)
    per_class = per_class_prf(rows)
    return {
        "label": label,
        "items": len(rows),
        "accuracy": round(accuracy(rows), 4),
        "top3": round(top_k_accuracy(rows, 3), 4),
        "macro_f1": round(macro_f1(rows), 4),
        "abstention": abstention_metrics(rows),
        "ece": ece(rows),
        "brier": brier(rows),
        "grounding": grounding_summary(rows),
        "per_fault": per_class,
        "accuracy_ci": run_clustered_bootstrap(rows, accuracy),
        "macro_f1_ci": run_clustered_bootstrap(rows, macro_f1),
        "confusion": confusion_matrix(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default=None)
    parser.add_argument("--compare", default=None,
                        help="second prediction file for a paired McNemar test")
    args = parser.parse_args()
    rows = load_predictions(Path(args.predictions))
    result = report(rows, label=args.label or Path(args.predictions).stem)
    if args.compare:
        other = load_predictions(Path(args.compare))
        result["mcnemar"] = mcnemar(rows, other)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
