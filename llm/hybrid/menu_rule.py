"""The classifier's decision, and how a menu turns it into an answer.

In the hybrid, the fault identity is not the adapter's to choose: a classifier predicts the
true fault of the event, and this rule decides whether that prediction may be named. The two
ways a diagnosis can be withheld are different and are kept apart on purpose:

* the prediction is a fault the item's menu does not contain — the item is an open-set
  episode and naming the fault would be answering off-menu, which the prompt forbids;
* the prediction is on the menu but the calibrated confidence is below the abstention
  threshold — the classifier is saying the evidence is thin.

A prediction of ``NORMAL_TRANSIENT`` is a third case: there is no fault to name, so the
verdict is ``no_fault`` and the diagnosis is the non-fault class. It is not an abstention,
and it is not a low-confidence fault either.

The threshold itself is derived, not chosen: :func:`largest_threshold` finds the largest value
that keeps false abstention inside a pre-registered budget (PROTOCOL §16.3).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

UNCLASSIFIED = "UNCLASSIFIED_OPEN_SET"
NORMAL = "NORMAL_TRANSIENT"
VERDICTS = ("classified", "unclassified_open_set", "no_fault")


def menu_rule(label: Optional[str], confidence: Optional[float],
              menu_ids: Sequence[str], threshold: float) -> Dict:
    """Map one classifier prediction to ``(diagnosis, verdict, reason)``.

    ``menu_ids`` is the item's own menu, so the same prediction becomes a named fault on a
    menu-present item and an abstention on its open-set twin. That is the whole point of
    scoring open-set episodes: the classifier sees no menu, and the rule is what makes the
    answer menu-legal.
    """
    menu = list(menu_ids or [])
    if not label:
        return {"diagnosis": UNCLASSIFIED, "verdict": "unclassified_open_set",
                "reason": "no_prediction", "abstained": True, "confidence": None}
    if label == NORMAL:
        return {"diagnosis": NORMAL, "verdict": "no_fault", "reason": "no_fault_predicted",
                "abstained": False, "confidence": confidence}
    if label not in menu:
        return {"diagnosis": UNCLASSIFIED, "verdict": "unclassified_open_set",
                "reason": "not_on_menu", "abstained": True, "confidence": confidence}
    if confidence is None or float(confidence) < float(threshold):
        return {"diagnosis": UNCLASSIFIED, "verdict": "unclassified_open_set",
                "reason": "below_threshold", "abstained": True, "confidence": confidence}
    return {"diagnosis": label, "verdict": "classified", "reason": "on_menu_above_threshold",
            "abstained": False, "confidence": confidence}


def largest_threshold(confidences: Iterable[float], budget: float = 0.05) -> Dict:
    """The largest threshold whose false-abstention rate stays inside ``budget``.

    "False abstention" is declining an item whose fault is on the menu and was seen in
    training — the classifier had the class and the answer was allowed. The grid is the
    observed confidences themselves, so the returned value is a data point rather than a
    round number, and the achieved rate is returned with it so the budget cannot be
    silently exceeded.

    A threshold of 0.0 would abstain on nothing; the search is for the largest value, i.e.
    the one that spends the whole budget.
    """
    values = sorted(float(value) for value in confidences)
    if not values:
        return {"threshold": 1.0, "achieved_false_abstention": 0.0, "budget": budget,
                "items": 0}
    allowed = int(len(values) * float(budget))  # how many may fall below the threshold
    if allowed <= 0:
        threshold = values[0]
        achieved = 0.0
    else:
        threshold = values[allowed - 1] if allowed <= len(values) else values[-1]
        achieved = sum(1 for value in values if value < threshold) / len(values)
    return {"threshold": round(float(threshold), 6),
            "achieved_false_abstention": round(float(achieved), 6),
            "budget": budget, "items": len(values),
            "abstained": sum(1 for value in values if value < threshold)}


__all__ = ["UNCLASSIFIED", "NORMAL", "VERDICTS", "menu_rule", "largest_threshold"]
