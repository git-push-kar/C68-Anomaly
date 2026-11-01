"""§17.1-17.2 and §17.6: out-of-fold calibration, a threshold without validation, and the
natural wrong-label counts.

Everything fitted here is fitted on **train events**, out of fold. The classifier that scores the
validation rows is the §16 Protocol A model, unchanged; only its calibration and its threshold
are rebuilt, so the comparison with §16 isolates the thing that changed.

Outputs (new files, no §16 artifact is touched):

* ``crossfit.json`` — out-of-fold ECE/Brier per method, the chosen method, the threshold and its
  achieved false abstention, the saturation share on validation, the seven-stratum table, and the
  natural wrong-label counts;
* ``crossfit_val_rows.jsonl`` — per validation item: calibrated probabilities, signals, rule
  output. Written so later steps read numbers instead of recomputing them.

Usage::

    python scripts/hybrid_crossfit.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hybrid_classifier as hc  # noqa: E402
import hybrid_comparison as hcmp  # noqa: E402
from llm.hybrid import calibration as cal  # noqa: E402
from llm.hybrid.menu_rule import NORMAL, UNCLASSIFIED  # noqa: E402

OUT = ROOT / "outputs/analysis/phase3a/hybrid"
VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"


def label_of(item: Dict) -> str:
    return hc.true_label(item)


def fit_predict(features: np.ndarray, labels: Sequence[str],
                query: np.ndarray) -> List[Dict[str, float]]:
    """One fold's model: fit on the kept rows, predict the held-out ones."""
    model, ordered = hc.fit(features, labels)
    return hc.predict_proba(model, ordered, query)


def per_stratum(rows: Sequence[Dict]) -> Dict[str, Dict]:
    out = {}
    for name, keep in hcmp.STRATA.items():
        subset = [row for row in rows if keep(row)]
        if subset:
            out[name] = hc.summarise(subset, label=name)
    return out


def wrong_label_counts(items: Sequence[Dict], probs: Sequence[Dict], fold: Sequence[int],
                       threshold: float, predictions: Dict[int, Dict]) -> Dict:
    """§17.6: what the out-of-fold classifier gets wrong on its own training distribution."""
    wrong, abstained, both = Counter(), Counter(), Counter()
    by_variant: Dict[str, Counter] = defaultdict(Counter)
    lookalike: Dict[str, Counter] = defaultdict(Counter)
    distinct = set()
    total = 0
    for index, (item, prob) in enumerate(zip(items, probs)):
        truth = label_of(item)
        pred = max(prob, key=prob.get)
        confidence = float(prob[pred])
        on_menu = pred in set(item.get("menu_ids") or [])
        is_abstained = (pred != NORMAL and pred != UNCLASSIFIED
                        and (not on_menu or confidence < threshold))
        is_wrong = pred != truth
        total += 1
        if is_wrong:
            wrong[truth] += 1
            both[truth] += 1
            lookalike[truth][pred] += 1
            by_variant[str(item.get("variant"))]["wrong"] += 1
        if is_abstained:
            abstained[truth] += 1
            both[truth] += 1
            by_variant[str(item.get("variant"))]["abstained"] += 1
        if is_wrong or is_abstained:
            distinct.add(cal.event_key(item))
    return {
        "train_events": total,
        "wrong": sum(wrong.values()), "abstained": sum(abstained.values()),
        "wrong_or_abstained_events": len(distinct),
        "by_true_fault": {key: {"wrong": wrong[key], "abstained": abstained[key],
                                "union": both[key]} for key in sorted(both)},
        "by_predicted_fault": dict(Counter(
            label for truth, preds in lookalike.items() for label, count in preds.items()
            for _ in range(count))),
        "by_variant": {key: dict(value) for key, value in by_variant.items()},
        "lookalike_split": {truth: dict(preds.most_common()) for truth, preds in
                            sorted(lookalike.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default=str(OUT / "crossfit.json"))
    args = parser.parse_args()

    features, labels, items = hc.dataset(hc.TRAIN)
    print(json.dumps({"train_events": len(items), "features": features.shape[1]}), flush=True)

    oof, fold = cal.cross_fit(features, labels, items, fit_predict, k=args.folds)
    truth = [label_of(item) for item in items]
    model, metadata, _old_calibrator, _old_threshold = hc.load_variant("protocolA", hc.OUT)
    class_labels = metadata["train_classes"]

    choice = cal.cross_fit_calibration(oof, truth, fold, class_labels, k=args.folds)
    calibrator = choice["calibrator"]
    oof_calibrated = cal.apply_calibrator(calibrator, oof)

    # §17.2: the threshold comes from out-of-fold *train* confidences only.
    confidences = []
    for item, prob in zip(items, oof_calibrated):
        if item.get("variant") != "menu":
            continue
        if label_of(item) not in set(class_labels):
            continue
        confidences.append(cal.top_two(prob)[0])
    threshold = cal.threshold_for(confidences, budget=0.05)

    # --- clean evaluation on the 834 validation rows -----------------------
    val_matrix, _val_labels, val_items = hc.dataset(VAL, dedupe=False)
    raw_val = hc.probabilities(model, metadata, calibrator, val_matrix)
    calibrated_val = raw_val  # probabilities() already applies the calibrator
    val_truth = [label_of(item) for item in val_items]
    saturation = sum(1 for prob in calibrated_val if cal.top_two(prob)[0] == 1.0)
    rows = hc.rows_with_rule(val_items, calibrated_val, threshold["threshold"])
    entry = [{"row": row, "item": item} for row, item in zip(rows, val_items)]
    novelty = cal.novelty_scores(val_matrix, features, k=10)
    signals = cal.signal_values(calibrated_val, novelty)
    for index, row in enumerate(rows):
        row["signals"] = {name: float(values[index]) for name, values in signals.items()}

    # Isotonic maps each class's probability separately and then renormalises, which is not
    # order-preserving: the calibrated argmax can differ from the raw one. Both are reported,
    # because a label decision and a confidence decision are two different uses of the vector.
    raw_only = hc.probabilities(model, metadata, {"method": "identity"}, val_matrix)
    raw_rows = hc.rows_with_rule(val_items, raw_only, threshold["threshold"])
    raw_labelled = [max(prob, key=prob.get) for prob in raw_only]
    calibrated_labelled = [max(prob, key=prob.get) for prob in calibrated_val]
    flips = [{"item_id": row["item_id"], "gold": row["gold"], "raw": raw_labelled[index],
              "calibrated": calibrated_labelled[index], "unobservable": row["unobservable"]}
             for index, row in enumerate(rows) if raw_labelled[index] != calibrated_labelled[index]]

    outcome = {
        "folds": args.folds,
        "train_events": len(items),
        "calibrator_full": calibrator,
        "raw_vs_calibrated_argmax": {
            "disagreements": len(flips),
            "of": len(rows),
            "examples": flips[:15],
            "raw_label_accuracy": round(sum(1 for label, row in zip(raw_labelled, rows)
                                          if label == row["gold"]) / max(len(rows), 1), 4),
            "calibrated_label_accuracy": round(
                sum(1 for label, row in zip(calibrated_labelled, rows)
                    if label == row["gold"]) / max(len(rows), 1), 4),
            "raw_rule_accuracy_observable": hc.summarise(
                [row for row in raw_rows if not row["unobservable"]],
                "crossfit:raw")["accuracy"],
        },
        "out_of_fold": {
            "ece": {method: round(score, 5) for method, score in choice["scores"].items()},
            "brier_isotonic": cal.brier(cal.apply_calibrator(
                cal.FITTERS["isotonic"](oof, truth, class_labels), oof), truth),
            "brier_temperature": cal.brier(cal.apply_calibrator(
                cal.FITTERS["temperature"](oof, truth, class_labels), oof), truth),
            "chosen": choice["chosen"],
            "chosen_ece": round(choice["scores"][choice["chosen"]], 5),
            "ece_chosen_on_oof": cal.ece(oof_calibrated, truth),
            "brier_chosen_on_oof": cal.brier(oof_calibrated, truth),
        },
        "calibrator": {key: value for key, value in calibrator.items()
                       if key in ("method", "temperature", "oof_ece", "chosen_by")},
        "threshold": {**threshold, "source": "out-of-fold train events, menu-present seen-fault"},
        "validation": {
            "items": len(val_items),
            "saturation_share": round(saturation / max(len(val_items), 1), 4),
            "saturated": saturation,
            "observable": hc.summarise([row for row in rows if not row["unobservable"]],
                                       "crossfit:observable"),
            "ece": cal.ece(calibrated_val, val_truth),
            "brier": cal.brier(calibrated_val, val_truth),
            "rule_reasons": dict(Counter(row["rule_reason"] for row in rows)),
            "strata": per_stratum(rows),
        },
        "natural_wrong_labels": wrong_label_counts(
            items, oof_calibrated, fold, threshold["threshold"], {}),
    }
    (OUT / "crossfit_val_rows.jsonl").write_text(
        "\n".join(json.dumps(row, default=str) for row in rows), encoding="utf-8")
    Path(args.out).write_text(json.dumps(outcome, indent=2, default=str), encoding="utf-8")

    print(json.dumps({
        "oof_ece": outcome["out_of_fold"]["ece"],
        "chosen": choice["chosen"],
        "threshold": threshold["threshold"],
        "achieved_false_abstention": threshold["achieved_false_abstention"],
        "saturation_share": outcome["validation"]["saturation_share"],
        "observable": {key: outcome["validation"]["observable"][key] for key in
                       ("items", "accuracy", "macro_f1", "top3")},
        "abstention": outcome["validation"]["observable"]["abstention"],
        "ece_val": outcome["validation"]["ece"], "brier_val": outcome["validation"]["brier"],
        "wrong": outcome["natural_wrong_labels"]["wrong"],
        "abstained": outcome["natural_wrong_labels"]["abstained"],
        "events": outcome["natural_wrong_labels"]["wrong_or_abstained_events"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
