"""§17.5: the fold classifiers on the faults they never saw.

`fold_1` was trained without faults 4, 11 and 15; `fold_2` without 14 and 15. Both are
**unchanged models** — only their calibration and threshold are rebuilt by §17.1/§17.2 on their
own training events — and they are scored on the held-out faults' items and, as a control, on
the kept faults' items from the same split.

A held-out fault cannot be named by its own fold model: the class is not in its vocabulary. So
the interesting numbers are how often it declines, what it says instead, and whether any of the
four signals would separate the held-out items from the kept ones.

**Correction to the task's premise, recorded before scoring:** the fold test rows in
`val_eval.jsonl` are *validation-split* items (`EVT-val-…`, `simulationRun` ≥ 301), not
`source_dataset: train` rows from runs 1-300, and that export has no `source_dataset` field at
all. Every fold test id resolves in `val.jsonl` and none in `train.jsonl`; this script asserts
both before it scores anything.

Usage::

    python scripts/hybrid_fold_holdout.py
"""
from __future__ import annotations

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
import hybrid_crossfit as cf  # noqa: E402
from llm.hybrid import calibration as cal  # noqa: E402
from llm.hybrid.menu_rule import UNCLASSIFIED, menu_rule  # noqa: E402

OUT = ROOT / "outputs/analysis/phase3a/hybrid"
EVAL = ROOT / "outputs/benchmark_v2/dataset/val_eval.jsonl"
VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
TRAIN = ROOT / "outputs/benchmark_v2/dataset/train.jsonl"


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def raw_probabilities(model, labels: Sequence[str], matrix: np.ndarray) -> List[Dict[str, float]]:
    proba = model.predict(matrix)
    ordered = [labels[index] for index in model.classes_] if hasattr(model, "classes_") else labels
    return [{label: float(value) for label, value in zip(ordered, row)} for row in proba]


def fold_block(variant: str, held: Sequence[int], eval_rows: List[Dict], val_by_id: Dict[str, Dict],
               train_ids: set) -> Dict:
    component_test, component_val = f"{variant}/test", f"{variant}/val"
    test_ids = [row["item_id"] for row in eval_rows if row["component"] == component_test]
    keept_ids = [row["item_id"] for row in eval_rows if row["component"] == component_val]

    dataset = lambda ids: [val_by_id[i] for i in ids]  # noqa: E731
    held_labels = {f"FAULT_{fault:02d}" for fault in held}
    for item in dataset(test_ids):
        assert hc.true_label(item) in held_labels, (item["item_id"], hc.true_label(item))
    for item in dataset(keept_ids):
        assert hc.true_label(item) not in held_labels, item["item_id"]
    assert not (set(test_ids) | set(keept_ids)) & train_ids, "a fold test row is a train row"

    # The fold's own training events, used only to cross-fit the calibrator and threshold.
    features, labels, items = hc.dataset(hc.TRAIN, drop_faults=held)
    assert held_labels.isdisjoint(set(labels)), "a held-out fault leaked into the fold's train"
    oof, fold = cal.cross_fit(features, labels, items, cf.fit_predict, k=5)
    choice = cal.cross_fit_calibration(oof, labels, fold, sorted(set(labels)), k=5)
    calibrator = choice["calibrator"]
    oof_calibrated = cal.apply_calibrator(calibrator, oof)
    confidences = [cal.top_two(prob)[0] for item, prob in zip(items, oof_calibrated)
                   if item.get("variant") == "menu" and hc.true_label(item) in set(labels)]
    threshold = cal.threshold_for(confidences, budget=0.05)

    model, metadata, _cal, _thr = hc.load_variant(variant, hc.OUT)
    classes = metadata["train_classes"]
    assert held_labels.isdisjoint(set(classes)), "the fold model has a held-out class"

    def score(ids: List[str]) -> Dict:
        wanted = dataset(ids)
        matrix = np.asarray([hc.prompt_features(item["prompt"]) for item in wanted],
                            dtype=np.float32)
        probs = cal.apply_calibrator(calibrator, raw_probabilities(model, classes, matrix))
        novelty = cal.novelty_scores(matrix, features, k=10)
        signals = cal.signal_values(probs, novelty)
        rows = []
        for index, (item, prob) in enumerate(zip(wanted, probs)):
            pred = max(prob, key=prob.get)
            decision = menu_rule(pred, prob.get(pred), item.get("menu_ids") or [],
                                 threshold["threshold"])
            rows.append({"item_id": item["item_id"], "variant": item.get("variant"),
                         "gold": str((item.get("target") or {}).get("fault_id") or ""),
                         "true_label": hc.true_label(item), "classifier_label": pred,
                         "pred": decision["diagnosis"], "abstained": decision["abstained"],
                         "confidence": cal.top_two(prob)[0],
                         "signals": {name: float(values[index])
                                     for name, values in signals.items()}})
        return {"rows": rows, "model": metadata["variant"]}

    test = score(test_ids)
    kept = score(keept_ids)

    def summarise(rows: List[Dict], label: str) -> Dict:
        """Declined / named / correct.

        `UNCLASSIFIED_OPEN_SET` is a menu entry, so a correct "open set" answer can arrive either
        through the menu rule's withholding or through the classifier naming that class; both are
        a decline to a reader, and on a menu-removed twin a decline *is* the gold answer. Counting
        the two separately, per variant, is what keeps a correct abstention from being recorded
        as a mislabel.
        """
        if not rows:
            return {"items": 0}
        declined = [row for row in rows if row["pred"] == UNCLASSIFIED]
        named = [row for row in rows if row["pred"].startswith("FAULT")]
        counts = Counter(row["pred"] for row in named)
        return {
            "items": len(rows),
            "declined": len(declined),
            "declined_rate": round(len(declined) / len(rows), 4),
            "named_a_fault": len(named),
            "named_rate": round(len(named) / len(rows), 4),
            "named_as": dict(counts.most_common(6)),
            "correctly_named_held_out_fault": sum(
                1 for row in rows if row["pred"] == row["true_label"]),
            "accuracy_against_gold": round(
                sum(1 for row in rows if row["pred"] == row["gold"]) / len(rows), 4),
            "mean_confidence": round(float(np.mean([row["confidence"] for row in rows])), 4),
        }

    def auroc_for(variant_name: str) -> Dict[str, float]:
        """Held-out faults against kept faults, in §17.3's orientation.

        A class the fold never saw gets a *lower* calibrated maximum and a *narrower*
        margin, so both are negated first; otherwise they report 1 - AUROC and read as
        worthless when they are in fact the two strongest signals here.
        """
        positives = [row for row in test["rows"] if row["variant"] == variant_name]
        negatives = [row for row in kept["rows"] if row["variant"] == variant_name]
        if not positives or not negatives:
            return {}
        out = {}
        for name in positives[0]["signals"]:
            positive = cal.orient_values(name, [row["signals"][name] for row in positives])
            negative = cal.orient_values(name, [row["signals"][name] for row in negatives])
            out[name] = round(cal.auroc(list(positive), list(negative)), 4)
        return out

    kept_false_abstention = {}
    for variant_name in ("menu", "removed"):
        subset = [row for row in kept["rows"] if row["variant"] == variant_name]
        answerable = [row for row in subset if row["gold"] != UNCLASSIFIED]
        if answerable:
            kept_false_abstention[variant_name] = round(
                sum(1 for row in answerable if row["abstained"]) / len(answerable), 4)

    return {
        "variant": variant, "held_out_faults": list(held),
        "train_events": len(items), "calibration": {
            "chosen": choice["chosen"],
            "oof_ece": {name: round(value, 5) for name, value in choice["scores"].items()}},
        "threshold": threshold,
        "test": {"menu": summarise([row for row in test["rows"] if row["variant"] == "menu"],
                                   "test:menu"),
                 "removed": summarise([row for row in test["rows"] if row["variant"] != "menu"],
                                      "test:removed"),
                 "all": summarise(test["rows"], "test:all")},
        "kept_faults": {"menu": summarise([row for row in kept["rows"]
                                           if row["variant"] == "menu"], "kept:menu"),
                        "removed": summarise([row for row in kept["rows"]
                                              if row["variant"] != "menu"], "kept:removed")},
        "kept_accuracy": round(sum(1 for row in kept["rows"] if row["pred"] == row["gold"])
                               / max(len(kept["rows"]), 1), 4),
        "false_abstention_on_kept": kept_false_abstention,
        "auroc_heldout_vs_kept": {"menu": auroc_for("menu"), "removed": auroc_for("removed")},
        "rows": {"test": test["rows"], "kept": kept["rows"]},
    }


def main() -> int:
    eval_rows = load_jsonl(EVAL)
    val_by_id = {item["item_id"]: item for item in load_jsonl(VAL)}
    train_ids = {item["item_id"] for item in load_jsonl(TRAIN)}

    components = Counter(row["component"] for row in eval_rows)
    fields = sorted(eval_rows[0])
    test_ids = {name: {row["item_id"] for row in eval_rows if row["component"] == name}
                for name in components if name.endswith("/test")}
    outcome = {
        "export_check": {
            "components": dict(components),
            "fields": fields,
            "source_dataset_present": "source_dataset" in fields,
            "fold_test_ids_in_val": len({row["item_id"] for row in eval_rows
                                         if row["component"].endswith("/test")} & set(val_by_id)),
            "fold_test_ids_in_train": len({row["item_id"] for row in eval_rows
                                           if row["component"].endswith("/test")} & train_ids),
            "fold_test_total": sum(count for name, count in components.items()
                                   if name.endswith("/test")),
            "fold_test_per_fold": {name: len(ids) for name, ids in test_ids.items()},
            "fold_test_ids_shared_by_both_folds": len(
                set.intersection(*test_ids.values())) if len(test_ids) > 1 else 0,
            "fold_test_rows_are_validation_range": all(
                ids <= set(val_by_id) for ids in test_ids.values()),
            "note": "the fold test components are the held-out faults' validation-range items "
                    "(menu-present and menu-removed twins); the export has no source_dataset "
                    "field, so the runs 1-300 additions described in PROTOCOL §17.5 are not "
                    "in it. Nothing in a fold's test component is in that fold's train file "
                    "-- the faults are held out -- so these are validation numbers, not test "
                    "numbers.",
        },
    }
    for variant, held in (("fold_1", (4, 11, 15)), ("fold_2", (14, 15))):
        block = fold_block(variant, held, eval_rows, val_by_id, train_ids)
        outcome[variant] = {key: block[key] for key in
                            ("held_out_faults", "train_events", "calibration", "threshold",
                             "test", "kept_faults", "false_abstention_on_kept",
                             "auroc_heldout_vs_kept")}
        print(json.dumps({variant: {"test": outcome[variant]["test"]["all"],
                                    "auroc": outcome[variant]["auroc_heldout_vs_kept"]}},
                         default=str), flush=True)
    (OUT / "fold_holdout.json").write_text(json.dumps(outcome, indent=2, default=str),
                                          encoding="utf-8")
    print(json.dumps(outcome["export_check"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
