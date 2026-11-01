"""The classifier half of the hybrid, and the calibration that decides when to abstain.

Three models are trained — Protocol A, fold 1 (held-out faults 4/11/15), fold 2 (held-out
14/15) — on **train events only**, one row per event. Both menu variants of an episode carry
the same evidence, so they are deduplicated: a classifier trained on both would see the same
features twice and would also learn that one event has two different gold labels.

Classes are the event's **true** fault, not the menu-relative answer: faults 1-15 as
``FAULT_nn``, faults 16-20 as ``UNCLASSIFIED_OPEN_SET``, fault-free events as
``NORMAL_TRANSIENT``. Turning a true fault into an abstention is the menu rule's job
(``llm/hybrid/menu_rule.py``), and keeping the two apart is what lets the same classifier be
scored on a menu-present and a menu-removed twin of one event.

Calibration is fitted on validation only, and the *method* is chosen by 5-fold
cross-validation **within** validation (PROTOCOL §16.3): isotonic one-vs-rest renormalised,
or temperature scaling. The abstention threshold is then the largest value that keeps false
abstention inside 5 % of seen-fault, menu-present items.

Usage::

    python scripts/hybrid_classifier.py train --variant protocolA
    python scripts/hybrid_classifier.py train --variant fold_1
    python scripts/hybrid_classifier.py surrogate
    python scripts/hybrid_classifier.py evaluate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

from llm.benchmark import evaluate_v2 as ev  # noqa: E402
from llm.hybrid.menu_rule import (  # noqa: E402
    NORMAL,
    UNCLASSIFIED,
    largest_threshold,
    menu_rule,
)
from scripts.evidence_classifier import (  # noqa: E402
    feature_names,
    identifier_audit,
    prompt_features,
)

TRAIN = ROOT / "outputs/benchmark_v2/dataset/train.jsonl"
VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
OUT = ROOT / "outputs/analysis/phase3a/hybrid"
ADAPTER_PREDS = ROOT / "outputs/analysis/phase3a/instrument_check/preds_decode_step500_full.jsonl"

#: Held-out faults per fold, from the dataset manifest (§ Protocol B).
FOLD_HOLDOUT = {"protocolA": (), "fold_1": (4, 11, 15), "fold_2": (14, 15)}
SEED = 42


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def event_key(item: Dict) -> str:
    """The event's identity — never a feature, only the deduplication key."""
    return "|".join(str(item.get(name)) for name in
                    ("source_file", "fault", "simulationRun"))


def true_label(item: Dict) -> str:
    """The event's true class, menu-independent."""
    fault = item.get("fault")
    if fault is None:
        return NORMAL
    number = int(fault)
    if 16 <= number <= 20:
        return UNCLASSIFIED
    return f"FAULT_{number:02d}"


def dataset(split: Path, drop_faults: Iterable[int] = (),
            dedupe: bool = True) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """Features parsed from the rendered evidence block.

    ``dedupe`` collapses an event's two menu variants to one row. It is for *training*:
    a classifier that saw both would see identical features twice. Validation must not
    collapse them, because the two variants are two different questions with two different
    gold answers (``FAULT_07`` and ``UNCLASSIFIED_OPEN_SET``) and the menu rule needs both.
    In the train split there is nothing to collapse anyway -- an open-set episode is emitted
    only in its removed form, so the 1,919 rows are 1,919 distinct events.
    """
    drop = {int(value) for value in drop_faults}
    seen: Dict[str, Dict] = {}
    items: List[Dict] = []
    for item in load_jsonl(split):
        if item.get("fault") is not None and int(item["fault"]) in drop:
            continue
        if not dedupe:
            items.append(item)
            continue
        key = event_key(item)
        if key in seen:
            continue
        seen[key] = item
        items.append(item)
    items.sort(key=lambda item: str(item.get("item_id")))
    matrix = np.asarray([prompt_features(item["prompt"]) for item in items], dtype=np.float32)
    return matrix, [true_label(item) for item in items], items


def fit(x_train: np.ndarray, y_train: Sequence[str], seed: int = SEED):
    import lightgbm as lgb

    labels = sorted(set(y_train))
    encoder = {label: index for index, label in enumerate(labels)}
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(labels), n_estimators=400,
        learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.9,
        subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, random_state=seed,
        verbose=-1,
        # Determinism is a requirement, not a preference: the surrogate models are compared
        # against this one and a rerun must reproduce it exactly.
        deterministic=True, force_row_wise=True, n_jobs=1,
    )
    model.fit(x_train, [encoder[label] for label in y_train])
    return model, labels


def predict_proba(model, labels: Sequence[str], x: np.ndarray) -> List[Dict[str, float]]:
    proba = model.predict_proba(x)
    ordered = [labels[index] for index in model.classes_]
    return [{label: float(p) for label, p in zip(ordered, row)} for row in proba]


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def _row_vector(probs: Dict[str, float], labels: Sequence[str]) -> np.ndarray:
    return np.asarray([float(probs.get(label, 0.0)) for label in labels], dtype=np.float64)


def _softmax(matrix: np.ndarray) -> np.ndarray:
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def fit_isotonic(probs: Sequence[Dict], truths: Sequence[str], labels: Sequence[str]) -> Dict:
    from sklearn.isotonic import IsotonicRegression

    matrix = np.asarray([_row_vector(row, labels) for row in probs])
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


def _nll(probs: Sequence[Dict], truths: Sequence[str]) -> float:
    total = 0.0
    for row, truth in zip(probs, truths):
        total -= float(np.log(max(float(row.get(truth, 0.0)), 1e-12)))
    return total / max(len(probs), 1)


def fit_temperature(probs: Sequence[Dict], truths: Sequence[str], labels: Sequence[str]) -> Dict:
    from scipy.optimize import minimize_scalar

    matrix = np.asarray([_row_vector(row, labels) for row in probs])
    logits = np.log(np.clip(matrix, 1e-12, 1.0))

    def objective(temperature: float) -> float:
        scaled = _softmax(logits / max(temperature, 1e-3))
        rows = [{label: float(value) for label, value in zip(labels, row)} for row in scaled]
        return _nll(rows, truths)

    result = minimize_scalar(objective, bounds=(0.05, 20.0), method="bounded")
    return {"method": "temperature", "labels": list(labels),
            "temperature": float(result.x), "nll": float(result.fun)}


def apply_temperature(model: Dict, probs: Sequence[Dict]) -> List[Dict[str, float]]:
    labels = model["labels"]
    matrix = np.asarray([_row_vector(row, labels) for row in probs])
    logits = np.log(np.clip(matrix, 1e-12, 1.0)) / max(model["temperature"], 1e-3)
    scaled = _softmax(logits)
    return [{label: float(value) for label, value in zip(labels, row)} for row in scaled]


APPLIERS = {"isotonic": apply_isotonic, "temperature": apply_temperature}
FITTERS = {"isotonic": fit_isotonic, "temperature": fit_temperature}


def apply_calibrator(model: Dict, probs: Sequence[Dict]) -> List[Dict[str, float]]:
    if model.get("method") == "identity":
        return [dict(row) for row in probs]
    return APPLIERS[model["method"]](model, probs)


def ece_of(probs: Sequence[Dict], truths: Sequence[str], gold: Sequence[str]) -> float:
    rows = []
    for row, truth in zip(probs, truths):
        pred = max(row, key=row.get)
        rows.append({"gold": truth, "pred": pred, "probs": row, "item_id": "x"})
    return ev.ece(rows)


def cross_validated_ece(probs: Sequence[Dict], truths: Sequence[str], labels: Sequence[str],
                        folds: int = 5, seed: int = SEED) -> Dict[str, float]:
    """ECE per calibration method, fitted on 4 folds and scored on the fifth."""
    import random

    index = list(range(len(probs)))
    random.Random(seed).shuffle(index)
    chunks = [index[i::folds] for i in range(folds)]
    out = {}
    for method, fitter in FITTERS.items():
        scores = []
        for fold in range(folds):
            held = set(chunks[fold])
            train_index = [i for i in index if i not in held]
            model = fitter([probs[i] for i in train_index],
                           [truths[i] for i in train_index], labels)
            calibrated = APPLIERS[method](model, [probs[i] for i in chunks[fold]])
            scores.append(ece_of(calibrated, [truths[i] for i in chunks[fold]],
                                 [truths[i] for i in chunks[fold]]))
        out[method] = float(np.mean(scores))
    return out


# --------------------------------------------------------------------------- #
# artefacts
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def train_variant(variant: str, out: Path) -> Dict:
    drop = FOLD_HOLDOUT[variant]
    x_train, y_train, _items = dataset(TRAIN, drop_faults=drop)
    model, labels = fit(x_train, y_train)
    directory = out / variant
    directory.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(directory / "classifier.txt"))

    # Calibration on validation (all rows, both variants; the features are the event's).
    x_val, y_val, val_items = dataset(VAL, dedupe=False)
    probs = predict_proba(model, labels, x_val)
    truths = [true_label(item) for item in val_items]
    cv = cross_validated_ece(probs, truths, labels)
    chosen = min(cv, key=cv.get)
    calibrator = FITTERS[chosen](probs, truths, labels)
    (directory / "calibration.json").write_text(json.dumps(calibrator, indent=2),
                                                encoding="utf-8")

    calibrated = apply_calibrator(calibrator, probs)
    # The threshold is fitted on what the rule will see: menu-present items whose true fault
    # is a class this model was trained on.
    confidences = []
    for item, row, truth in zip(val_items, calibrated, truths):
        if item.get("variant") != "menu":
            continue
        if truth not in set(labels):
            continue
        confidences.append(max(row.values()))
    threshold = largest_threshold(confidences, budget=0.05)
    (directory / "threshold.json").write_text(json.dumps(threshold, indent=2), encoding="utf-8")

    metadata = {
        "variant": variant, "held_out_faults": list(drop), "seed": SEED,
        "train_events": len(y_train), "train_classes": sorted(set(y_train)),
        "features": int(x_train.shape[1]),
        "feature_audit": identifier_audit(feature_names()),
        "calibration": {"chosen": chosen, "cv_ece": cv},
        "threshold": threshold,
        "files": {name: sha256_file(directory / name)
                  for name in ("classifier.txt", "calibration.json", "threshold.json")},
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_variant(variant: str, out: Path):
    import lightgbm as lgb

    directory = out / variant
    model = lgb.Booster(model_file=str(directory / "classifier.txt"))
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    calibrator = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
    threshold = json.loads((directory / "threshold.json").read_text(encoding="utf-8"))
    return model, metadata, calibrator, threshold


def probabilities(model, metadata: Dict, calibrator: Dict, x: np.ndarray) -> List[Dict[str, float]]:
    raw = model.predict(x)
    labels = metadata["train_classes"]
    rows = [{label: float(value) for label, value in zip(labels, row)} for row in raw]
    return apply_calibrator(calibrator, rows)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def rows_with_rule(items: Sequence[Dict], probs: Sequence[Dict], threshold: float) -> List[Dict]:
    rows = []
    for item, prob in zip(items, probs):
        pred = max(prob, key=prob.get)
        decision = menu_rule(pred, prob.get(pred), item.get("menu_ids") or [], threshold)
        rows.append({
            "item_id": item["item_id"],
            "gold": str((item.get("target") or {}).get("fault_id") or ""),
            "pred": decision["diagnosis"],
            "probs": prob,
            "run": event_key(item),
            "unobservable": bool(item.get("unobservable")),
            "variant": item.get("variant"),
            "true_label": true_label(item),
            "classifier_label": pred,
            "confidence": decision["confidence"],
            "rule_reason": decision["reason"],
            "verdict": decision["verdict"],
        })
    return rows


def summarise(rows: Sequence[Dict], label: str) -> Dict:
    report = ev.report(rows, label=label)
    return {
        "label": label, "items": report["items"], "accuracy": report["accuracy"],
        "top3": report["top3"], "macro_f1": report["macro_f1"],
        "abstention": report["abstention"], "ece": report["ece"], "brier": report["brier"],
        "per_class": report["per_fault"],
        "accuracy_ci": report["accuracy_ci"], "macro_f1_ci": report["macro_f1_ci"],
    }


def evaluate(out: Path, variant: str = "protocolA") -> Dict:
    model, metadata, calibrator, threshold = load_variant(variant, out)
    x_val, y_val, items = dataset(VAL, dedupe=False)
    probs = probabilities(model, metadata, calibrator, x_val)
    rows = rows_with_rule(items, probs, threshold["threshold"])

    (out / f"preds_{variant}_val.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    observable = [row for row in rows if not row["unobservable"]]
    unobservable = [row for row in rows if row["unobservable"]]
    present = [row for row in observable if row["variant"] == "menu"]
    removed = [row for row in observable if row["variant"] != "menu"]
    result = {
        "variant": variant, "threshold": threshold["threshold"],
        "held_out_faults": metadata["held_out_faults"],
        "observable": summarise(observable, f"hybrid:{variant}:observable"),
        "unobservable": summarise(unobservable, f"hybrid:{variant}:unobservable"),
        "menu_present": summarise(present, f"hybrid:{variant}:present"),
        "menu_removed": summarise(removed, f"hybrid:{variant}:removed"),
        "menu_present_classifier_only": summarise(
            [dict(row, pred=row["classifier_label"]) for row in present],
            f"classifier:{variant}:present"),
        "rule_reasons": dict(Counter(row["rule_reason"] for row in rows)),
        "verdicts": dict(Counter(row["verdict"] for row in rows)),
    }
    # The adapter on exactly the same rows, so the comparison is paired.
    if ADAPTER_PREDS.exists():
        adapter = {row["item_id"]: row for row in load_jsonl(ADAPTER_PREDS)}
        same = [row for row in observable if row["item_id"] in adapter]
        result["adapter_same_rows"] = summarise(
            [adapter[row["item_id"]] for row in same], "adapter:same_rows")
        result["hybrid_same_rows"] = summarise(same, "hybrid:same_rows")
        result["mcnemar_adapter_vs_hybrid"] = ev.mcnemar(
            [adapter[row["item_id"]] for row in same], same)
    return result


def surrogate(out: Path, faults: Sequence[int] = (5, 8, 12, 13)) -> Dict:
    """Leave-one-fault-out: does the rule abstain when a class was never trained?"""
    x_train_all, y_train_all, _ = dataset(TRAIN)
    x_val, y_val, items = dataset(VAL, dedupe=False)
    result = {}
    for fault in faults:
        label = f"FAULT_{fault:02d}"
        keep = [index for index, value in enumerate(y_train_all) if value != label]
        model, labels = fit(x_train_all[keep], [y_train_all[index] for index in keep])
        probs = predict_proba(model, labels, x_val)
        truths = [true_label(item) for item in items]
        # Calibrate on the other classes only: a class the model never saw must not tune
        # its own calibration.
        others = [index for index, value in enumerate(truths) if value != label]
        cv = cross_validated_ece([probs[i] for i in others],
                                 [truths[i] for i in others], labels)
        chosen = min(cv, key=cv.get)
        calibrator = FITTERS[chosen]([probs[i] for i in others],
                                     [truths[i] for i in others], labels)
        calibrated = apply_calibrator(calibrator, probs)
        confidences = [max(calibrated[i].values()) for i in others
                       if items[i].get("variant") == "menu" and truths[i] in set(labels)]
        threshold = largest_threshold(confidences)["threshold"]

        held = [i for i, value in enumerate(truths)
                if value == label and items[i].get("variant") == "menu"]
        abstained = 0
        mislabelled = Counter()
        for index in held:
            pred = max(calibrated[index], key=calibrated[index].get)
            decision = menu_rule(pred, calibrated[index].get(pred),
                                 items[index].get("menu_ids") or [], threshold)
            if decision["abstained"]:
                abstained += 1
            else:
                mislabelled[pred] += 1
        false_abstention = [
            max(calibrated[i].values()) < threshold for i in others
            if items[i].get("variant") == "menu" and truths[i] in set(labels)
        ]
        result[label] = {
            "held_out_fault": fault, "validation_items": len(held),
            "abstained": abstained,
            "abstained_rate": round(abstained / max(len(held), 1), 4),
            "mislabelled": dict(mislabelled.most_common(5)),
            "threshold": threshold,
            "false_abstention_on_other_faults": round(
                sum(false_abstention) / max(len(false_abstention), 1), 4),
        }
    (out / "surrogate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    sub = parser.add_subparsers(dest="command", required=True)
    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--variant", default="all",
                           choices=["all", *FOLD_HOLDOUT.keys()])
    sub.add_parser("evaluate")
    sub.add_parser("surrogate")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        variants = list(FOLD_HOLDOUT) if args.variant == "all" else [args.variant]
        summary = {}
        for variant in variants:
            metadata = train_variant(variant, out)
            summary[variant] = {key: metadata[key] for key in
                                ("train_events", "features", "calibration", "threshold",
                                 "feature_audit", "files")}
            print(json.dumps({variant: summary[variant]}, indent=2)[:1200], flush=True)
        manifest = {"variants": summary, "seed": SEED,
                    "train_file": str(TRAIN), "val_file": str(VAL)}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elif args.command == "evaluate":
        result = evaluate(out)
        (out / "evaluation.json").write_text(json.dumps(result, indent=2, default=str),
                                             encoding="utf-8")
        print(json.dumps({key: (value if not isinstance(value, dict) else
                                {name: value[name] for name in
                                 ("items", "accuracy", "macro_f1", "top3") if name in value})
                          for key, value in result.items()}, indent=2, default=str)[:2500])
    else:
        result = surrogate(out)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
