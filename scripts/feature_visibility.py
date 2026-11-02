"""Which evidence does the classifier use, and is it visible to the adapter too?

The classifier in the hybrid reads 323 numbers parsed out of the rendered prompt, and it beats
the adapter on the same items. Two explanations survive that fact:

* the classifier uses something the prompt does not show (it cannot — every feature is parsed
  from the prompt text, this script re-derives them and the truncation check in
  ``scripts/prompt_budget.py`` shows nothing is cut), or
* the classifier uses the same evidence, aggregated, and the adapter does not use it.

This script settles which by measuring, per feature group, how much the classifier leans on it
for the **gold class** of an item (LightGBM's per-class SHAP contributions, which are additive
and signed — a gain importance would be class-blind), and then checking where that group is
printed in the prompt: in full, summarised to the top-k sensors, or absent.

Rows come from the adapter's own record: the 48 menu-present validation items the step-500
checkpoint got wrong, and a seeded sample of 100 it got right. CPU only, no GPU, no model.

Usage::

    python scripts/feature_visibility.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

from scripts.evidence_classifier import (  # noqa: E402
    GLOBALS,
    PER_TAG,
    TAGS,
    feature_names,
    fit_trees,
    identifier_audit,
    prompt_features,
)

DATASET = ROOT / "outputs" / "benchmark_v2" / "dataset"
PREDS = ROOT / "outputs/analysis/phase3a/instrument_check/preds_decode_step500_full.jsonl"
OUT = ROOT / "outputs/analysis/phase3a/fit"
SEED = 42
CORRECT_SAMPLE = 100

#: Feature-group → the group's own visibility in the rendered prompt.
VISIBILITY = {
    "detector_globals": "in full (overall score, both detector scores and their ratios, lag, "
                        "slow-drift ratio, fused trigger, number of listed sensors)",
    "z": "summarised: the top-5 sensors by |z| are listed, the other 47 tags are only implicit",
    "deviation": "summarised: same 5 sensor lines, deviation % printed per line",
    "trend": "summarised: same 5 sensor lines, trend word printed where recorded",
    "onset": "summarised: the ONSET ORDER block lists the earliest sensors with +min",
    "residual": "summarised: the CVA residual block lists its top variables' shares",
    "present": "implicit: a tag that is absent from the list is absent from the prompt too",
}
GROUP_ORDER = ("detector_globals", "z", "deviation", "trend", "onset", "residual", "present")


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def groups() -> Dict[str, List[int]]:
    """Column indices per group, derived from the feature names themselves."""
    names = feature_names()
    out: Dict[str, List[int]] = {"detector_globals": []}
    for kind in ("z", "deviation", "trend", "onset", "residual", "present"):
        out[kind] = []
    for index, name in enumerate(names):
        if name in GLOBALS:
            out["detector_globals"].append(index)
            continue
        out[name.split(":")[-1]].append(index)
    total = sum(len(value) for value in out.values())
    assert total == len(names), (total, len(names))
    return out


def normalise(label: str) -> str:
    if str(label).startswith("FAULT_"):
        number = int(str(label).split("_")[-1])
        if 16 <= number <= 20:
            return "UNCLASSIFIED_OPEN_SET"
    return str(label)


def prompt_shape(prompt: str) -> Dict[str, int]:
    """How many entries each summarised block actually prints."""
    sensors = sum(1 for line in prompt.splitlines()
                  if line.startswith("- ") and " z=" in line)
    onset = sum(1 for line in prompt.splitlines()
                if line[:1].isdigit() and "] at +" in line)
    residual = sum(1 for line in prompt.splitlines()
                   if line.startswith("- ") and "residual share" in line)
    return {"sensor_lines": sensors, "onset_lines": onset, "residual_lines": residual}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train = [row for row in load_jsonl(DATASET / "train.jsonl") if row.get("variant") == "menu"]
    val = [row for row in load_jsonl(DATASET / "val.jsonl") if row.get("variant") == "menu"]
    x_train = np.asarray([prompt_features(row["prompt"]) for row in train], dtype=np.float32)
    y_train = [normalise(row.get("label") or (row.get("target") or {}).get("fault_id"))
               for row in train]
    x_val = np.asarray([prompt_features(row["prompt"]) for row in val], dtype=np.float32)
    names = feature_names()

    # Same estimator, same seed as the recorded classifier comparison: this is a read of that
    # model, not a new tuning.
    import lightgbm as lgb

    labels = sorted(set(y_train))
    encoder = {label: index for index, label in enumerate(labels)}
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(labels), n_estimators=400,
        learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.9,
        subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, random_state=args.seed,
        verbose=-1,
    )
    model.fit(x_train, [encoder[label] for label in y_train])
    ordered = [labels[int(index)] for index in model.classes_]

    probing = model.booster_.predict(x_val[:2], pred_contrib=True)
    n_features = x_val.shape[1] + 1
    shaped = probing.reshape(2, len(labels), n_features) if probing.ndim == 2 else probing
    assert shaped.shape == (2, len(labels), n_features), shaped.shape

    preds = load_jsonl(PREDS)
    gold = {row["item_id"]: row["gold"] for row in preds}
    correct_ids = [row["item_id"] for row in preds
                   if row.get("variant") == "menu" and not row.get("unobservable")
                   and row.get("pred") == row.get("gold")]
    error_ids = [row["item_id"] for row in preds
                 if row.get("variant") == "menu" and not row.get("unobservable")
                 and row.get("pred") != row.get("gold")]
    sample = sorted(random.Random(args.seed).sample(correct_ids, CORRECT_SAMPLE))

    index_of = {row["item_id"]: index for index, row in enumerate(val)}
    group_index = groups()

    def profile(ids: Sequence[str], label: str) -> Dict:
        rows = [index_of[item] for item in ids if item in index_of]
        matrix = x_val[rows]
        contrib = model.booster_.predict(matrix, pred_contrib=True)
        shaped = (contrib.reshape(len(rows), len(labels), n_features)
                  if contrib.ndim == 2 else contrib)
        per_group: Dict[str, float] = {}
        per_group_share: Dict[str, float] = {}
        for name in GROUP_ORDER:
            columns = group_index[name]
            magnitude = []
            for position, item_id in enumerate(ids):
                klass = normalise(gold.get(item_id, ""))
                if klass not in labels:
                    continue
                block = shaped[position, labels.index(klass)]
                magnitude.append(float(np.abs(block[columns]).sum()))
            per_group[name] = round(float(np.mean(magnitude)), 4) if magnitude else 0.0
        total = sum(per_group.values()) or 1.0
        per_group_share = {name: round(value / total, 4) for name, value in per_group.items()}
        shapes = [prompt_shape(val[index_of[item]]["prompt"]) for item in ids if item in index_of]
        return {
            "items": len(rows),
            "mean_abs_shap_for_gold_class_by_group": per_group,
            "share_of_total": per_group_share,
            "ranking": sorted(per_group_share, key=per_group_share.get, reverse=True),
            "prompt_shape": {
                key: {"min": min(shape[key] for shape in shapes),
                      "median": sorted(shape[key] for shape in shapes)[len(shapes) // 2],
                      "max": max(shape[key] for shape in shapes)}
                for key in ("sensor_lines", "onset_lines", "residual_lines")},
        }

    report = {
        "model": {"n_features": len(names), "classes": len(labels), "seed": args.seed,
                  "feature_audit": identifier_audit(names)},
        "adapter_errors": profile(error_ids, "errors"),
        "adapter_correct_sample": profile(sample, "correct"),
        "visibility": VISIBILITY,
        "prompt_feature_basis": {
            "globals_printed": ["overall anomaly score", "LSTM reconstruction error",
                                "CVA Q-statistic", "dominant lag", "slow-drift ratio",
                                "fused trigger", "number of listed sensors"],
            "globals_derived": ["LSTM ratio = score / threshold; the ratio is printed, so the "
                                "classifier's ratio column is a re-parse, not new information"],
            "per_tag_printed": ["z", "deviation %", "trend word", "onset minute "
                                "(ONSET ORDER block)", "residual share (CVA block)"],
            "not_in_features": ["the candidate menu (the classifier never sees it)",
                                "the fault names and KB text",
                                "the question/instruction text and the schema line",
                                "the sensor display names and the tag kind (valve vs measurement)",
                                "any event, run, item or split identifier"],
            "not_in_prompt": ["the full 52-tag matrix: only the top-5 sensors by |z| are printed, "
                              "so a tag's absence is implicit on both sides"],
        },
        "error_pairs": {f"{gold_} -> {pred}": count for (gold_, pred), count in
                        Counter((row["gold"], row["pred"]) for row in preds
                                if row.get("variant") == "menu"
                                and not row.get("unobservable")
                                and row.get("pred") != row.get("gold")).most_common()},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "feature_visibility.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "features": len(names), "classes": len(labels),
        "errors": report["adapter_errors"]["share_of_total"],
        "correct": report["adapter_correct_sample"]["share_of_total"],
        "ranking_errors": report["adapter_errors"]["ranking"][:4],
        "sensor_lines": report["adapter_errors"]["prompt_shape"]["sensor_lines"],
    }, indent=2))
    print(f"written: {out / 'feature_visibility.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
