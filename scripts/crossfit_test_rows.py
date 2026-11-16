"""Per-item candidate probabilities for the **test** split, mirroring `crossfit_val_rows.jsonl`.

D33 needs condition A rebuilt on the test rows, and `fallback_v31` builds A from the classifier's candidate
probabilities. The test split does not carry them (`test.jsonl` has no `probs` field), so they are computed
the same way validation's were: the §16 Protocol A model and the §17 cross-fitted calibrator, applied to the
test feature matrix. Nothing is refitted on test.

This reproduces the validation path exactly; `--check-val` re-runs validation through the same code and
asserts the resulting probabilities match the stored `crossfit_val_rows.jsonl`, so the test numbers cannot
come from a different construction than the ones the cell table used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hybrid_classifier as hc  # noqa: E402
from llm.hybrid import calibration as cal  # noqa: E402

OUT = ROOT / "outputs/analysis/phase3a/hybrid"
VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
TEST = ROOT / "outputs/benchmark_v2/dataset/test.jsonl"


def label_of(item: dict) -> str:
    return hc.true_label(item)


def fit_predict(features, labels, query):
    model, ordered = hc.fit(features, labels)
    return hc.predict_proba(model, ordered, query)


def build(matrix, items, model, metadata, calibrator, threshold):
    probs = hc.probabilities(model, metadata, calibrator, matrix)
    rows = hc.rows_with_rule(items, probs, threshold)
    out = []
    for row, item, prob in zip(rows, items, probs):
        out.append({"item_id": item["item_id"], "pred": row.get("pred"),
                    "verdict": row.get("verdict"),
                    "probs": {str(k): float(v) for k, v in dict(prob).items()}})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=str(OUT / "crossfit_test_rows.jsonl"))
    ap.add_argument("--check-val", action="store_true")
    args = ap.parse_args()

    features, labels, items = hc.dataset(hc.TRAIN)
    oof, fold = cal.cross_fit(features, labels, items, fit_predict, k=args.folds)
    truth = [label_of(item) for item in items]
    model, metadata, _old_cal, _old_thr = hc.load_variant("protocolA", hc.OUT)
    class_labels = metadata["train_classes"]
    choice = cal.cross_fit_calibration(oof, truth, fold, class_labels, k=args.folds)
    calibrator = choice["calibrator"]
    oof_cal = cal.apply_calibrator(calibrator, oof)
    confidences = [cal.top_two(prob)[0] for item, prob in zip(items, oof_cal)
                   if item.get("variant") == "menu" and label_of(item) in set(class_labels)]
    threshold = cal.threshold_for(confidences, budget=0.05)["threshold"]

    if args.check_val:
        stored = {json.loads(line)["item_id"]: json.loads(line)
                  for line in (OUT / "crossfit_val_rows.jsonl").read_text(encoding="utf-8").splitlines()
                  if line.strip()}
        matrix, _labels, val_items = hc.dataset(VAL, dedupe=False)
        rebuilt = build(matrix, val_items, model, metadata, calibrator, threshold)
        same = sum(1 for row in rebuilt
                   if abs(max(stored[row["item_id"]]["probs"].values())
                          - max(row["probs"].values())) < 1e-6)
        print(json.dumps({"check": "val", "rows": len(rebuilt), "probs_match": same}), flush=True)
        if same != len(rebuilt):
            raise SystemExit("validation rebuild does not match the stored rows")

    matrix, _labels, test_items = hc.dataset(TEST, dedupe=False)
    built = build(matrix, test_items, model, metadata, calibrator, threshold)
    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in built:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"rows": len(built), "threshold": round(float(threshold), 4),
                      "out": str(out)}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
