"""§17's cross-fitted rule on the *training* events, so the prose set can exclude its disagreements.

The pilot supplies the gold label in the prompt. A row whose supplied label the §17 rule would not
have produced teaches prose for a situation the system never creates, so those rows are counted and
dropped (PROTOCOL §23.2, 3B-11 step 4). The rule has to be the cross-fitted one — the same out-of-fold
classifier and threshold — not the in-sample model, which would be optimistic about itself.

Writes one row per training event: the out-of-fold verdict, the label it would print, and its top-2
with calibrated probabilities (the prompt's "supplied" numbers).

Usage::  python scripts/prose_crossfit_train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hybrid_classifier as hc  # noqa: E402
from llm.hybrid import calibration as cal  # noqa: E402
from llm.hybrid.menu_rule import NORMAL, UNCLASSIFIED  # noqa: E402

OUT = ROOT / "outputs/analysis/phase3a/hybrid"
CROSSFIT = OUT / "crossfit.json"
DEST = OUT / "crossfit_train_oof.jsonl"


def fit_predict(features: np.ndarray, labels, query: np.ndarray) -> List[Dict[str, float]]:
    model, ordered = hc.fit(features, labels)
    return hc.predict_proba(model, ordered, query)


def verdict_of(item: Dict, probs: Dict[str, float], threshold: float) -> Dict:
    pred = max(probs, key=probs.get)
    confidence = float(probs[pred])
    on_menu = pred in set(item.get("menu_ids") or [])
    abstained = (pred not in (NORMAL, UNCLASSIFIED)
                 and (not on_menu or confidence < threshold))
    if abstained:
        label = NORMAL if pred == NORMAL else UNCLASSIFIED
        verdict = "no_fault" if label == NORMAL else "unclassified_open_set"
    else:
        label, verdict = pred, "classified"
    top = sorted(probs.items(), key=lambda item_: -item_[1])[:2]
    return {"verdict": verdict, "label": label, "raw_pred": pred, "confidence": confidence,
            "abstained": abstained, "top2": [[str(name), round(float(value), 4)]
                                             for name, value in top]}


def main() -> int:
    threshold = float(json.loads(CROSSFIT.read_text(encoding="utf-8"))["threshold"]["threshold"])
    features, labels, items = hc.dataset(hc.TRAIN)
    print(json.dumps({"train_events": len(items), "features": int(features.shape[1]),
                      "threshold": threshold}), flush=True)
    oof, fold = cal.cross_fit(features, labels, items, fit_predict, k=5)
    rows = []
    for index, item in enumerate(items):
        row = {"event_key": cal.event_key(item), "gold": hc.true_label(item),
               "item_id": item.get("item_id"),
               "event": str(item.get("item_id") or "").split("::")[0],
               "fold": int(fold[index]) if fold is not None else None}
        row.update(verdict_of(item, oof[index], threshold))
        rows.append(row)
    DEST.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    agree = sum(1 for row in rows if row["label"] == row["gold"])
    print(json.dumps({"events": len(rows), "label_agrees_with_gold": agree,
                      "written": str(DEST)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
