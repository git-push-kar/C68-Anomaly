"""§17.4: the leave-one-fault-out surrogate, widened to every fault, with the four signals.

For each fault the model is retrained without it, the calibrator is cross-fitted **without that
fault's events**, and the threshold is derived by §17.2 from out-of-fold confidences of the
remaining events. The held-out fault's validation items are then scored: the honest question is
not "does it name the fault" (it cannot) but "does it refuse, and which signal would have told
it to".

Shards are deterministic: each model is a single-threaded, deterministic LightGBM, so running
several faults in parallel processes cannot change a number.

Usage::

    python scripts/hybrid_surrogate_wide.py --faults 1,2,4,5 --out outputs/.../surrogate_shard1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"


def fault_of(item: Dict) -> str:
    return hc.true_label(item)


def one_fault(fault: int, features: np.ndarray, labels: Sequence[str], items: Sequence[Dict],
              val_matrix: np.ndarray, val_items: Sequence[Dict],
              val_truth: Sequence[str], k: int = 5) -> Dict:
    label = f"FAULT_{fault:02d}"
    keep = [index for index, value in enumerate(labels) if value != label]
    kept_features = features[keep]
    kept_labels = [labels[index] for index in keep]
    kept_items = [items[index] for index in keep]

    oof, fold = cal.cross_fit(kept_features, kept_labels, kept_items, cf.fit_predict, k=k)
    choice = cal.cross_fit_calibration(oof, kept_labels, fold, sorted(set(kept_labels)), k=k)
    calibrator = choice["calibrator"]
    oof_calibrated = cal.apply_calibrator(calibrator, oof)
    confidences = [cal.top_two(prob)[0] for item, prob in zip(kept_items, oof_calibrated)
                   if item.get("variant") == "menu"
                   and fault_of(item) in set(kept_labels)]
    threshold = cal.threshold_for(confidences, budget=0.05)

    model, ordered = hc.fit(kept_features, kept_labels)
    raw = hc.predict_proba(model, ordered, val_matrix)
    calibrated = cal.apply_calibrator(calibrator, raw)
    novelty = cal.novelty_scores(val_matrix, kept_features, k=10)
    signals = cal.signal_values(calibrated, novelty)

    held = [i for i, truth in enumerate(val_truth) if truth == label]
    others = [i for i, truth in enumerate(val_truth) if truth in set(kept_labels)]
    # `UNCLASSIFIED_OPEN_SET` is itself a menu entry, so "unclassified" can arrive either as the
    # menu rule's withholding or as the classifier's own class. Both mean the same thing to a
    # reader, and neither is a name, so the split that matters is named / declined / correct.
    held_menu = 0
    per_variant: Dict[str, Dict] = {}
    named_by_variant: Dict[str, Counter] = {}
    for variant in ("menu", "removed"):
        rows = [i for i in held if (val_items[i].get("variant") == "menu") == (variant == "menu")]
        if variant == "menu":
            held_menu = len(rows)
        declined = named = correct = 0
        counts: Counter = Counter()
        for index in rows:
            pred = max(calibrated[index], key=calibrated[index].get)
            decision = menu_rule(pred, calibrated[index].get(pred),
                                 val_items[index].get("menu_ids") or [], threshold["threshold"])
            diagnosis = decision["diagnosis"]
            if diagnosis == UNCLASSIFIED:
                declined += 1
            else:
                named += 1
                counts[diagnosis] += 1
            gold = str((val_items[index].get("target") or {}).get("fault_id") or "")
            if diagnosis == gold:
                correct += 1
        per_variant[variant] = {
            "items": len(rows), "declined": declined,
            "declined_rate": round(declined / max(len(rows), 1), 4),
            "named": named, "named_rate": round(named / max(len(rows), 1), 4),
            "correct": correct,
            "accuracy": round(correct / max(len(rows), 1), 4),
        }
        named_by_variant[variant] = counts

    false_abstention = 0
    for index in others:
        if val_items[index].get("variant") != "menu":
            continue
        pred = max(calibrated[index], key=calibrated[index].get)
        if menu_rule(pred, calibrated[index].get(pred),
                     val_items[index].get("menu_ids") or [],
                     threshold["threshold"])["abstained"]:
            false_abstention += 1
    others_menu = sum(1 for index in others if val_items[index].get("variant") == "menu")

    # Signals are oriented first (§17.3: higher = more likely unseen), so AUROC is comparable
    # across them. Without the orientation, max_prob and margin score at 1 - AUROC and look
    # useless when they are the two that matter most.
    oriented = {name: cal.orient_values(name, values) for name, values in signals.items()}
    aurocs = {name: cal.auroc([values[i] for i in held], [values[i] for i in others])
              for name, values in oriented.items()}
    # The same comparison restricted to menu-present rows, which is what deployment sees.
    held_m = [i for i in held if val_items[i].get("variant") == "menu"]
    others_m = [i for i in others if val_items[i].get("variant") == "menu"]
    aurocs_menu = {name: cal.auroc([values[i] for i in held_m],
                                   [values[i] for i in others_m])
                   for name, values in oriented.items()}
    return {
        "fault": fault,
        "train_events_without_fault": len(kept_items),
        "calibration": {"chosen": choice["chosen"],
                        "oof_ece": {name: round(score, 5) for name, score in
                                    choice["scores"].items()}},
        "threshold": threshold,
        "held_out_items": len(held),
        "held_out_menu_items": held_menu,
        "by_variant": per_variant,
        "named_as": {variant: dict(counts.most_common(6))
                     for variant, counts in named_by_variant.items()},
        "false_abstention_on_others": round(false_abstention / max(others_menu, 1), 4),
        "auroc": {name: round(value, 4) for name, value in aurocs.items()},
        "auroc_menu_only": {name: round(value, 4) for name, value in aurocs_menu.items()},
        "counts": {"held": len(held), "others": len(others), "others_menu": others_menu},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faults", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default=str(OUT / "surrogate_wide.json"))
    args = parser.parse_args()
    faults = [int(value) for value in args.faults.split(",") if value.strip()]

    features, labels, items = hc.dataset(hc.TRAIN)
    val_matrix, _val_labels, val_items = hc.dataset(VAL, dedupe=False)
    val_truth = [fault_of(item) for item in val_items]

    result: Dict[str, Dict] = {}
    for fault in faults:
        block = one_fault(fault, features, labels, items, val_matrix, val_items,
                          val_truth, k=args.folds)
        result[f"FAULT_{fault:02d}"] = block
        print(json.dumps({f"FAULT_{fault:02d}": {
            key: block[key] for key in ("by_variant", "named_as", "threshold",
                                        "false_abstention_on_others", "auroc")}}), flush=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
