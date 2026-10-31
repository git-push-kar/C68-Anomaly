"""The classifier and the adapter on the items they were both asked about.

``scripts/evidence_classifier.py`` compares the gradient-boosted trees and the
small MLP against whatever adapter prediction file it is handed. This script
pins that comparison to the *frozen* closed-set instrument -- instrument (c),
greedy decoding of the first answer field, on the full validation split -- and
adds the paired test, because two accuracies that differ by a handful of items
mean nothing without knowing which items moved.

It also corrects an arithmetic slip in the run-1 report: 0.9940 over 497 items is
three errors, not two, and the classifier's own accuracy over 503 items implies
nine.

Usage::

    python scripts/classifier_vs_adapter.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark import evaluate_v2 as ev  # noqa: E402

OUT = ROOT / "outputs/analysis/phase3a/instrument_check"
CLASSIFIER = ROOT / "outputs/analysis/phase3a/evidence_classifier"


def load(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def score(rows: Sequence[Dict], label: str) -> Dict:
    report = ev.report([{key: row.get(key) for key in ("item_id", "gold", "pred", "probs", "run")}
                        for row in rows], label=label)
    return {
        "label": label,
        "items": report["items"],
        "errors": report["items"] - round(report["accuracy"] * report["items"]),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "abstention": report["abstention"],
        "per_class": {name: {"precision": info["precision"], "recall": info["recall"],
                             "f1": info["f1"], "support": info["support"]}
                      for name, info in report["per_fault"].items()},
    }


def accuracy_ci(rows: Sequence[Dict], n_resamples: int = 1000, seed: int = 42) -> Dict:
    """Run-clustered percentile interval for plain accuracy.

    ``evaluate_v2.run_clustered_bootstrap`` re-derives the whole per-class report
    on every resample; over 822 rows and two models that is half an hour for a
    number this script only needs to two decimals. Resampling the runs and
    averaging a boolean is the same estimator, minus the tables.
    """
    import random

    clusters: Dict[str, List[int]] = {}
    for row in rows:
        clusters.setdefault(str(row.get("run") or row["item_id"]), []).append(
            int(row["pred"] == row["gold"]))
    keys = sorted(clusters)
    if not keys:
        return {"point": float("nan"), "low": None, "high": None, "clusters": 0}
    rng = random.Random(seed)
    draw: List[float] = []
    for _ in range(n_resamples):
        hit = total = 0
        for _ in range(len(keys)):
            hits = clusters[keys[rng.randrange(len(keys))]]
            hit += sum(hits)
            total += len(hits)
        draw.append(hit / max(total, 1))
    draw.sort()
    point = sum(sum(v) for v in clusters.values()) / sum(len(v) for v in clusters.values())
    return {"point": round(point, 4),
            "low": round(draw[int(0.025 * (len(draw) - 1))], 4),
            "high": round(draw[int(0.975 * (len(draw) - 1))], 4),
            "clusters": len(keys), "resamples": len(draw)}


def per_class_delta(a: Dict, b: Dict) -> Dict:
    """``a`` minus ``b``, per class f1, on the shared classes only."""
    out = {}
    for name, info in a.get("per_class", {}).items():
        other = b.get("per_class", {}).get(name)
        if other is None:
            continue
        out[name] = {"adapter_f1": info["f1"], "classifier_f1": other["f1"],
                     "delta": round(info["f1"] - other["f1"], 4),
                     "support": info["support"]}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", default="500")
    parser.add_argument("--out", default=str(OUT / "classifier_vs_adapter.json"))
    args = parser.parse_args()

    adapter = [row for row in load(OUT / f"preds_decode_step{args.step}_full.jsonl")
               if not row.get("unobservable")]
    wanted = {row["item_id"] for row in adapter}
    trees = [row for row in load(CLASSIFIER / "preds_gbdt_val503.jsonl")
             if row["item_id"] in wanted]
    mlp = [row for row in load(CLASSIFIER / "preds_mlp_val503.jsonl")
           if row["item_id"] in wanted]
    shared = wanted & {row["item_id"] for row in trees}
    adapter_shared = [row for row in adapter if row["item_id"] in shared]
    trees_shared = [row for row in trees if row["item_id"] in shared]
    mlp_shared = [row for row in mlp if row["item_id"] in shared]

    result: Dict[str, object] = {
        "adapter_instrument": "decode",
        "adapter_step": args.step,
        "adapter_observable": len(adapter),
        "classifier_items": len(trees),
        "intersection": len(shared),
        "class_counts": dict(Counter(row["gold"] for row in adapter_shared)),
        "adapter": score(adapter_shared, "adapter:decode"),
        "gbdt": score(trees_shared, "gbdt"),
        "mlp": score(mlp_shared, "mlp"),
        "mcnemar_adapter_vs_gbdt": ev.mcnemar(adapter_shared, trees_shared),
        "mcnemar_adapter_vs_mlp": ev.mcnemar(adapter_shared, mlp_shared),
        "delta_adapter_minus_gbdt": per_class_delta(
            score(adapter_shared, "adapter:decode"), score(trees_shared, "gbdt")),
        "delta_adapter_minus_mlp": per_class_delta(
            score(adapter_shared, "adapter:decode"), score(mlp_shared, "mlp")),
        "bootstrap_adapter": accuracy_ci(adapter_shared),
        "bootstrap_gbdt": accuracy_ci(trees_shared),
        "bootstrap_mlp": accuracy_ci(mlp_shared),
    }

    # Pull the feature ranking and identifier audit from the classifier's own run
    # instead of recomputing them: the ranking is a property of that fitted model.
    summary_path = CLASSIFIER / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result["top_features"] = summary.get("top_features", [])[:25]
        result["feature_audit"] = summary.get("feature_audit")

    banned = ("item", "run", "event", "evt", "index", "order", "seed", "split",
              "hash", "simulation", "position")
    names = [row["feature"] for row in result.get("top_features", [])]
    offenders = [name for name in names if any(tok in name.lower() for tok in banned)]
    result["top_features_identifier_check"] = {"n": len(names),
                                              "offenders": offenders,
                                              "clean": not offenders}

    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(f"{'model':10s}{'n':>5s}{'errors':>8s}{'acc':>8s}{'mF1':>8s}")
    for name in ("adapter", "gbdt", "mlp"):
        row = result[name]
        print(f"{name:10s}{row['items']:5d}{row['errors']:8d}"
              f"{row['accuracy']:8.4f}{row['macro_f1']:8.4f}")
    for pair in ("gbdt", "mlp"):
        m = result[f"mcnemar_adapter_vs_{pair}"]
        print(f"McNemar adapter vs {pair}: n={m['items']} b={m['b']} c={m['c']} "
              f"p={m['p_value']} ({m['method']})")
    print("top features clean:", result["top_features_identifier_check"])
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
