"""Cluster bootstrap for the cross-fitted accuracy and macro-F1.

The RP quotes point estimates (0.9878 / 0.9843) without intervals. This computes them from the stored
cross-fitted rows, resampling **events**, not rows: each event contributes a menu-present and a menu-removed
page, so resampling rows would treat the same physical incident as two independent draws and narrow the interval
for no reason.

    python scripts/crossfit_bootstrap.py
    python scripts/crossfit_bootstrap.py --resamples 5000 --out .../crossfit_bootstrap.json

The population is the one the stratified table uses: every cross-fitted validation row except the two
unobservable faults (3 and 9), 822 rows over 417 events. The point estimate is recomputed here and must match
`crossfit.json`'s `validation` entry before the interval means anything; a mismatch is reported as such rather
than silently bootstrapped.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import defaultdict
from typing import Dict, List, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_ROWS = ROOT / "outputs/analysis/phase3a/hybrid/vocabfix/crossfit_val_rows.jsonl"  # 822 observable rows
DEFAULT_OUT = ROOT / "outputs/analysis/phase3a/hybrid/vocabfix/crossfit_bootstrap.json"
CROSSFIT = ROOT / "outputs/analysis/phase3a/hybrid/vocabfix/crossfit.json"
UNOBSERVABLE = {"FAULT_03", "FAULT_09"}


def cluster_of(row: Dict) -> str:
    return str(row.get("item_id", "")).split("::")[0] or str(row.get("run"))


def observable(rows: Sequence[Dict]) -> List[Dict]:
    """The 822-row population the stratified table uses: the stored ``unobservable`` flag decides.

    The flag is what the crossfit script wrote; the gold label is the *menu answer* for open-set twins, so
    filtering on ``gold`` here would silently select a different population.
    """
    return [row for row in rows if not row.get("unobservable")]


def scores(rows: Sequence[Dict]) -> Dict[str, float]:
    ordered = [row for row in rows if row.get("pred")]
    if not ordered:
        return {"accuracy": 0.0, "macro_f1": 0.0, "items": 0}
    correct = sum(1 for row in ordered if row["pred"] == row["gold"])
    labels = sorted({row["gold"] for row in ordered})
    f1_values = []
    for label in labels:
        tp = sum(1 for row in ordered if row["gold"] == label and row["pred"] == label)
        fp = sum(1 for row in ordered if row["gold"] != label and row["pred"] == label)
        fn = sum(1 for row in ordered if row["gold"] == label and row["pred"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": round(correct / len(ordered), 6),
            "macro_f1": round(sum(f1_values) / max(len(f1_values), 1), 6),
            "items": len(ordered)}


def interval(values: Sequence[float], level: float = 0.95) -> Dict[str, float]:
    ordered = sorted(values)
    tail = (1.0 - level) / 2.0
    low = ordered[int(tail * (len(ordered) - 1))]
    high = ordered[int((1.0 - tail) * (len(ordered) - 1))]
    return {"low": round(low, 6), "high": round(high, 6)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=pathlib.Path, default=DEFAULT_ROWS)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = observable([json.loads(line) for line in
                       args.rows.read_text(encoding="utf-8").splitlines()])
    clustered: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        clustered[cluster_of(row)].append(row)
    clusters = sorted(clustered)
    point = scores(rows)

    rng = random.Random(args.seed)
    accuracy_draws: List[float] = []
    f1_draws: List[float] = []
    for _ in range(args.resamples):
        draw = [row for key in (rng.choice(clusters) for _ in clusters) for row in clustered[key]]
        values = scores(draw)
        accuracy_draws.append(values["accuracy"])
        f1_draws.append(values["macro_f1"])

    quoted = None
    if CROSSFIT.exists():
        validation = json.loads(CROSSFIT.read_text(encoding="utf-8")).get("validation", {})
        observable_entry = validation.get("observable") or {}
        quoted = {"accuracy": observable_entry.get("accuracy"),
                  "macro_f1": observable_entry.get("macro_f1")}

    outcome = {
        "rows": len(rows),
        "clusters": len(clusters),
        "resamples": args.resamples,
        "seed": args.seed,
        "population": "cross-fitted validation rows, faults 3 and 9 excluded",
        "unit": "event (cluster), not row",
        "point": point,
        "accuracy_ci": interval(accuracy_draws),
        "macro_f1_ci": interval(f1_draws),
        "quoted_in_crossfit_json": quoted,
        "matches_quoted_point_estimate": bool(
            quoted and quoted.get("accuracy") is not None
            and abs(float(quoted["accuracy"]) - point["accuracy"]) < 1e-4),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
