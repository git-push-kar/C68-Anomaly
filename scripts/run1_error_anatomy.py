"""What the 48 closed-set errors at step 500 are made of.

The instrument report says where the adapter fails: 48 wrong answers on the 497
validation items whose true entry is on the menu, 46 of which the GBDT gets right.
That is a rate, not a diagnosis. This script asks the questions that separate the
plausible causes:

* which faults are confused with which -- a block structure points at the KB's own
  differential pairs, a diagonal-free scatter points at the evidence;
* whether the errors are the *weak* events (lower overall score, lower fused ratio,
  smaller top |z|) or indistinguishable from the ones it gets right;
* whether the errors depend on where the true entry sits in the menu (a position
  effect would mean the model is reading the list, not the evidence);
* what the strongest tabular baseline thought of the right answer on exactly those
  rows, which distinguishes "this event is genuinely ambiguous" from "the adapter
  threw the information away".

No model is called here: it reads the step-500 decode file, the validation prompts
and the GBDT prediction file.

Usage::

    python scripts/run1_error_anatomy.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
DECODE = ROOT / "outputs/analysis/phase3a/instrument_check/preds_decode_step500_full.jsonl"
GBDT = ROOT / "outputs/analysis/phase3a/evidence_classifier/preds_gbdt_val503.jsonl"
OUT = ROOT / "outputs/analysis/phase3a/run1_error_anatomy.json"

SCORE = re.compile(r"Overall anomaly score:\s*([\d.]+)")
LSTM_RATIO = re.compile(r"LSTM autoencoder reconstruction error\s*[\d.]+\s*"
                        r"\(threshold\s*[\d.]+,\s*ratio\s*([\d.]+)x")
CVA_RATIO = re.compile(r"CVA Q-statistic\s*[\d.]+\s*\(threshold\s*[\d.]+,\s*ratio\s*([\d.]+)x")
Z_VALUE = re.compile(r"z=(-?[\d.]+)")
ONSET = re.compile(r"^\d+\.\s+.*?at \+([\d.]+) min", re.MULTILINE)
MENU_LINE = re.compile(r"^-\s+(FAULT_\d{2}|UNCLASSIFIED_OPEN_SET):\s*(.+)$", re.MULTILINE)
SENSOR_LINE = re.compile(r"^-\s+(.+?)\s*\[([A-Z]+_\d+)[^\]]*\]:\s*z=(-?[\d.]+),\s*"
                         r"deviation\s*(-?[\d.]+)%.*?trend\s+(\w+)", re.MULTILINE)


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def evidence_of(prompt: str) -> Dict:
    """The evidence numbers a reader would use, pulled out of the rendered prompt."""
    sensors = [{"name": match.group(1).strip(), "tag": match.group(2),
                "z": float(match.group(3)), "deviation": float(match.group(4)),
                "trend": match.group(5)} for match in SENSOR_LINE.finditer(prompt)]
    lstm = LSTM_RATIO.search(prompt)
    cva = CVA_RATIO.search(prompt)
    onsets = [float(value) for value in ONSET.findall(prompt)]
    score = SCORE.search(prompt)
    menu = [(match.group(1), match.group(2).strip()) for match in MENU_LINE.finditer(prompt)]
    return {
        "overall_score": float(score.group(1)) if score else None,
        "lstm_ratio": float(lstm.group(1)) if lstm else None,
        "cva_ratio": float(cva.group(1)) if cva else None,
        "fused_ratio": max([value for value in
                            (float(lstm.group(1)) if lstm else None,
                             float(cva.group(1)) if cva else None) if value is not None],
                           default=None),
        "top_abs_z": max((abs(sensor["z"]) for sensor in sensors), default=None),
        "sensors": sensors,
        "onset_delay": onsets[0] if onsets else None,
        "onset_count": len(onsets),
        "menu": menu,
        "menu_size": len(menu),
    }


def permutation_p(a: Sequence[float], b: Sequence[float], seed: int = 42,
                  n: int = 2000) -> Optional[float]:
    """Two-sided permutation test on the difference in means.

    The sample sizes here are small (48 vs 449) and the distributions are skewed, so
    a t-test would be an assertion. Shuffling the labels makes no distributional
    assumption beyond exchangeability.
    """
    a = [value for value in a if value is not None]
    b = [value for value in b if value is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    observed = abs(statistics.mean(a) - statistics.mean(b))
    pooled = a + b
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        rng.shuffle(pooled)
        left = pooled[:len(a)]
        right = pooled[len(a):]
        if abs(statistics.mean(left) - statistics.mean(right)) >= observed:
            hits += 1
    return round((hits + 1) / (n + 1), 4)


def describe(values: Sequence[Optional[float]]) -> Dict:
    clean = [value for value in values if value is not None]
    if not clean:
        return {"n": 0}
    clean.sort()
    return {"n": len(clean), "mean": round(statistics.mean(clean), 3),
            "median": round(statistics.median(clean), 3),
            "p10": round(clean[int(0.1 * (len(clean) - 1))], 3),
            "p90": round(clean[int(0.9 * (len(clean) - 1))], 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode", default=str(DECODE))
    parser.add_argument("--gbdt", default=str(GBDT))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    items = {item["item_id"]: item for item in load_jsonl(VAL)}
    predictions = [row for row in load_jsonl(Path(args.decode))
                   if row.get("variant") == "menu" and not row.get("unobservable")]
    gbdt = {row["item_id"]: row for row in load_jsonl(Path(args.gbdt))}

    rows = []
    for prediction in predictions:
        item = items[prediction["item_id"]]
        evidence = evidence_of(item["prompt"])
        gold = prediction["gold"]
        position = None
        for index, (fault_id, _signature) in enumerate(evidence["menu"], start=1):
            if fault_id == gold:
                position = index
                break
        gbdt_row = gbdt.get(prediction["item_id"]) or {}
        rows.append({
            "item_id": prediction["item_id"],
            "gold": gold, "pred": prediction["pred"], "correct": prediction["pred"] == gold,
            "run": prediction.get("run"),
            "fault": item.get("fault"), "simulationRun": item.get("simulationRun"),
            "menu_position": position, "menu_size": evidence["menu_size"],
            "overall_score": evidence["overall_score"],
            "fused_ratio": evidence["fused_ratio"],
            "lstm_ratio": evidence["lstm_ratio"], "cva_ratio": evidence["cva_ratio"],
            "top_abs_z": evidence["top_abs_z"],
            "onset_delay": evidence["onset_delay"], "onset_count": evidence["onset_count"],
            "top_sensors": [f"{sensor['name']} [{sensor['tag']}] z={sensor['z']:+.2f} "
                            f"trend {sensor['trend']}" for sensor in
                            evidence["sensors"][:3]],
            "menu_lines": [f"{fault_id}: {signature}"
                           for fault_id, signature in evidence["menu"]],
            "gbdt_pred": gbdt_row.get("pred"),
            "gbdt_gold_probability": (gbdt_row.get("probs") or {}).get(gold),
            "gbdt_pred_probability": max((gbdt_row.get("probs") or {"": 0.0}).values())
            if gbdt_row.get("probs") else None,
        })

    errors = [row for row in rows if not row["correct"]]
    correct = [row for row in rows if row["correct"]]

    confusion = Counter((row["gold"], row["pred"]) for row in errors)
    by_gold = Counter(row["gold"] for row in errors)
    by_pred = Counter(row["pred"] for row in errors)

    strength = {}
    for name in ("overall_score", "fused_ratio", "lstm_ratio", "cva_ratio", "top_abs_z",
                 "onset_delay", "onset_count"):
        strength[name] = {
            "errors": describe([row[name] for row in errors]),
            "correct": describe([row[name] for row in correct]),
            "permutation_p": permutation_p([row[name] for row in errors],
                                           [row[name] for row in correct]),
        }

    # Menu position: the menu is presented in fault order, so a position effect that
    # survives per-fault pooling would be a list-order artefact, not a fault effect.
    position = defaultdict(lambda: {"items": 0, "errors": 0})
    for row in rows:
        key = row["menu_position"]
        position[key]["items"] += 1
        position[key]["errors"] += int(not row["correct"])

    by_fault = defaultdict(lambda: {"items": 0, "errors": 0})
    for row in rows:
        entry = by_fault[row["gold"]]
        entry["items"] += 1
        entry["errors"] += int(not row["correct"])
    for entry in by_fault.values():
        entry["error_rate"] = round(entry["errors"] / entry["items"], 4)

    examples = {}
    for fault in ("FAULT_10", "FAULT_15"):
        picked = [row for row in errors if row["gold"] == fault][: args.examples]
        examples[fault] = [{key: row[key] for key in
                            ("item_id", "gold", "pred", "overall_score", "fused_ratio",
                             "lstm_ratio", "cva_ratio", "top_abs_z", "onset_delay",
                             "onset_count", "top_sensors", "gbdt_pred",
                             "gbdt_gold_probability", "menu_position")} |
                           {"gold_menu_line": next((line for line in row["menu_lines"]
                                                    if line.startswith(row["gold"] + ":")), None)}
                           for row in picked]

    gbdt_on_errors = [row for row in errors if row["gbdt_pred"] is not None]
    result = {
        "items": len(rows), "errors": len(errors),
        "accuracy": round(len(correct) / max(len(rows), 1), 4),
        "confusion": [{"gold": gold, "pred": pred, "count": count}
                      for (gold, pred), count in confusion.most_common()],
        "errors_by_gold": dict(by_gold.most_common()),
        "errors_by_prediction": dict(by_pred.most_common()),
        "by_fault": {key: by_fault[key] for key in sorted(by_fault)},
        "menu_position": {str(key): value for key, value in
                          sorted(position.items(), key=lambda pair: (pair[0] is None,
                                                                     pair[0] or 0))},
        "menu_position_note": ("one item per (event, fault) pair, so the true entry's "
                               "position is the fault's index in the KB order -- a "
                               "position effect and a fault effect are the same "
                               "direction here and cannot be separated"),
        "strength": strength,
        "gbdt": {
            "rows_with_prediction": len(gbdt_on_errors),
            "gbdt_correct_on_adapter_errors": sum(
                1 for row in gbdt_on_errors if row["gbdt_pred"] == row["gold"]),
            "gbdt_gold_probability": describe([row["gbdt_gold_probability"]
                                               for row in gbdt_on_errors]),
            "gbdt_gold_probability_below_0_5": sum(
                1 for row in gbdt_on_errors
                if (row["gbdt_gold_probability"] or 0.0) < 0.5),
        },
        "examples": examples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("items", "errors", "accuracy", "errors_by_gold",
                       "errors_by_prediction", "gbdt")}, indent=2))
    print(json.dumps({"strength": {name: {"errors_median": value["errors"].get("median"),
                                          "correct_median": value["correct"].get("median"),
                                          "p": value["permutation_p"]}
                                   for name, value in strength.items()}}, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
