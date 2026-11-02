"""A fit score with no fitted parameters: does the evidence show what the fault must show?

The hybrid's prose writer is given a diagnosis it did not choose, and §17 measured what that
costs: it contradicted the prefix on 0 of 242 items and rationalised a deliberately wrong one
on 58 of 58. A prose target that can disagree needs something to disagree *with*, and this
module asks whether the evidence itself can supply it at no cost.

The score is a fraction of required elements, nothing else:

1. the initiating variable the KB names is listed among the deviating sensors;
2. its printed trend runs in the KB's direction;
3. it appears in the onset block (the fault starts somewhere);
4. the KB's first cascading variable is listed;
5. … in the KB's direction;
6. its second cascading variable is listed;
7. … in the KB's direction;
8. the initiating variable's onset is not later than the cascading ones'.

No weight, threshold or parameter is fitted: eight binary tests, averaged. A fault whose
initiating variable has no canonical tag in the KB (or no measurement in this sensor set)
returns ``None`` — "cannot be computed" is a result, not a zero.

§18.4 fixes the verdict rule before this runs: the score is feasible as the basis of a
system-computed grounding verdict iff the **mean paired AUROC on train is ≥ 0.80**, where each
pair is (true fault, its §17 lookalike) on the events of the true fault.

CPU only. No model, no GPU, no training data.

Usage::

    python scripts/hybrid_fit_score.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.hybrid import calibration as cal  # noqa: E402
from scripts.fault_ids import fault_id_for  # noqa: E402
from scripts.fault_knowledge import FAULT_KNOWLEDGE  # noqa: E402
from scripts.tags import canonical_tag, iter_tags  # noqa: E402

DATASET = ROOT / "outputs" / "benchmark_v2" / "dataset"
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
OUT = ROOT / "outputs/analysis/phase3a/fit"

#: (true fault, lookalike) from §17's absorption table: the pairs the surrogate showed being
#: confused. ``15``'s absorber was ``NORMAL_TRANSIENT``, which has no KB signature to fit, so
#: its KB differential (``FAULT_11``, the inlet temperature that moves the same valve) is used
#: and labelled as such in the output.
PAIRS: Tuple[Tuple[int, int, str], ...] = (
    (1, 8, "surrogate: 25 of 25 absorbed by FAULT_08"),
    (2, 8, "surrogate: 11 of 25 absorbed by FAULT_08"),
    (6, 8, "surrogate: 9 of 25 absorbed by FAULT_08"),
    (7, 8, "surrogate: 13 of 25 absorbed by FAULT_08"),
    (8, 1, "surrogate: 3 of 25 absorbed by FAULT_01"),
    (8, 7, "surrogate: 1 of 25 absorbed by FAULT_07"),
    (5, 12, "surrogate: 25 of 25 absorbed by FAULT_12"),
    (14, 11, "surrogate: 25 of 25 absorbed by FAULT_11"),
    (15, 11, "KB differential, not the surrogate absorber (that was NORMAL_TRANSIENT)"),
)

ELEMENTS = ("initiator_listed", "initiator_direction", "initiator_in_onset",
            "cascade1_listed", "cascade1_direction",
            "cascade2_listed", "cascade2_direction", "onset_order_consistent")

_SENSOR = re.compile(r"^- (.+?) \[((?:XMEAS|XMV)_\d+), [^\]]*\]: z=(-?[\d.]+), "
                     r"deviation (-?[\d.]+)%(?:, trend (\w+))?")
_ONSET = re.compile(r"^\d+\. (.+?) \[((?:XMEAS|XMV)_\d+), [^\]]*\] at \+([\d.]+) min")
_RESIDUAL = re.compile(r"^- .*?: residual share ([\d.]+)")


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def parse_prompt(prompt: str) -> Dict:
    """The evidence block as the score needs it: tag → trend, and tag → onset minute."""
    trend: Dict[str, str] = {}
    onset: Dict[str, float] = {}
    residual: Dict[str, float] = {}
    for line in str(prompt or "").splitlines():
        match = _SENSOR.match(line)
        if match:
            tag = match.group(2)
            trend[tag] = (match.group(5) or "").lower()
            continue
        match = _ONSET.match(line)
        if match:
            onset[match.group(2)] = float(match.group(3))
            continue
        match = _RESIDUAL.match(line)
        if match and line.startswith("- "):
            tags = iter_tags(line)
            if tags:
                residual[tags[0]] = float(match.group(1))
    return {"trend": trend, "onset": onset, "residual": residual}


def _tags_for(entry: Dict) -> List[str]:
    tags = iter_tags(entry.get("tag"))
    if not tags:
        canonical = canonical_tag(entry.get("sensor"))
        if canonical:
            tags = [canonical]
    return tags


def _expected(direction: Optional[str]) -> str:
    text = str(direction or "").lower()
    if text.startswith("dec"):
        return "decreasing"
    if text.startswith("inc"):
        return "increasing"
    return ""


def signature(fault: int) -> Optional[Dict]:
    """The KB's required elements for one fault, or ``None`` when they cannot be read."""
    entry = FAULT_KNOWLEDGE.get(int(fault)) or {}
    initiating = entry.get("initiating") or []
    if not initiating:
        return None
    first = initiating[0] if isinstance(initiating[0], dict) else {}
    init_tags = _tags_for(first)
    if not init_tags:
        return None
    cascading = [item for item in (entry.get("cascading") or []) if isinstance(item, dict)]
    cascade = []
    for item in cascading[:2]:
        tags = _tags_for(item)
        cascade.append({"tags": tags, "direction": _expected(item.get("direction"))})
    return {"initiator": init_tags, "initiator_direction": _expected(first.get("direction")),
            "cascade": cascade}


def fit_score(prompt: str, fault: int, parsed: Optional[Dict] = None) -> Optional[Dict]:
    """Fraction of the KB's required elements that the evidence shows."""
    spec = signature(fault)
    if spec is None:
        return None
    evidence = parsed if parsed is not None else parse_prompt(prompt)
    trend, onset = evidence["trend"], evidence["onset"]
    met: Dict[str, bool] = {name: False for name in ELEMENTS}

    init = spec["initiator"]
    listed = [tag for tag in init if tag in trend]
    met["initiator_listed"] = bool(listed)
    if listed:
        met["initiator_direction"] = trend.get(listed[0], "") == spec["initiator_direction"]
    init_onsets = [onset[tag] for tag in init if tag in onset]
    met["initiator_in_onset"] = bool(init_onsets)

    cascade_onsets: List[float] = []
    for index, item in enumerate(spec["cascade"], start=1):
        tags = item["tags"]
        present = [tag for tag in tags if tag in trend]
        met[f"cascade{index}_listed"] = bool(present)
        if present:
            met[f"cascade{index}_direction"] = trend.get(present[0], "") == item["direction"]
        cascade_onsets += [onset[tag] for tag in tags if tag in onset]

    if init_onsets and cascade_onsets:
        met["onset_order_consistent"] = min(init_onsets) <= min(cascade_onsets)

    return {"score": round(sum(1 for value in met.values() if value) / len(ELEMENTS), 4),
            "met": met, "elements": len(ELEMENTS)}


def paired_auroc(differences: Sequence[float]) -> float:
    """P(true fit > lookalike fit), ties half — a paired comparison, not a pooled one."""
    if not differences:
        return float("nan")
    return sum(1.0 if value > 0 else 0.5 if value == 0 else 0.0
               for value in differences) / len(differences)


def train_pairs() -> Dict:
    """Per pair: AUROC of fit(true) against fit(lookalike), on the true fault's train events."""
    rows = [row for row in load_jsonl(DATASET / "train.jsonl")]
    by_fault: Dict[int, Dict[str, Dict]] = {}
    for row in rows:
        fault = row.get("fault")
        if fault is None:
            continue
        key = cal.event_key(row)
        if key in by_fault.setdefault(int(fault), {}):
            continue
        by_fault[int(fault)][key] = row

    out: Dict[str, Dict] = {}
    scores: List[float] = []
    for true_fault, lookalike, why in PAIRS:
        events = list(by_fault.get(true_fault, {}).values())
        differences: List[float] = []
        cannot = 0
        detail: List[Dict] = []
        for row in events:
            parsed = parse_prompt(row["prompt"])
            left = fit_score(row["prompt"], true_fault, parsed)
            right = fit_score(row["prompt"], lookalike, parsed)
            if left is None or right is None:
                cannot += 1
                continue
            differences.append(left["score"] - right["score"])
            if len(detail) < 3:
                detail.append({"item_id": row.get("item_id"), "fit_true": left["score"],
                               "fit_lookalike": right["score"]})
        value = paired_auroc(differences)
        scores.append(value)
        out[f"FAULT_{true_fault:02d}_vs_FAULT_{lookalike:02d}"] = {
            "true_fault": true_fault, "lookalike": lookalike, "why": why,
            "events": len(events), "compared": len(differences), "not_computable": cannot,
            "mean_fit_true": round(float(sum(differences) / len(differences)), 4) if differences else None,
            "auroc": None if value != value else round(value, 4),
            "examples": detail,
        }
    valid = [value for value in scores if value == value]
    return {"pairs": out,
            "mean_pair_auroc": round(float(sum(valid) / len(valid)), 4) if valid else None,
            "pairs_scored": len(valid),
            "cannot_compute_share": round(
                sum(entry["not_computable"] for entry in out.values())
                / max(1, sum(entry["events"] for entry in out.values())), 4)}


def uncomputable_faults() -> Dict:
    """Faults whose KB signature cannot be read at all — stated, not assumed."""
    out = {}
    for fault in range(1, 21):
        if signature(fault) is None:
            out[f"FAULT_{fault:02d}"] = "no resolvable initiating tag in the KB"
    return out


def validation() -> Dict:
    """Does the fit score separate the classifier's correct diagnoses from its wrong ones?

    The score is computed for the label the classifier *chose*, which is the only version
    available at run time. Rows whose prediction names no fault (an abstention or a normal
    answer) have no candidate to fit and are counted, not scored.
    """
    prompts = {row["item_id"]: row["prompt"] for row in load_jsonl(DATASET / "val.jsonl")}
    rows = [row for row in load_jsonl(HYBRID / "crossfit_val_rows.jsonl")
            if not row.get("unobservable") and row.get("variant") == "menu"
            and str(row.get("gold", "")).startswith("FAULT")
            and row.get("item_id") in prompts]
    correct: List[float] = []
    wrong: List[float] = []
    no_candidate = 0
    pairs = Counter()
    for row in rows:
        pred = str(row.get("pred") or "")
        if not pred.startswith("FAULT"):
            no_candidate += 1
            continue
        number = int(pred.split("_")[-1])
        parsed = parse_prompt(prompts[row["item_id"]])
        scored = fit_score(prompts[row["item_id"]], number, parsed)
        if scored is None:
            no_candidate += 1
            continue
        if pred == row["gold"]:
            correct.append(scored["score"])
        else:
            wrong.append(scored["score"])
            pairs[f"{row['gold']} -> {pred}"] += 1
    return {
        "stratum": "fault_named (menu-present, gold is a named fault)",
        "rows": len(rows),
        "scored_correct": len(correct),
        "scored_wrong": len(wrong),
        "no_candidate_prediction": no_candidate,
        "error_pool": len(wrong),
        "mean_fit_correct": round(float(sum(correct) / len(correct)), 4) if correct else None,
        "mean_fit_wrong": round(float(sum(wrong) / len(wrong)), 4) if wrong else None,
        "auroc_fit": round(float(cal.auroc(correct, wrong)), 4) if correct and wrong else None,
        "error_pairs": dict(pairs.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    report = {
        "definition": {
            "elements": list(ELEMENTS),
            "score": "met elements / 8; no fitted parameter, no threshold",
            "verdict_rule": "feasible iff mean paired AUROC on train >= 0.80 (§18.4, fixed first)",
            "threshold": 0.80,
        },
        "uncomputable_faults": uncomputable_faults(),
        "train": train_pairs(),
        "validation": validation(),
    }
    mean = report["train"]["mean_pair_auroc"]
    report["verdict"] = {
        "mean_pair_auroc": mean,
        "threshold": 0.80,
        "feasible": bool(mean is not None and mean >= 0.80),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fit_score.json").write_text(json.dumps(report, indent=2, default=str),
                                        encoding="utf-8")
    print(json.dumps({"mean_pair_auroc": mean, "pairs": report["train"]["pairs_scored"],
                      "feasible": report["verdict"]["feasible"],
                      "validation": {key: report["validation"][key] for key in
                                     ("rows", "scored_correct", "scored_wrong",
                                      "auroc_fit", "no_candidate_prediction")},
                      "uncomputable": report["uncomputable_faults"]}, indent=2, default=str))
    print(f"written: {out / 'fit_score.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
