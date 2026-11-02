"""§18.3: novelty as a second abstention channel, evaluated on every instrument at once.

The §17 result this starts from is uncomfortable: the confidence threshold abstained on 22 of
834 validation rows, while the surrogate showed a never-seen fault being answered as its lookalike
25 times out of 25. Confidence cannot say "I know something like this"; distance to the training
set can at least say "this is far from anything I was trained on".

The channel is defined once, in PROTOCOL §18.3, and applied without a single tuned number:

* **score** — mean Euclidean distance to the 10 nearest **train** events, standardised with the
  reference set's own mean and SD (``llm/hybrid/calibration.novelty_scores``, unchanged);
* **threshold** — the largest value keeping out-of-fold false-novel ≤ 5 % on train, menu-present,
  seen-fault events; the folds are grouped by event, so an event never sits in its own reference
  set;
* **rule** — a named diagnosis becomes ``UNCLASSIFIED_OPEN_SET`` when the §17 rule abstained **or**
  novelty exceeds the threshold. ``NORMAL_TRANSIENT`` keeps precedence: there is no fault to
  withhold, and a far-from-training normal event is still a normal event.

Everything is measured: the 834 validation rows with before/after metrics, the 13-fault surrogate,
and both held-out folds. The verdict against §18.6's two rejection criteria is computed, not
asserted.

CPU only. No model training, no GPU, no test component.

Usage::

    python scripts/hybrid_novelty.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hybrid_classifier as hc  # noqa: E402
import hybrid_crossfit as cf  # noqa: E402
from llm.hybrid import calibration as cal  # noqa: E402
from llm.hybrid.menu_rule import NORMAL, UNCLASSIFIED, menu_rule  # noqa: E402

OUT = ROOT / "outputs/analysis/phase3a/hybrid"
FIT = ROOT / "outputs/analysis/phase3a/fit"
EVAL = ROOT / "outputs/benchmark_v2/dataset/val_eval.jsonl"
FOLDS = {"fold_1": (4, 11, 15), "fold_2": (14, 15)}
BUDGET = 0.05
K = 10
FOLD_COUNT = 5
SEED = 42


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def novelty_threshold(values: Sequence[float], budget: float = BUDGET) -> Dict:
    """The **largest** novelty value whose false-novel rate stays inside the budget.

    Mirrors ``menu_rule.largest_threshold`` but in the other direction: novelty rises for an
    unseen class, so the threshold is the value above which at most ``budget`` of the seen-fault
    events fall. Taking the largest such value spends the whole budget.
    """
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"threshold": float("inf"), "achieved_false_novel": 0.0, "budget": budget,
                "items": 0, "flagged": 0}
    allowed = int(len(ordered) * float(budget))
    index = max(0, len(ordered) - 1 - allowed)
    threshold = ordered[index]
    flagged = sum(1 for value in ordered if value > threshold)
    return {"threshold": round(float(threshold), 6),
            "achieved_false_novel": round(flagged / len(ordered), 6),
            "budget": budget, "items": len(ordered), "flagged": flagged}


def oof_novelty(features: np.ndarray, labels: Sequence[str], items: Sequence[Dict],
                mask: Optional[np.ndarray] = None, k: int = K,
                folds: int = FOLD_COUNT, seed: int = SEED) -> np.ndarray:
    """Out-of-fold novelty: each event is scored against the folds it was not in."""
    fold = cal.grouped_folds(items, k=folds, seed=seed)
    out = np.full(len(items), np.nan, dtype=np.float64)
    keep = np.ones(len(items), dtype=bool) if mask is None else mask
    for number in range(folds):
        held = [index for index, value in enumerate(fold)
                if value == number and keep[index]]
        kept = [index for index, value in enumerate(fold)
                if value != number and keep[index]]
        if not held or len(kept) <= k:
            continue
        out[held] = cal.novelty_scores(features[held], features[kept], k=k)
    return out


def seen_fault_mask(items: Sequence[Dict], mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Train events that are menu-present with a seen fault — the population §18.3 uses."""
    flags = [item.get("variant") == "menu" and hc.true_label(item).startswith("FAULT")
             for item in items]
    array = np.asarray(flags, dtype=bool)
    return array if mask is None else (array & mask)


def novelty_summary(values: Sequence[float], label: str) -> Dict:
    clean = [float(value) for value in values if value == value]
    if not clean:
        return {"label": label, "items": 0}
    ordered = sorted(clean)
    return {"label": label, "items": len(clean), "min": round(ordered[0], 4),
            "p50": round(ordered[len(ordered) // 2], 4),
            "p95": round(ordered[int(0.95 * (len(ordered) - 1))], 4),
            "max": round(ordered[-1], 4),
            "mean": round(float(sum(clean) / len(clean)), 4)}


# --------------------------------------------------------------------------- #
# validation (834 rows, both variants)
# --------------------------------------------------------------------------- #
def validation_block(train_features: np.ndarray, threshold: Dict) -> Dict:
    prompts = {row["item_id"]: row["prompt"] for row in load_jsonl(hc.VAL)}
    rows = load_jsonl(OUT / "crossfit_val_rows.jsonl")
    matrix = np.asarray([hc.prompt_features(prompts[row["item_id"]]) for row in rows],
                        dtype=np.float32)
    novelty = cal.novelty_scores(matrix, train_features, k=K)
    stored = [float((row.get("signals") or {}).get("novelty", float("nan"))) for row in rows]
    parity = max((abs(a - b) for a, b in zip(novelty, stored) if a == a and b == b), default=None)

    tau = threshold["threshold"]
    before, after, changed, improved, worsened = [], [], 0, 0, 0
    for row, value in zip(rows, novelty):
        old = str(row["pred"])
        flagged = bool(value > tau) and old != NORMAL
        new = UNCLASSIFIED if (old == UNCLASSIFIED or flagged) else old
        gold = str(row["gold"])
        before.append({**row, "pred": old})
        after.append({**row, "pred": new, "novelty": float(value), "flagged": flagged})
        if new != old:
            changed += 1
            if new == gold:
                improved += 1
            elif old == gold:
                worsened += 1
    return {
        "parity_vs_stored_novelty_max_abs": None if parity is None else round(parity, 6),
        "before": hc.summarise(_observable(before), "before"),
        "after": hc.summarise(_observable(after), "after"),
        "changed_decisions": changed,
        "changed_improved": improved,
        "changed_worsened": worsened,
        "fault_named_false_abstention": {
            "before": _false_abstention(before),
            "after": _false_abstention(after),
        },
        "open_set_recall": {
            "before": _open_set(before), "after": _open_set(after),
        },
        "strata_after": {name: hc.summarise([row for row in _observable(after) if keep(row)], name)
                         for name, keep in _STRATA.items()
                         if [row for row in _observable(after) if keep(row)]},
        "flagged_on_named_fault": sum(1 for row in after
                                      if row["flagged"] and str(row["gold"]).startswith("FAULT")),
        "flagged_on_normal": sum(1 for row in after
                                 if row["flagged"] and row["gold"] == NORMAL),
    }


_STRATA = {
    "observable": lambda row: True,
    "menu_present": lambda row: row.get("variant") == "menu",
    "menu_removed": lambda row: row.get("variant") != "menu",
    "fault_named": lambda row: row.get("variant") == "menu"
                              and str(row.get("gold")).startswith("FAULT"),
    "normal": lambda row: row.get("gold") == NORMAL,
    "open_set_on_menu": lambda row: row.get("variant") == "menu"
                                    and row.get("gold") == UNCLASSIFIED,
}


def _observable(rows: Sequence[Dict]) -> List[Dict]:
    return [row for row in rows if not row.get("unobservable")]


def _false_abstention(rows: Sequence[Dict]) -> Optional[float]:
    named = [row for row in rows if row.get("variant") == "menu"
             and str(row.get("gold")).startswith("FAULT")]
    if not named:
        return None
    return round(sum(1 for row in named if row["pred"] == UNCLASSIFIED) / len(named), 4)


def _open_set(rows: Sequence[Dict]) -> Optional[float]:
    opens = [row for row in rows if row.get("variant") == "menu"
             and row.get("gold") == UNCLASSIFIED]
    if not opens:
        return None
    return round(sum(1 for row in opens if row["pred"] == UNCLASSIFIED) / len(opens), 4)


# --------------------------------------------------------------------------- #
# surrogate (13 leave-one-fault-out refits)
# --------------------------------------------------------------------------- #
def surrogate_block(features: np.ndarray, labels: Sequence[str], items: Sequence[Dict],
                    faults: Sequence[int]) -> Dict:
    val_items = [row for row in hc.load_jsonl(hc.VAL)]
    val_labels = [hc.true_label(row) for row in val_items]
    val_matrix = np.asarray([hc.prompt_features(row["prompt"]) for row in val_items],
                            dtype=np.float32)
    stored = json.loads((OUT / "surrogate_wide.json").read_text(encoding="utf-8"))["per_fault"]
    out = {}
    for fault in faults:
        name = f"FAULT_{int(fault):02d}"
        keep = np.asarray([label != name for label in labels], dtype=bool)
        novelty = oof_novelty(features, labels, items, mask=keep)
        mask = seen_fault_mask(items, keep)
        threshold = novelty_threshold([value for value, flag in zip(novelty, mask) if flag])
        tau = threshold["threshold"]
        held = [index for index, label in enumerate(val_labels) if label == name
                and val_items[index].get("variant") == "menu"]
        held_novelty = cal.novelty_scores(val_matrix[held], features[keep], k=K) if held else []
        flagged = sum(1 for value in held_novelty if value > tau)
        kept_val = [index for index, label in enumerate(val_labels)
                    if label.startswith("FAULT") and label != name
                    and val_items[index].get("variant") == "menu"]
        kept_novelty = cal.novelty_scores(val_matrix[kept_val], features[keep], k=K) if kept_val else []
        old = stored.get(name, {}).get("by_variant", {}).get("menu", {})
        named = old.get("named", 0)
        declined = old.get("declined", 0)
        out[name] = {
            "threshold": threshold,
            "held_out_menu_items": len(held),
            "novelty_flagged": flagged,
            "novelty_flag_rate": round(flagged / len(held), 4) if held else None,
            "s17_named": named,
            "s17_declined": declined,
            # A previously named item that novelty flags becomes a decline; a previously
            # declined one stays declined. Without per-item rule decisions for these refits the
            # overlap is unknown, so the *gain* in declines is bounded by min(named, flagged).
            "decline_gain_upper_bound": min(named, flagged),
            "kept_fault_validation_items": len(kept_val),
            "false_novel_on_kept": round(sum(1 for value in kept_novelty if value > tau)
                                         / len(kept_val), 4) if kept_val else None,
            "novelty_by_stratum": {
                "held_out": novelty_summary(held_novelty, "held_out"),
                "kept_faults": novelty_summary(kept_novelty, "kept_faults"),
            },
        }
    return out


# --------------------------------------------------------------------------- #
# folds (reproduced per item, then checked against the stored aggregates)
# --------------------------------------------------------------------------- #
def fold_block(variant: str, held: Sequence[int]) -> Dict:
    stored = json.loads((OUT / "fold_holdout.json").read_text(encoding="utf-8"))[variant]
    features, labels, items = hc.dataset(hc.TRAIN, drop_faults=held)
    oof, fold = cal.cross_fit(features, labels, items, cf.fit_predict, k=FOLD_COUNT)
    choice = cal.cross_fit_calibration(oof, labels, fold, sorted(set(labels)), k=FOLD_COUNT)
    calibrator = choice["calibrator"]
    oof_calibrated = cal.apply_calibrator(calibrator, oof)
    confidences = [cal.top_two(prob)[0] for item, prob in zip(items, oof_calibrated)
                   if item.get("variant") == "menu" and hc.true_label(item) in set(labels)]
    rule_threshold = cal.threshold_for(confidences, budget=BUDGET)["threshold"]

    novelty_oof = oof_novelty(features, labels, items)
    mask = seen_fault_mask(items)
    novelty_thr = novelty_threshold([value for value, flag in zip(novelty_oof, mask) if flag])

    model, metadata, _c, _t = hc.load_variant(variant, hc.OUT)
    classes = metadata["train_classes"]
    val_by_id = {row["item_id"]: row for row in load_jsonl(hc.VAL)}
    eval_rows = load_jsonl(EVAL)

    def score(component: str) -> List[Dict]:
        wanted = [val_by_id[row["item_id"]] for row in eval_rows
                  if row.get("component") == component]
        matrix = np.asarray([hc.prompt_features(item["prompt"]) for item in wanted],
                            dtype=np.float32)
        raw = model.predict(matrix)
        ordered = [classes[index] for index in model.classes_] if hasattr(model, "classes_") else classes
        probs = cal.apply_calibrator(calibrator,
                                     [{label: float(value) for label, value in zip(ordered, row)}
                                      for row in raw])
        novelty = cal.novelty_scores(matrix, features, k=K)
        out = []
        for item, prob, value in zip(wanted, probs, novelty):
            pred = max(prob, key=prob.get)
            decision = menu_rule(pred, prob.get(pred), item.get("menu_ids") or [], rule_threshold)
            flagged = bool(value > novelty_thr["threshold"]) and decision["diagnosis"] != NORMAL
            new = UNCLASSIFIED if (decision["diagnosis"] == UNCLASSIFIED or flagged) \
                else decision["diagnosis"]
            out.append({"item_id": item["item_id"], "variant": item.get("variant"),
                        "gold": str((item.get("target") or {}).get("fault_id") or ""),
                        "old": decision["diagnosis"], "new": new, "novelty": float(value),
                        "flagged": flagged, "verdict": decision["verdict"]})
        return out

    test_rows = score(f"{variant}/test")
    kept_rows = score(f"{variant}/val")
    menu_test = [row for row in test_rows if row["variant"] == "menu"]
    menu_kept = [row for row in kept_rows if row["variant"] == "menu"]
    reproduced = {
        "declined": sum(1 for row in menu_test if row["old"] == UNCLASSIFIED),
        "named": sum(1 for row in menu_test if str(row["old"]).startswith("FAULT")),
        "items": len(menu_test),
    }
    return {
        "novelty_threshold": novelty_thr,
        "rule_threshold": rule_threshold,
        "reproduced_vs_stored": {
            "reproduced": reproduced,
            "stored": stored.get("test", {}).get("menu", {}),
            "matches": (reproduced["declined"] == stored.get("test", {}).get("menu", {}).get("declined")
                        and reproduced["named"] == stored.get("test", {}).get("menu", {}).get("named_a_fault")),
        },
        "held_out_menu": {
            "items": len(menu_test),
            "old_declined": reproduced["declined"],
            "old_named": reproduced["named"],
            "novelty_flagged": sum(1 for row in menu_test if row["flagged"]),
            "flags_on_previously_named": sum(1 for row in menu_test
                                             if row["flagged"] and str(row["old"]).startswith("FAULT")),
            "new_declined": sum(1 for row in menu_test if row["new"] == UNCLASSIFIED),
            "new_declined_rate": round(sum(1 for row in menu_test if row["new"] == UNCLASSIFIED)
                                       / len(menu_test), 4) if menu_test else None,
        },
        "kept_faults_menu": {
            "items": len(menu_kept),
            "old_declined": sum(1 for row in menu_kept if row["old"] == UNCLASSIFIED),
            "novelty_flagged": sum(1 for row in menu_kept if row["flagged"]),
            "false_novel_rate": round(sum(1 for row in menu_kept if row["flagged"])
                                      / len(menu_kept), 4) if menu_kept else None,
            "old_accuracy": round(sum(1 for row in menu_kept if row["old"] == row["gold"])
                                  / len(menu_kept), 4) if menu_kept else None,
            "new_accuracy": round(sum(1 for row in menu_kept if row["new"] == row["gold"])
                                  / len(menu_kept), 4) if menu_kept else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(FIT))
    args = parser.parse_args()

    features, labels, items = hc.dataset(hc.TRAIN)
    novelty_oof = oof_novelty(features, labels, items)
    mask = seen_fault_mask(items)
    threshold = novelty_threshold([value for value, flag in zip(novelty_oof, mask) if flag])

    report = {
        "definition": {
            "score": "mean Euclidean distance to the 10 nearest train events, standardised",
            "threshold_rule": "largest value with out-of-fold false-novel <= 5 % on train "
                              "menu-present seen-fault events",
            "decision": "UNCLASSIFIED if the §17 rule abstained OR novelty > threshold; "
                        "NORMAL_TRANSIENT keeps precedence",
            "k": K, "folds": FOLD_COUNT, "seed": SEED, "budget": BUDGET,
        },
        "threshold": threshold,
        "train_novelty": {
            "all_events": novelty_summary(novelty_oof, "all"),
            "seen_fault_menu": novelty_summary([value for value, flag in
                                                zip(novelty_oof, mask) if flag], "seen_fault"),
        },
        "validation": validation_block(features, threshold),
        "surrogate": surrogate_block(features, labels, items, sorted({int(f) for f in
                                                                     [item.get("fault") for item in items
                                                                      if item.get("fault") is not None
                                                                      and 1 <= int(item.get("fault")) <= 15]})),
        "folds": {name: fold_block(name, held) for name, held in FOLDS.items()},
    }

    # §18.6, applied rather than argued.
    before = report["validation"]["before"]
    after = report["validation"]["after"]
    accuracy_drop = round(float(before["accuracy"]) - float(after["accuracy"]), 4)
    false_before = report["validation"]["fault_named_false_abstention"]["before"] or 0.0
    false_after = report["validation"]["fault_named_false_abstention"]["after"] or 0.0
    # Measured instruments only: per-item decisions exist for the folds and for validation, so
    # their gains are counted. The surrogate has counts, not items, so its bound is reported
    # beside the verdict and is not allowed to decide it.
    fold_gains = {name: (block["held_out_menu"]["new_declined"] - block["held_out_menu"]["old_declined"])
                  for name, block in report["folds"].items()}
    surrogate_gain_bounds = {name: block["decline_gain_upper_bound"]
                             for name, block in report["surrogate"].items()}
    validation_gain = report["validation"]["changed_improved"] - \
        report["validation"]["changed_worsened"]
    measured_gains = {**fold_gains, "validation": validation_gain}
    improved_any = any(value > 0 for value in measured_gains.values())
    # §18.6's criterion (a) is per instrument: the accuracy cost and the decline gain must be
    # read from the same rows, because a channel can be worth its cost on one instrument and
    # pure loss on another.
    instruments = {
        "validation": {
            "seen_fault_accuracy_before": float(before["accuracy"]),
            "seen_fault_accuracy_after": float(after["accuracy"]),
            "accuracy_drop": accuracy_drop,
            "decline_gain": validation_gain,
        },
        "fold_1": {
            "seen_fault_accuracy_before": report["folds"]["fold_1"]["kept_faults_menu"]["old_accuracy"],
            "seen_fault_accuracy_after": report["folds"]["fold_1"]["kept_faults_menu"]["new_accuracy"],
            "accuracy_drop": round(report["folds"]["fold_1"]["kept_faults_menu"]["old_accuracy"]
                                   - report["folds"]["fold_1"]["kept_faults_menu"]["new_accuracy"], 4),
            "decline_gain": fold_gains["fold_1"],
        },
        "fold_2": {
            "seen_fault_accuracy_before": report["folds"]["fold_2"]["kept_faults_menu"]["old_accuracy"],
            "seen_fault_accuracy_after": report["folds"]["fold_2"]["kept_faults_menu"]["new_accuracy"],
            "accuracy_drop": round(report["folds"]["fold_2"]["kept_faults_menu"]["old_accuracy"]
                                   - report["folds"]["fold_2"]["kept_faults_menu"]["new_accuracy"], 4),
            "decline_gain": fold_gains["fold_2"],
        },
    }
    for entry in instruments.values():
        entry["triggers_criterion_a"] = bool(entry["accuracy_drop"] > 0.02
                                             and entry["decline_gain"] <= 0)

    report["verdict"] = {
        "criterion_a": {
            "rule": "reject if seen-fault accuracy drops by more than 2 points on an instrument "
                    "without a higher held-out decline rate on the same instrument",
            "instruments": instruments,
            "validation_accuracy_drop": accuracy_drop,
            "false_abstention_before": false_before,
            "false_abstention_after": false_after,
            "improved_any_held_out": improved_any,
            "triggered": any(entry["triggers_criterion_a"] for entry in instruments.values()),
            "triggering_instruments": [name for name, entry in instruments.items()
                                       if entry["triggers_criterion_a"]],
        },
        "criterion_b": {
            "rule": "reject if it fails to beat the §17 rule on all three instruments",
            "measured_gains": measured_gains,
            "surrogate_gain_bounds": surrogate_gain_bounds,
            "instruments_beaten": sum(1 for value in measured_gains.values() if value > 0),
            "instruments": len(measured_gains),
            "triggered": not improved_any,
            "note": "the surrogate's per-fault figure is an upper bound (min(named, flagged)) "
                    "because its stored artefact has counts, not per-item decisions; it is "
                    "reported and not used to decide",
        },
        "accepted": None,
    }
    report["verdict"]["accepted"] = not (report["verdict"]["criterion_a"]["triggered"]
                                         or report["verdict"]["criterion_b"]["triggered"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "novelty_channel.json").write_text(json.dumps(report, indent=2, default=str),
                                              encoding="utf-8")
    print(json.dumps({
        "threshold": report["threshold"],
        "validation": {"accuracy": [before["accuracy"], after["accuracy"]],
                       "macro_f1": [before["macro_f1"], after["macro_f1"]],
                       "changed": report["validation"]["changed_decisions"],
                       "improved": report["validation"]["changed_improved"],
                       "worsened": report["validation"]["changed_worsened"],
                       "false_abstention": [false_before, false_after]},
        "fold_gains": fold_gains,
        "verdict": report["verdict"]["accepted"],
        "criterion_a": report["verdict"]["criterion_a"]["triggered"],
        "criterion_b": report["verdict"]["criterion_b"]["triggered"],
    }, indent=2, default=str))
    print(f"written: {out / 'novelty_channel.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
