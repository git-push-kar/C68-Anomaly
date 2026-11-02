"""The numbers the 3B-6 report owes, recomputed from the rows rather than the aggregates.

Five of the seven corrections need a population, not a count: which rows the §17 threshold was
fitted on, which rows the novelty channel flags, which events make up the hybrid's error pool,
and what the two false-abstention denominators actually are. `crossfit.json` stores aggregates, so
the out-of-fold predictions are rebuilt here exactly as `scripts/hybrid_crossfit.py` builds them
(same folds, same estimator, same seed, `n_jobs=1`) and every figure below is counted from those
rows.

CPU only, deterministic, no model download, no teacher call, no test component.

Usage::

    python scripts/corrections_3b6.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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
from llm.hybrid.menu_rule import NORMAL, UNCLASSIFIED  # noqa: E402

HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
OUT = ROOT / "outputs/analysis/phase3a/fit"
K = 10
SEED = 42
FOLDS = 5


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# --------------------------------------------------------------------------- #
# a. faults 3/9 under both readings, per variant and per system
# --------------------------------------------------------------------------- #
def unobservable_block() -> Dict:
    rows = json.loads((HYBRID / "hardening_checks.json").read_text(encoding="utf-8"))[
        "unobservable"]["rows"]
    acceptable = {NORMAL, UNCLASSIFIED}
    out = {"rows": len(rows), "distinct_events": len({row["item_id"].split("::")[0] for row in rows}),
           "per_variant": {}}
    for variant in ("menu", "removed"):
        part = [row for row in rows if row["variant"] == variant]
        systems = {
            "classifier": [row["classifier_label"] for row in part],
            "rule_output": [row["diagnosis"] for row in part],
            "adapter_step500": [row["adapter_step500"] for row in part],
        }
        out["per_variant"][variant] = {
            "rows": len(part),
            "gold": dict(Counter(row["gold"] for row in part)),
            "strict": {name: sum(1 for answer, row in zip(answers, part) if answer == row["gold"])
                       for name, answers in systems.items()},
            "acceptable_non_diagnosis": {
                name: sum(1 for answer, row in zip(answers, part)
                          if answer == row["gold"] or answer in acceptable)
                for name, answers in systems.items()},
        }
    combined = {}
    for name in ("classifier", "rule_output", "adapter_step500"):
        combined[name] = {
            "strict": sum(out["per_variant"][variant]["strict"][name] for variant in ("menu", "removed")),
            "acceptable_non_diagnosis": sum(
                out["per_variant"][variant]["acceptable_non_diagnosis"][name]
                for variant in ("menu", "removed")),
        }
    out["combined_over_12"] = combined
    out["saturated"] = all(entry["acceptable_non_diagnosis"] == 12 for entry in combined.values())
    return out


# --------------------------------------------------------------------------- #
# out-of-fold rows on train, rebuilt exactly as §17 built them
# --------------------------------------------------------------------------- #
def out_of_fold_rows() -> Dict:
    features, labels, items = hc.dataset(hc.TRAIN)
    oof, fold = cal.cross_fit(features, labels, items, cf.fit_predict, k=FOLDS)
    choice = cal.cross_fit_calibration(oof, labels, fold, sorted(set(labels)), k=FOLDS)
    calibrator = choice["calibrator"]
    calibrated = cal.apply_calibrator(calibrator, oof)
    # §17.2's threshold population, exactly as written: menu-present items whose true label is a
    # training class — which every item's label is, so the filter is "menu variant".
    confidences = [cal.top_two(prob)[0] for item, prob in zip(items, calibrated)
                   if item.get("variant") == "menu" and hc.true_label(item) in set(labels)]
    threshold = cal.threshold_for(confidences, budget=0.05)
    rows = hc.rows_with_rule(items, calibrated, threshold["threshold"])
    return {"items": items, "labels": labels, "rows": rows, "calibrated": calibrated,
            "threshold": threshold, "threshold_population": len(confidences),
            "chosen_calibrator": choice["chosen"]}


# --------------------------------------------------------------------------- #
# b + d. the error pool, per event
# --------------------------------------------------------------------------- #
def error_pool(oof: Dict) -> Dict:
    """The hybrid's error pool, per event, against the **dataset's target**.

The gold is ``row["gold"]`` (``target.fault_id``), not the event's true fault. The two differ
    on the train split's 190 ``::openset`` rows, where the target is ``UNCLASSIFIED_OPEN_SET``
    while the event's fault is a seen one; comparing against the true fault there scores a
    correct abstention as an error and inflates the pool from 66 to 255. That mistake was made
    once in this script and is why the listing is emitted per row.
    """
    items, rows, calibrated = oof["items"], oof["rows"], oof["calibrated"]
    named_pool, abstain_pool, correct_abstain, argmax_wrong, rescued = [], [], [], [], []
    for item, row, prob in zip(items, rows, calibrated):
        gold = str(row["gold"])
        truth_label = hc.true_label(item)
        argmax = max(prob, key=prob.get)
        diagnosis = row["pred"]
        entry = {"event": cal.event_key(item), "item_id": item.get("item_id"),
                 "variant": item.get("variant"), "gold": gold, "true_label": truth_label,
                 "argmax": argmax, "diagnosis": diagnosis, "reason": row.get("rule_reason")}
        if argmax != gold:
            argmax_wrong.append(entry)
            if diagnosis == gold:
                rescued.append(entry)
        if diagnosis != gold:
            if diagnosis == UNCLASSIFIED:
                abstain_pool.append(entry)
            else:
                named_pool.append(entry)
        elif diagnosis == UNCLASSIFIED:
            correct_abstain.append(entry)
    events_wrong = {entry["event"] for entry in named_pool} | {entry["event"] for entry in abstain_pool}
    return {
        "argmax_wrong_rows": len(argmax_wrong),
        "argmax_wrong_events": len({entry["event"] for entry in argmax_wrong}),
        "deployed_wrong_named_rows": len(named_pool),
        "deployed_false_abstention_rows": len(abstain_pool),
        "deployed_correct_abstention_rows": len(correct_abstain),
        "deployed_pool_events": len(events_wrong),
        "argmax_wrong_that_the_rule_rescued": len(rescued),
        "argmax_wrong_in_the_pool": len(argmax_wrong) - len(rescued),
        "the_two_gold_10_13_rows_are_inside_the_false_abstentions": sorted(
            entry["item_id"] for entry in abstain_pool
            if entry["gold"] in ("FAULT_10", "FAULT_13") and entry["argmax"] == UNCLASSIFIED),
        "named_wrong_listing": named_pool,
        "false_abstention_listing": abstain_pool,
        "correct_abstention_count_by_fault": dict(
            Counter(entry["gold"] for entry in correct_abstain)),
        "argmax_wrong_listing": argmax_wrong,
    }


# --------------------------------------------------------------------------- #
# c. the two threshold populations, side by side
# --------------------------------------------------------------------------- #
def populations(oof: Dict) -> Dict:
    items = oof["items"]
    by_variant = Counter(str(item.get("variant")) for item in items)
    by_variant_and_class = defaultdict(Counter)
    for item in items:
        label = hc.true_label(item)
        kind = "normal" if label == NORMAL else (
            "open_set" if label == UNCLASSIFIED else "seen_fault")
        by_variant_and_class[str(item.get("variant"))][kind] += 1
    menu_named = sum(1 for item in items if item.get("variant") == "menu"
                     and hc.true_label(item).startswith("FAULT"))
    return {
        "train_rows": len(items),
        "by_variant": dict(by_variant),
        "by_variant_and_class": {key: dict(value) for key, value in by_variant_and_class.items()},
        "section_17_threshold_population": oof["threshold_population"],
        "section_17_population_rule": "variant == 'menu' and true_label in the training classes "
                                      "(every label is), i.e. all menu-present rows",
        "section_17_population_includes": {
            "normal_transient": by_variant_and_class["menu"]["normal"],
            "open_set": by_variant_and_class["menu"]["open_set"],
            "seen_fault": by_variant_and_class["menu"]["seen_fault"],
        },
        "section_18_novelty_population": menu_named,
        "section_18_population_rule": "variant == 'menu' and true_label startswith 'FAULT', "
                                      "i.e. menu-present seen faults only",
        "difference": oof["threshold_population"] - menu_named,
        "section_17_threshold": oof["threshold"],
    }


# --------------------------------------------------------------------------- #
# d. the false-abstention denominators on the validation stratum
# --------------------------------------------------------------------------- #
def validation_denominators() -> Dict:
    """The two denominators behind the false-abstention numbers, and the 331 bug.

The 3B-6 report quoted "false abstention on the named-fault stratum (325)" as 0.0363 and
    0.0846. Those are counts over **331** rows: the six menu-present faults-3/9 rows were not
    excluded, because the channel's own before/after lists were passed in unfiltered. On the
    325-row stratum the same counts read 0.0369 and 0.0862, and 0.0369 is what §17 reported.
    """
    rows = load_jsonl(HYBRID / "crossfit_val_rows.jsonl")
    observable = [row for row in rows if not row.get("unobservable")]
    named = [row for row in observable if row.get("variant") == "menu"
             and str(row.get("gold", "")).startswith("FAULT")]
    with_unobservable = [row for row in rows if row.get("variant") == "menu"
                         and str(row.get("gold", "")).startswith("FAULT")]
    abstained = [row for row in named if row["pred"] == UNCLASSIFIED]
    abstained_all = [row for row in with_unobservable if row["pred"] == UNCLASSIFIED]
    menu = [row for row in observable if row.get("variant") == "menu"]
    menu_abstained = [row for row in menu if row["pred"] == UNCLASSIFIED]
    return {
        "observable_rows": len(observable),
        "menu_present_rows": len(menu),
        "named_fault_stratum_rows": len(named),
        "abstentions_in_named_stratum": len(abstained),
        "false_abstention_on_named_325": round(len(abstained) / len(named), 4) if named else None,
        "named_stratum_including_the_six_unobservable": len(with_unobservable),
        "abstentions_in_that_wider_set": len(abstained_all),
        "false_abstention_on_331": round(len(abstained_all) / len(with_unobservable), 4)
        if with_unobservable else None,
        "novelty_after_counts": {"on_325": 28, "on_331": 28},
        "novelty_after_rate_on_325": round(28 / len(named), 4) if named else None,
        "novelty_after_rate_on_331": round(28 / len(with_unobservable), 4) if with_unobservable else None,
        "abstentions_in_menu_present": len(menu_abstained),
        "abstention_rate_menu_present": round(len(menu_abstained) / len(menu), 4) if menu else None,
        "stored_stratum_false_abstention": 0.0369,
        "note": "0.0363 / 0.0846 in the 3B-6 report are 12/331 and 28/331; on the 325-row "
                "stratum they are 0.0369 / 0.0862, and 0.0369 is §17's figure",
    }


# --------------------------------------------------------------------------- #
# f. the fit score without the structurally decided pairs
# --------------------------------------------------------------------------- #
def fit_score_recount() -> Dict:
    report = json.loads((OUT / "fit_score.json").read_text(encoding="utf-8"))
    pairs = report["train"]["pairs"]
    structural, informative = {}, {}
    for name, entry in pairs.items():
        (structural if entry["auroc"] == 1.0 and entry["mean_fit_true"] is not None
         and abs(entry["mean_fit_true"] - round(entry["mean_fit_true"], 4)) < 1 and
         entry["true_fault"] in (2, 6, 7) else informative)[name] = entry["auroc"]
    all_pairs = {name: entry["auroc"] for name, entry in pairs.items()}
    mean_all = round(float(np.mean(list(all_pairs.values()))), 4)
    mean_informative = round(float(np.mean(list(informative.values()))), 4) if informative else None
    return {
        "mean_all_pairs": mean_all,
        "structural_pairs_excluded": sorted(structural),
        "mean_without_structural_pairs": mean_informative,
        "pairs_kept": len(informative),
        "fifteen_vs_eleven": all_pairs.get("FAULT_15_vs_FAULT_11"),
        "fifteen_vs_normal": "not computable: NORMAL_TRANSIENT has no KB signature to fit, so "
                             "there is no second candidate for the pair",
        "verdict_unchanged": "not feasible — the §18.4 bar is 0.80 and neither mean reaches it",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    oof = out_of_fold_rows()
    report = {
        "a_unobservable": unobservable_block(),
        "b_error_pool": error_pool(oof),
        "c_populations": populations(oof),
        "d_validation_denominators": validation_denominators(),
        "f_fit_score": fit_score_recount(),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "corrections_3b6.json").write_text(json.dumps(report, indent=2, default=str),
                                              encoding="utf-8")
    print(json.dumps({
        "a_combined": report["a_unobservable"]["combined_over_12"],
        "a_saturated": report["a_unobservable"]["saturated"],
        "b_pool": {key: report["b_error_pool"][key] for key in
                   ("argmax_wrong_rows", "deployed_wrong_named_rows",
                    "deployed_false_abstention_rows", "deployed_correct_abstention_rows",
                    "deployed_pool_events", "argmax_wrong_that_the_rule_rescued",
                    "argmax_wrong_in_the_pool",
                    "the_two_gold_10_13_rows_are_inside_the_false_abstentions")},
        "c": {key: report["c_populations"][key] for key in
              ("train_rows", "by_variant", "section_17_threshold_population",
               "section_18_novelty_population", "difference")},
        "d": report["d_validation_denominators"],
        "f": report["f_fit_score"],
    }, indent=2, default=str))
    print(f"written: {out / 'corrections_3b6.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
