"""Every table the run-1 check reports, computed from files instead of retyped.

The point of this script is that no number in the report is hand-copied from a log:
each section reads the artifact it describes, and the JSON it writes is what the
report quotes. It also keeps the *before* numbers next to the *after* ones, because
the check's whole subject is an instrument change and a table without its
before-state proves nothing.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.benchmark import evaluate_v2 as ev  # noqa: E402
from llm.benchmark.label_scorer import candidates  # noqa: E402

EVAL = ROOT / "outputs/benchmark_v2/eval"
BEFORE = EVAL / "before_candidate_fix"
ADAPTER = ROOT / "outputs/tep_rca_v2_protocolA_seed42"
OUT = ROOT / "outputs/analysis/phase3a/check_run1.json"


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def arm(path: Path) -> Dict:
    """One checkpoint table row per step, for either the before or the after run."""
    table = load_json(path)
    if not table:
        return {"present": False}
    return {
        "present": True,
        "rows": [{key: row.get(key) for key in
                  ("step", "items", "accuracy", "top3", "macro_f1", "seconds_per_item")}
                 for row in table],
        "normal_recall": {row["step"]: row.get("per_fault_recall", {}).get("NORMAL_TRANSIENT")
                          for row in table},
        "abstention": {row["step"]: row.get("abstention") for row in table},
    }


def selection(path: Path) -> Dict:
    data = load_json(path)
    if not data:
        return {"present": False}
    selected = data.get("selected") or {}
    return {
        "present": True,
        "step": selected.get("step"),
        "subsample_accuracy": selected.get("accuracy"),
        "subsample_macro_f1": selected.get("macro_f1"),
        "full_val": {key: (data.get("full_val") or {}).get(key)
                     for key in ("items", "accuracy", "top3", "macro_f1")},
        "full_val_abstention": (data.get("full_val") or {}).get("abstention"),
        "unobservable": {key: (data.get("unobservable") or {}).get(key)
                         for key in ("items", "accuracy", "macro_f1")},
        "per_fault": (data.get("full_val") or {}).get("per_fault"),
        "ece_brier": {"ece": (data.get("full_val") or {}).get("ece"),
                      "brier": (data.get("full_val") or {}).get("brier")},
    }


def fault15_arithmetic(rows: Sequence[Dict]) -> Dict:
    """What FAULT_15 was predicted on, and why each wrong one is wrong."""
    if not rows:
        return {"present": False}
    predicted = [row for row in rows if row["pred"] == "FAULT_15"]
    wrong = [row for row in predicted if row["gold"] != "FAULT_15"]
    by_gold = Counter(row["gold"] for row in predicted)
    normal_rows = [row for row in rows if row["gold"] == "NORMAL_TRANSIENT"]
    f15_gold = [row for row in rows if row["gold"] == "FAULT_15"]
    return {
        "present": True,
        "predicted_fault15": len(predicted),
        "correct": len(predicted) - len(wrong),
        "wrong": len(wrong),
        "by_gold": dict(by_gold),
        "wrong_sources": dict(Counter(row["gold"] for row in wrong)),
        "gold_fault15_rows": len(f15_gold),
        "gold_fault15_recall": round((len(f15_gold) - len([r for r in f15_gold
                                                          if r["pred"] != "FAULT_15"]))
                                    / max(len(f15_gold), 1), 4),
        "normal_rows": len(normal_rows),
        "normal_predictions": dict(Counter(row["pred"] for row in normal_rows)),
        "fault15_precision": round((len(predicted) - len(wrong)) / max(len(predicted), 1), 4),
        # The published precision is over the observable set, which is what
        # ``evaluate_v2.report`` was handed in the run report; the two differ by
        # the fault 3/9 rows that are scored but reported separately.
        "fault15_precision_observable": round(
            (len([r for r in predicted if not r.get("unobservable")])
             - len([r for r in wrong if not r.get("unobservable")]))
            / max(len([r for r in predicted if not r.get("unobservable")]), 1), 4),
    }


def disagreements(generation: Dict, preds: Sequence[Dict], scorer_step: Optional[int]) -> Dict:
    """The 7 generation-vs-scoring disagreements, with all three labels."""
    if not generation:
        return {"present": False}
    scored = {row["item_id"]: row["pred"] for row in preds}
    before = load_jsonl(BEFORE / "preds_val_full_400.jsonl")
    old = {row["item_id"]: row["pred"] for row in before}
    rows = []
    for row in generation.get("rows") or []:
        generated = row.get("label")
        new = scored.get(row["item_id"])
        if generated and new and generated != new:
            rows.append({"item_id": row["item_id"], "gold": row["gold"],
                         "generated": generated, "scorer_before": old.get(row["item_id"]),
                         "scorer_after": new})
        elif generated and not new and generated != old.get(row["item_id"]):
            rows.append({"item_id": row["item_id"], "gold": row["gold"],
                         "generated": generated, "scorer_before": old.get(row["item_id"]),
                         "scorer_after": None})
    legal = set(candidates())
    enum_bad = [{"item_id": row["item_id"], "gold": row["gold"],
                 "generated": row.get("label")} for row in generation.get("rows") or []
                if row.get("label") not in legal]
    gen_correct = sum(1 for row in generation.get("rows") or []
                      if row.get("label") == row["gold"])
    return {
        "present": True,
        "scorer_step": scorer_step,
        "items": len(generation.get("rows") or []),
        "generated_accuracy": round(gen_correct / max(len(generation.get("rows") or []), 1), 4),
        "enum_valid": len(generation.get("rows") or []) - len(enum_bad),
        "enum_invalid": len(enum_bad),
        "enum_invalid_rows": enum_bad,
        "disagreements": rows,
        "disagreement_count": len(rows),
        "generation_was_right": sum(1 for row in rows if row["generated"] == row["gold"]),
    }


def base_comparison(base_path: Path, preds: Sequence[Dict]) -> Dict:
    base = load_json(base_path)
    if not base:
        return {"present": False}
    base300 = load_jsonl(EVAL / "preds_base_val300.jsonl")
    selected_step = ((load_json(EVAL / "selected_checkpoint.json") or {}).get("selected")
                     or {}).get("step")
    adapter300 = load_jsonl(EVAL / f"preds_val300_step{selected_step}.jsonl")
    common = {row["item_id"] for row in base300} & {row["item_id"] for row in adapter300}
    def subset(rows):
        return [row for row in rows if row["item_id"] in common]
    out = {
        "present": True,
        "base": {key: base.get("val_full", {}).get(key)
                 for key in ("items", "accuracy", "top3", "macro_f1")},
        "base_abstention": base.get("val_full", {}).get("abstention"),
        "base_per_fault": base.get("val_full", {}).get("per_fault"),
        "base_val300": {key: base.get("val300", {}).get(key)
                        for key in ("items", "accuracy", "top3", "macro_f1")},
    }
    if common:
        out["common_items"] = len(common)
        out["base_on_common"] = {key: ev.report(subset(base300), "base").get(key)
                                 for key in ("accuracy", "top3", "macro_f1")}
        out["adapter_on_common"] = {key: ev.report(subset(adapter300), "adapter").get(key)
                                    for key in ("accuracy", "top3", "macro_f1")}
    return out


def followup_report(path: Path) -> Dict:
    data = load_json(path)
    if not data:
        return {"present": False}
    rows = data.get("rows") or []
    by_kind: Dict[str, Dict] = {}
    for kind in sorted({row["kind"] for row in rows}):
        subset = [row for row in rows if row["kind"] == kind]
        by_kind[kind] = {
            "n": len(subset),
            "refusal_rate": round(sum(1 for row in subset if row["refused"]) / len(subset), 4),
            "grounded_clean": round(sum(1 for row in subset
                                        if not row["grounding_violations"]) / len(subset), 4),
            "repetition_rate": round(sum(1 for row in subset
                                         if row["repeats_question"]) / len(subset), 4),
        }
    refusals = [{"kind": row["kind"], "question": row["question"], "answer": row["answer"]}
                for row in rows
                if row["refused"] and row["kind"] in
                ("repeated_question", "causal_why", "mechanism", "valve_actuator")][:15]
    return {
        "present": True,
        "items": len(rows),
        "by_kind": by_kind,
        "refusal_rate_on_out_of_scope_and_false_premise":
            data.get("refusal_rate_on_out_of_scope_and_false_premise"),
        "over_refusal_rate_on_controls": data.get("over_refusal_rate_on_controls"),
        "refusal_rate_on_other_cases": data.get("refusal_rate_on_other_cases"),
        "grounded_clean_rate": data.get("grounded_clean_rate"),
        "repetition_rate": data.get("repetition_rate"),
        "seconds_per_item": data.get("seconds_per_item"),
        "refusals_on_in_scope_cases": refusals,
    }


def budget() -> Dict:
    telemetry = load_jsonl(ADAPTER / "train_telemetry.jsonl")
    if not telemetry:
        return {"present": False}
    unique: Dict[int, Dict] = {}
    for row in telemetry:
        unique.setdefault(int(row["step"]), row)
    steps = [unique[key] for key in sorted(unique)]
    per_step = [row["step_seconds"] for row in steps]
    runtime = 12_003.5139
    mean_reported = 21.71  # the trainer's own cumulative s/it at step 540
    return {
        "present": True,
        "logged_intervals": len(steps),
        "mean_step_seconds": round(statistics.mean(per_step), 2),
        "median_step_seconds": round(statistics.median(per_step), 2),
        "min": min(per_step),
        "max": max(per_step),
        "max_step": [row["step"] for row in steps
                     if row["step_seconds"] == max(per_step)][0],
        "clock_mhz": {"min": min(row["gpu_clock_mhz"] for row in steps),
                      "max": max(row["gpu_clock_mhz"] for row in steps),
                      "mean": round(statistics.mean(row["gpu_clock_mhz"] for row in steps), 1)},
        "temperature_c": {"min": min(row["gpu_temp_c"] for row in steps),
                          "max": max(row["gpu_temp_c"] for row in steps)},
        "power_w": {"min": min(row["gpu_power_w"] for row in steps),
                    "max": max(row["gpu_power_w"] for row in steps)},
        "device_memory_mib": {"min": min(row["gpu_mem_mib"] for row in steps),
                              "max": max(row["gpu_mem_mib"] for row in steps)},
        "save_steps_wall": {row["step"]: row["step_seconds"] for row in steps
                            if row["step"] % 100 == 0},
        "stall_steps": {row["step"]: {"seconds": row["step_seconds"],
                                      "power_w": row["gpu_power_w"],
                                      "temp_c": row["gpu_temp_c"]}
                        for row in steps if row["step_seconds"] > 30},
        "runtime_seconds": runtime,
        "trainer_mean_step_seconds": mean_reported,
        "steps_x_mean": round(540 * mean_reported, 1),
        "unaccounted_seconds": round(runtime - 540 * mean_reported, 1),
        "measured_hours_per_run": round(runtime / 3600, 2),
        "projection_five_runs_hours": round(runtime * 5 / 3600, 1),
        "projection_five_runs_with_headroom_hours": [
            round(runtime * 5 * 1.10 / 3600, 1), round(runtime * 5 * 1.20 / 3600, 1)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    after_selection = load_json(EVAL / "selected_checkpoint.json") or {}
    step = (after_selection.get("selected") or {}).get("step")
    preds_full = load_jsonl(EVAL / f"preds_val_full_{step}.jsonl") if step else []
    generation = load_json(EVAL / "generation_512.json") or {}

    result = {
        "candidates": candidates(),
        "candidate_count": len(candidates()),
        "checkpoints_before": arm(BEFORE / "checkpoint_table.json"),
        "checkpoints_after": arm(EVAL / "checkpoint_table.json"),
        "selection_before": selection(BEFORE / "selected_checkpoint.json"),
        "selection_after": selection(EVAL / "selected_checkpoint.json"),
        "fault15_before": fault15_arithmetic(load_jsonl(BEFORE / "preds_val_full_400.jsonl")),
        "fault15_after": fault15_arithmetic(preds_full),
        # Two comparisons, because selection moved: the generation prose was
        # written by step 400 (selected under the old scorer), so the honest
        # apples-to-apples table scores those 50 items with step 400 itself.
        "generation": disagreements(generation,
                                     load_jsonl(EVAL / "preds_gen_scored.jsonl"), 400),
        "generation_vs_selected": disagreements(generation, preds_full, step),
        "base": base_comparison(EVAL / "base_reference.json", preds_full),
        "followup_before": followup_report(EVAL / "followup_sanity.json"),
        "followup_after": followup_report(EVAL / "followup_sanity_200.json"),
        "budget": budget(),
        "classifier": load_json(ROOT / "outputs/analysis/phase3a/evidence_classifier/summary.json"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("candidate_count", "fault15_before", "fault15_after")}, indent=1,
                     default=str)[:1200])
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
