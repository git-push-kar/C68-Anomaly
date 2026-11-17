"""Audit the wired diagnosis over every stored event that carries detector evidence.

Answers three questions from the stored records, without touching them:

* what the verdicts and calibrated confidences look like on live traffic;
* how often the classifier's fault group disagrees with the deterministic fallback's prose;
* what a different menu threshold would do, so the stored 1.0 can be compared against the cross-fitted value;
* live accuracy on every event whose run provenance names a fault (``fault_number``), with the misses listed.

    python scripts/audit_live_diagnosis.py
    python scripts/audit_live_diagnosis.py --thresholds 1.0 0.9792 0.95 0.9

The disagreement comparison is at **subsystem-group** level, because the fallback names a group ("B composition
step change or purge variation"), not a fault id. Grouping both sides is the only comparison that does not
pretend the prose is a label.

Read-only: events.db is opened with ``mode=ro`` and nothing is written back.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_live_event import DEFAULT_ARTIFACTS, DEFAULT_DB, reconstruct  # noqa: E402
from llm.hybrid.menu_rule import menu_rule  # noqa: E402
from main import TEPApp  # noqa: E402
from utils import load_config  # noqa: E402

#: TEP taxonomy, grouped by the subsystem the fault acts on. 3 and 9 are unobservable in this dataset and are
#: not on the deployment menu. (The partial dict in scripts/validate_detection.py is a different, stale naming.)
FAULT_GROUP = {
    "FAULT_01": "feed", "FAULT_02": "feed", "FAULT_06": "feed", "FAULT_07": "feed",
    "FAULT_08": "feed", "FAULT_10": "feed",
    "FAULT_04": "reactor", "FAULT_11": "reactor", "FAULT_13": "reactor", "FAULT_14": "reactor",
    "FAULT_05": "condenser", "FAULT_12": "condenser", "FAULT_15": "condenser",
}

PROSE_GROUP_HINTS = (
    ("A/C feed ratio", "feed"),
    ("B composition", "feed"),
    ("C header", "feed"),
    ("C feed", "feed"),
    ("A feed loss", "feed"),
    ("feed_system", "feed"),
    ("Reactor cooling", "reactor"),
    ("reactor_cooling", "reactor"),
    ("Condenser cooling", "condenser"),
    ("condenser_cooling", "condenser"),
    ("Stripper", "stripper"),
    ("stripper", "stripper"),
)


def prose_group(root_cause: str, affected: str) -> str:
    text = f"{root_cause} {affected}"
    for needle, group in PROSE_GROUP_HINTS:
        if needle in text:
            return group
    return affected or "other"


def load_events(db: pathlib.Path) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT event_json FROM events ORDER BY rowid").fetchall()
    finally:
        con.close()
    records = [json.loads(payload) for (payload,) in rows]
    return [r for r in records if (r.get("evidence") or {}).get("detector_evidence") is not None]


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]
    return {"min": ordered[0], "p10": at(0.10), "p25": at(0.25), "median": at(0.50),
            "p75": at(0.75), "p90": at(0.90), "max": ordered[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=pathlib.Path, default=DEFAULT_DB)
    parser.add_argument("--artifacts", type=pathlib.Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[1.0, 0.9792, 0.95, 0.9])
    parser.add_argument("--out", type=pathlib.Path,
                        default=ROOT / "outputs/analysis/phase3a/hybrid/live_diagnosis_audit.json")
    args = parser.parse_args()

    records = load_events(args.db)
    config = load_config()
    config["hybrid"] = {"enabled": True, "artifact_dir": str(args.artifacts)}
    app = TEPApp(config, enable_llm=False)
    threshold = app.hybrid.threshold

    rows = []
    for record in records:
        event = reconstruct(record)
        report = app._generate_report(event)
        diagnosis = report["diagnosis"]
        evidence = event.evidence.to_dict()
        try:
            probabilities = app.hybrid.probabilities(evidence)
        except Exception:
            # Refused before scoring (degenerate probabilities): no ranking exists to compare.
            probabilities = None
        if probabilities is None:
            predicted, confidence, decisions = None, None, {}
        else:
            predicted = max(probabilities, key=probabilities.get)
            confidence = float(probabilities[predicted])
            decision = menu_rule(predicted, confidence, app.hybrid.menu_ids, threshold)
            assert decision["verdict"] == diagnosis["verdict"], (decision, diagnosis)
            decisions = {str(t): menu_rule(predicted, confidence, app.hybrid.menu_ids, t)
                         for t in args.thresholds}
        rows.append({
            "event_id": event.event_id,
            "source_file": record.get("source_file"),
            "simulation_run": record.get("simulation_run"),
            "truth_fault_number": record.get("fault_number"),
            "verdict": diagnosis["verdict"],
            "reason": diagnosis["reason"],
            "fault_id": diagnosis["fault_id"],
            "predicted_class": predicted,
            "confidence": confidence,
            "decisions": decisions,
            "prose_root_cause": report["root_cause"],
            "prose_group": prose_group(report["root_cause"], report.get("affected_subsystem") or ""),
            "classifier_group": FAULT_GROUP.get(predicted),
        })

    confidences = [r["confidence"] for r in rows if r["confidence"] is not None]
    summary = {
        "events": len(rows),
        "threshold_in_force": threshold,
        "verdicts": dict(Counter(r["verdict"] for r in rows)),
        "reasons": dict(Counter(r["reason"] for r in rows)),
        "fault_ids": dict(Counter(r["fault_id"] for r in rows if r["fault_id"]).most_common()),
        "confidence": quantiles(confidences),
        "share_at_ceiling": sum(1 for c in confidences if c == 1.0) / max(len(rows), 1),
        "share_above_0.999": sum(1 for c in confidences if c > 0.999) / max(len(rows), 1),
    }

    compared = [r for r in rows if r["fault_id"] and r["classifier_group"]]
    agree = [r for r in compared if r["classifier_group"] == r["prose_group"]]
    summary["group_comparison"] = {
        "compared": len(compared),
        "agree": len(agree),
        "disagree": len(compared) - len(agree),
        "agreement_rate": len(agree) / max(len(compared), 1),
        "confusion_pairs": {
            f"{a}->{b}": n for (a, b), n in Counter(
                (r["classifier_group"], r["prose_group"]) for r in compared
                if r["classifier_group"] != r["prose_group"]).most_common()
        },
    }

    # Live accuracy, only over events whose run provenance names a fault. Old events (and
    # unfaulted runs) carry no ``fault_number`` and are counted separately, not scored.
    scored = [r for r in rows if r["truth_fault_number"] is not None]
    hits = [r for r in scored
            if r["fault_id"] == f"FAULT_{int(r['truth_fault_number']):02d}"]
    summary["live_accuracy"] = {
        "events_scored": len(scored),
        "events_without_truth": len(rows) - len(scored),
        "correct": len(hits),
        "accuracy": (len(hits) / len(scored)) if scored else None,
        "misses": [
            {"event_id": r["event_id"], "truth": f"FAULT_{int(r['truth_fault_number']):02d}",
             "predicted": r["fault_id"], "source_file": r["source_file"],
             "simulation_run": r["simulation_run"]}
            for r in scored if r not in hits
        ],
    }

    by_threshold = {}
    for t in args.thresholds:
        refused = {"verdict": "unavailable", "reason": "degenerate_scores", "abstained": True,
                   "diagnosis": None, "confidence": None}
        decisions = [r["decisions"].get(str(t), refused) for r in rows]
        labelled = [r for r in rows if r["decisions"].get(str(t), refused)["diagnosis"]]
        pairs = [(FAULT_GROUP.get(r["predicted_class"]), r["prose_group"]) for r in labelled]
        usable = [(a, b) for a, b in pairs if a]
        by_threshold[str(t)] = {
            "classified": sum(1 for d in decisions if d["verdict"] == "classified"),
            "open_set": sum(1 for d in decisions if d["verdict"] == "unclassified_open_set"),
            "abstained": sum(1 for d in decisions if d["abstained"]),
            "reasons": dict(Counter(d["reason"] for d in decisions)),
            "group_agreement_rate": (sum(1 for a, b in usable if a == b) / len(usable)) if usable else None,
        }
    summary["by_threshold"] = by_threshold

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nrow-level detail: {args.out}")


if __name__ == "__main__":
    main()
