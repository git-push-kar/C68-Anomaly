"""Decide the deployment menu policy from measured probabilities, and test renderer parity.

Two questions, one script.

1. **Menu policy.** The benchmark menu is 16 entries — the 15 fault classes plus an explicit
   ``UNCLASSIFIED_OPEN_SET`` option — while the "removed" twin is the same 15 faults with the
   abstention option stripped. The deployment choice is whether to keep the abstention option.
   Both policies are simulated on the stored per-item probabilities of the 834 validation rows, so
   the comparison is arithmetic on the same predictions, not a re-run.

2. **Renderer parity.** The classifier reads the rendered evidence page, so the live renderer and the
   benchmark renderer must produce the same page. Both are applied to the same stored evidence dicts
   from ``events.db``; the 323 features are compared element by element.

Usage:  python scripts/check_menu_and_parity.py [--events 25]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.benchmark.menu import build_menu  # noqa: E402
from llm.benchmark.prompt_v2 import render_prompt  # noqa: E402
from llm.dataset import format_evidence_question  # noqa: E402
from llm.hybrid.menu_rule import menu_rule  # noqa: E402
from scripts.evidence_classifier import prompt_features  # noqa: E402

VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
ROWS = ROOT / "outputs/analysis/phase3a/hybrid/crossfit_val_rows.jsonl"
DB = ROOT / "outputs/events.db"
THRESHOLD = 0.9792  # cross-fitted threshold, the one the paper quotes


def menu_policy(menus_of: dict) -> dict:
    """Score the same predictions under two deployment menus.

    The evaluation set contains two constructions: ``menu`` rows, whose menu lists every fault plus the
    abstention option, and ``removed`` rows, a *constructed* open-set case whose gold assumes the true fault
    was withheld. Only the ``menu`` rows correspond to something a deployment can present, so the policy
    comparison runs on those (plus the 47 normal rows), and the removed rows are reported separately as an
    evaluation artefact rather than scored.
    """
    rows = [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]
    full = build_menu()
    faults_only = [entry["fault_id"] for entry in full if entry["fault_id"] != "UNCLASSIFIED_OPEN_SET"]
    fault_ids = [entry["fault_id"] for entry in full]

    deployed = [row for row in rows if row["variant"] == "menu"]
    policies = {"deployment A: 15 faults + abstain option": fault_ids,
                "deployment B: 15 faults, no abstain option": faults_only}
    out = {}
    for name, menu in policies.items():
        correct = abstained = 0
        reasons: Counter = Counter()
        verdicts: Counter = Counter()
        for row in deployed:
            probs = row["probs"]
            label = max(probs, key=probs.get)
            decision = menu_rule(label, float(probs[label]), menu, THRESHOLD)
            correct += int(decision["diagnosis"] == row["gold"])
            abstained += int(bool(decision["abstained"]))
            reasons[decision["reason"]] += 1
            verdicts[decision["verdict"]] += 1
        out[name] = {"accuracy": round(correct / len(deployed), 4), "correct": correct,
                     "rows": len(deployed), "abstentions": abstained,
                     "reasons": dict(reasons), "verdicts": dict(verdicts)}
    # reproduction check: the measured pipeline uses each row's own menu
    own_correct = 0
    for row in rows:
        probs = row["probs"]
        label = max(probs, key=probs.get)
        decision = menu_rule(label, float(probs[label]), menus_of[row["item_id"]], THRESHOLD)
        own_correct += int(decision["diagnosis"] == row["gold"])
    out["_reproduction"] = {"accuracy_with_own_menus": round(own_correct / len(rows), 4),
                            "rows": len(rows)}
    out["_strata"] = {
        "menu rows (deployable)": len(deployed),
        "removed rows (constructed, not deployable)": sum(1 for r in rows if r["variant"] == "removed"),
        "unobservable": sum(1 for r in rows if r["unobservable"]),
        "predicted unclassified": sum(1 for r in rows
                                      if max(r["probs"], key=r["probs"].get) == "UNCLASSIFIED_OPEN_SET"),
    }
    return out


def renderer_parity(limit: int) -> dict:
    con = sqlite3.connect(DB)
    evidence_rows = [json.loads(value) for (value,) in
                     con.execute("select evidence_json from events order by rowid desc limit ?", (limit,))]
    con.close()
    menu = build_menu()
    findings = []
    for evidence in evidence_rows:
        live = format_evidence_question(evidence)
        bench = render_prompt(evidence, menu)
        a = np.asarray(prompt_features(live), dtype=np.float64)
        b = np.asarray(prompt_features(bench), dtype=np.float64)
        differing = int(np.sum(~np.isclose(a, b)))
        findings.append({
            "event": str(evidence.get("event_id"))[:24],
            "identical_features": bool(differing == 0),
            "differing_features": differing,
            "features": int(len(a)),
            "nonzero_live": int(np.count_nonzero(a)),
            "nonzero_benchmark": int(np.count_nonzero(b)),
            "markers": {name: (name in live, name in bench) for name in
                        ("DETECTORS", "Top sensor deviations:", "TOP DEVIATING SENSORS",
                         "z=", "deviation ", "residual share", "Fused trigger", "Onset order", "ONSET ORDER")},
        })
    identical = sum(1 for f in findings if f["identical_features"])
    return {"events": len(findings), "identical": identical, "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=25)
    args = parser.parse_args()

    print("=" * 78)
    print("1) MENU POLICY — simulated on the stored per-item probabilities (834 validation rows)")
    print("=" * 78)
    menus_of = {}
    for line in VAL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            menus_of[item["item_id"]] = item["menu_ids"]
    policy = menu_policy(menus_of)
    print("\nreproduction with each row's own menu: accuracy %.4f over %d rows" % (
        policy["_reproduction"]["accuracy_with_own_menus"], policy["_reproduction"]["rows"]))
    for name in ("deployment A: 15 faults + abstain option", "deployment B: 15 faults, no abstain option"):
        block = policy[name]
        print("\n%s" % name)
        print("  accuracy %.4f (%d/%d)   abstentions %d" % (block["accuracy"], block["correct"],
                                                            block["rows"], block["abstentions"]))
        print("  reasons  %s" % block["reasons"])
        print("  verdicts %s" % block["verdicts"])
    print("\n  strata: %s" % policy["_strata"])

    print("\n" + "=" * 78)
    print("2) RENDERER PARITY — live format_evidence_question vs benchmark render_prompt")
    print("=" * 78)
    parity = renderer_parity(args.events)
    print("\nevents compared: %d   identical feature vectors: %d" % (parity["events"], parity["identical"]))
    first = parity["findings"][0]
    print("first event: %s" % first["event"])
    print("  differing features %d/%d   nonzero( live %d, benchmark %d )" % (
        first["differing_features"], first["features"], first["nonzero_live"], first["nonzero_benchmark"]))
    print("  markers (live, benchmark):")
    for name, (in_live, in_bench) in first["markers"].items():
        print("    %-24s %-5s %-5s %s" % (name, in_live, in_bench, "" if in_live == in_bench else "← DIFFERS"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
