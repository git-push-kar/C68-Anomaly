"""The six-rung selection rule, with a receipt and a refusal.

Refuses to emit a receipt unless `m` is supplied and the rule is marked committed.
Rung 2 (preference) is skipped when no preference input exists, and the receipt says so.
A choice made only by rung 6 is marked provisional: "A wins ties" is a default, not a
measured win, and a receipt that hid that would be a claim the data cannot carry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def sha(path: str) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:24]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--m", type=float, default=None)
    ap.add_argument("--rule-committed", action="store_true")
    ap.add_argument("--preference", default=None, help="json: {condition: {wins, n}}")
    ap.add_argument("--mechanism", default=None, help="json: {condition: validated_count}")
    ap.add_argument("--invariance", default=None, help="json: {condition: rate}")
    ap.add_argument("--cost", default=None, help="json: {condition: usd_per_1000}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.m is None:
        print("refusing: m is unset (D3 requires a declared margin)", file=sys.stderr)
        return 2
    if not args.rule_committed:
        print("refusing: the six-rung rule is not marked committed", file=sys.stderr)
        return 2

    summary = json.loads(pathlib.Path(args.cells).read_text(encoding="utf-8"))["summary"]
    base = summary.get("A")
    if not base:
        print("refusing: no A row to anchor the gate", file=sys.stderr)
        return 2

    ceiling = base["numeric_rate"] + args.m
    passers = sorted(c for c, v in summary.items()
                     if v["numeric_rate"] <= ceiling + 1e-9)
    skipped, chosen, rung = [], None, None

    if args.preference:
        pref = json.loads(pathlib.Path(args.preference).read_text(encoding="utf-8"))
        scored = [(c, pref[c]["wins"] / max(pref[c]["n"], 1))
                  for c in passers if c in pref and pref[c].get("n")]
        if scored:
            best = max(v for _, v in scored)
            top = [c for c, v in scored if abs(v - best) < 1e-12]
            if len(top) == 1:
                chosen, rung = top[0], 2
            else:
                passers = top
                skipped.append("rung 2: tie among passers -> next rung")
        else:
            skipped.append("rung 2: preference file names no passer")
    else:
        skipped.append("rung 2: no preference data; tier removed per the plan")

    for key, source, better, rung_id in (
            ("mechanism", args.mechanism, max, 3),
            ("invariance", args.invariance, max, 4),
            ("cost", args.cost, min, 5)):
        if chosen is not None or not source:
            continue
        data = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
        scored = [(c, data[c]) for c in passers if c in data]
        if not scored:
            continue
        best = better(v for _, v in scored)
        top = [c for c, v in scored if v == best]
        if len(top) == 1:
            chosen, rung = top[0], rung_id
        else:
            passers = top

    if chosen is None:
        chosen = "A" if "A" in passers else passers[0]
        rung = 6

    receipt = {
        "rule": "six rungs: numeric gate, preference, mechanism validity, order "
                "invariance, cost, A wins ties",
        "m": args.m,
        "gate_ceiling": round(ceiling, 6),
        "candidates": sorted(summary),
        "passers": passers,
        "chosen": chosen,
        "decided_by_rung": rung,
        "provisional": rung == 6,
        "skipped": skipped,
        "inputs": {k: {"path": v, "sha256_24": sha(v)} for k, v in
                   (("cells", args.cells), ("preference", args.preference),
                    ("mechanism", args.mechanism), ("invariance", args.invariance),
                    ("cost", args.cost)) if v},
        "note": ("rung 6 is the deterministic default, not a measured win; this receipt "
                 "stays provisional until rungs 2 and 3 carry data" if rung == 6 else
                 "decided by a measured rung"),
    }
    pathlib.Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
