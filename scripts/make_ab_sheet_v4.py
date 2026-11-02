"""Build the balanced v4 A/B sheet. Scaffold: runs on a dummy stand-in, never the real texts."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATA = {"abstention_menu": 10, "abstention_removed": 10, "normal": 20}
TEXT_COLUMNS = ("text_A", "text_B")
FREE_COLUMNS = ("score_A", "score_B", "preference", "notes")
GROUP_FIELDS = ("stratum", "family", "diagnosis", "verdict")


def pair_candidates(candidates):
    """Pair adjacent candidates inside a stratum after sorting on the visible fields.

    Adjacent pairing means every column value receives one adapter side and one fallback side per
    pair that shares it, and at most one unpaired contribution at a boundary, so no column can
    separate the sides by more than one row. Exact-group pairing would refuse odd groups, which
    the real strata can easily produce.
    """
    groups = defaultdict(list)
    for item in candidates:
        groups[str(item.get("stratum"))].append(item)
    pairs = []
    for stratum, members in sorted(groups.items()):
        if len(members) % 2:
            raise ValueError(f"odd stratum {stratum}: {len(members)} candidates")
        members.sort(key=lambda row: (str(row.get("family")), str(row.get("verdict")),
                                      str(row.get("diagnosis")), str(row["item_id"])))
        pairs.extend(members[i:i + 2] for i in range(0, len(members), 2))
    return pairs


def build_sheet(candidates, seed=42):
    counts = Counter(str(c.get("stratum")) for c in candidates)
    if dict(counts) != STRATA:
        raise ValueError(f"stratum counts {dict(counts)} != {STRATA}")
    rng = random.Random(seed)
    pairs = []
    for first, second in pair_candidates(candidates):
        a, b = (first, second) if rng.random() < 0.5 else (second, first)
        row_a = {"item_id": "", "stratum": a["stratum"], "family": a.get("family"),
                 "diagnosis": a.get("diagnosis"), "verdict": a.get("verdict"),
                 "menu_present": a.get("menu_present"), "text_A": a["text_adapter"],
                 "text_B": b["text_adapter"], "score_A": "", "score_B": "",
                 "preference": "", "notes": ""}
        row_b = {"item_id": "", "stratum": b["stratum"], "family": b.get("family"),
                 "diagnosis": b.get("diagnosis"), "verdict": b.get("verdict"),
                 "menu_present": b.get("menu_present"), "text_A": b["text_fallback"],
                 "text_B": a["text_fallback"], "score_A": "", "score_B": "",
                 "preference": "", "notes": ""}
        pairs.append((row_a, row_b, str(a["item_id"]), str(b["item_id"])))
    rows = [row for pair in pairs for row in pair[:2]]
    rng.shuffle(rows)
    for position, row in enumerate(rows, start=1):
        row["item_id"] = f"v4-{position:02d}"
    key = []
    for index, (row_a, row_b, id_a, id_b) in enumerate(pairs, start=1):
        key.append({"pair": index, "item_id": row_a["item_id"], "candidate": id_a,
                    "side_A_source": "adapter", "side_B_source": "fallback"})
        key.append({"pair": index, "item_id": row_b["item_id"], "candidate": id_b,
                    "side_A_source": "fallback", "side_B_source": "adapter"})
    return rows, key


def leak_report(rows, key):
    """No column may tell a rater which side carries the adapter.

    Balance is checked *inside* each stratum, not across the sheet: a rater sees the row's stratum,
    so a column may legitimately differ between strata. Across the sheet the boundary pairs of two
    strata can also cancel out and hide a real imbalance, which is how the first version of this
    check passed a key that leaked diagnosis.
    """
    side = {entry["item_id"]: entry["side_A_source"] for entry in key}
    for row in rows:
        row["_side"] = side[row["item_id"]]
    columns = [c for c in rows[0] if c not in TEXT_COLUMNS + FREE_COLUMNS + ("_side",)]
    report = {"rows": len(rows), "columns": {}, "per_stratum": {}}
    ok = True
    for stratum in STRATA:
        subset = [row for row in rows if row["stratum"] == stratum]
        report["per_stratum"][stratum] = {
            "rows": len(subset),
            "adapter_on_A": sum(1 for r in subset if r["_side"] == "adapter"),
            "fallback_on_A": sum(1 for r in subset if r["_side"] == "fallback")}
        ok = ok and report["per_stratum"][stratum]["adapter_on_A"] * 2 == len(subset)
    for column in columns:
        if column == "item_id":
            positional = all(row["item_id"] == f"v4-{i:02d}" for i, row in enumerate(rows, 1))
            report["columns"][column] = {"positional": positional, "tested": False}
            ok = ok and positional
            continue
        per_stratum = {}
        for stratum in STRATA:
            table = defaultdict(lambda: {"adapter": 0, "fallback": 0})
            for row in rows:
                if row["stratum"] == stratum:
                    table[str(row[column])][row["_side"]] += 1
            balanced = all(abs(c["adapter"] - c["fallback"]) <= 1 for c in table.values())
            per_stratum[stratum] = {
                "values": len(table), "balanced": balanced,
                "max_gap": max((abs(c["adapter"] - c["fallback"]) for c in table.values()), default=0),
                "side_totals": {"adapter": sum(c["adapter"] for c in table.values()),
                                "fallback": sum(c["fallback"] for c in table.values())}}
            ok = ok and balanced
        report["columns"][column] = {
            "balanced": all(v["balanced"] for v in per_stratum.values()),
            "per_stratum": per_stratum}
    report["passed"] = bool(ok)
    for row in rows:
        row.pop("_side", None)
    return report


def write_sheet(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dummy_candidates():
    profiles = {"abstention_menu": ("heat_balance", "unclassified", True),
                "abstention_removed": ("removed", "unclassified", False),
                "normal": ("normal", "normal_transient", True)}
    out, index = [], 0
    for stratum, total in STRATA.items():
        family, verdict, menu = profiles[stratum]
        for i in range(total):
            index += 1
            out.append({"item_id": f"dummy-{index:02d}", "stratum": stratum, "family": family,
                        "diagnosis": f"diag-{i % 2}", "verdict": verdict, "menu_present": menu,
                        "text_adapter": f"adapter prose {index}",
                        "text_fallback": f"fallback prose {index}"})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--out", default=str(ROOT / "outputs/analysis/phase3a/fit/_v4_scaffold_dummy.csv"))
    parser.add_argument("--key", default=str(ROOT / "outputs/analysis/phase3a/fit/_v4_scaffold_dummy_key.csv"))
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()
    candidates = (dummy_candidates() if args.dummy or not args.candidates else
                  [json.loads(line) for line in Path(args.candidates).open(encoding="utf-8") if line.strip()])
    rows, key = build_sheet(candidates)
    report = leak_report(rows, key)
    write_sheet(rows, Path(args.out))
    write_sheet(key, Path(args.key))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
