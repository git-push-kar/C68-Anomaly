"""Cohen's kappa between two raters' blind preference choices.

Each rater CSV needs `pair_id` and `choice`, with choice in {left, right, tie}.
Reports observed agreement, Cohen's kappa over those three categories, the per-category
marginals, and the disagreement list for adjudication.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter

CATEGORIES = ("left", "right", "tie")


def read(path: pathlib.Path) -> dict:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return {row["pair_id"]: (row.get("choice") or "").strip().lower()
                for row in csv.DictReader(handle)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rater1", required=True)
    ap.add_argument("--rater2", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    one, two = read(pathlib.Path(args.rater1)), read(pathlib.Path(args.rater2))
    shared = sorted(set(one) & set(two))
    if not shared:
        print("no shared pair_id between the two sheets", file=sys.stderr)
        return 2
    pairs = [(one[k], two[k]) for k in shared
             if one[k] in CATEGORIES and two[k] in CATEGORIES]
    if len(pairs) < len(shared):
        print("dropped %d blank or illegal choices" % (len(shared) - len(pairs)),
              file=sys.stderr)
    n = len(pairs)
    if n == 0:
        print("no usable choices after dropping blanks", file=sys.stderr)
        return 2

    agree = sum(1 for a, b in pairs if a == b)
    observed = agree / n
    m1, m2 = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum((m1[c] / n) * (m2[c] / n) for c in CATEGORIES)
    kappa = 1.0 if expected >= 1.0 else (observed - expected) / (1.0 - expected)

    payload = {
        "pairs": n,
        "agree": agree,
        "observed_agreement": round(observed, 4),
        "expected_agreement": round(expected, 4),
        "cohen_kappa": round(kappa, 4),
        "meets_0_60": kappa >= 0.60,
        "marginals_rater1": {c: m1[c] for c in CATEGORIES},
        "marginals_rater2": {c: m2[c] for c in CATEGORIES},
        "disagreements": [{"pair_id": k, "rater1": one[k], "rater2": two[k]}
                          for k in shared if one[k] in CATEGORIES
                          and two[k] in CATEGORIES and one[k] != two[k]],
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
