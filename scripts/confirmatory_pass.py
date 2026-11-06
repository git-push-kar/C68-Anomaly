"""The one-shot confirmatory gate: refunds to run without a decided winner.

D4 allows exactly one paired pass of winner + A on the test export, receipted before
launch. This script refuses while the selection receipt is provisional (rung 6, i.e. no
measured rung decided), because a one-shot pass spent on a default cannot be repeated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", default=str(ROOT / "outputs/overnight_2026-11-05/selection_receipt.json"))
    ap.add_argument("--test", default=str(ROOT / "outputs/benchmark_v2/dataset/test.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "outputs/overnight_2026-11-05/confirmatory_prepass.json"))
    args = ap.parse_args()

    receipt_path, test_path = pathlib.Path(args.receipt), pathlib.Path(args.test)
    if not receipt_path.exists():
        print("refusing: no selection receipt", file=sys.stderr)
        return 2
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("provisional") or receipt.get("decided_by_rung") == 6:
        print("refusing: the selection receipt is provisional (rung %s, no measured win); "
              "the one-shot pass would be spent on a default"
              % receipt.get("decided_by_rung"), file=sys.stderr)
        return 2
    if not test_path.exists():
        print("refusing: test export missing at %s" % test_path, file=sys.stderr)
        return 2

    prepass = {
        "status": "authorised-to-launch",
        "winner": receipt.get("chosen"),
        "baseline": "A",
        "decided_by_rung": receipt.get("decided_by_rung"),
        "receipt_sha256": sha(receipt_path),
        "test_sha256": sha(test_path),
        "test_bytes": test_path.stat().st_size,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rule": "one pass; a second run is a new dated section, never a re-run",
    }
    pathlib.Path(args.out).write_text(json.dumps(prepass, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(prepass, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
