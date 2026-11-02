"""3B-11 step 3: the blind copy of the fresh 40-row sheet, and the leak check beside it.

The scored sheet carried a `source` column naming which side was the adapter, so the rater was
never blind (PROTOCOL §23.1). This script writes `audit_ab_100_v3_blind.csv`: the same 40 rows, in
the same order, with `source` removed and every scoring column blank. The scored file is kept as it
was scored; this is a new file, not a correction of it.

The assignment is recoverable only from the key, which is not part of either sheet.

Usage::  python scripts/make_blind_sheet.py [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIT = ROOT / "outputs/analysis/phase3a/fit"
SCORE_COLUMNS = ["correct_A", "grounded_A", "consistent_A", "useful_A",
                 "correct_B", "grounded_B", "consistent_B", "useful_B", "preference"]
HEADER = ["item_id", "family", "diagnosis", "verdict", "text_A", "text_B", *SCORE_COLUMNS, "notes"]


def load(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(FIT))
    args = parser.parse_args()
    out = Path(args.out)

    rows = load(out / "audit_ab_100_v3.csv")
    with open(out / "audit_ab_100_v3_blind.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            record = {key: row.get(key, "") for key in HEADER}
            for column in SCORE_COLUMNS:
                record[column] = ""
            writer.writerow(record)

    blind = load(out / "audit_ab_100_v3_blind.csv")
    report = {
        "rows": len(blind),
        "columns": HEADER,
        "source_column_present": "source" in (blind[0] if blind else {}),
        "scoring_columns_filled": sum(1 for row in blind for column in SCORE_COLUMNS
                                      if str(row[column]).strip()),
        "texts_identical_to_scored_sheet": sum(
            1 for before, after in zip(rows, blind)
            if before["text_A"] == after["text_A"] and before["text_B"] == after["text_B"]),
        "non_text_columns_identical": sum(
            1 for before, after in zip(rows, blind)
            if all(before[key] == after[key]
                   for key in ("item_id", "family", "diagnosis", "verdict"))),
        "kept": str(out / "audit_ab_100_v3.csv"),
        "written": str(out / "audit_ab_100_v3_blind.csv"),
    }
    (out / "audit_ab_100_v3_blind_meta.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
