"""Convert the filled blind workbook into the scored CSV the decision reads.

The rater works in `tep_blind_answer_evaluation.xlsx` (a CSV export of its Annotation sheet
is accepted too); `audit_v5_sheet.py join|reveal` and
`tests/test_v5_scoring_decision.py` read `audit_ab_100_v5_scored_rater1.csv`. This script
is the only bridge, and it refuses rather than repairs:

- the workbook must carry the same 40 item ids as the keyless sheet;
- all eight score cells must be exactly 1 or 0 - no blanks, no "y", no "0.0";
- every preference must be A, B or TIE;
- both text columns must still match `audit_ab_100_v5.csv` byte for byte (after newline
  normalisation), because an edited answer is no longer the text under test.

Nothing here is written unless every row passes; a refusal prints what is missing.

Example:
    python scripts/import_blind_scores.py --check
    python scripts/import_blind_scores.py          # writes the scored CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FIT = ROOT / "outputs/analysis/phase3a/fit"
WORKBOOK = FIT / "tep_blind_answer_evaluation.xlsx"
SHEET = FIT / "audit_ab_100_v5.csv"
TARGET = FIT / "audit_ab_100_v5_scored_rater1.csv"
SCORE_CELLS = ("correct_A", "grounded_A", "consistent_A", "useful_A",
               "correct_B", "grounded_B", "consistent_B", "useful_B")
PREFERENCES = ("A", "B", "TIE")


def read_sheet() -> list[dict]:
    with open(SHEET, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalise(text) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def load_rows(path: Path) -> list[dict]:
    """The filled sheet, from the workbook or from an export of its Annotation sheet."""
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return load_workbook_rows(path)
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle)
                if any(str(value or "").strip() for value in row.values())]


def load_workbook_rows(path: Path) -> list[dict]:
    try:
        import openpyxl
    except ImportError:
        raise SystemExit("openpyxl is not installed: pip install openpyxl")
    book = openpyxl.load_workbook(path, data_only=True)
    sheet = book["Annotation"] if "Annotation" in book.sheetnames else book.worksheets[0]
    header = [cell.value for cell in sheet[1]]
    missing = [c for c in ("item_id", *SCORE_CELLS, "preference") if c not in header]
    if missing:
        raise SystemExit("workbook is missing columns: " + ", ".join(missing))
    index = {name: i for i, name in enumerate(header)}
    rows = []
    for values in sheet.iter_rows(min_row=2, values_only=True):
        if all(value in (None, "") for value in values):
            continue
        rows.append({name: values[index[name]] if index[name] < len(values) else None
                     for name in header if name})
    return rows


def check(rows: list[dict], sheet_rows: list[dict]) -> dict:
    expected = [row["item_id"] for row in sheet_rows]
    got = [str(row.get("item_id") or "").strip() for row in rows]
    report = {
        "workbook_rows": len(rows),
        "sheet_rows": len(expected),
        "ids_match": got == expected,
        "unexpected_ids": [i for i in got if i not in set(expected)][:5],
        "missing_ids": [i for i in expected if i not in set(got)][:5],
        "score_cells_filled": 0,
        "score_cells_total": len(expected) * len(SCORE_CELLS),
        "blank_cells": Counter(),
        "illegal_cells": [],
        "illegal_preferences": [],
        "text_edits": [],
        "preferences": Counter(),
    }
    canonical = {row["item_id"]: row for row in sheet_rows}
    for row in rows:
        item = str(row.get("item_id") or "").strip()
        for column in SCORE_CELLS:
            value = row.get(column)
            text = "" if value is None else str(value).strip()
            if text:
                report["score_cells_filled"] += 1
            else:
                report["blank_cells"][column] += 1
                continue
            if text not in ("0", "1"):
                report["illegal_cells"].append(item + ":" + column + "=" + text[:12])
        preference = str(row.get("preference") or "").strip().upper()
        if preference:
            report["preferences"][preference] += 1
            if preference not in PREFERENCES:
                report["illegal_preferences"].append(item + "=" + preference[:12])
        else:
            report["blank_cells"]["preference"] += 1
        base = canonical.get(item)
        if base:
            for side in ("A", "B"):
                column = "text_" + side
                if column in row and normalise(row.get(column)) != normalise(base[column]):
                    report["text_edits"].append(item + ":" + side)
    report["blank_cells"] = dict(report["blank_cells"])
    report["preferences"] = dict(report["preferences"])
    report["complete"] = (report["ids_match"] and not report["blank_cells"]
                          and not report["illegal_cells"]
                          and not report["illegal_preferences"]
                          and not report["text_edits"])
    return report


def write_csv(rows: list[dict], sheet_rows: list[dict], target: Path) -> None:
    columns = list(sheet_rows[0].keys())
    by_item = {str(row.get("item_id") or "").strip(): row for row in rows}
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for base in sheet_rows:
            source = by_item[base["item_id"]]
            out = dict(base)
            for column in SCORE_CELLS:
                out[column] = str(source.get(column)).strip()
            out["preference"] = str(source.get("preference")).strip().upper()
            notes = source.get("notes")
            if notes not in (None, ""):
                out["notes"] = str(notes).strip()
            writer.writerow(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", default=str(WORKBOOK))
    ap.add_argument("--out", default=str(TARGET))
    ap.add_argument("--check", action="store_true",
                    help="validate only; never writes the scored CSV")
    args = ap.parse_args()

    path = Path(args.workbook)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        print(json.dumps({"error": "workbook not found", "path": str(path)}, indent=1))
        return 2
    sheet_rows = read_sheet()
    rows = load_rows(path)
    report = check(rows, sheet_rows)
    report["workbook"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    if not report["complete"]:
        report["action"] = "refused: nothing written"
        print(json.dumps(report, indent=1, default=str))
        return 2
    if args.check:
        report["action"] = "valid; pass without --check to write the scored CSV"
        print(json.dumps(report, indent=1))
        return 0
    target = Path(args.out)
    if not target.is_absolute():
        target = ROOT / target
    write_csv(rows, sheet_rows, target)
    report["action"] = "wrote scored CSV"
    report["out"] = str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(target)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
