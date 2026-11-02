"""The A/B sheet must not name the source anywhere a rater can see it.

The first fresh sheet carried a `source` column that said which side was the adapter, so the rater
was not blind (PROTOCOL §23.1). This test fails if a sheet column other than the two text columns
separates the sides, which is exactly what that column did.
"""
from __future__ import annotations

import csv
from pathlib import Path

FIT = Path(__file__).resolve().parents[1] / "outputs/analysis/phase3a/fit"
SCORED = FIT / "audit_ab_100_v3.csv"
KEY = FIT / "audit_ab_100_v3_key.csv"
BLIND_SHEET = FIT / "audit_ab_100_v3_blind.csv"
TEXT_COLUMNS = {"text_A", "text_B", "notes"}
LEAKY_WORDS = ("adapter", "fallback", "v3", "v3.1")
SCORES = ("correct_A", "grounded_A", "consistent_A", "useful_A", "correct_B", "grounded_B",
          "consistent_B", "useful_B", "preference")


def read(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sides() -> dict:
    return {row["item_id"]: "A" if row["A"] == "adapter" else "B" for row in read(KEY)}


def leaky_columns(rows, side) -> list:
    """A column leaks when some non-unique value occurs only on one side of the key."""
    leaks = []
    for column in rows[0]:
        if column in TEXT_COLUMNS:
            continue
        values = [str(row[column]) for row in rows]
        if len(set(values)) == len(values):
            continue                      # unique per row: no group signal
        groups = {}
        for row, value in zip(rows, values):
            groups.setdefault(value, set()).add(side[row["item_id"]])
        if any(len(sides_) == 1 and values.count(group) > 1 for group, sides_ in groups.items()):
            leaks.append(column)
    return leaks


#: The three stratum columns separate the sides in this key: `removed` is 0 A / 5 B and `normal`
#: is 9 A / 1 B, so a rater who knows the strata can guess the side from them (reported in
#: docs/reports/3b11_report.md). They are pinned here, not excused: any *other* leaking column fails.
STRATUM_COLUMNS = ["family", "diagnosis", "verdict"]


def test_scored_sheet_leaks_the_assignment_and_blind_sheet_does_not():
    side = sides()
    scored, blind = read(SCORED), read(BLIND_SHEET)
    assert "source" in scored[0], "the scored sheet is expected to carry the leaking column"
    assert "source" not in blind[0]
    blind_leaks = set(leaky_columns(blind, side))
    assert set(leaky_columns(scored, side)) == blind_leaks | {"source"}
    assert blind_leaks <= set(STRATUM_COLUMNS), blind_leaks


def test_blind_sheet_keeps_rows_texts_and_blank_scores():
    scored, blind = read(SCORED), read(BLIND_SHEET)
    assert len(blind) == len(scored) == 40
    assert [row["item_id"] for row in blind] == [row["item_id"] for row in scored]
    for before, after in zip(scored, blind):
        assert before["text_A"] == after["text_A"] and before["text_B"] == after["text_B"]
        assert all(not str(after[column]).strip() for column in SCORES)


def test_text_columns_do_not_name_the_source():
    for row in read(BLIND_SHEET):
        for column in ("text_A", "text_B"):
            lowered = row[column].lower()
            assert not any(word in lowered for word in LEAKY_WORDS), (row["item_id"], column)
