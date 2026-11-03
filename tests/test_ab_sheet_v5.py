"""The v4 sheet defect is detectable, and the v5 sheet does not have it (PROTOCOL 26.3, 26.4)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.ab_sheet_checks import (FIT, LEAK_LABELS, Sources, audit_sheet,  # noqa: E402
                                     format_features, read_rows)

V4 = FIT / "audit_ab_100_v4.csv"
V4_KEY = FIT / "audit_ab_100_v4_key.csv"
V5 = FIT / "audit_ab_100_v5.csv"
V5_KEY = FIT / "audit_ab_100_v5_key.csv"
V5_PROV = FIT / "audit_ab_100_v5_provenance.json"
SOURCES = None


def sources() -> Sources:
    global SOURCES
    if SOURCES is None:
        SOURCES = Sources()
    return SOURCES


def test_v4_regression_every_row_pairs_two_different_items():
    report = audit_sheet(V4, V4_KEY, sources())
    assert report["rows"] == 40
    assert report["rows_with_different_items"] == 40
    assert report["same_item_rows"] == 0


def test_v4_regression_key_label_never_matches_the_text_style():
    report = audit_sheet(V4, V4_KEY, sources())
    assert report["rows_with_style_mismatch"] == 40


def test_v4_regression_classified_rows_sit_in_the_normal_stratum():
    report = audit_sheet(V4, V4_KEY, sources())
    assert report["classified_rows"] == 7
    key = read_rows(V4_KEY)
    assert sum(1 for row in key if row["verdict"] == "no_fault") == 13


def test_v5_pairs_the_same_item_on_both_sides():
    provenance = json.loads(V5_PROV.read_text(encoding="utf-8"))
    assert len(provenance) == 40
    assert all(entry["adapter_item"] == entry["fallback_item"] == entry["corpus_item_id"]
               for entry in provenance.values())
    report = audit_sheet(V5, V5_KEY, sources(), provenance)
    assert report["rows"] == 40
    assert report["same_item_rows"] == 40
    assert report["rows_with_different_items"] == 0
    assert report["rows_with_style_mismatch"] == 0
    assert report["classified_rows"] == 0


def test_v5_strata_are_exactly_ten_ten_twenty():
    key = read_rows(V5_KEY)
    families = defaultdict(int)
    for row in key:
        families[row["family"]] += 1
    assert dict(families) == {"open_set": 10, "removed": 10, "normal": 20}
    normal = [row for row in key if row["family"] == "normal"]
    assert all(row["verdict"] == "no_fault" and row["diagnosis"] == "NORMAL_TRANSIENT"
               for row in normal)
    assert not any(row["diagnosis"].startswith("FAULT") for row in key)


def test_v5_key_is_fifty_fifty_inside_every_stratum():
    key = read_rows(V5_KEY)
    counts = defaultdict(lambda: [0, 0])
    for row in key:
        counts[row["family"]][1] += 1
        counts[row["family"]][0] += int(row["A"] == "adapter")
    for family, (adapter_on_a, rows) in counts.items():
        assert adapter_on_a * 2 == rows, family


def test_v5_columns_and_blank_scoring():
    expected = ["item_id", "text_A", "text_B", "correct_A", "grounded_A", "consistent_A",
                "useful_A", "correct_B", "grounded_B", "consistent_B", "useful_B", "preference",
                "notes"]
    rows = read_rows(V5)
    assert list(rows[0].keys()) == expected
    assert not ({"family", "diagnosis", "verdict", "source"} & set(rows[0].keys()))
    assert all(row[column] == "" for row in rows for column in expected[3:])


def test_v5_layout_cannot_identify_the_source():
    rows = read_rows(V5)
    for row in rows:
        left, right = format_features(row["text_A"]), format_features(row["text_B"])
        assert left == right, row["item_id"]
        assert left["labels"] == ("Diagnosis", "Reasoning", "Narrative"), row["item_id"]
        assert left["braces"] == 0 and left["bullets"] == 0, row["item_id"]
        assert left["leak_labels"] == (), row["item_id"]
        for side in ("text_A", "text_B"):
            body = row[side]
            assert "Diagnosis: " in body and "Reasoning: " in body and "Narrative: " in body
            assert not any(label in body for label in LEAK_LABELS), (row["item_id"], side)
