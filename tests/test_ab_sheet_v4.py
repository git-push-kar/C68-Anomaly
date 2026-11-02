"""The v4 sheet must be balanced within stratum and must not reveal which side is the adapter."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_ab_sheet_v4 import (STRATA, build_sheet, dummy_candidates, leak_report,
                                      write_sheet)


def test_dummy_sheet_is_balanced_and_wide():
    candidates = dummy_candidates()
    rows, key = build_sheet(candidates)
    assert len(rows) == 40
    assert len(key) == 40
    assert [row["item_id"] for row in rows] == [f"v4-{i:02d}" for i in range(1, 41)]
    for stratum, total in STRATA.items():
        assert sum(1 for row in rows if row["stratum"] == stratum) == total
    for column in ("score_A", "score_B", "preference", "notes"):
        assert all(row[column] == "" for row in rows)
    assert "source" not in rows[0]


def test_leak_report_passes_for_every_column():
    rows, key = build_sheet(dummy_candidates())
    report = leak_report(rows, key)
    assert report["passed"], report
    for stratum, info in report["per_stratum"].items():
        assert info["adapter_on_A"] == info["fallback_on_A"] == info["rows"] // 2
    for column, info in report["columns"].items():
        if column == "item_id":
            assert info["positional"]
        else:
            assert info["balanced"], column


def test_tampered_key_fails_the_leak_test():
    rows, key = build_sheet(dummy_candidates())
    for entry in key:
        entry["side_A_source"] = "adapter"
    assert not leak_report(rows, key)["passed"]


def test_odd_stratum_is_refused():
    candidates = dummy_candidates()[:-1]
    with pytest.raises(ValueError):
        build_sheet(candidates)


def test_wrong_stratum_counts_are_refused():
    candidates = dummy_candidates()
    candidates[0]["stratum"] = "normal"
    with pytest.raises(ValueError):
        build_sheet(candidates)


def test_written_sheet_round_trips(tmp_path):
    rows, key = build_sheet(dummy_candidates())
    path = tmp_path / "sheet.csv"
    write_sheet(rows, path)
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0].split(",")[0] == "item_id"
    assert len(text) == 41
    assert all(row["text_A"] != row["text_B"] for row in rows)
