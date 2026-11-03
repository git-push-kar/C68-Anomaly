"""The v5 sheet is auditable without the key, and the decision reading reproduces (PROTOCOL 29)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.audit_v5_sheet import (FIT, KEY, PROVENANCE, read_rows,  # noqa: E402
                                    stage_join, stage_structure)

SCORED = FIT / "audit_ab_100_v5_scored_rater1.csv"


def test_structure_gate_passes_and_pairs_every_row():
    report = stage_structure()
    assert report["passed"] is True
    assert report["rows"] == 40
    assert report["pair_recovered"] == 40
    assert report["pair_wrong_item"] == 0
    assert report["layout_structural_mismatch"] == 0
    assert report["leaks"] == 0


def test_structure_report_carries_no_side_information():
    """The audit runs before scoring, so nothing in it may say which side is which."""
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    for entry in provenance.values():
        assert set(entry) == {"corpus_item_id", "adapter_item", "fallback_item", "tag", "z"}
    for entry in stage_structure()["per_row"]:
        assert "adapter_side" not in entry
        assert "source" not in entry
    with open(FIT / "audit_ab_100_v5.csv", newline="", encoding="utf-8") as handle:
        columns = next(csv.reader(handle))
    for forbidden in ("family", "diagnosis", "verdict", "source"):
        assert forbidden not in columns


def test_decision_reading_reproduces_from_the_stored_scoring():
    report = stage_join(SCORED)
    assert report["rows"] == 40
    assert report["pooled"] == {"adapter": 14, "fallback": 11, "tie": 15, "non_tie": 25,
                               "adapter_share_non_tie": 0.56}
    assert report["decision"] == "fallback_v31"
    assert report["per_stratum"]["removed"]["adapter_share_non_tie"] == 0.8889
    assert report["per_stratum"]["normal"]["adapter_share_non_tie"] == 0.2857
    # The decision is one row from the threshold; the record must keep saying so.
    assert report["pooled"]["adapter"] + 1 == round(0.60 * report["pooled"]["non_tie"])


def test_every_keyed_row_is_scored_once_with_a_legal_preference():
    scored = {row["item_id"]: row for row in read_rows(SCORED)}
    assert set(scored) == {row["item_id"] for row in read_rows(KEY)}
    assert len(scored) == 40
    for row in scored.values():
        assert row["preference"].strip().upper() in ("A", "B", "TIE")
        for column in ("correct", "grounded", "consistent", "useful"):
            for side in ("A", "B"):
                assert row[column + "_" + side].strip() in ("0", "1")
