"""Menu provenance: which cached answers a knowledge-base edit invalidates.

The point of these tests is the distinction the filter hash cannot make. A KB
edit changes the menu text, so *some* answers are stale; folding the KB into the
filter hash would declare *all* of them stale. The rule is: filter hash must
match, and the menu revision must match unless the record is explicitly marked
unaffected by the revision(s) in between.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from llm.benchmark import provenance as P
from scripts.build_benchmark_dataset import select_pending


def _prov(**overrides) -> dict:
    base = {"filter_sha256": "f" * 64, "menu_revision": "r1", "menu_sha256": "m" * 64}
    base.update(overrides)
    return base


def _record(**overrides) -> dict:
    base = {"item_id": "EVT-1::menu", "filter_sha256": "f" * 64, "menu_revision": "r1",
            "menu_sha256": "m" * 64, "accepted": True}
    base.update(overrides)
    return base


def test_matching_filter_and_revision_is_reusable():
    assert P.reuse_decision(_record(), _prov()) == (True, "revision_match")


def test_a_legacy_record_is_read_as_r1():
    """Records written before the field existed came from the r1 text."""
    record = _record()
    record.pop("menu_revision")
    assert P.reuse_decision(record, _prov()) == (True, "revision_match")


def test_a_filter_change_invalidates_even_at_the_same_revision():
    record = _record(filter_sha256="a" * 64)
    assert P.reuse_decision(record, _prov()) == (False, "filter_changed")


def test_a_revision_bump_invalidates_an_unmarked_record():
    record = _record(menu_revision="r1")
    assert P.reuse_decision(record, _prov(menu_revision="r2")) == (
        False, "menu_revision_changed")


def test_an_explicit_unaffected_marker_survives_the_bump():
    record = _record(menu_revision="r1",
                     menu_revision_unaffected=["r2"])
    assert P.reuse_decision(record, _prov(menu_revision="r2")) == (
        True, "revision_unaffected")


def test_the_marker_is_per_revision_not_a_free_pass():
    """Marked for r2 does not mean marked for r3."""
    record = _record(menu_revision="r1", menu_revision_unaffected=["r2"])
    assert P.reuse_decision(record, _prov(menu_revision="r3")) == (
        False, "menu_revision_changed")


def test_the_marker_accepts_the_dict_form_with_a_reason():
    record = _record(menu_revision="r1",
                     menu_revision_unaffected={"revisions": ["r2"],
                                               "reason": "prose never names XMEAS_7"})
    assert P.reuse_decision(record, _prov(menu_revision="r2")) == (
        True, "revision_unaffected")


def test_a_filter_change_beats_an_unaffected_marker():
    """Unchanged semantics do not exempt an answer from the rules themselves."""
    record = _record(filter_sha256="a" * 64, menu_revision="r1",
                     menu_revision_unaffected=["r2"])
    assert P.reuse_decision(record, _prov(menu_revision="r2")) == (False, "filter_changed")


# -- the replay stamp --------------------------------------------------------
def test_a_stale_hash_without_a_stamp_is_pending():
    """The old behaviour, kept: a hash mismatch alone means a call."""
    record = _record(filter_sha256="a" * 64)
    assert P.reuse_decision(record, _prov()) == (False, "filter_changed")


def test_a_stamp_under_the_current_filter_makes_it_reusable():
    record = _record(filter_sha256="a" * 64,
                     validated_filter_sha256="f" * 64)
    assert P.reuse_decision(record, _prov()) == (True, "validated_revision_match")


def test_a_stamp_under_a_different_filter_does_not_help():
    record = _record(filter_sha256="a" * 64,
                     validated_filter_sha256="b" * 64)
    assert P.reuse_decision(record, _prov()) == (False, "filter_changed")


def test_a_stamp_does_not_excuse_a_menu_revision_bump():
    """Rules passed is not the same claim as menu text unchanged."""
    record = _record(filter_sha256="a" * 64, validated_filter_sha256="f" * 64,
                     menu_revision="r1")
    assert P.reuse_decision(record, _prov(menu_revision="r2")) == (
        False, "menu_revision_changed")


def test_a_stamped_and_exempted_record_survives_both_moves():
    record = _record(filter_sha256="a" * 64, validated_filter_sha256="f" * 64,
                     menu_revision="r1", menu_revision_unaffected=["r2"])
    assert P.reuse_decision(record, _prov(menu_revision="r2")) == (
        True, "validated_revision_unaffected")


def test_the_stamp_records_when_and_how_it_was_checked():
    stamp = P.validation_stamp("c" * 64)
    assert stamp["validated_filter_sha256"] == "c" * 64
    assert stamp["validation_method"] == "replay"
    assert len(stamp["validated_at"]) >= 10


# -- the sidecar the stamps live in ------------------------------------------
def test_stamps_and_exemptions_reach_the_record_through_the_sidecar(tmp_path):
    """A cache line is append-only evidence; claims about it live beside it."""
    from scripts.build_benchmark_dataset import (
        append_cache_meta,
        load_teacher_cache,
        meta_path,
    )

    cache = tmp_path / "teacher_cache.jsonl"
    cache.write_text(json.dumps(_record(item_id="a", filter_sha256="a" * 64)) + "\n",
                     encoding="utf-8")
    cache.write_text(cache.read_text(encoding="utf-8")
                     + json.dumps(_record(item_id="b", filter_sha256="a" * 64)) + "\n",
                     encoding="utf-8")
    append_cache_meta(cache, [
        {"item_id": "a", **P.validation_stamp("f" * 64)},
        {"item_id": "b", "menu_revision_unaffected": {"revisions": ["r2"],
                                                       "reason": "names no XMEAS_7"}},
    ])
    assert cache.read_text(encoding="utf-8").count("validated_filter_sha256") == 0
    assert meta_path(cache).exists()
    merged = load_teacher_cache(cache)
    assert merged["a"]["validated_filter_sha256"] == "f" * 64
    assert merged["b"]["menu_revision_unaffected"]["revisions"] == ["r2"]
    assert P.reuse_decision(merged["a"], _prov()) == (True, "validated_revision_match")
    assert P.reuse_decision(merged["b"], _prov(filter_sha256="a" * 64,
                                             menu_revision="r2")) == (
        True, "revision_unaffected")


def test_only_replayed_answers_are_stamped(tmp_path):
    from scripts.build_benchmark_dataset import load_cache_meta, stamp_replay

    cache = tmp_path / "teacher_cache.jsonl"
    cache.write_text("", encoding="utf-8")
    written = stamp_replay(cache, {"a": {"prose": {}}}, "f" * 64)
    assert written == 1
    assert list(load_cache_meta(cache)) == ["a"]


# -- the ordering used by run_teacher ---------------------------------------
def _items(*ids):
    return [{"item_id": item_id} for item_id in ids]


def test_select_pending_counts_why_each_item_needs_a_call():
    done = {"a": _record(item_id="a", menu_revision="r1",
                         menu_revision_unaffected=["r2"]),
            "b": _record(item_id="b", menu_revision="r1")}
    pending, reasons = select_pending(_items("a", "b", "c"), done,
                                      _prov(menu_revision="r2"))
    assert [item["item_id"] for item in pending] == ["b", "c"]
    assert reasons == {"menu_revision_changed": 1, "not_called": 1,
                       "reuse_revision_unaffected": 1}


def test_select_pending_keeps_everything_under_matching_provenance():
    done = {"a": _record(item_id="a"), "b": _record(item_id="b")}
    pending, reasons = select_pending(_items("a", "b"), done, _prov())
    assert pending == []
    assert reasons == {"reuse_revision_match": 2}


# -- the hashes the records carry -------------------------------------------
def test_menu_hash_covers_every_entry_and_moves_with_the_kb(monkeypatch):
    from scripts.fault_knowledge import FAULT_KNOWLEDGE

    before = P.menu_sha256()
    text = P.menu_text()
    assert "FAULT_13" in text and "UNCLASSIFIED_OPEN_SET" in text

    entry = copy.deepcopy(FAULT_KNOWLEDGE[13])
    patched = [dict(item) for item in entry["initiating"]]
    patched[0]["direction"] = "increase"
    monkeypatch.setitem(FAULT_KNOWLEDGE, 13, {**entry, "initiating": patched})
    after = P.menu_sha256()

    assert after != before, "a KB direction edit must move the menu hash"
    assert "decreases first" in text and "increases first" in P.menu_text()


def test_kb_hash_is_the_file_hash():
    path = Path(P.ROOT) / P.KB_PATH
    assert P.kb_sha256() == hashlib.sha256(path.read_bytes()).hexdigest()


def test_menu_provenance_carries_all_three_fields():
    prov = P.menu_provenance()
    assert set(prov) == {"menu_revision", "menu_sha256", "kb_sha256"}
    assert prov["menu_revision"] == P.MENU_REVISION


@pytest.mark.parametrize("field", ["menu_revision", "menu_sha256", "kb_sha256"])
def test_a_record_round_trips_its_menu_fields_through_json(field):
    """The cache is JSONL; the fields must survive the write/read cycle."""
    record = _record(**{field: "sentinel"})
    assert json.loads(json.dumps(record))[field] == "sentinel"
