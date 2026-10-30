"""Tests for the rejection reconciliation.

The two tables in the full-run report did not reconcile because a code count was
printed under an ``items`` heading. These tests pin the definitions: an item is
counted once, a rule firing is counted separately, and the primary attribution
sums back to the item count.
"""
from __future__ import annotations

from scripts.analyze_teacher_rejections import (
    reconcile,
    train_policy,
)


def _item(item_id, fault=1, variant="menu", fault_id="FAULT_01"):
    return {
        "item_id": item_id,
        "event_id": item_id.split("::")[0],
        "fault": fault,
        "variant": variant,
        "target": {"fault_id": fault_id},
    }


def _record(item_id, *, family="fault_01", first_codes=(), codes=(), accepted=True,
            attempts=None):
    codes = list(codes)
    return {
        "item_id": item_id,
        "family": family,
        "first_codes": list(first_codes),
        "codes": codes,
        "accepted": accepted,
        "attempts": attempts if attempts is not None else (2 if first_codes else 1),
    }


def test_one_item_with_two_codes_is_one_rejected_item():
    items = [_item("e1::menu")]
    cache = {"e1::menu": _record("e1::menu",
                                 first_codes=["number_not_in_store:1.0",
                                              "direction_conflict:XMV_51"],
                                 codes=["number_not_in_store:1.0",
                                        "direction_conflict:XMV_51",
                                        "retry:number_not_in_store:1.0"],
                                 accepted=False)}
    result = reconcile(items, cache)
    totals = result["totals"]
    assert totals["first_rejected_items"] == 1
    assert totals["first_codes"] == 2
    assert totals["items_with_two_first_codes"] == 1
    assert totals["dropped"] == 1
    # Final attribution counts only what the retry failed on.
    assert totals["dropped_codes"] == 1
    row = result["rows"]["FAULT 01"]
    assert row["primary_first"] == {"number_not_in_store": 1}
    assert row["primary_final"] == {"number_not_in_store": 1}


def test_open_set_episode_is_counted_once_under_its_own_family():
    items = [_item("e1::removed"), _item("e2::menu")]
    cache = {
        "e1::removed": _record("e1::removed", family="open_set",
                               first_codes=["number_not_in_store:2.0"],
                               codes=["number_not_in_store:2.0"], accepted=False, attempts=1),
        "e2::menu": _record("e2::menu", first_codes=["unknown_tag:Foo"],
                            codes=[], accepted=True),
    }
    result = reconcile(items, cache)
    assert result["totals"]["first_rejected_items"] == 2
    # The dropped episode was recorded without a retry code: counted separately
    # rather than silently breaking the primary-attribution invariant.
    assert result["totals"]["dropped_no_retry"] == 1
    assert result["rows"]["OPEN_SET"]["first_rejected"] == 1
    # The episode's event fault must not also be charged for it.
    assert result["rows"]["FAULT 01"]["first_rejected"] == 1
    assert result["totals"]["recovered"] == 1


def test_accepted_and_dropped_sum_to_items():
    items = [_item(f"e{i}::menu") for i in range(5)]
    cache = {f"e{i}::menu": _record(f"e{i}::menu", accepted=i < 3,
                                    first_codes=[] if i < 3 else ["reasoning_length:2"],
                                    codes=[] if i < 3 else ["retry:reasoning_length:2"],
                                    attempts=1 if i < 3 else 2)
             for i in range(5)}
    totals = reconcile(items, cache)["totals"]
    assert totals["items"] == 5
    assert totals["accepted"] == 3
    assert totals["dropped"] == 2
    assert totals["recovered"] + totals["dropped"] == totals["retried_items"]


def test_items_without_a_cache_record_are_not_counted():
    items = [_item("e1::menu")]
    assert reconcile(items, {})["totals"]["items"] == 0


def test_train_policy_excludes_dropped_items_and_reports_the_share():
    items = [_item("a::menu", fault=1, fault_id="FAULT_01"),
             _item("b::menu", fault=1, fault_id="FAULT_01"),
             _item("c::removed", fault=1, variant="removed"),
             _item("d::menu", fault=None, fault_id="NORMAL_TRANSIENT")]
    cache = {
        "a::menu": _record("a::menu", accepted=True),
        "b::menu": _record("b::menu", accepted=False, first_codes=["x"], codes=["x"]),
        "c::removed": _record("c::removed", accepted=True),
        "d::menu": _record("d::menu", accepted=True),
    }
    policy = train_policy(items, cache)
    assert policy["items"] == 3
    assert policy["counts"] == {"classified": 1, "UNCLASSIFIED_OPEN_SET": 1,
                                "NORMAL_TRANSIENT": 1}
    assert policy["unclassified_share_of_faulty"] == 0.5
    assert policy["unclassified_share_of_all"] == round(1 / 3, 4)
