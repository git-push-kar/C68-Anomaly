"""End-to-end tests for the follow-up dialogue builder.

The event set here is a *fixture* set in the Phase 2B record contract, generated
from the three stored response fixtures: the detector artifacts in this checkout
cannot rebuild the real Phase 2B events (see the phase 2C report). Every case type
is built through the same code path the real build uses, with the fake teacher.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_followup_dataset as builder  # noqa: E402
import make_followup_fixture_events as fixture_events  # noqa: E402
from build_benchmark_dataset import load_records  # noqa: E402
from llm.followup import cases  # noqa: E402
from llm.followup.cases import DEFAULT_MIX, FOLLOWUP_CASE_ORDER  # noqa: E402
from llm.followup.filters import check_answer  # noqa: E402
from llm.followup.leak import (  # noqa: E402
    LeakError,
    assert_clean,
    check_event_overlap,
    check_paraphrase_disjointness,
)
from llm.followup.paraphrases import heldout_questions, normalise, train_questions  # noqa: E402
from llm.followup.teacher import FakeTeacher, TeacherCache  # noqa: E402
from llm.response.lookup_answers import lookup_answer  # noqa: E402
from llm.response.schemas import Intent, table_allowed_for  # noqa: E402
from llm.response.store_access import StoreAccess, UnknownKeyError  # noqa: E402


@pytest.fixture(scope="session")
def events_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("followup_events")
    fixture_events.build(out, per_split=8, normal_per_split=2)
    return out


@pytest.fixture(scope="session")
def records(events_dir) -> dict:
    return load_records(events_dir)


def pick_record(records: dict, split: str = "train", fault: int = 15) -> dict:
    for record in records.get(split, []):
        if record.get("fault") == fault:
            return record
    raise AssertionError(f"no fault-{fault} record in {split}")


def make_dialogue(case_type: str, *, split: str = "train", variant: str = "menu",
                  teacher=None, cache=None, index: int = 0, mix=None):
    record = pick_record(records_fixture(), split)
    return builder.build_dialogue(
        record, split, variant, case_type, teacher or FakeTeacher(), cache,
        mix=mix or DEFAULT_MIX, seed=42, index=index,
    )


#: Records are loaded once per session; the fixture directory lives in tmp.
_RECORDS: dict = {}


def records_fixture() -> dict:
    return _RECORDS


@pytest.fixture(autouse=True, scope="session")
def _load_records(records):
    _RECORDS.clear()
    _RECORDS.update(records)
    yield


# ---------------------------------------------------------------------------
# one dialogue per case type
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case_type", FOLLOWUP_CASE_ORDER)
def test_every_case_type_builds_with_the_fake_teacher(case_type):
    dialogue = make_dialogue(case_type)
    assert 2 <= len(dialogue["turns"]) <= 4, "a dialogue is 2-4 turns"
    assert dialogue["case_type"] in FOLLOWUP_CASE_ORDER
    for turn in dialogue["turns"]:
        assert turn["accepted"], (case_type, turn["kind"], turn.get("filter_codes"))
        assert turn["answer"], "an accepted turn carries an answer"
        assert 1 <= cases.sentence_count(turn["answer"]) <= 4
        assert "?" not in turn["answer"]
        # deterministic fields are present and resolvable
        for key in turn["citations"]:
            assert key.count(".") >= 1
        if turn["intent"] is not None:
            intent = Intent(turn["intent"])
            assert turn["table"] in {t.value for t in table_allowed_for(intent)} | {"none"} or \
                turn["table"] == "none"
        for suggestion in turn["suggestions"]:
            from llm.response.router import ResponseRouter
            from llm.response.router import LOOKUP_INTENTS

            candidates, _ = ResponseRouter().candidates(suggestion)
            assert any(c in LOOKUP_INTENTS for c in candidates), suggestion


@pytest.mark.parametrize("case_type", FOLLOWUP_CASE_ORDER)
def test_deterministic_fields_resolve_in_the_store(case_type):
    """Every citation is a key the store answers, and the table is allowed."""
    dialogue = make_dialogue(case_type)
    record = pick_record(records_fixture(), "train")
    store = StoreAccess(record["event"], {
        "root_cause": dialogue["turns"][0].get("intent") or "FAULT_15",
    })
    for turn in dialogue["turns"]:
        for key in turn["citations"]:
            value = store.resolve(key)
            assert value is not None or key.endswith(".alarmed")
        if turn["intent"] is not None and turn["table"] != "none":
            assert turn["table"] in {t.value for t in table_allowed_for(Intent(turn["intent"]))}


def test_template_sample_answers_are_the_lookup_text():
    dialogue = make_dialogue("template_sample")
    assert dialogue["turns"], "template sample builds turns"
    for turn in dialogue["turns"]:
        assert turn["answer_source"] == "lookup"
        record = pick_record(records_fixture(), "train")
        hit = lookup_answer(record["event"], {
            "root_cause": "FAULT_15",
            "affected_subsystem": record["event"]["evidence"].get("candidate_subsystem"),
            "hypotheses": [],
            "recommended_action": None,
        }, turn["question"])
        assert hit is not None
        assert turn["answer"] == hit.answer.answer
        assert turn["intent"] == hit.answer.intent.value


# ---------------------------------------------------------------------------
# open-set
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case_type", ["valve_actuator", "hypothesis_comparison", "out_of_scope"])
def test_open_set_dialogue_never_names_the_removed_fault(case_type):
    dialogue = make_dialogue(case_type, variant="removed")
    assert dialogue["open_set"] is True
    # The context and the answers must not establish the removed family. The
    # questions are checked for the id only: an operator may describe the
    # physics ("is the condenser cooling water valve sticking?") without
    # naming a family, and fault 15's KB name happens to be that description.
    answers = dialogue["context"] + "\n" + "\n".join(turn["answer"] or "" for turn in dialogue["turns"])
    for needle in ("FAULT_15", "fault 15", "Condenser cooling water valve sticking"):
        assert needle.lower() not in answers.lower(), needle
    for turn in dialogue["turns"]:
        assert "FAULT_15" not in turn["question"]
        assert "fault 15" not in turn["question"].lower()


def test_open_set_answers_are_still_grounded():
    dialogue = make_dialogue("valve_actuator", variant="removed")
    record = pick_record(records_fixture(), "train")
    for turn in dialogue["turns"]:
        assert turn["accepted"], turn.get("filter_codes")
        assert "FAULT_15" not in (turn["answer"] or "")


# ---------------------------------------------------------------------------
# refusal and clarification guardrails
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flavour_index", [0, 1, 2])
def test_false_premise_is_checked_against_the_store(flavour_index):
    dialogue = make_dialogue("false_premise", index=flavour_index)
    turn = dialogue["turns"][0]
    premise = turn["premise"]
    assert premise["verdict"] in ("missing", "contradicted")
    assert premise["verdict"] != "supported", "a false premise must actually be false"
    refusal = turn["refusal"]
    assert refusal["required"] is True
    assert refusal["alternatives"], "a refusal offers something the store answers"
    from llm.response.router import ResponseRouter

    for alternative in refusal["alternatives"]:
        assert ResponseRouter().live_intent(
            pick_record(records_fixture(), "train")["event"], {"root_cause": "FAULT_15"},
            alternative["question"],
        ) == alternative["intent"]


def test_out_of_scope_refusal_is_required_and_offers_an_alternative():
    dialogue = make_dialogue("out_of_scope", index=0)
    turn = dialogue["turns"][0]
    assert turn["answerability"] == "refuse"
    assert turn["refusal"]["reason"] in ("key_missing", "off_domain")
    assert turn["refusal"]["alternatives"]


def test_a_confirming_answer_to_a_false_premise_is_rejected():
    dialogue = make_dialogue("false_premise", teacher=FakeTeacher(profile="confirm"))
    turn = dialogue["turns"][0]
    assert not turn["accepted"]
    assert "refusal_missing" in turn["filter_codes"]


def test_a_bare_refusal_without_an_alternative_is_rejected():
    dialogue = make_dialogue("out_of_scope", teacher=FakeTeacher(profile="no_alternative"))
    turn = dialogue["turns"][0]
    assert "refusal_no_alternative" in turn["filter_codes"]


def test_clarification_is_asked_at_most_once_per_dialogue():
    dialogue = make_dialogue("ambiguous_question")
    questions = [turn["clarifying_question"] for turn in dialogue["turns"] if turn["clarifying_question"]]
    assert len(questions) == 1
    first = dialogue["turns"][0]
    assert first["answerability"] == "clarify"
    assert first["answer"] and "?" not in first["answer"]
    assert first["table"] == "none"


def test_clarification_reply_never_re_asks():
    dialogue = make_dialogue("clarification_reply")
    first, reply = dialogue["turns"][0], dialogue["turns"][1]
    assert first["clarifying_question"]
    assert reply["clarifying_question"] is None
    assert "?" not in (reply["answer"] or "")
    assert reply["intent"] == "clarification_reply"
    assert reply["resolved_intent"] in {i.value for i in Intent}


def test_repeated_question_is_not_answered_with_the_same_words():
    dialogue = make_dialogue("repeated_question")
    first, repeat = dialogue["turns"][0], dialogue["turns"][1]
    assert repeat["question"] == first["question"]
    assert repeat["repeat_of"] == 0
    assert repeat["answer"] != first["answer"]
    from llm.followup.filters import is_near_duplicate

    assert not is_near_duplicate(repeat["answer"], first["answer"])


def test_a_verbatim_repeat_is_rejected_by_the_filter():
    dialogue = make_dialogue("repeated_question", teacher=FakeTeacher(profile="repeat"))
    repeat = dialogue["turns"][1]
    assert any(code.startswith("near_duplicate") for code in repeat["filter_codes"])


# ---------------------------------------------------------------------------
# leaks, splits, hold-out
# ---------------------------------------------------------------------------
def test_leak_check_aborts_on_an_event_id():
    with pytest.raises(LeakError):
        assert_clean("See EVT-train-f15-r001 for details.", permitted_faults=[15])


def test_leak_check_aborts_on_a_split_name_and_on_a_run_number():
    for text in ("This came from the train split.", "Run 312 looked the same.", "seed 42"):
        with pytest.raises(LeakError):
            assert_clean(text, permitted_faults=[15])


def test_leak_check_aborts_on_a_fault_outside_the_menu():
    with pytest.raises(LeakError):
        assert_clean("The signature is FAULT_15.", permitted_faults=list(range(1, 15)))


def test_context_is_clean_and_carries_an_opaque_reference():
    dialogue = make_dialogue("valve_actuator")
    assert "INC-" in dialogue["context"]
    assert_clean(dialogue["context"], permitted_faults=list(range(1, 16)),
                 event_id=pick_record(records_fixture(), "train")["event_id"], allow_opaque=True)


def test_event_overlap_between_splits_is_detected(records, events_dir):
    duplicated = {"train": [dict(pick_record(records, "train"))],
                  "test": [dict(pick_record(records, "train"))]}
    with pytest.raises(LeakError):
        check_event_overlap(duplicated, None)


def test_split_integrity_is_clean_on_the_fixture_set(records):
    report = check_event_overlap(
        {split: records.get(split, []) for split in ("train", "val", "test")}, builder.SPLITS_FILE
    )
    assert report["zero_overlap"] is True
    assert report["events_per_split"]["train"] > 0


def test_paraphrase_holdout_is_disjoint():
    report = check_paraphrase_disjointness()
    assert report["heldout_total"] > 0
    for label, entry in report["per_label"].items():
        assert entry["disjoint"] is True, label
        train = {normalise(q) for q in train_questions(label)}
        heldout = {normalise(q) for q in heldout_questions(label)}
        assert not (train & heldout)


def test_val_and_test_use_the_heldout_paraphrases():
    for split in ("val", "test"):
        dialogue = make_dialogue("valve_actuator", split=split)
        for turn in dialogue["turns"]:
            label = turn["paraphrase"].split("#")[0]
            if label not in {"valve_actuator", "causal_why", "mechanism"}:
                continue
            heldout = {normalise(q) for q in heldout_questions(label)}
            train = {normalise(q) for q in train_questions(label)}
            assert normalise(turn["question"]) in heldout
            assert normalise(turn["question"]) not in train


def test_train_uses_only_train_paraphrases():
    dialogue = make_dialogue("valve_actuator", split="train")
    for turn in dialogue["turns"]:
        label = turn["paraphrase"].split("#")[0]
        if label not in {"valve_actuator", "causal_why", "mechanism"}:
            continue
        assert normalise(turn["question"]) in {normalise(q) for q in train_questions(label)}


# ---------------------------------------------------------------------------
# reproducibility, cache, mix, audit
# ---------------------------------------------------------------------------
def test_same_seed_reproduces_the_whole_dialogue():
    first = make_dialogue("hypothesis_comparison")
    second = make_dialogue("hypothesis_comparison")
    def strip(dialogue):
        return [
            {k: turn[k] for k in ("kind", "question", "intent", "resolved_intent", "table",
                                  "citations", "clarifying_question", "answer", "suggestions")}
            for turn in dialogue["turns"]
        ]
    assert strip(first) == strip(second)
    assert first["context"] == second["context"]


def test_cache_resume_skips_the_teacher(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    first_teacher = FakeTeacher()
    cache = TeacherCache(cache_path, model=first_teacher.model)
    first = make_dialogue("causal_why", teacher=first_teacher, cache=cache)
    assert first_teacher.calls > 0

    second_teacher = FakeTeacher()
    resumed = TeacherCache(cache_path, model=second_teacher.model)
    second = make_dialogue("causal_why", teacher=second_teacher, cache=resumed)
    assert second_teacher.calls == 0, "a resumed build must not call the teacher again"
    assert resumed.hits > 0
    assert [t["answer"] for t in first["turns"]] == [t["answer"] for t in second["turns"]]


def test_mix_override_changes_the_case_distribution():
    mix = {name: 0.0 for name in FOLLOWUP_CASE_ORDER}
    mix["valve_actuator"] = 1.0
    chosen = {cases.choose_case_type(mix, f"key-{i}") for i in range(25)}
    assert chosen == {"valve_actuator"}
    default = {cases.choose_case_type(DEFAULT_MIX, f"key-{i}") for i in range(200)}
    assert "template_sample" in default
    assert len(default) >= 8


def test_audit_sheet_is_stratified_by_case_type(tmp_path):
    dialogues = {name: [make_dialogue(name, index=i)] for i, name in enumerate(FOLLOWUP_CASE_ORDER)}
    path = tmp_path / "audit.csv"
    builder.write_audit(path, dialogues, limit=100, seed=42)
    import csv

    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {"correct", "grounded", "refusal_appropriate", "not_repetitive", "useful", "notes"} <= set(rows[0])
    assert all(row["correct"] == "" and row["notes"] == "" for row in rows)
    assert {row["case_type"] for row in rows} == set(FOLLOWUP_CASE_ORDER)


def test_full_build_writes_every_split_and_manifest(events_dir, tmp_path, monkeypatch):
    out = tmp_path / "followup"
    argv = [
        "build_followup_dataset.py",
        "--events", str(events_dir),
        "--out", str(out),
        "--limit", "6",
        "--no-cache",
        "--audit", "20",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert builder.main() == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        path = out / f"{split}.jsonl"
        assert path.exists()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows, split
        assert all(2 <= len(row["turns"]) <= 4 for row in rows)
        assert manifest["counts"][split]["dialogues"] == len(rows)
    assert manifest["leak_check"]["violations"] == 0
    assert manifest["split_integrity"]["zero_overlap"] is True
    assert manifest["paraphrase_disjointness"]["per_label"]["sensor_values"]["disjoint"] is True
    assert manifest["teacher"] == "fake"
    assert manifest["cost_estimate"]["seconds_per_call"] == 10.9
    assert (out / "audit_sample.csv").exists()
