"""3B-12 pre-flight (CPU): follow-up reconciliation (a) and collator mask proof (b)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NL = chr(10)
OUT = ROOT / "outputs/benchmark_v2/eval/prose_pilot"
FOLLOW = ROOT / "outputs/benchmark_v2/followup/val.jsonl"
TRAIN = ROOT / "outputs/benchmark_v2/sft/train_prose_v1.jsonl"
FIT = ROOT / "outputs/analysis/phase3a/fit"
REFUSAL = ("out_of_scope", "false_premise")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def part_a():
    exported = load_jsonl(OUT / "prose_eval_followups.jsonl")
    tally = Counter()
    for row in exported:
        kind = str(row.get("case_type"))
        tally["refusal" if kind in REFUSAL else ("control" if kind == "in_scope_control" else "other")] += 1
    import scripts.eval_protocolA as ev
    args = argparse.Namespace(seed=20261031, items=40, per_case=15, repeats=25, controls=40)
    selected = ev.followup_questions(args)
    by_kind = Counter(row["kind"] for row in selected)
    keys = {(str(r["dialogue_id"]), str(r["question"]).strip()) for r in exported}
    refus = [r for r in selected if r["kind"] in REFUSAL]
    ctrl = [r for r in selected if r["kind"] == "in_scope_control"]
    refus_in = sum(1 for r in refus if (str(r["dialogue_id"]), str(r["question"]).strip()) in keys)
    ctrl_in = sum(1 for r in ctrl if (str(r["dialogue_id"]), str(r["question"]).strip()) in keys)
    contained = refus_in == len(refus) and ctrl_in == len(ctrl)
    corpus = {str(d["dialogue_id"]): d for d in load_jsonl(FOLLOW)}
    rows = []
    for index, row in enumerate(selected):
        dialogue = corpus.get(str(row["dialogue_id"])) or {}
        turns = dialogue.get("turns") or []
        control = row["kind"] == "in_scope_control"
        answer = None if control else (str(turns[0].get("answer")) if turns else None)
        rows.append({"index": index, "dialogue_id": row["dialogue_id"], "kind": row["kind"],
                     "question": row["question"], "answer": answer,
                     "in_scope": row.get("in_scope"),
                     "source": "constructed_control" if control
                     else ("corpus" if answer else "missing")})
    extended = OUT / "prose_eval_followups_179.jsonl"
    if not contained:
        extended.write_text(NL.join(json.dumps(r, ensure_ascii=False) for r in rows) + NL,
                            encoding="utf-8")
    return {
        "exported_97": {"turns": len(exported), "categories": dict(tally),
                        "refusal_kinds": list(REFUSAL),
                        "sha256": sha256(OUT / "prose_eval_followups.jsonl")},
        "selected_179": {"turns": len(selected), "by_kind": dict(sorted(by_kind.items())),
                         "refusal_population": len(refus), "control_population": len(ctrl)},
        "containment": {"refusal_turns_in_export": refus_in, "controls_in_export": ctrl_in,
                        "fully_contained": contained},
        "extended": ({"path": str(extended.relative_to(ROOT)), "sha256": sha256(extended),
                      "turns": len(rows),
                      "constructed_controls": sum(1 for r in rows if not r["answer"])}
                     if not contained else None),
    }


def part_b():
    import torch  # noqa: F401
    from transformers import AutoTokenizer
    from llm.dataset import TEPInstructionDataset, make_collator

    cfg = yaml.safe_load(open(ROOT / "configs/config_v2.yaml", encoding="utf-8"))
    base = cfg["llm"]["base_model"]
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    def is_report(row):
        try:
            return "physical_reasoning" in json.loads(row["answer"])
        except (ValueError, TypeError):
            return False

    all_rows = load_jsonl(TRAIN)
    rows = ([r for r in all_rows if is_report(r)][:4]
            + [r for r in all_rows if not is_report(r)][:4])
    ds = TEPInstructionDataset(rows, tok, max_length=4096, dummy_image_size=448)
    coll = make_collator(tok.pad_token_id)
    batch = coll([ds[i] for i in range(len(rows))])
    assert batch["labels"].shape == batch["input_ids"].shape

    kinds = [is_report(row) for row in rows]
    proof = {}
    for label, index in (("report", kinds.index(True) if True in kinds else None),
                         ("dialogue", kinds.index(False) if False in kinds else None)):
        if index is None:
            continue
        item = ds[index]
        labels = item["labels"]
        supervised = labels[labels != -100].tolist()
        body = tok.decode(supervised, skip_special_tokens=True).strip()
        try:
            keys = sorted(json.loads(body).keys())
        except ValueError:
            keys = ["<not json>"]
        answer = json.loads(rows[index]["answer"])
        gold = str(answer.get("verdict") or "")
        proof[label] = {
            "row": index,
            "prompt_tokens_masked": int((labels == -100).sum()),
            "answer_tokens_supervised": len(supervised),
            "total_tokens": int(item["input_ids"].shape[0]),
            "supervised_keys": keys,
            "supervised_chars": len(body),
            "supervised_head": body[:180],
            "verdict_string_in_supervised": bool(gold) and gold in body,
            "question_in_supervised": str(rows[index]["question"])[:60] in body,
            "supervised_equals_answer": body == str(rows[index]["answer"]).strip()
            or body.split(NL)[0] == str(rows[index]["answer"]).strip().split(NL)[0],
        }
    errors = [k for k, p in proof.items() if isinstance(p, dict)
              and (p["verdict_string_in_supervised"] or p["question_in_supervised"])]
    for label in ("report", "dialogue"):
        if label not in proof:
            errors.append("missing " + label + " row in first 8")
    proof["errors"] = errors
    proof["passed"] = not errors
    return proof


def main():
    report = {"part_a": part_a(), "part_b": part_b()}
    FIT.mkdir(parents=True, exist_ok=True)
    (FIT / "preflight_prose_pilot.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:4000])
    return 0 if report["part_b"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
