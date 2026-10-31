"""Protocol A evaluation: parity, checkpoint sweep, generation, follow-up sanity.

Four subcommands, all validation-only. Nothing here reads a test component, and
nothing here fits or tunes anything: the checkpoint rule (macro-F1, pre-registered)
is applied, not chosen.

    python scripts/eval_protocolA.py parity --items 40
    python scripts/eval_protocolA.py checkpoints --per-checkpoint 300
    python scripts/eval_protocolA.py generate --items 50 --cap 512
    python scripts/eval_protocolA.py followup --items 60

``checkpoints`` loads the base model once and swaps adapters into it, writes one
prediction file per checkpoint (so every later comparison is a McNemar over saved
files rather than a re-run), selects by macro-F1 on the subsample, and then scores
that checkpoint on the full Protocol A validation.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.benchmark import evaluate_v2 as ev  # noqa: E402
from llm.benchmark.menu import UNCLASSIFIED  # noqa: E402
from llm.benchmark.prompt_v2 import SYSTEM_PROMPT_V2  # noqa: E402
from llm.benchmark.hydrate import SLIM_KEYS  # noqa: E402
from llm.followup.filters import declines, overlap_ratio  # noqa: E402
from llm.followup.teacher import SYSTEM_PROMPT_FOLLOWUP  # noqa: E402

VAL_ITEMS = Path("outputs/benchmark_v2/dataset/val.jsonl")
FOLLOWUP_VAL = Path("outputs/benchmark_v2/followup/val.jsonl")
ADAPTER_DIR = Path("outputs/tep_rca_v2_protocolA_seed42")
OUT_DIR = Path("outputs/benchmark_v2/eval")
CONFIG = Path("configs/config_v2_protocolA_seed42.yaml")


# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def gold_of(item: Dict) -> str:
    return str((item.get("target") or {}).get("fault_id") or "")


def stratify(items: Sequence[Dict], total: int, seed: int,
             key=lambda item: (gold_of(item), item.get("variant"))) -> List[Dict]:
    """A seeded stratified subsample of exactly ``total`` items.

    Proportional to each stratum, at least one item per stratum where the stratum is
    non-empty, then trimmed or topped up so the size is exact. Stratifying on
    (label, variant) keeps both menu variants present in every class, which is what
    stops the subsample from measuring one variant's prose and calling it accuracy.
    """
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    rng = random.Random(seed)
    for group in groups.values():
        group.sort(key=lambda item: str(item.get("item_id")))
        rng.shuffle(group)
    size = len(items)
    quota = {name: max(1, min(len(group), round(total * len(group) / max(size, 1))))
             for name, group in groups.items()}
    order = sorted(groups, key=lambda name: (-len(groups[name]), str(name)))
    while sum(quota.values()) > total:
        reduced = False
        for name in order:
            if quota[name] > 1:
                quota[name] -= 1
                reduced = True
                break
        if not reduced:
            break
    while sum(quota.values()) < total:
        extended = False
        for name in order:
            if quota[name] < len(groups[name]):
                quota[name] += 1
                extended = True
                break
        if not extended:
            break
    picked = [item for name in sorted(quota) for item in groups[name][: quota[name]]]
    return picked[:total]


def prompt_tags(prompt: str) -> List[str]:
    return sorted({tag.upper() for tag in ev.TAG_PATTERN.findall(prompt or "")})


def prompt_numbers(prompt: str) -> List[float]:
    return [float(match) for match in ev.NUMBER_PATTERN.findall(prompt or "")]


def run_of(item: Dict) -> str:
    return "|".join(str(item.get(key)) for key in ("source_file", "fault", "simulationRun"))


def load_base(config: Path):
    from llm.benchmark.decode_bench import build_model

    return build_model(config, None, merged=False)


def adapter_root(path: Path) -> Path:
    """The directory that actually holds ``adapter_config.json``.

    A trainer checkpoint keeps its adapter one level down, in
    ``<checkpoint>/<adapter_name>/``, because ``save_pretrained`` is handed a
    directory named after the adapter. The caller therefore names the container,
    not the adapter, and handing that straight to ``PeftModel.from_pretrained``
    fails on a directory that looks complete but is not.
    """
    path = Path(path)
    if (path / "adapter_config.json").exists():
        return path
    if path.is_dir():
        nested = [child for child in sorted(path.iterdir())
                  if child.is_dir() and (child / "adapter_config.json").exists()]
        if len(nested) == 1:
            return nested[0]
    raise SystemExit(f"no adapter_config.json at or under {path}")


def adapter_steps(directory: Path) -> List[tuple]:
    """Every checkpoint and the final adapter, in training order."""
    found = []
    for path in sorted(directory.glob("checkpoint-*")):
        try:
            found.append((int(path.name.split("-")[-1]), adapter_root(path)))
        except (ValueError, SystemExit):
            continue
    summary = directory / "training_summary.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        step = int(data.get("global_step") or 0)
        try:
            found.append((step, adapter_root(directory)))
        except SystemExit:
            pass
    return sorted(set(found), key=lambda pair: pair[0])


def score(model, tokenizer, items: Sequence[Dict], dummy_size: int, batch: int,
          via: str = "wrapper") -> List[Dict]:
    from llm.benchmark.label_scorer import score_batch

    questions = [item["prompt"] for item in items]
    scored = score_batch(model, tokenizer, questions, dummy_size, batch=batch, via=via)
    rows = []
    for item, result in zip(items, scored):
        rows.append({
            "item_id": item.get("item_id"),
            "gold": gold_of(item),
            "pred": result["label"],
            "probability": result["probability"],
            "probs": result.get("probs"),
            "run": run_of(item),
            "variant": item.get("variant"),
            "unobservable": bool(item.get("unobservable")),
            "seconds": result.get("seconds"),
        })
    return rows


# ---------------------------------------------------------------------------
def cmd_parity(args) -> int:
    """Batch-16 label scoring through the wrapper and through the language model."""
    items = [item for item in load_jsonl(VAL_ITEMS) if not item.get("unobservable")]
    sample = stratify(items, args.items, args.seed)
    model, tokenizer, config, _applied = load_base(Path(args.config))
    dummy = int(config["llm"].get("dummy_image_size", 448))
    wrapper = score(model, tokenizer, sample, dummy, args.batch, via="wrapper")
    direct = score(model, tokenizer, sample, dummy, args.batch, via="language_model")
    agree = sum(1 for a, b in zip(wrapper, direct) if a["pred"] == b["pred"])
    deltas = []
    for a, b in zip(wrapper, direct):
        for label in set(a["probs"] or {}) & set(b["probs"] or {}):
            deltas.append(abs(a["probs"][label] - b["probs"][label]))
    result = {
        "items": len(sample), "batch": args.batch,
        "argmax_agree": agree,
        "argmax_agree_rate": round(agree / max(len(sample), 1), 4),
        "max_abs_delta_probability": round(max(deltas) if deltas else 0.0, 6),
        "threshold": 0.01,
        "passes": agree == len(sample) and (max(deltas) if deltas else 0.0) < 0.01,
        "seconds_per_item_wrapper": round(
            sum(row["seconds"] or 0 for row in wrapper) / max(len(wrapper), 1), 4),
        "seconds_per_item_direct": round(
            sum(row["seconds"] or 0 for row in direct) / max(len(direct), 1), 4),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scoring_parity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT_DIR / "preds_parity_wrapper.jsonl").write_text(
        "\n".join(json.dumps(row) for row in wrapper), encoding="utf-8")
    (OUT_DIR / "preds_parity_direct.jsonl").write_text(
        "\n".join(json.dumps(row) for row in direct), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def _metric_row(report: Dict) -> Dict:
    return {key: report[key] for key in ("items", "accuracy", "top3", "macro_f1")}


def cmd_checkpoints(args) -> int:
    from peft import PeftModel

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    observable = [item for item in load_jsonl(VAL_ITEMS) if not item.get("unobservable")]
    sample = stratify(observable, args.per_checkpoint, args.seed)
    (OUT_DIR / "val_subsample_ids.json").write_text(
        json.dumps([item["item_id"] for item in sample], indent=2), encoding="utf-8")
    steps = adapter_steps(Path(args.adapter))
    if not steps:
        raise SystemExit(f"no checkpoints under {args.adapter}")

    model, tokenizer, config, _applied = load_base(Path(args.config))
    dummy = int(config["llm"].get("dummy_image_size", 448))
    # Wrap once and swap adapters into the same model. Wrapping repeatedly would
    # attach LoRA layers to modules that already have them, and the parameter count
    # would silently stop matching the run.
    first_step, first_path = steps[0]
    peft_model = PeftModel.from_pretrained(model, str(first_path),
                                           adapter_name=f"step{first_step}")
    history = []
    for step, path in steps:
        name = f"step{step}"
        if name not in peft_model.peft_config:
            peft_model.load_adapter(str(path), adapter_name=name)
        peft_model.set_adapter(name)
        started = time.time()
        rows = score(peft_model, tokenizer, sample, dummy, args.batch, via=args.via)
        seconds = time.time() - started
        report = ev.report(rows, label=name)
        (OUT_DIR / f"preds_val300_{name}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        entry = {"step": step, "path": str(path), **_metric_row(report),
                 "abstention": report["abstention"],
                 "per_fault_recall": {label: info["recall"]
                                      for label, info in report["per_fault"].items()},
                 "seconds": round(seconds, 1),
                 "seconds_per_item": round(seconds / max(len(sample), 1), 3)}
        history.append(entry)
        print(json.dumps(entry))

    # The rule is macro-F1; accuracy breaks an exact tie, then the later step.
    best = max(history, key=lambda row: (row["macro_f1"], row["accuracy"], row["step"]))
    (OUT_DIR / "checkpoint_table.json").write_text(json.dumps(history, indent=2),
                                                   encoding="utf-8")

    selected = dict(best)
    chosen = selected["path"]
    peft_model.set_adapter(f"step{selected['step']}")
    full_items = load_jsonl(VAL_ITEMS)
    started = time.time()
    full_rows = score(peft_model, tokenizer, full_items, dummy, args.batch, via=args.via)
    seconds = time.time() - started
    observables = [row for row in full_rows if not row["unobservable"]]
    unobservable = [row for row in full_rows if row["unobservable"]]
    full_report = ev.report(observables, label=f"{chosen}:observable")
    unobservable_report = ev.report(unobservable, label=f"{chosen}:unobservable")
    (OUT_DIR / f"preds_val_full_{selected['step']}.jsonl").write_text(
        "\n".join(json.dumps(row) for row in full_rows), encoding="utf-8")
    (OUT_DIR / "selected_checkpoint.json").write_text(json.dumps({
        "selected": selected,
        "rule": "macro-F1 on the 300-item stratified validation subsample, accuracy "
                "as tie-break, later step as second tie-break",
        "scoring_via": args.via,
        "full_val": full_report,
        "unobservable": unobservable_report,
        "full_val_seconds": round(seconds, 1),
    }, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"selected": selected["step"], "macro_f1": selected["macro_f1"],
                      "accuracy": selected["accuracy"],
                      "full_val_accuracy": full_report["accuracy"],
                      "full_val_macro_f1": full_report["macro_f1"]}, indent=2))
    return 0


def cmd_base(args) -> int:
    """The unadapted base model, scored with the same prompt and candidate set.

The reference row every adapter number needs: a 2 B chat model that has never
seen this dataset can still be right about some of it, and without measuring
that, "accuracy 0.84" has no denominator.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    observable = [item for item in load_jsonl(VAL_ITEMS) if not item.get("unobservable")]
    sample = stratify(observable, args.per_checkpoint, args.seed)
    model, tokenizer, config, applied = load_base(Path(args.config))
    if applied:
        raise SystemExit(f"base reference must run without an adapter, got {applied}")
    dummy = int(config["llm"].get("dummy_image_size", 448))
    started = time.time()
    rows300 = score(model, tokenizer, sample, dummy, args.batch, via=args.via)
    seconds300 = time.time() - started
    report300 = ev.report(rows300, label="base:val300")
    (OUT_DIR / "preds_base_val300.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows300), encoding="utf-8")

    started = time.time()
    full_rows = score(model, tokenizer, load_jsonl(VAL_ITEMS), dummy, args.batch,
                      via=args.via)
    seconds_full = time.time() - started
    observables = [row for row in full_rows if not row["unobservable"]]
    unobservable = [row for row in full_rows if row["unobservable"]]
    report_full = ev.report(observables, label="base:val_full")
    report_unobservable = ev.report(unobservable, label="base:val_unobservable")
    (OUT_DIR / "preds_base_val_full.jsonl").write_text(
        "\n".join(json.dumps(row) for row in full_rows), encoding="utf-8")

    from llm.benchmark.label_scorer import candidates as scorer_candidates

    outcome = {
        "candidate_count": len(scorer_candidates()),
        "val300": report300,
        "val_full": report_full,
        "unobservable": report_unobservable,
        "seconds_val300": round(seconds300, 1),
        "seconds_val_full": round(seconds_full, 1),
        "seconds_per_item": round(seconds300 / max(len(sample), 1), 3),
    }
    (OUT_DIR / "base_reference.json").write_text(json.dumps(outcome, indent=2, default=str),
                                                 encoding="utf-8")
    print(json.dumps({key: outcome[key] for key in
                      ("seconds_val300", "seconds_val_full", "seconds_per_item")}, indent=2))
    print(json.dumps({"val300": _metric_row(report300),
                      "val_full": _metric_row(report_full),
                      "abstention_full": report_full["abstention"]}, indent=2))
    return 0


# ---------------------------------------------------------------------------
MONTHS = re.compile(r"\b(?:definitely|certainly|undoubtedly|absolutely)\b", re.I)


def asks_schema(text: str) -> Dict:
    """Validate one generated answer against the slim schema."""
    body = (text or "").strip()
    parsed, error = None, None
    match = re.search(r"\{.*\}", body, re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except ValueError as exc:
            error = f"invalid_json: {exc}"
    elif body:
        error = "no_json_object"
    else:
        error = "empty"
    if error:
        return {"schema_valid": False, "error": error, "label": None}
    missing = [key for key in SLIM_KEYS if key not in parsed]
    if missing:
        return {"schema_valid": False, "error": f"missing:{','.join(missing)}", "label": None}
    from llm.benchmark.label_scorer import candidates

    menu = set(candidates())
    label = str(parsed.get("fault_id") or "")
    enum_ok = label in menu
    ranked = parsed.get("ranked_hypotheses") or []
    ranked_ok = all(str(entry) in menu for entry in ranked) if isinstance(ranked, list) else False
    absolute = bool(MONTHS.search(json.dumps(parsed)))
    return {"schema_valid": True, "error": None, "label": label,
            "enum_valid": enum_ok and ranked_ok, "absolute_wording": absolute,
            "prose": " ".join(str(parsed.get(key) or "") for key in
                              ("physical_reasoning", "narrative"))}


def bare(model):
    """The chat model under a PEFT wrapper.

    ``decode_kv`` drives the vision-aware forward itself (pixel_values, image_flags,
    past_key_values), and ``PeftModel.forward`` does not take those arguments, so a
    wrapped model has to be handed over unwrapped. The LoRA layers are injected into
    the base model's own modules, so calling the base still applies the adapter;
    the wrapper is kept only to load and select it.
    """
    if hasattr(model, "peft_config"):
        return model.base_model.model
    return model


def generate_one(model, tokenizer, prompt: str, cap: int, dummy: int) -> Dict:
    from llm.benchmark.decode_bench import decode_kv
    from llm.benchmark.label_scorer import prompt_for

    started = time.time()
    tokens = decode_kv(model, tokenizer, prompt_for(prompt), cap, dummy)
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return {"text": text, "tokens": len(tokens), "seconds": round(time.time() - started, 2)}


def cmd_generate(args) -> int:
    from peft import PeftModel

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = json.loads((OUT_DIR / "selected_checkpoint.json").read_text(encoding="utf-8"))
    adapter = selected["selected"]["path"]
    model, tokenizer, config, _applied = load_base(Path(args.config))
    dummy = int(config["llm"].get("dummy_image_size", 448))
    wrapped = PeftModel.from_pretrained(model, str(adapter_root(Path(adapter))),
                                        adapter_name="gen")
    wrapped.set_adapter("gen")

    observable = [item for item in load_jsonl(VAL_ITEMS) if not item.get("unobservable")]
    sample = stratify(observable, args.items, args.seed + 1)

    def run(cap: int) -> Dict:
        rows = []
        for item in sample:
            produced = generate_one(bare(wrapped), tokenizer, item["prompt"], cap, dummy)
            checked = asks_schema(produced["text"])
            violations = ev.grounding_violations(checked.get("prose") or "",
                                                 prompt_tags(item["prompt"]),
                                                 prompt_numbers(item["prompt"]))
            rows.append({"item_id": item["item_id"], "gold": gold_of(item),
                         "cap": cap, "tokens": produced["tokens"],
                         "seconds": produced["seconds"], **checked,
                         "grounding_violations": violations,
                         "complete": checked["schema_valid"]})
        return {"cap": cap, "rows": rows}

    first = run(args.cap)
    rows = first["rows"]
    completion = sum(1 for row in rows if row["complete"]) / max(len(rows), 1)
    outcome = {
        "selected_adapter": adapter,
        "cap_used": args.cap,
        "items": len(rows),
        "completion_rate": round(completion, 4),
        "schema_valid": sum(1 for row in rows if row["schema_valid"]),
        "enum_valid": sum(1 for row in rows if row.get("enum_valid")),
        "absolute_wording": sum(1 for row in rows if row.get("absolute_wording")),
        "grounded_clean": sum(1 for row in rows if not row["grounding_violations"]),
        "seconds_per_item": round(sum(row["seconds"] for row in rows) / max(len(rows), 1), 2),
        "tokens_mean": round(sum(row["tokens"] for row in rows) / max(len(rows), 1), 1),
        "errors": dict(Counter(row.get("error") for row in rows if row.get("error"))),
        "rows": rows,
    }
    if completion < 0.95:
        second = run(args.cap_768)
        rows2 = second["rows"]
        completion2 = sum(1 for row in rows2 if row["complete"]) / max(len(rows2), 1)
        outcome["cap_768"] = {
            "completion_rate": round(completion2, 4),
            "seconds_per_item": round(sum(row["seconds"] for row in rows2) / max(len(rows2), 1), 2),
            "schema_valid": sum(1 for row in rows2 if row["schema_valid"]),
            "enum_valid": sum(1 for row in rows2 if row.get("enum_valid")),
            "grounded_clean": sum(1 for row in rows2 if not row["grounding_violations"]),
            "tokens_mean": round(sum(row["tokens"] for row in rows2) / max(len(rows2), 1), 1),
            "errors": dict(Counter(row.get("error") for row in rows2 if row.get("error"))),
        }
        (OUT_DIR / f"preds_gen_768.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows2), encoding="utf-8")
        outcome["rows"] = rows  # the 512 rows stay the reported set

    # Agreement with the label-scoring path on the same items.
    preds = OUT_DIR / f"preds_val_full_{selected['selected']['step']}.jsonl"
    if preds.exists():
        scored = {row["item_id"]: row["pred"] for row in load_jsonl(preds)}
        gold = {row["item_id"]: row["gold"] for row in rows}
        pairs = [(row["item_id"], row.get("label"), scored.get(row["item_id"])) for row in rows]
        usable = [pair for pair in pairs if pair[1]]
        outcome["generation_vs_scoring"] = {
            "items": len(usable),
            "agree": sum(1 for _id, generated, model_label in usable if generated == model_label),
            "generated_accuracy": round(
                sum(1 for item_id, generated, _model in usable if generated == gold.get(item_id))
                / max(len(usable), 1), 4),
        }
    (OUT_DIR / "generation_512.json").write_text(json.dumps(outcome, indent=2, default=str),
                                                 encoding="utf-8")
    summary = {key: outcome[key] for key in
               ("cap_used", "items", "completion_rate", "schema_valid", "enum_valid",
                "grounded_clean", "seconds_per_item", "tokens_mean")}
    print(json.dumps(summary, indent=2))
    if "cap_768" in outcome:
        print(json.dumps({"cap_768": outcome["cap_768"]}, indent=2))
    return 0


def cmd_speed(args) -> int:
    """Cached decode speed: adapter-on-base versus the same adapter merged."""
    from llm.benchmark.decode_bench import build_model, decode_kv
    from llm.benchmark.label_scorer import prompt_for

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = json.loads((OUT_DIR / "selected_checkpoint.json").read_text(encoding="utf-8"))
    adapter = selected["selected"]["path"]
    observable = [item for item in load_jsonl(VAL_ITEMS) if not item.get("unobservable")]
    sample = stratify(observable, args.items, args.seed + 2)
    results = {}
    for label, merged in (("adapter", False), ("merged", True)):
        model, tokenizer, config, applied = build_model(Path(args.config), adapter, merged=merged)
        model = bare(model)
        dummy = int(config["llm"].get("dummy_image_size", 448))
        started = time.time()
        produced = 0
        for item in sample:
            produced += len(decode_kv(model, tokenizer, prompt_for(item["prompt"]),
                                      args.cap, dummy))
        seconds = time.time() - started
        results[label] = {"applied": applied, "items": len(sample), "cap": args.cap,
                          "seconds": round(seconds, 2),
                          "seconds_per_item": round(seconds / len(sample), 2),
                          "tokens_per_second": round(produced / seconds, 2) if seconds else None}
        del model
    (OUT_DIR / "decode_speed.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


# ---------------------------------------------------------------------------
CONTROL_TEMPLATES = (
    "What does {tag} read in this incident, and how far is that from its baseline?",
    "Is {tag} moving away from its baseline, and in which direction?",
    "How does {tag} compare with its normal value here?",
    "Which reading of {tag} supports the diagnosis, and what is it?",
)
OUT_OF_SCOPE_CASES = ("out_of_scope", "false_premise")


def followup_questions(args) -> List[Dict]:
    """40 stratified validation questions plus 20 in-scope controls.

    The controls are the same kind of question in the same voice, but about a
    sensor the event actually holds: without them a refusal rate measures how often
    the model declines, not whether it declines the right things.
    """
    from llm.benchmark.to_sft_followup import question_for

    dialogues = load_jsonl(FOLLOWUP_VAL)
    by_case: Dict[str, List[Dict]] = defaultdict(list)
    for dialogue in dialogues:
        by_case[str(dialogue.get("case_type"))].append(dialogue)
    rng = random.Random(args.seed)
    picked: List[Dict] = []
    per_case = int(getattr(args, "per_case", 0) or 0)
    if not per_case:
        per_case = max(1, args.items // max(len(by_case), 1)) if by_case else 0
    for case in sorted(by_case):
        bucket = sorted(by_case[case], key=lambda row: str(row.get("dialogue_id")))
        rng.shuffle(bucket)
        for dialogue in bucket[:per_case]:
            picked.append({"kind": case, "dialogue_id": dialogue.get("dialogue_id"),
                           "question": question_for(dialogue, 0), "in_scope": None})
    # A case can be worth more than its flat share: a re-ask is where the model
    # either adds detail or starts declining, so the re-ask bucket is topped up on
    # top of the per-case floor rather than being fixed by it.
    repeats = int(getattr(args, "repeats", 0) or 0)
    if repeats:
        have = sum(1 for row in picked if row["kind"] == "repeated_question")
        bucket = sorted(by_case.get("repeated_question", []),
                        key=lambda row: str(row.get("dialogue_id")))
        rng.shuffle(bucket)
        used = {row["dialogue_id"] for row in picked}
        for dialogue in bucket:
            if have >= repeats:
                break
            if dialogue.get("dialogue_id") in used:
                continue
            picked.append({"kind": "repeated_question",
                           "dialogue_id": dialogue.get("dialogue_id"),
                           "question": question_for(dialogue, 0), "in_scope": None})
            have += 1
    controls = args.controls
    control_rows = []
    for dialogue in dialogues:
        context = str(dialogue.get("context") or "")
        tags = prompt_tags(context)
        if len(tags) < 2:
            continue
        tag = tags[len(control_rows) % len(tags)]
        template = CONTROL_TEMPLATES[len(control_rows) % len(CONTROL_TEMPLATES)]
        control_rows.append({
            "kind": "in_scope_control",
            "dialogue_id": dialogue.get("dialogue_id"),
            "tag": tag,
            "question": "INCIDENT CONTEXT\n" + context + "\n\nQUESTION\n"
                        + template.format(tag=tag),
            "in_scope": True,
        })
        if len(control_rows) >= controls:
            break
    return picked + control_rows


def cmd_followup(args) -> int:
    from peft import PeftModel

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = json.loads((OUT_DIR / "selected_checkpoint.json").read_text(encoding="utf-8"))
    model, tokenizer, config, _applied = load_base(Path(args.config))
    dummy = int(config["llm"].get("dummy_image_size", 448))
    wrapped = PeftModel.from_pretrained(
        model, str(adapter_root(Path(selected["selected"]["path"]))), adapter_name="fu")
    wrapped.set_adapter("fu")
    from llm.benchmark.decode_bench import decode_kv
    from llm.dataset import build_chat_prompt

    questions = followup_questions(args)
    rows = []
    for entry in questions:
        prompt = build_chat_prompt(SYSTEM_PROMPT_FOLLOWUP, [("user", entry["question"])])
        started = time.time()
        tokens = decode_kv(bare(wrapped), tokenizer, prompt, args.cap, dummy)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        seconds = round(time.time() - started, 2)
        body = text
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                body = str(json.loads(match.group(0)).get("answer") or text)
            except ValueError:
                body = text
        refused = declines(body)
        violations = ev.grounding_violations(body, prompt_tags(entry["question"]),
                                            prompt_numbers(entry["question"]))
        rows.append({"kind": entry["kind"], "dialogue_id": entry["dialogue_id"],
                     "question": entry["question"], "answer": body,
                     "refused": refused, "grounding_violations": violations,
                     "repeats_question": overlap_ratio(body, entry["question"]) >= 0.5,
                     "seconds": seconds, "tokens": len(tokens)})

    def rate(subset, predicate):
        return round(sum(1 for row in subset if predicate(row)) / max(len(subset), 1), 4)

    refusal_cases = [row for row in rows if row["kind"] in OUT_OF_SCOPE_CASES]
    controls = [row for row in rows if row["kind"] == "in_scope_control"]
    in_scope_others = [row for row in rows
                       if row["kind"] not in OUT_OF_SCOPE_CASES
                       and row["kind"] != "in_scope_control"]
    outcome = {
        "items": len(rows),
        "by_kind": dict(Counter(row["kind"] for row in rows)),
        "refusal_rate_on_out_of_scope_and_false_premise": rate(refusal_cases,
                                                              lambda row: row["refused"]),
        "over_refusal_rate_on_controls": rate(controls, lambda row: row["refused"]),
        "refusal_rate_on_other_cases": rate(in_scope_others, lambda row: row["refused"]),
        "grounded_clean_rate": rate(rows, lambda row: not row["grounding_violations"]),
        "repetition_rate": rate(rows, lambda row: row["repeats_question"]),
        "seconds_per_item": round(sum(row["seconds"] for row in rows) / max(len(rows), 1), 2),
        "rows": rows,
    }
    (OUT_DIR / str(getattr(args, "out", "followup_sanity.json"))).write_text(
        json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in outcome.items() if key != "rows"}, indent=2))
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--adapter", default=str(ADAPTER_DIR))
    parser.add_argument("--seed", type=int, default=20261031)
    sub = parser.add_subparsers(dest="command", required=True)

    parity = sub.add_parser("parity")
    parity.add_argument("--items", type=int, default=40)
    parity.add_argument("--batch", type=int, default=16)
    parity.set_defaults(func=cmd_parity)

    check = sub.add_parser("checkpoints")
    check.add_argument("--per-checkpoint", type=int, default=300)
    check.add_argument("--batch", type=int, default=16)
    check.add_argument("--via", default="wrapper", choices=("wrapper", "language_model"),
                       help="which scoring path to use; language_model only after the "
                            "parity test passes, because it is several times faster")
    check.set_defaults(func=cmd_checkpoints)

    gen = sub.add_parser("generate")
    gen.add_argument("--items", type=int, default=50)
    gen.add_argument("--cap", type=int, default=512)
    gen.add_argument("--cap-768", dest="cap_768", type=int, default=768)
    gen.set_defaults(func=cmd_generate)

    speed = sub.add_parser("speed")
    speed.add_argument("--items", type=int, default=10)
    speed.add_argument("--cap", type=int, default=128)
    speed.set_defaults(func=cmd_speed)

    follow = sub.add_parser("followup")
    follow.add_argument("--items", type=int, default=40)
    follow.add_argument("--controls", type=int, default=20)
    follow.add_argument("--per-case", dest="per_case", type=int, default=0,
                       help="floor per case type; 0 keeps items/cases")
    follow.add_argument("--repeats", type=int, default=0,
                        help="top up repeated_question to this many turns")
    follow.add_argument("--out", default="followup_sanity.json")
    follow.add_argument("--cap", type=int, default=256)
    follow.set_defaults(func=cmd_followup)

    base = sub.add_parser("base")
    base.add_argument("--per-checkpoint", dest="per_checkpoint", type=int, default=300)
    base.add_argument("--batch", type=int, default=16)
    base.add_argument("--via", default="language_model", choices=("wrapper", "language_model"))
    base.set_defaults(func=cmd_base)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
