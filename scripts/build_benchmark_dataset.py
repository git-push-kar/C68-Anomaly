#!/usr/bin/env python
"""Assemble the response-young benchmark dataset from built events.

Input: the event records produced by ``build_benchmark_events.py`` (one closed
event per run, under the frozen v2 operating point) plus the oracle records for
detector misses and the NORMAL_TRANSIENT supply.

Output, per split, a JSONL file where each line is one item:

    item_id, event_id, source_file, fault, simulationRun, split, seed,
    variant (menu | removed), prompt, target, prose (teacher output or null),
    teacher metadata (model, seconds, rejection codes)

Design rules that this script is responsible for:

* **the label is never the teacher's.** ``target`` comes from
  ``llm/benchmark/targets.py``; the teacher supplies two strings that are
  rejected if they disagree with it.
* **every generated prompt is checked**, not a sample: ``assert_prompt_clean``
  runs on each one and would fail the build on a subsystem hint or the word
  "severity".
* **open-set episodes are a stable property of the event id**, so the same event
  is an open-set episode or not regardless of which Protocol B fold selects it.
* **evaluation splits carry no teacher prose.** Val/test items are scored against
  the deterministic target, so generating reference prose for them would only
  add cost and a second place for a label to disagree.

Usage:
    python scripts/build_benchmark_dataset.py --pilot 10          # ~200 train items
    python scripts/build_benchmark_dataset.py                     # full, no teacher
    python scripts/build_benchmark_dataset.py --teacher           # full, with teacher
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.benchmark.filters import TEACHER_ANSWER_KEYS, check_answer  # noqa: E402
from llm.benchmark.menu import CLASSIFIABLE_FAULTS, UNCLASSIFIED, build_menu  # noqa: E402
from llm.benchmark.prompt_v2 import (  # noqa: E402
    SYSTEM_PROMPT_V2,
    assert_prompt_clean,
    render_prompt,
    teacher_instruction,
)
from llm.benchmark.targets import (  # noqa: E402
    NORMAL_ID,
    build_target,
    eligible_for_open_set,
    open_set_assignment,
)
from llm.response.allowed_set import build_allowed_set  # noqa: E402
from scripts.fault_ids import SYNTHETIC_FAULTS  # noqa: E402

logger = logging.getLogger("build_dataset")

OUT = "outputs/benchmark_v2/dataset"
EVENTS = "outputs/benchmark_v2/events"
SPLITS = Path("data/splits_v2.json")
LABEL_SPACE = "configs/detector_v2_operating_point.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_records(events: Path) -> Dict[str, List[Dict]]:
    """Every event record on disk, grouped by split, oracles kept separate."""
    out: Dict[str, List[Dict]] = defaultdict(list)
    for path in sorted((events / "faulty").glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if path.stem.endswith("_oracle"):
            record["oracle"] = True
        out[record.get("split") or "train"].append(record)
    for path in sorted((events / "normal").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record.setdefault("label", NORMAL_ID)
        out[record.get("split") or "train"].append(record)
    return out


def record_key(record: Dict) -> str:
    fault = record.get("fault")
    run = record.get("simulationRun")
    source = record.get("source_file") or ("TEP_FaultFree_Testing.csv" if fault is None else "")
    return f"{source}|{fault if fault is not None else 'normal'}|{run}"


def variant_for(record: Dict, split: str) -> str:
    """Which menu an item is shown.

    Train: a stable 12% of classifiable episodes lose their entry (open-set
    training). Val/test: the plain menu; the removed-variant copy is emitted
    alongside for every classifiable item so open-set scoring always has a pair.
    """
    if record.get("fault") is None:
        return "menu"
    if split == "train" and eligible_for_open_set(record, split) and \
            open_set_assignment(str(record.get("event_id"))):
        return "removed"
    return "menu"


def evidence_of(record: Dict) -> Dict:
    """The event's evidence block.

    ``evidence.event_builder.build_event`` returns an object whose ``to_dict``
    nests the prompt-facing material under ``event.evidence``; the flat form is
    accepted too so a hand-built record still works.
    """
    event = record.get("event") or {}
    nested = event.get("evidence")
    return nested if isinstance(nested, dict) else event


def make_item(record: Dict, split: str, variant: str) -> Dict:
    evidence = evidence_of(record)
    target = build_target(record, variant=variant, split=split)
    menu = build_menu(remove=target.get("menu_removed") or [])
    prompt = render_prompt(evidence, menu, instruction=teacher_instruction(
        [entry["fault_id"] for entry in menu]))
    assert_prompt_clean(prompt, where=f"prompt[{record.get('event_id')}]")
    item = {
        "item_id": f"{record.get('event_id')}::{'openset' if variant == 'removed' else 'menu'}",
        "event_id": record.get("event_id"),
        "source_file": record.get("source_file"),
        "fault": record.get("fault"),
        "simulationRun": record.get("simulationRun"),
        "split": split,
        "seed": record.get("seed", 42),
        "oracle": bool(record.get("oracle")),
        "confirmed_false_alarm": record.get("confirmed_false_alarm"),
        "label": record.get("label") or ("NORMAL_TRANSIENT" if record.get("fault") is None else None),
        "variant": variant,
        "menu_ids": [entry["fault_id"] for entry in menu],
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "target": target,
        "prose": None,
        "teacher": None,
    }
    return item


def teacher_decision(record: Dict, item: Dict, teacher, allowed_cache: Dict[str, object]) -> None:
    """Call the teacher for one train item and keep it only if it passes."""
    evidence = evidence_of(record)
    menu = [{"fault_id": fid, "signature": ""} for fid in item["menu_ids"]]
    target = item["target"]
    answer, response, parse_error = teacher.answer(evidence, build_menu(remove=target.get("menu_removed") or []), target)
    key = item["item_id"]
    allowed = allowed_cache.get(key)
    if allowed is None:
        allowed = build_allowed_set(evidence, extra_faults=[record["fault"]] if record.get("fault") else [])
    codes = check_answer(answer, target, evidence, allowed, parse_error=parse_error,
                         prompt=item.get("prompt"))
    item["teacher"] = {
        "model": response.model,
        "seconds": round(response.seconds, 3),
        "prompt_chars": response.prompt_chars,
        "raw_chars": len(response.text or ""),
        "codes": codes,
        "accepted": not codes,
        # Kept verbatim so a filter change can be replayed against the same
        # answers instead of re-running the model and getting different text.
        "raw_text": (response.text or "")[:4000],
        "temperature": teacher.temperature,
        "seed": teacher.seed,
    }
    if not codes and isinstance(answer, dict):
        item["prose"] = {key_name: str(answer.get(key_name)).strip() for key_name in TEACHER_ANSWER_KEYS}


def _family(item: Dict) -> str:
    fault = item.get("fault")
    if fault is None:
        return "normal"
    if item.get("variant") == "removed":
        return "open_set"
    return f"fault_{int(fault):02d}"


def stratified_train_order(items: List[Dict], limit: Optional[int]) -> List[Dict]:
    """Train items in an order that round-robins across families.

    With no budget the order is unchanged (so a full run is reproducible in file
    order); with a budget the first ``limit`` items are one per family at a time.
    """
    if limit is None:
        return list(items)
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for item in items:
        buckets[_family(item)].append(item)
    order: List[Dict] = []
    keys = sorted(buckets)
    index = 0
    while len(order) < len(items):
        added = False
        for key in keys:
            bucket = buckets[key]
            if index < len(bucket):
                order.append(bucket[index])
                added = True
        if not added:
            break
        index += 1
    return order


def _expand_component(block: Dict) -> set:
    """Run keys of one component, expanded from the file's (fault, run-range) form."""
    source = Path(str(block.get("source_file") or "")).name
    keys = set()
    for entry in block.get("faults") or []:
        number = int(entry["faultNumber"])
        low, high = (int(x) for x in entry["simulationRunRange"])
        for run in range(low, high + 1):
            keys.add(f"{source}|{number}|{run}")
    return keys


def _check_components(label: str, components: Dict[str, Dict], built: set,
                      held_out: Optional[List[int]] = None) -> Dict:
    expanded = {name: _expand_component(block) for name, block in components.items()}
    names = sorted(expanded)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = expanded[left] & expanded[right]
            if overlap:
                raise SystemExit(
                    f"SPLIT OVERLAP in {label}: {left} and {right} share "
                    f"{len(overlap)} run keys, e.g. {sorted(overlap)[:3]}")
    result = {
        "components": {name: len(keys) for name, keys in expanded.items()},
        "pairwise_disjoint": True,
        "built_keys_matched": sum(len(keys & built) for keys in expanded.values()),
        "built_keys_total": len(built),
    }
    if held_out:
        seen_components = [n for n in ("train", "val", "test_seen") if n in expanded]
        clash = set()
        for name in seen_components:
            clash |= {int(key.split("|")[1]) for key in expanded[name]
                      if key.split("|")[1].isdigit()} & {int(f) for f in held_out}
        if clash:
            raise SystemExit(f"HELD-OUT LEAK in {label}: faults {sorted(clash)} in a seen component")
        result["held_out_faults"] = sorted(int(f) for f in held_out)
        result["held_out_absent_from_seen"] = not clash
    return result


def verify_split_integrity(items: Dict[str, List[Dict]], splits: Dict) -> Dict[str, Dict]:
    """Prove no run key lands in two components of the same protocol or fold.

    The split file is the authority and stores run *ranges*; they are expanded
    here and compared pairwise. An overlap raises instead of being reported,
    because a leaking split invalidates every number computed from it.
    """
    built = {record_key_of(item) for split_items in items.values() for item in split_items}
    report: Dict[str, Dict] = {}
    for protocol, block in (splits.get("protocols") or {}).items():
        block = block or {}
        if isinstance(block.get("splits"), dict):
            report[protocol] = _check_components(protocol, block["splits"], built)
        for fold_name, fold in (block.get("folds") or {}).items():
            report[f"{protocol}/{fold_name}"] = _check_components(
                f"{protocol}/{fold_name}", fold.get("splits") or {}, built,
                held_out=list(fold.get("held_out_faults") or []))
    return report


def record_key_of(item: Dict) -> str:
    fault = item.get("fault")
    source = item.get("source_file") or ("TEP_FaultFree_Testing.csv" if fault is None else "")
    return f"{source}|{fault if fault is not None else 'normal'}|{item.get('simulationRun')}"


def split_keys(splits: Dict, protocol: str, component: str) -> List[str]:
    block = (splits.get("protocols") or {}).get(protocol) or {}
    return list(block.get(component) or [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default=EVENTS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pilot", type=int, default=None,
                    help="max train items per fault (pilot mode)")
    ap.add_argument("--teacher", action="store_true", help="call the Ollama teacher")
    ap.add_argument("--model", default="qwen2.5:32b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--teacher-limit", type=int, default=None,
                    help="stop after N teacher calls (pilot budget guard)")
    ap.add_argument("--audit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)-8s %(message)s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(Path(args.events))
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))

    teacher = None
    if args.teacher:
        from llm.benchmark.teacher import OllamaTeacher, resolve_model

        model = resolve_model(args.model, args.host)
        teacher = OllamaTeacher(model=model, host=args.host)
        logger.info("teacher: %s", model)

    items: Dict[str, List[Dict]] = defaultdict(list)
    rejected: List[Dict] = []
    coverage: Dict[str, Counter] = defaultdict(Counter)
    teacher_calls = 0
    teacher_seconds = 0.0
    started = time.time()

    for split in ("train", "val", "test"):
        per_fault: Counter = Counter()
        for record in records.get(split, []):
            fault = record.get("fault")
            # Oracle windows exist because the detector missed the run. They are
            # only used for training: an evaluation item should be an event the
            # system actually produced.
            if record.get("oracle") and split != "train":
                coverage[split]["oracle_skipped"] += 1
                continue
            if args.pilot is not None and split == "train" and fault is not None:
                if per_fault[fault] >= args.pilot:
                    continue
                per_fault[fault] += 1
            variant = variant_for(record, split)
            item = make_item(record, split, variant)
            items[split].append(item)
            coverage[split]["items"] += 1
            if variant == "removed":
                coverage[split]["open_set"] += 1
            if item["oracle"]:
                coverage[split]["oracle"] += 1
            if item["label"] == NORMAL_ID:
                coverage[split]["normal"] += 1
            # Evaluation items also get the removed-menu copy, so an open-set
            # score can always be paired with the closed-set one.
            if split != "train" and fault is not None and variant == "menu":
                twin = make_item(record, split, "removed")
                items[split].append(twin)
                coverage[split]["items"] += 1
                coverage[split]["open_set"] += 1

    if teacher is not None:
        allowed_cache: Dict[str, object] = {}
        by_event = {r.get("event_id"): r for r in records["train"]}
        # A budgeted pilot must not measure the rejection rate on whichever fault
        # happens to come first in file order. Round-robin across families (each
        # fault, the open-set episodes, the NORMAL_TRANSIENT items) so the rate is
        # measured on a representative mix.
        ordered = stratified_train_order(items["train"], args.teacher_limit)
        for item in ordered:
            if args.teacher_limit is not None and teacher_calls >= args.teacher_limit:
                logger.info("teacher budget reached (%d calls)", teacher_calls)
                break
            record = by_event.get(item["event_id"])
            if record is None:
                continue
            teacher_decision(record, item, teacher, allowed_cache)
            teacher_calls += 1
            teacher_seconds += float((item.get("teacher") or {}).get("seconds") or 0.0)
            if not (item.get("teacher") or {}).get("accepted"):
                rejected.append({"item_id": item["item_id"],
                                 "codes": (item.get("teacher") or {}).get("codes")})
            if teacher_calls % 10 == 0:
                logger.info("teacher calls %d, elapsed %.0fs, rejected %d",
                            teacher_calls, time.time() - started, len(rejected))

    # ---- write JSONL ------------------------------------------------------
    hashes: Dict[str, str] = {}
    counts: Dict[str, Dict] = {}
    for split, rows in items.items():
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        hashes[path.name] = sha256_file(path)
        counts[split] = dict(coverage[split])
        counts[split]["teacher_accepted"] = sum(
            1 for row in rows if (row.get("teacher") or {}).get("accepted"))
        counts[split]["teacher_rejected"] = sum(
            1 for row in rows if row.get("teacher") and not row["teacher"]["accepted"])
        logger.info("%-6s %4d items -> %s", split, len(rows), path)

    if rejected:
        with open(out / "teacher_rejections.jsonl", "w", encoding="utf-8") as handle:
            for row in rejected:
                handle.write(json.dumps(row) + "\n")

    # ---- coverage against the split file --------------------------------
    built = {split: {record_key(r) for r in records.get(split, [])} for split in
             ("train", "val", "test")}
    fidelity: Dict[str, Dict] = {}
    for protocol, block in (splits.get("protocols") or {}).items():
        for component, keys in (block or {}).items():
            if not isinstance(keys, list):
                continue
            key_set = set(keys)
            fidelity[f"{protocol}/{component}"] = {
                "keys_in_split_file": len(key_set),
                "sha256": sha256_text("\n".join(sorted(key_set))),
                "months_built": len(key_set & built.get("train", set())),
            }

    # ---- audit sheet -----------------------------------------------------
    population = [item for split in ("train", "val", "test") for item in items.get(split, [])]
    rng = random.Random(args.seed)
    families: Dict[str, List[Dict]] = defaultdict(list)
    for item in population:
        fault = item.get("fault")
        family = "normal" if fault is None else ("open_set" if int(fault) >= 16 else f"fault_{fault}")
        families[family].append(item)
    audit_rows: List[Dict] = []
    targets = max(1, args.audit)
    while len(audit_rows) < min(targets, len(population)):
        progressed = False
        for family in sorted(families):
            bucket = families[family]
            if not bucket:
                continue
            audit_rows.append(rng.choice(bucket))
            progressed = True
            if len(audit_rows) >= targets:
                break
        if not progressed:
            break
    audit_path = out / "audit_sample.csv"
    with open(audit_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "event_id", "split", "variant", "truth_fault_number",
                         "predicted_fault_placeholder", "correct", "grounded", "useful",
                         "notes"])
        for row in audit_rows:
            writer.writerow([row["item_id"], row["event_id"], row["split"], row["variant"],
                             row["target"].get("truth_fault_number"), "", "", "", "", ""])
    hashes[audit_path.name] = sha256_file(audit_path)

    # ---- manifest --------------------------------------------------------
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "events_dir": str(Path(args.events)),
        "counts": counts,
        "files_sha256": hashes,
        "operating_point_sha256": sha256_file(Path(LABEL_SPACE)),
        "kb_sha256": sha256_file(Path("scripts/fault_knowledge.py")),
        "prompt_template_sha256": sha256_text(SYSTEM_PROMPT_V2),
        "prompt_rules": {
            "forbidden_fragments_enforced": True,
            "subsystem_hint": "removed",
            "severity_rubric": "removed",
        },
        "teacher": {
            "enabled": teacher is not None,
            "model": getattr(teacher, "model", None),
            "calls": teacher_calls,
            "mean_seconds": round(teacher_seconds / teacher_calls, 3) if teacher_calls else None,
            "rejections": len(rejected),
            "rejection_rate": round(len(rejected) / teacher_calls, 4) if teacher_calls else None,
            "rejection_codes": dict(Counter(
                code.split(":")[0] for row in rejected for code in row["codes"])),
        },
        "split_file_fidelity": fidelity,
        "split_integrity": verify_split_integrity(items, splits),
        "synthetic_faults_present": sorted({int(item["fault"]) for item in population
                                            if item.get("fault") in SYNTHETIC_FAULTS}),
        "open_set_rate_train": round(
            sum(1 for item in items.get("train", []) if item["variant"] == "removed")
            / max(1, sum(1 for item in items.get("train", []) if item.get("fault") is not None)), 4),
        # The brief asks for 10-15% of *train events*; only faults 1-15 can have
        # their entry removed, so the eligible denominator is the honest one.
        "open_set_rate_train_eligible": round(
            sum(1 for item in items.get("train", []) if item["variant"] == "removed")
            / max(1, sum(1 for item in items.get("train", [])
                        if item.get("fault") in CLASSIFIABLE_FAULTS)), 4),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("manifest written: %s", out / "manifest.json")
    logger.info("counts: %s", json.dumps(counts))
    logger.info("teacher: calls=%d mean=%.2fs rejections=%d",
                teacher_calls,
                teacher_seconds / teacher_calls if teacher_calls else 0.0, len(rejected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
