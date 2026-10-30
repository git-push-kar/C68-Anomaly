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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.benchmark.filters import (  # noqa: E402
    TEACHER_ANSWER_KEYS,
    check_answer,
    direction_inconsistent,
)
from llm.benchmark.inputs import verify_inputs  # noqa: E402
from llm.benchmark.menu import CLASSIFIABLE_FAULTS, UNCLASSIFIED, build_menu  # noqa: E402
from llm.benchmark.prompt_v2 import (  # noqa: E402
    SYSTEM_PROMPT_V2,
    assert_prompt_clean,
    prompt_leaks,
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
ROOT = Path(__file__).resolve().parents[1]

#: Faults the fused detector does not reliably see (fused FDR far below 5% on
#: validation). They stay in val/test so the miss is measured, carry an explicit
#: ``unobservable`` flag, and are kept out of train and out of teacher input: a
#: model trained to name a fault the detector cannot flag is trained on evidence
#: that does not exist.
UNOBSERVABLE_FAULTS = (3, 9)

#: Faults with no menu entry: naming one is wrong by construction.
OPEN_SET_FAMILY = (16, 17, 18, 19, 20)

#: Cap on UNCLASSIFIED_OPEN_SET as a share of faulty TRAIN items.
OPEN_SET_CAP = 0.25


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
        # Carried on the item so a metric can report the unobservable subset
        # without having to re-derive it from the fault number.
        "unobservable": (record.get("fault") is not None
                         and int(record["fault"]) in UNOBSERVABLE_FAULTS),
        "variant": variant,
        "menu_ids": [entry["fault_id"] for entry in menu],
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "target": target,
        "prose": None,
        "teacher": None,
    }
    return item


#: Files whose contents decide accept/reject. Their hash is written into every
#: cached teacher record, so a filter edit cannot silently mix two regimes.
FILTER_CODE_FILES = (
    "llm/benchmark/filters.py",
    "llm/benchmark/prompt_v2.py",
    "llm/response/allowed_set.py",
    "llm/response/validators.py",
)


def filter_code_hash() -> str:
    digest = hashlib.sha256()
    for rel in FILTER_CODE_FILES:
        path = ROOT / rel
        if path.exists():
            digest.update(rel.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def load_teacher_cache(path: Path) -> Dict[str, Dict]:
    """Every finished teacher call, keyed by item id."""
    done: Dict[str, Dict] = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                # A truncated last line is what a kill mid-write looks like.
                continue
            if record.get("item_id"):
                done[record["item_id"]] = record
    return done


def rejection_feedback(codes: Sequence[str]) -> str:
    """The retry instruction: what failed, stated so it can be fixed without help."""
    return (
        "REVISION REQUIRED. Your previous answer was rejected for: "
        + "; ".join(codes)
        + ". Rewrite both fields to fix exactly those points. Keep the diagnosis "
          "unchanged, quote a number only if it is printed in the evidence above "
          "and quote it exactly, name only sensors the evidence lists, and use the "
          "disturbance wording given under DECIDED DIAGNOSIS when naming the cause."
    )


def teacher_call(item: Dict, record: Dict, teacher, provenance: Dict) -> Dict:
    """One teacher item: one call, plus one retry if the first is rejected.

    The retry is checked against the *original* prompt, never the retry prompt:
    the feedback names the offending token, and if that token counted as "shown"
    the model could quote it back and pass on the second attempt.
    """
    evidence = evidence_of(record)
    target = item["target"]
    menu = build_menu(remove=target.get("menu_removed") or [])
    allowed = build_allowed_set(evidence,
                                extra_faults=[record["fault"]] if record.get("fault") else [])
    answer, response, parse_error = teacher.answer(evidence, menu, target)
    codes = check_answer(answer, target, evidence, allowed, parse_error=parse_error,
                         prompt=item.get("prompt"))
    attempts = 1
    seconds = float(response.seconds)
    first_codes = list(codes)
    raw_text = response.text or ""
    if codes:
        feedback = rejection_feedback(codes)
        answer2, response2, parse_error2 = teacher.answer(
            evidence, menu, target, extra_instruction=feedback, seed=teacher.seed + 1
        )
        codes2 = check_answer(answer2, target, evidence, allowed, parse_error=parse_error2,
                              prompt=item.get("prompt"))
        attempts = 2
        seconds += float(response2.seconds)
        if not codes2:
            answer, codes = answer2, codes2
            raw_text = response2.text or ""
        else:
            codes = codes + [f"retry:{code}" for code in codes2]
            answer = None
    prose = None
    if not codes and isinstance(answer, dict):
        prose = {key: str(answer.get(key)).strip() for key in TEACHER_ANSWER_KEYS}
    flags = direction_inconsistent(prose, evidence, allowed) if prose else []
    return {
        "item_id": item["item_id"],
        "event_id": item["event_id"],
        "family": _family(item),
        "model": response.model,
        "seconds": round(seconds, 3),
        "attempts": attempts,
        "first_codes": first_codes,
        "prompt_chars": response.prompt_chars,
        "raw_chars": len(raw_text),
        "codes": codes,
        "accepted": not codes,
        # A twice-rejected item is dropped from the dataset rather than kept as an
        # accepted-looking row with bad prose.
        "dropped": bool(codes),
        "flags": flags,
        "prose": prose,
        # Kept verbatim so a filter change can be replayed against the same
        # answers instead of re-running the model and getting different text.
        "raw_text": raw_text[:4000],
        **provenance,
    }


def run_teacher(items: List[Dict], records: Dict[str, Dict], teacher, cache_path: Path,
                provenance: Dict, limit: Optional[int], concurrency: int,
                item_ids: Optional[set] = None, max_calls: Optional[int] = None) -> Dict:
    """Call the teacher for every pending item, appending each result immediately.

    The cache is the durable record: one JSONL line per finished call, written as
    soon as the call returns. A run that dies at item 1,900 restarts at item 1,901
    instead of losing five hours of work.
    """
    done = load_teacher_cache(cache_path)
    ordered = stratified_train_order(items, limit)
    if limit is not None:
        # A budget is a cap on *calls*, not on the order they are taken in. The
        # order interleaves families so a partial run is representative.
        ordered = ordered[:limit]
    if item_ids is not None:
        ordered = [item for item in ordered if item["item_id"] in item_ids]
    pending = [item for item in ordered
               if item["item_id"] not in done
               or done[item["item_id"]].get("filter_sha256") != provenance["filter_sha256"]]
    stats = {"cached": len(done), "pending": len(pending), "calls": 0, "api_calls": 0,
             "errors": 0, "rejected": 0, "retried": 0, "dropped": 0, "seconds": 0.0,
             "max_calls": max_calls}
    if not pending:
        return stats
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as handle:
        def work(item: Dict) -> Optional[Dict]:
            record = records.get(item["event_id"])
            if record is None:
                return None
            return teacher_call(item, record, teacher, provenance)

        if concurrency <= 1:
            results = map(work, pending)
            pool = None
        else:
            pool = ThreadPoolExecutor(max_workers=concurrency)
            results = pool.map(work, pending)
        try:
            for result in results:
                if result is None:
                    stats["errors"] += 1
                    continue
                attempts = int(result.get("attempts") or 1)
                if max_calls is not None and stats["api_calls"] + attempts > max_calls:
                    logger.info("teacher call budget reached (%d api calls)", stats["api_calls"])
                    break
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                stats["api_calls"] += attempts
                stats["calls"] += 1
                stats["rejected"] += 0 if result.get("accepted") else 1
                stats["retried"] += 1 if int(result.get("attempts") or 1) > 1 else 0
                stats["dropped"] += 1 if result.get("dropped") else 0
                stats["seconds"] += float(result.get("seconds") or 0.0)
                if stats["calls"] % 10 == 0:
                    logger.info("teacher %d/%d calls, rejected %d, mean %.1fs",
                                stats["calls"], len(pending), stats["rejected"],
                                stats["seconds"] / max(stats["calls"], 1))
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
    return stats


def attach_cached_prose(items: List[Dict], cache: Dict[str, Dict]) -> None:
    """Publish cached verdicts onto the items that produced them."""
    for item in items:
        entry = cache.get(item["item_id"])
        if not entry:
            continue
        item["teacher"] = {
            "model": entry.get("model"),
            "seconds": entry.get("seconds"),
            "attempts": entry.get("attempts"),
            "first_codes": entry.get("first_codes") or [],
            "prompt_chars": entry.get("prompt_chars"),
            "raw_chars": entry.get("raw_chars"),
            "codes": entry.get("codes") or [],
            "accepted": bool(entry.get("accepted")),
            "dropped": bool(entry.get("dropped")),
            # Flags are diagnostics, not verdicts: they never remove an item.
            "flags": entry.get("flags") or [],
            "raw_text": entry.get("raw_text"),
            "temperature": entry.get("temperature"),
            "seed": entry.get("seed"),
            "filter_sha256": entry.get("filter_sha256"),
        }
        if entry.get("accepted") and entry.get("prose"):
            item["prose"] = dict(entry["prose"])


def fault_coverage(train_items: List[Dict], floor: float = 0.70) -> Dict:
    """Accepted teacher prose per classifiable fault, with a floor.

    A fault whose items mostly failed the filters trains nothing: it contributes
    rows without prose and disappears from any prose-conditioned metric. The floor
    makes that visible in the manifest instead of leaving it to be noticed later.
    """
    per: Dict[str, Dict] = {}
    for item in train_items:
        fault = item.get("fault")
        if fault is None:
            continue
        key = f"FAULT_{int(fault):02d}"
        row = per.setdefault(key, {"items": 0, "with_prose": 0})
        row["items"] += 1
        if item.get("prose"):
            row["with_prose"] += 1
    for row in per.values():
        row["share"] = round(row["with_prose"] / max(row["items"], 1), 3)
        row["below_floor"] = row["share"] < floor
    return {
        "floor": floor,
        "faults": per,
        "below_floor": sorted(key for key, row in per.items() if row["below_floor"]),
    }


def cap_open_set(train_items: List[Dict], seed: int) -> tuple:
    """Keep UNCLASSIFIED_OPEN_SET at or below ``OPEN_SET_CAP`` of faulty train.

    The lever is the faults 16-20 supply, not the 1-15 open-set episodes: the
    episodes are the training signal for abstention and are spread across every
    classifiable fault, while 16-20 are a bulk family that can crowd them out.
    """
    faulty = [item for item in train_items if item.get("fault") is not None]
    unclassified = [item for item in train_items
                    if item["target"].get("fault_id") == UNCLASSIFIED]
    # The share is of the faulty total *after* dropping, so the bound is
    # u <= cap * (classified) / (1 - cap), not cap * faulty.
    classified = len(faulty) - len(unclassified)
    limit = int(OPEN_SET_CAP * classified / (1.0 - OPEN_SET_CAP))
    report = {
        "faulty_train_items": len(faulty),
        "classified_train_items": classified,
        "unclassified_before": len(unclassified),
        "cap_items": limit,
        "capped": False,
        "dropped": 0,
    }
    if len(unclassified) <= limit:
        report["unclassified_after"] = len(unclassified)
        report["share_after"] = round(len(unclassified) / max(len(faulty), 1), 4)
        return train_items, report
    droppable = [item for item in unclassified
                 if item.get("fault") is not None and int(item["fault"]) in OPEN_SET_FAMILY]
    excess = len(unclassified) - limit
    if excess > len(droppable):
        excess = len(droppable)
        logger.warning("open-set cap cannot be met without dropping 1-15 episodes")

    # Equal per-fault budget, so the 16-20 family is thinned evenly instead of a
    # single shuffle that can empty one fault and leave another untouched.
    by_fault: Dict[int, List[Dict]] = defaultdict(list)
    for item in droppable:
        by_fault[int(item["fault"])].append(item)
    ordered_faults = sorted(by_fault)
    rng = random.Random(seed)
    victims: set = set()
    if excess and ordered_faults:
        quotas = {f: excess // len(ordered_faults) for f in ordered_faults}
        for fault in ordered_faults[: excess % len(ordered_faults)]:
            quotas[fault] += 1
        remaining = excess
        for fault in ordered_faults:
            bucket = sorted((i for i in by_fault[fault] if i["item_id"] not in victims),
                            key=lambda i: i["item_id"])
            take = min(quotas[fault], len(bucket))
            if take:
                victims.update(i["item_id"] for i in rng.sample(bucket, take))
                remaining -= take
        # A fault that could not fill its quota pushes the difference onto the rest.
        while remaining > 0:
            progressed = False
            for fault in ordered_faults:
                if remaining == 0:
                    break
                bucket = [i for i in by_fault[fault] if i["item_id"] not in victims]
                if not bucket:
                    continue
                victims.add(rng.choice(bucket)["item_id"])
                remaining -= 1
                progressed = True
            if not progressed:
                break

    per_fault_16_20 = {f"FAULT_{fault:02d}": {
        "items": len(by_fault.get(fault, [])),
        "kept": sum(1 for i in by_fault.get(fault, []) if i["item_id"] not in victims),
    } for fault in ordered_faults}
    kept = [item for item in train_items if item["item_id"] not in victims]
    after = len(unclassified) - len(victims)
    report.update({
        "capped": True,
        "dropped": len(victims),
        "unclassified_after": after,
        "share_after": round(after / max(classified + after, 1), 4),
        "budget_per_fault": (excess // len(ordered_faults)) if ordered_faults else 0,
        "per_fault_16_20": per_fault_16_20,
    })
    return kept, report


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
    ap.add_argument("--model", default="qwen3:30b",
                    help="teacher tag; must be installed (no nearest-match fallback)")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--teacher-limit", type=int, default=None,
                    help="stop after N teacher calls (pilot budget guard)")
    ap.add_argument("--teacher-cache", default=None,
                    help="resume file, one JSON line per finished call (default <out>/teacher_cache.jsonl)")
    ap.add_argument("--teacher-items", default=None,
                    help="file with one item_id per line; restrict the teacher to those items")
    ap.add_argument("--max-calls", type=int, default=None,
                    help="hard cap on teacher calls, retries included")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel teacher calls; Ollama serves them one at a time")
    ap.add_argument("--audit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)-8s %(message)s")

    inputs = verify_inputs(ROOT)
    logger.info("input artifacts verified: %s", ", ".join(sorted(inputs)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(Path(args.events))
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    events_summary = {}
    summary_path = Path(args.events) / "build_summary.json"
    if summary_path.exists():
        events_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    teacher = None
    license_sha = None
    if args.teacher:
        from llm.benchmark.teacher import OllamaTeacher, license_hash, resolve_model

        model = resolve_model(args.model, args.host)
        teacher = OllamaTeacher(model=model, host=args.host, seed=args.seed)
        license_sha = license_hash(model, args.host)
        logger.info("teacher: %s (license hash %s)", model, license_sha)

    items: Dict[str, List[Dict]] = defaultdict(list)
    rejected: List[Dict] = []
    coverage: Dict[str, Counter] = defaultdict(Counter)
    prompts_checked = 0
    leak_findings = 0
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
            # An unobservable fault cannot be diagnosed from an event that the
            # detector produced only by chance: keep it out of train entirely, and
            # keep it in val/test so the miss is measured.
            if split == "train" and fault is not None and int(fault) in UNOBSERVABLE_FAULTS:
                coverage[split]["unobservable_excluded"] += 1
                continue
            if args.pilot is not None and split == "train" and fault is not None:
                if per_fault[fault] >= args.pilot:
                    continue
                per_fault[fault] += 1
            variant = variant_for(record, split)
            item = make_item(record, split, variant)
            prompts_checked += 1
            leak_findings += len(prompt_leaks(item["prompt"]))
            items[split].append(item)
            coverage[split]["items"] += 1
            if variant == "removed":
                coverage[split]["open_set"] += 1
            if item["unobservable"]:
                coverage[split]["unobservable"] += 1
            if item["oracle"]:
                coverage[split]["oracle"] += 1
            if item["label"] == NORMAL_ID:
                coverage[split]["normal"] += 1
            # Evaluation items also get the removed-menu copy, so an open-set
            # score can be paired with the closed-set one -- but only when
            # removing the entry actually changes the menu. For faults 16-20 the
            # entry was never there, so the twin would be a byte-identical
            # duplicate of the item it was meant to contrast with.
            if (split != "train" and fault is not None and variant == "menu"
                    and int(fault) in CLASSIFIABLE_FAULTS):
                twin = make_item(record, split, "removed")
                items[split].append(twin)
                coverage[split]["items"] += 1
                coverage[split]["open_set"] += 1

    items["train"], class_balance = cap_open_set(items["train"], args.seed)
    if class_balance["capped"]:
        logger.info("open-set cap applied: dropped %d, share now %.3f",
                    class_balance["dropped"], class_balance["share_after"])

    teacher_stats: Dict[str, object] = {}
    if teacher is not None:
        from llm.benchmark.teacher import SYSTEM_PROMPT_TEACHER

        provenance = {
            "teacher_tag": teacher.model,
            "temperature": teacher.temperature,
            "seed": teacher.seed,
            "filter_sha256": filter_code_hash(),
            "license_sha256": license_sha,
            "system_prompt_sha256": sha256_text(SYSTEM_PROMPT_TEACHER),
        }
        cache_path = (Path(args.teacher_cache) if args.teacher_cache
                      else out / "teacher_cache.jsonl")
        by_event = {r.get("event_id"): r for r in records["train"]}
        item_ids = None
        if args.teacher_items:
            item_ids = {line.strip() for line in
                        Path(args.teacher_items).read_text(encoding="utf-8").splitlines()
                        if line.strip()}
            logger.info("teacher restricted to %d requested item ids", len(item_ids))
        teacher_stats = run_teacher(items["train"], by_event, teacher, cache_path,
                                    provenance, args.teacher_limit, args.concurrency,
                                    item_ids=item_ids, max_calls=args.max_calls)
        cache = load_teacher_cache(cache_path)
        attach_cached_prose(items["train"], cache)
        for row in items["train"]:
            verdict = row.get("teacher")
            if verdict and not verdict.get("accepted"):
                rejected.append({"item_id": row["item_id"], "codes": verdict.get("codes")})
        teacher_stats.update({"cache_path": str(cache_path), **provenance,
                              "concurrency": args.concurrency})
        logger.info("teacher stats: %s", json.dumps(teacher_stats))

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
        # The coverage counter is filled while items are made, before the open-set
        # cap drops any; the file is the truth, so report both.
        counts[split]["items_before_cap"] = counts[split].get("items")
        counts[split]["items"] = len(rows)
        counts[split]["unclassified_open_set"] = sum(
            1 for row in rows if row["target"].get("fault_id") == UNCLASSIFIED)
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
            "model": teacher_stats.get("teacher_tag") or getattr(teacher, "model", None),
            "calls": teacher_stats.get("calls", 0),
            "cached_reused": teacher_stats.get("cached", 0),
            "mean_seconds": round(float(teacher_stats.get("seconds") or 0.0)
                                  / max(int(teacher_stats.get("calls") or 0), 1), 3),
            "rejections": len(rejected),
            "rejection_rate": (round(len(rejected) / teacher_stats["calls"], 4)
                               if teacher_stats.get("calls") else None),
            "retried": teacher_stats.get("retried", 0),
            "dropped_after_retry": teacher_stats.get("dropped", 0),
            "max_calls": teacher_stats.get("max_calls"),
            "rejection_codes": dict(Counter(
                code.split(":")[0] for row in rejected for code in row["codes"])),
            "provenance": {key: teacher_stats.get(key) for key in
                           ("teacher_tag", "temperature", "seed", "filter_sha256",
                            "license_sha256", "system_prompt_sha256", "concurrency",
                            "cache_path") if teacher_stats.get(key) is not None},
        },
        # The evidence view is part of the dataset's identity: a snapshot at a
        # different horizon is a different dataset, not a re-render.
        "event_view": {
            "horizon_windows": events_summary.get("horizon_windows"),
            "persistence_windows": events_summary.get("persistence_windows"),
            "onset_sample": events_summary.get("onset_sample"),
            "events_dir": str(args.events),
            "events_total": events_summary.get("events_total"),
        },
        "input_artifacts": inputs,
        "class_balance": class_balance,
        "teacher_coverage": fault_coverage(items["train"]),
        "prompt_check": {
            "prompts_checked": prompts_checked,
            "leak_violations": leak_findings,
            "rules": "no event id, split name, run number, EVT/NORM token, or fault "
                     "reference outside the menu; checked on every prompt",
        },
        "unobservable_faults": list(UNOBSERVABLE_FAULTS),
        "unobservable_items": {split: sum(1 for item in items.get(split, [])
                                          if item.get("unobservable"))
                               for split in ("train", "val", "test")},
        # Confidence is not a model field and is not in the target. It is computed
        # in code from the fault-id token probability and calibrated on validation;
        # until a calibrator is fitted the renderer says so instead of printing a
        # number that looks measured.
        "confidence_contract": {
            "in_target": False,
            "in_teacher_output": False,
            "source": "fault_id token probability at inference, isotonic-calibrated on validation",
            "calibrator_fitted": False,
            "renderer_until_calibrated": "uncalibrated (no calibrator fitted)",
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
    logger.info("teacher: calls=%d cached=%d rejections=%d",
                teacher_stats.get("calls", 0), teacher_stats.get("cached", 0), len(rejected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
