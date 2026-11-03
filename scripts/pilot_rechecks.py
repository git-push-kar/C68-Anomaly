"""Re-specified contradiction, tag-role and anchor checks for the prose pilot (PROTOCOL 28).

CPU only, read only: stored generations are re-read, nothing is generated, no model is loaded, no
teacher is called. The fallback v3.1 comparator is regenerated deterministically from the same
supplied diagnosis and measured by the same rules.
"""
from __future__ import annotations

import ast
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evidence.process_relationships import SENSOR_NAMES  # noqa: E402
from scripts.ab_sheet_checks import FIT  # noqa: E402
from scripts.adapter_checks_3b9 import (BIND_BRACK, BIND_PAREN, canonical_names,  # noqa: E402
                                        identity_tokens, variants)
from scripts.build_prose_dataset import load_jsonl  # noqa: E402
from scripts.eval_protocolA import prompt_numbers, prompt_tags  # noqa: E402
from scripts.fallback_v2 import parse_store  # noqa: E402
from scripts.fallback_v3 import load_overlay  # noqa: E402
from scripts.fallback_v31 import fallback_v31  # noqa: E402
from scripts.prose_pilot_prompts import EXPORT, build_orders, parse_prose  # noqa: E402
from scripts.prose_pilot_score import ORDERS  # noqa: E402
from scripts.recheck_cases import CASES  # noqa: E402
import llm.benchmark.evaluate_v2 as ev  # noqa: E402

HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
SENT = re.compile(r"(?<=[.!?])\s+")
FAULT = re.compile(r"FAULT_(\d{1,2})")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
CAUSE = re.compile(r"(?:\b(?:caused by|due to|the cause is|the fault is|diagnosed as|"
                   r"classified as|best fits|best fit is|responsible|attributable to)\b)|"
                   r"(?:diagnosis\s*[:=])", re.I)
IS_FAULT = re.compile(r"FAULT_(\d{1,2})\s+(?:is|was)\s+(?:the\s+)?(?:cause|fault|explanation)", re.I)
REJECT = re.compile(r"\b(?:not|never|neither|nor|contradict\w*|rules? out|ruled out|exclud\w*|"
                    r"unlike|rather than|instead of|would (?:show|produce|explain|give|cause|have)|"
                    r"if|unless|expected|should|could|alternative|candidate|differential|"
                    r"lookalike|resembl\w*)\b", re.I)
VERDICT = re.compile(r"verdict\s*(?:is|:)\s*(classified|no_fault|unclassified_open_set)", re.I)
NORMAL_CLAIM = re.compile(r"\b(this is a normal (?:operation|condition|event)|"
                          r"the process is normal|no fault (?:is )?present)\b", re.I)
FAULT_CLAIM = re.compile(r"\b(this is a (?:classified )?fault|a fault (?:is )?present|"
                         r"the fault is confirmed)\b", re.I)
CLASS_WORDS = (("temperature", "temperature"), ("pressure", "pressure"), ("flow", "flow"),
               ("rate", "flow"), ("level", "level"), ("composition", "composition"),
               ("fraction", "composition"), ("mole", "composition"), ("work", "work"),
               ("valve", "valve"), ("stream", "stream"))
EXPECTED = {"classified": "classified", "no_fault": "no_fault",
            "unclassified_open_set": "unclassified_open_set"}


def detector_key(detector: Dict) -> str:
    """The short name check_numbers uses for a detector (LSTM / CVA / DPCA)."""
    label = str((detector or {}).get("name") or "")
    lowered = label.lower()
    if "lstm" in lowered:
        return "LSTM"
    if "cva" in lowered:
        return "CVA"
    if "dpca" in lowered:
        return "DPCA"
    return label


def _tag_re(tag: str):
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(str(tag)) + r"(?![0-9])")


def top_sensor(prompt: str) -> Tuple[str, float]:
    store = parse_store(str(prompt or ""))
    sensors = sorted((store.get("sensors") or []),
                     key=lambda sensor: -abs(float(sensor.get("z") or 0.0)))
    if not sensors:
        return "", 0.0
    return str(sensors[0]["tag"]), round(float(sensors[0]["z"]), 2)


def expected_verdict(gold_fault_id: str, gold_verdict: str) -> str:
    """The verdict the supplied diagnosis implies, by the repository's own rule (v3.1's).

    An on-menu UNCLASSIFIED_OPEN_SET row carries the stored rule verdict "classified"; the
    answer's verdict is nevertheless an abstention, and both sources are read the same way.
    """
    if str(gold_fault_id).startswith("FAULT"):
        return "classified"
    if "NORMAL" in str(gold_fault_id) or str(gold_verdict) == "no_fault":
        return "no_fault"
    return "unclassified_open_set"


def contradiction_flags(prose: str, gold_fault_id: str, gold_verdict: str) -> List[Dict]:
    """A fault other than the supplied one claimed as the cause, or a different verdict (28.2)."""
    flags: List[Dict] = []
    want = expected_verdict(gold_fault_id, gold_verdict)
    for sentence in (part.strip() for part in SENT.split(str(prose or "")) if part.strip()):
        if REJECT.search(sentence):
            continue
        if CAUSE.search(sentence) or IS_FAULT.search(sentence):
            for number in FAULT.findall(sentence):
                label = "FAULT_" + str(number).zfill(2)
                if label != str(gold_fault_id):
                    flags.append({"kind": "asserts_other_fault", "detail": label,
                                  "sentence": sentence[:200]})
                    break
        claimed = VERDICT.search(sentence)
        if claimed and claimed.group(1).lower() != want:
            flags.append({"kind": "asserts_other_verdict", "detail": claimed.group(1),
                          "sentence": sentence[:200]})
            continue
        lowered = sentence.lower()
        if "listed" in lowered or "asserted" in lowered or "unclassified" in lowered:
            continue
        if NORMAL_CLAIM.search(sentence) and want != "no_fault":
            flags.append({"kind": "asserts_other_verdict", "detail": "no_fault",
                          "sentence": sentence[:200]})
        elif FAULT_CLAIM.search(sentence) and want != "classified":
            flags.append({"kind": "asserts_other_verdict", "detail": "classified",
                          "sentence": sentence[:200]})
    return flags


SUBSYSTEMS = {"reactor", "condenser", "separator", "stripper", "compressor", "purge",
              "feed", "product", "recycle", "column"}


def identity(text: str) -> set:
    """Identity words, with subsystem words kept even though they are generic on their own."""
    words = set(re.findall(r"[a-z0-9]+", str(text).lower().replace("_", " ")))
    return identity_tokens(str(text)) | (words & SUBSYSTEMS)


def kinds_of(text: str) -> set:
    lowered = str(text or "").lower().replace("_", " ")
    found = set()
    for word, kind in CLASS_WORDS:
        if re.search(r"\b" + word, lowered):
            found.add(kind)
    return found


def tag_role_flags(prose: str) -> List[Dict]:
    """A tag bound to a name whose physical class contradicts the tag table (28.3)."""
    flags: List[Dict] = []
    text = str(prose or "")
    for pattern in (BIND_PAREN, BIND_BRACK):
        for match in pattern.finditer(text):
            claimed, tag = match.group(1).strip(), match.group(2)
            canonical = SENSOR_NAMES.get(tag)
            if canonical is None:
                flags.append({"kind": "tag_unknown", "claimed": claimed, "tag": tag,
                              "canonical": ""})
                continue
            claimed_kinds = kinds_of(claimed)
            canonical_kinds = kinds_of(canonical.replace("_", " "))
            if tag.startswith("XMV"):
                # every manipulated variable is a valve position that meters a flow, whatever its
                # canonical name calls the stream
                canonical_kinds |= {"valve", "flow"}
            if claimed_kinds and canonical_kinds and not (claimed_kinds & canonical_kinds):
                flags.append({"kind": "class_mismatch", "claimed": claimed, "tag": tag,
                              "canonical": canonical,
                              "claimed_kinds": sorted(claimed_kinds),
                              "canonical_kinds": sorted(canonical_kinds)})
                continue
            claimed_tokens = identity(claimed)
            canonical_tokens = identity(canonical)
            if claimed_tokens and canonical_tokens and not (claimed_tokens & canonical_tokens):
                flags.append({"kind": "name_does_not_match_tag", "claimed": claimed, "tag": tag,
                              "canonical": canonical})
    return flags


def anchor_status(prose: str, tag: str, z: float) -> Tuple[str, str]:
    """exact / sensor_absent / z_omitted / contradicting_value, with the local window (28.4)."""
    if not tag:
        return "no_sensor", ""
    body = str(prose or "")
    match = _tag_re(tag).search(body)
    if not match:
        return "sensor_absent", ""
    if any(abs(float(value) - z) <= 0.01 + 1e-9 for value in NUMBER.findall(body)):
        return "exact", ""
    window = body[match.end():match.end() + 120]
    local = [float(value) for value in NUMBER.findall(window)]
    if any(abs(value - z) <= 0.5 for value in local):
        return "contradicting_value", " ".join(window.split())[:160]
    return "z_omitted", " ".join(window.split())[:160]


def numeric_flags(prose: str, store: Dict) -> List[Dict]:
    """Numeric failures split into direction, threshold and value kinds (28.5).

    A number is attributed to every detector named in its sentence: when one sentence quotes two
    detectors, one detector's ratio is not another's error. Direction is still checked per
    detector against its own stored value and threshold.
    """
    flags: List[Dict] = []
    detectors = store.get("detectors") or []
    for sentence in re.split(r";|(?<=[.!?])\s+", str(prose or "")):
        named = [entry for entry in detectors
                 if detector_key(entry).lower() in sentence.lower()]
        # a tag is not a quantity: XMEAS_17 and XMV_52 must not read as the numbers 17 and 52
        bare = re.sub(r"(?:XMEAS|XMV)_[0-9]{1,2}", " ", sentence)
        if not named:
            continue
        allowed = set()
        for entry in named:
            for field in ("value", "threshold", "ratio"):
                allowed |= variants(entry.get(field) or "")
        quoted = [number for number in re.findall(r"-?[0-9]+(?:[.][0-9]+)?", bare)
                  if number not in allowed]
        if quoted:
            threshold_variants = variants(named[0].get("threshold") or "")
            if "threshold" in sentence.lower() and not (set(quoted) & threshold_variants):
                kind = "threshold_misquoted"
            else:
                kind = "value_not_stored"
            flags.append({"kind": kind, "detector": detector_key(named[0]), "quoted": quoted,
                          "sentence": sentence[:220]})
        for entry in named:
            try:
                value, threshold = float(entry["value"]), float(entry["threshold"])
            except (KeyError, TypeError, ValueError):
                continue
            above = re.search(r"\b(above|exceeds|exceeded|over|greater)\b|>", bare, re.I)
            below = re.search(r"\b(below|under|beneath|less)\b|<", bare, re.I)
            if (above and value <= threshold) or (below and value >= threshold):
                flags.append({"kind": "wrong_inequality_direction",
                              "detector": detector_key(entry), "quoted": [],
                              "sentence": sentence[:220]})
    return flags


def grounding_flags(prose: str, prompt: str) -> List[str]:
    return ev.grounding_violations(str(prose or ""), prompt_tags(prompt), prompt_numbers(prompt))


def measure(prose: str, prompt: str, gold_fault_id: str, gold_verdict: str,
            store: Dict) -> Dict:
    tag, z = top_sensor(prompt)
    status, context = anchor_status(prose, tag, z)
    return {"contradiction": contradiction_flags(prose, gold_fault_id, gold_verdict),
            "tag_role": tag_role_flags(prose),
            "numeric": numeric_flags(prose, store),
            "grounding": grounding_flags(prose, prompt),
            "anchor": {"status": status, "tag": tag, "z": z, "context": context}}


def run_case(case: Dict) -> bool:
    check = case["check"]
    if check == "contradiction":
        return bool(contradiction_flags(case["text"], case["gold_fault_id"],
                                        case["gold_verdict"]))
    if check == "tag_role":
        return bool(tag_role_flags(case["text"]))
    if check == "anchor":
        return anchor_status(case["text"], case["tag"], case["z"])[0] != "exact"
    if check == "numeric":
        return bool(numeric_flags(case["text"], case["store"]))
    raise ValueError("unknown check " + str(check))


def calibration(cases: Sequence[Dict] = CASES) -> Dict:
    per: Dict[str, Dict] = defaultdict(lambda: Counter())
    for case in cases:
        flagged, expected = run_case(case), bool(case["expect"])
        key = ("tp" if flagged and expected else "fp" if flagged
               else ("fn" if expected else "tn"))
        per[case["check"]][key] += 1
    out = {}
    for name, counts in per.items():
        precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        out[name] = {"cases": sum(counts.values()), "tp": counts["tp"], "fp": counts["fp"],
                     "fn": counts["fn"], "tn": counts["tn"],
                     "precision": round(precision, 3), "recall": round(recall, 3)}
    return out


def stratum_of(row: Dict) -> str:
    if str(row["gold_fault_id"]).startswith("FAULT"):
        return "classified"
    if str(row["gold_verdict"]) == "no_fault":
        return "normal"
    return "menu" if str(row.get("variant")) == "menu" else "removed"


def phrase_of(check: str, flag) -> str:
    if check == "tag_role":
        return str(flag.get("claimed")) + " (" + str(flag.get("tag")) + ")"
    if check == "numeric":
        return str(flag.get("sentence"))[:110]
    if check == "grounding":
        return str(flag)[:110]
    return str(flag.get("sentence") or flag.get("detail") or "")[:110]


def normalise(text: str) -> str:
    body = re.sub(r"(?:XMEAS|XMV)_\d{1,2}", "<TAG>", str(text))
    body = re.sub(r"-?\d+(?:\.\d+)?", "<N>", body)
    return re.sub(r"\s+", " ", body).strip()[:100]


def collect(rows: Sequence[Dict], prompts: Dict[str, Dict], order: str, limit: int = 10) -> Dict:
    counts = Counter()
    strata = defaultdict(Counter)
    anchors = Counter()
    examples = defaultdict(list)
    phrases = Counter()
    for row in rows:
        item_id = row["item_id"]
        prompt = prompts[item_id][order]
        store = parse_store(prompt)
        parsed = parse_prose(row["text"])
        prose = parsed["prose"] if parsed["schema_valid"] else str(row["text"])
        result = measure(prose, prompt, row["gold_fault_id"], row["gold_verdict"], store)
        stratum = stratum_of(row)
        for check in ("contradiction", "tag_role", "numeric", "grounding"):
            found = result[check]
            if not found:
                continue
            counts[check] += 1
            strata[check][stratum] += 1
            for flag in found:
                phrases[normalise(phrase_of(check, flag))] += 1
                if len(examples[check]) < limit:
                    examples[check].append({"item_id": item_id, "stratum": stratum,
                                            "phrase": phrase_of(check, flag)})
        status = result["anchor"]["status"]
        anchors[status] += 1
        z = abs(float(result["anchor"]["z"] or 0.0))
        band = "ge_5" if z >= 5 else ("1_to_5" if z >= 1 else "lt_1")
        strata["anchor_" + band][status] += 1
        if status != "exact" and len(examples["anchor_" + status]) < limit:
            examples["anchor_" + status].append({"item_id": item_id, "stratum": stratum,
                                                 "top_tag": result["anchor"]["tag"],
                                                 "top_z": result["anchor"]["z"],
                                                 "phrase": result["anchor"]["context"]})
    return {"rows": len(rows), "counts": dict(counts), "strata": {name: dict(value)
                                                                 for name, value in strata.items()},
            "anchors": dict(anchors), "examples": dict(examples), "phrases": phrases}


def main() -> int:
    built = build_orders()
    prompts = {row["item_id"]: row["prompts"] for row in built}
    rows_by_order = {order: load_jsonl(EXPORT / ("raw_pilot_" + order + ".jsonl"))
                     for order in ORDERS}
    report: Dict = {"rows": len(built), "calibration": calibration()}
    report["strata"] = dict(Counter(stratum_of(row) for row in built))
    per_order = {}
    for order in ORDERS:
        per_order[order] = collect(rows_by_order[order], prompts, order)
    report["orders"] = {order: {"counts": data["counts"], "anchors": data["anchors"]}
                        for order, data in per_order.items()}
    main_data = per_order["kb"]
    report["main"] = {"counts": main_data["counts"], "strata": main_data["strata"],
                      "anchors": main_data["anchors"], "examples": main_data["examples"]}

    total = len(built)
    anchor_rows = {band: sum(main_data["strata"].get("anchor_" + band, {}).values())
                   for band in ("ge_5", "1_to_5", "lt_1")}
    anchor_exact = {band: main_data["strata"].get("anchor_" + band, {}).get("exact", 0)
                    for band in anchor_rows}
    report["anchor_bands"] = {
        band: {"rows": anchor_rows[band], "exact": anchor_exact[band],
               "rate": round(anchor_exact[band] / max(anchor_rows[band], 1), 4)}
        for band in anchor_rows}
    gated = anchor_rows["ge_5"] + anchor_rows["1_to_5"]
    report["anchor_gate"] = {
        "rows": gated, "rows_top_abs_z_ge_1": gated, "exact": anchor_exact["ge_5"] + anchor_exact["1_to_5"],
        "rate": round((anchor_exact["ge_5"] + anchor_exact["1_to_5"]) / max(gated, 1), 4),
        "threshold": 0.90,
        "pass": bool(gated and (anchor_exact["ge_5"] + anchor_exact["1_to_5"]) / gated >= 0.90)}

    # numeric and grounding breakdown on the kb order
    numeric_kinds, numeric_rows = Counter(), Counter()
    grounding_kinds = Counter()
    grounding_rows = Counter()
    for row in rows_by_order["kb"]:
        prompt = prompts[row["item_id"]]["kb"]
        parsed = parse_prose(row["text"])
        prose = parsed["prose"] if parsed["schema_valid"] else str(row["text"])
        flags = numeric_flags(prose, parse_store(prompt))
        for flag in flags:
            numeric_kinds[flag["kind"]] += 1
        if flags:
            numeric_rows[flags[0]["kind"]] += 1
        codes = grounding_flags(prose, prompt)
        has_number = any(code.startswith("number_") for code in codes)
        has_tag = any(code.startswith("tag_") for code in codes)
        if has_number and has_tag:
            grounding_rows["both"] += 1
        elif has_number:
            grounding_rows["numbers"] += 1
        elif has_tag:
            grounding_rows["tags"] += 1
        for code in codes:
            grounding_kinds[code.split(":")[0]] += 1
    report["numeric_breakdown"] = {"by_flag": dict(numeric_kinds),
                                   "by_row_first_kind": dict(numeric_rows)}
    report["grounding_breakdown"] = {"by_row": dict(grounding_rows),
                                     "by_code": dict(grounding_kinds)}

    stored = json.loads((FIT / "pilot_eval_summary.json").read_text(encoding="utf-8"))
    old = stored["orders"]["kb"]["counts"]
    new = main_data["counts"]
    report["old_vs_new"] = [
        {"gate": "contradiction", "old_count": old["contradiction"],
         "old_rate": round(old["contradiction"] / total, 4),
         "new_count": new.get("contradiction", 0),
         "new_rate": round(new.get("contradiction", 0) / total, 4),
         "threshold": 0.02, "note": "re-specified 28.2; stored value not a prose defect"},
        {"gate": "tag_role", "old_count": old["tag_role"],
         "old_rate": round(old["tag_role"] / total, 4),
         "new_count": new.get("tag_role", 0),
         "new_rate": round(new.get("tag_role", 0) / total, 4),
         "threshold": 0.02, "note": "re-specified 28.3; stored value not a prose defect"},
        {"gate": "numeric", "old_count": old["numeric"],
         "old_rate": round(old["numeric"] / total, 4),
         "new_count": new.get("numeric", 0),
         "new_rate": round(new.get("numeric", 0) / total, 4),
         "threshold": 0.02, "note": "same checker, decomposed by cause"},
        {"gate": "grounding", "old_count": old["grounding"],
         "old_rate": round(old["grounding"] / total, 4),
         "new_count": new.get("grounding", 0),
         "new_rate": round(new.get("grounding", 0) / total, 4),
         "threshold": 0.02, "note": "same checker, split into numbers and tags"},
        {"gate": "anchor", "old_count": 187, "old_rate": 0.3596,
         "new_count": report["anchor_gate"]["exact"], "new_rate": report["anchor_gate"]["rate"],
         "threshold": 0.90, "note": "higher is better; rows with top |z| >= 1"}]

    # fallback v3.1 on the same 520 items
    val_prompts = {row["item_id"]: row["prompts"]["kb"] for row in built}
    probs = {row["item_id"]: row.get("probs")
             for row in load_jsonl(HYBRID / "crossfit_val_rows.jsonl")}
    overlay = load_overlay(ROOT / "configs/kb_overlay_v1.json")
    fallback_counts = Counter()
    fallback_anchors = Counter()
    fallback_strata: Dict[str, Counter] = defaultdict(Counter)
    fallback_examples = defaultdict(list)
    for row in built:
        prompt = val_prompts[row["item_id"]]
        store = parse_store(prompt)
        raw = probs.get(row["item_id"]) or {}
        if isinstance(raw, str):
            raw = ast.literal_eval(raw)
        answer = fallback_v31(str(row["gold_fault_id"]), str(row["gold_verdict"]),
                              row.get("menu_ids") or [], store, raw, overlay)
        prose = (str(answer.get("physical_reasoning") or "") + " "
                 + str(answer.get("narrative") or ""))
        result = measure(prose, prompt, row["gold_fault_id"], row["gold_verdict"], store)
        stratum = stratum_of(row)
        for check in ("contradiction", "tag_role", "numeric", "grounding"):
            if result[check]:
                fallback_counts[check] += 1
                fallback_strata[check][stratum] += 1
                if len(fallback_examples[check]) < 10:
                    fallback_examples[check].append({"item_id": row["item_id"],
                                                     "stratum": stratum,
                                                     "phrase": phrase_of(check, result[check][0])})
        fallback_anchors[result["anchor"]["status"]] += 1
        z = abs(float(result["anchor"]["z"] or 0.0))
        if z >= 1:
            fallback_strata["anchor_ge_1"][result["anchor"]["status"]] += 1
        if result["anchor"]["status"] != "exact" and len(fallback_examples[
                "anchor_" + result["anchor"]["status"]]) < 10:
            fallback_examples["anchor_" + result["anchor"]["status"]].append(
                {"item_id": row["item_id"], "stratum": stratum,
                 "phrase": result["anchor"]["context"]})
    fallback_ge1 = sum(fallback_strata["anchor_ge_1"].values())
    report["fallback_v31"] = {
        "counts": dict(fallback_counts), "anchors": dict(fallback_anchors),
        "by_stratum": {name: dict(value) for name, value in fallback_strata.items()},
        "anchor_gate": {"rows": fallback_ge1,
                        "exact": fallback_strata["anchor_ge_1"].get("exact", 0),
                        "rate": round(fallback_strata["anchor_ge_1"].get("exact", 0)
                                      / max(fallback_ge1, 1), 4)},
        "examples": dict(fallback_examples)}

    top = Counter()
    for order in ORDERS:
        top.update(per_order[order]["phrases"])
    report["top_phrases"] = [{"phrase": phrase, "count": count}
                             for phrase, count in top.most_common(20)]
    report["by_order"] = {order: per_order[order]["counts"] for order in ORDERS}

    (FIT / "pilot_rechecks.json").write_text(json.dumps(report, indent=2, default=str),
                                             encoding="utf-8")
    (FIT / "pilot_rechecks_summary.md").write_text(markdown(report), encoding="utf-8")
    with (FIT / "pilot_recheck_examples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "check", "item_id", "stratum", "phrase"])
        for check, rows in main_data["examples"].items():
            for row in rows:
                writer.writerow(["adapter", check, row["item_id"], row["stratum"],
                                 row.get("phrase", "")])
        for check, rows in fallback_examples.items():
            for row in rows:
                writer.writerow(["fallback_v31", check, row["item_id"], row["stratum"],
                                 row.get("phrase", "")])
    print(json.dumps({key: report[key] for key in ("calibration", "main", "anchor_bands",
                                                   "anchor_gate", "numeric_breakdown",
                                                   "grounding_breakdown", "by_order")},
                     indent=1, default=str)[:4000])
    print("fallback:", json.dumps(report["fallback_v31"]["counts"]),
          report["fallback_v31"]["anchor_gate"])
    return 0


def markdown(report: Dict) -> str:
    total = report["rows"]
    lines = ["# Pilot artefact checks, re-specified (PROTOCOL 28)", "",
             "Rows: " + str(total) + ". Strata: " + ", ".join(
                 name + " " + str(value) for name, value in sorted(report["strata"].items()))
             + ". CPU only, stored generations only.", ""]
    lines += ["## 1. Synthetic calibration (40 hand-written cases)", "",
              "| check | cases | tp | fp | fn | precision | recall |", "|---|---|---|---|---|---|---|"]
    for name, entry in report["calibration"].items():
        lines.append("| " + name + " | " + str(entry["cases"]) + " | " + str(entry["tp"]) + " | "
                     + str(entry["fp"]) + " | " + str(entry["fn"]) + " | "
                     + f"{entry['precision']:.2f} | {entry['recall']:.2f} |")
    lines += ["", "## 2. Old vs new", "",
              "| gate | stored | re-specified | threshold | note |", "|---|---|---|---|---|"]
    for row in report["old_vs_new"]:
        lines.append("| " + row["gate"] + " | " + str(row["old_count"]) + " / " + str(total)
                     + " = " + "%.2f%%" % (100.0 * row["old_rate"]) + " | "
                     + str(row["new_count"]) + " / " + str(total) + " = "
                     + "%.2f%%" % (100.0 * row["new_rate"]) + " | "
                     + "%.0f%%" % (100.0 * row["threshold"]) + " | " + row["note"] + " |")
    lines += ["", "## 3. Flagged rows by stratum (kb order)", "",
              "| check | flagged | open-set on menu | menu-removed | NORMAL_TRANSIENT |",
              "|---|---|---|---|---|"]
    stratum_names = (("menu", "open-set on menu"), ("removed", "menu-removed"),
                     ("normal", "NORMAL_TRANSIENT"))
    for check in ("contradiction", "tag_role", "numeric", "grounding"):
        row = report["main"]["strata"]
        cells = [str(report["main"]["counts"].get(check, 0))]
        for name, _label in stratum_names:
            cells.append(str((row.get(check) or {}).get(name, 0)))
        lines.append("| " + check + " | " + " | ".join(cells) + " |")
    lines += ["", "Per order: " + "; ".join(
        order + " " + ", ".join(name + " " + str(value)
                                for name, value in sorted(counts.items()))
        for order, counts in report["by_order"].items()) + ".", ""]
    lines += ["## 4. Anchor", "",
              "| rows | exact | rate |", "|---|---|---|"]
    for band, label in (("ge_5", "top |z| >= 5"), ("1_to_5", "1 <= top |z| < 5"),
                        ("lt_1", "top |z| < 1")):
        entry = report["anchor_bands"][band]
        lines.append("| " + label + " | " + str(entry["exact"]) + " of " + str(entry["rows"])
                     + " | " + "%.2f%%" % (100.0 * entry["rate"]) + " |")
    gate = report["anchor_gate"]
    lines += ["", "Anchor gate (top |z| >= 1): " + str(gate["exact"]) + " of " + str(gate["rows"])
              + " = " + "%.2f%%" % (100.0 * gate["rate"]) + " against a 90% threshold: "
              + ("pass" if gate["pass"] else "FAIL") + ".",
              "Statuses (kb): " + ", ".join(name + " " + str(value)
                                            for name, value in sorted(
                                                report["main"]["anchors"].items())) + ".", ""]
    lines += ["## 5. Numeric and grounding breakdown", "",
              "Numeric by cause (flags): " + ", ".join(
                  name + " " + str(value) for name, value in
                  sorted(report["numeric_breakdown"]["by_flag"].items())) + ".",
              "Numeric rows by first cause: " + ", ".join(
                  name + " " + str(value) for name, value in
                  sorted(report["numeric_breakdown"]["by_row_first_kind"].items())) + ".",
              "Grounding rows: " + ", ".join(
                  name + " " + str(value) for name, value in
                  sorted(report["grounding_breakdown"]["by_row"].items())) + ".",
              "Grounding codes: " + ", ".join(
                  name + " " + str(value) for name, value in
                  sorted(report["grounding_breakdown"]["by_code"].items())) + ".", ""]
    lines += ["## 6. Side by side with fallback v3.1", "",
              "| reading | adapter (kb) | fallback v3.1 |", "|---|---|---|"]
    fallback = report["fallback_v31"]
    for check in ("contradiction", "tag_role", "numeric", "grounding"):
        lines.append("| " + check + " rows | " + str(report["main"]["counts"].get(check, 0))
                     + " | " + str(fallback["counts"].get(check, 0)) + " |")
    lines.append("| anchor exact, top |z| >= 1 | " + str(gate["exact"]) + " of "
                 + str(gate["rows"]) + " = " + "%.2f%%" % (100.0 * gate["rate"]) + " | "
                 + str(fallback["anchor_gate"]["exact"]) + " of "
                 + str(fallback["anchor_gate"]["rows"]) + " = "
                 + "%.2f%%" % (100.0 * fallback["anchor_gate"]["rate"]) + " |")
    lines += ["", "## 7. Twenty most frequent offending phrases (all three orders)", "",
              "| phrase | count |", "|---|---|"]
    for entry in report["top_phrases"]:
        lines.append("| `" + entry["phrase"].replace("|", "/") + "` | " + str(entry["count"])
                     + " |")
    return chr(10).join(lines) + chr(10)


if __name__ == "__main__":
    raise SystemExit(main())
