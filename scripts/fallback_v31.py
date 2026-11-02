"""The deterministic explainer, version 3.1: the seven defects the scored sheet exposed.

v3 was preferred over the adapter for classified faults and lost the abstention stratum 18 of 19
non-ties. Reading the losing side shows why: its abstention explainer was **cut in half** by its own
grounding guard, and its narrative was a byte-for-byte copy of its reasoning. v3.1 repairs the text,
it does not re-score it: every judgement here is a deterministic check on the 834 validation rows,
and no model is asked whether the result reads better (PROTOCOL §23.3, written before the work).

The seven fixes, in the order §23.3 lists them:

(i)   **the truncated clause.** ``terminal_guard`` splits on ``,`` and drops any clause whose tags
      the page does not print; the abstention explainer listed its candidates as
      ``FAULT_16 (initiating XMEAS_9)``, that tag is by construction one that does **not** deviate,
      the guard cut the clause, and the opener survived as
      ``"…, of the listed faults, so neither is supported by this page."`` v3.1 prints the candidate
      ids only, keeps each sentence self-contained, and runs a structural checker over the output.
(ii)  **action selection.** The check action is chosen from the subsystem of the **leading |z|
      sensor**, and any sensor the action names must be among the printed leading deviations.
(iii) **the narrative.** Abstention rows get one distinct sentence — verdict, leading sensor, one
      check — never the reasoning again.
(iv)  **"largest deviations"** are sorted by |z|, and the sensor the narrative names is the one the
      reasoning calls largest.
(v)   **KB overlay v2** adds the duplicated "overhead condenser" and the ". but heat duty" fragment.
(vi)  **low signal.** When max |z| < 1 and every detector ratio < 3 the action reads *check /
      verify* and never dispatch / manually open / bypass.
(vii) **trend words** print only when the trend agrees with the sign of the printed deviation.

CPU only, deterministic, no model, no teacher call.

Usage::  python scripts/fallback_v31.py [--overlay configs/kb_overlay_v2.json] [--out DIR]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evidence.process_limits import compute_deterministic_severity  # noqa: E402
from scripts.fallback_v2 import (_top_two, identical_prose, load_jsonl, parse_store,  # noqa: E402
                                 scrub, schema_ok, supplied_values, words)
from scripts.fallback_v3 import (CLAUSE, SENT, apply_overlay, differential,  # noqa: E402
                                 drop_contradicting_clauses, fallback_v3, initiating_tags,
                                 load_overlay, sha256, strict_issues, strip_alarm_wording,
                                 terminal_guard)
from scripts.fault_knowledge import FAULT_KNOWLEDGE  # noqa: E402
from scripts.tags import iter_tags  # noqa: E402

DATASET = ROOT / "outputs/benchmark_v2/dataset"
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
FIT = ROOT / "outputs/analysis/phase3a/fit"
DEFAULT_OVERLAY = ROOT / "configs/kb_overlay_v2.json"
KB_FILE = ROOT / "scripts/fault_knowledge.py"

#: A sentence may not end on one of these: that is how a cut clause leaves a dangling opener.
DANGLING = {"and", "or", "so", "with", "of", "the", "a", "an", "because", "while", "that", "which",
            "as", "in", "for", "at", "by", "from", "than", "then"}
VERBISH = re.compile(r"\b(is|are|was|were|be|stands?|starts?|moves?|check|checks|verify|shows?|"
                     r"indicates?|deviates?|increas\w*|decreas\w*|recommend\w*|asserted|supported|"
                     r"means?|remain\w*|carries?|opens?|drops?|rises?|falls?|reads|listed|z=|ratios?|"
                     r"deviations?|faults?|explains?)\b", re.I)
#: Sensor-name keyword -> KB subsystem. "cooling" outranks "reactor" on purpose.
SUBSYSTEM_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("stripper", "stripper_system"), ("cooling", "reactor_cooling_system"),
    ("condenser", "separation_system"), ("separator", "separation_system"),
    ("purge", "recycle_system"), ("recycle", "recycle_system"), ("compressor", "recycle_system"),
    ("feed", "feed_system"), ("product", "product_system"), ("reactor", "reactor_system"),
)
CHECK_ACTION = {
    "feed_system": "check the feed header flow transmitter and its ratio controller",
    "reactor_cooling_system": "check the cooling water valve stroke and the jacket flow",
    "reactor_system": "check the reactor temperature controller and its jacket valve",
    "stripper_system": "check the stripper steam valve and the reboiler duty",
    "separation_system": "check the overhead condenser cooling water and the separator level",
    "recycle_system": "check the recycle compressor speed and the purge rate",
    "product_system": "check the product analyser and its sample line",
}
DEFAULT_ACTION = "check the instrument on the leading deviation and its loop"
HARD_VERBS = re.compile(r"\b(dispatch|dispatched|manually open|open the|bypass|override|"
                        r"emergency)\b", re.I)


# --------------------------------------------------------------------------- (i) structure
def structural_issues(text: str) -> List[str]:
    """Why a sentence cannot stand on its own: the checker the truncated clause fails."""
    issues: List[str] = []
    body = str(text or "").strip()
    if body.count("(") != body.count(")"):
        issues.append(f"unbalanced parentheses {body.count('(')}/{body.count(')')}")
    if body.count("[") != body.count("]"):
        issues.append(f"unbalanced brackets {body.count('[')}/{body.count(']')}")
    if re.search(r",\s*,", body) or re.search(r",\s*\.", body):
        issues.append("double punctuation")
    for sentence in (part.strip() for part in SENT.split(body) if part.strip()):
        tokens = sentence.split()
        if not sentence.endswith((".", "!", "?")):
            issues.append(f"unterminated: {sentence[:70]}")
        if len(tokens) < 4:
            issues.append(f"stub: {sentence[:70]}")
        last = re.sub(r"[^a-z]", "", tokens[-1].lower())
        if last in DANGLING:
            issues.append(f"dangling final word {last!r}: {sentence[:70]}")
        lowered = sentence.lower()
        if "of the listed faults" in lowered and not re.search(r"FAULT_\d", sentence):
            issues.append(f"candidate opener with no candidates: {sentence[:70]}")
        if re.search(r"\bneither is supported\b|\bso neither\b", lowered) and "would" not in lowered:
            issues.append(f"orphan conclusion: {sentence[:70]}")


    return issues
def close_text(text: str, stats: Dict, key: str = "sentences_dropped_by_structure") -> str:
    """Drop any sentence the structural checker rejects. Truncation is never repaired by guessing."""
    kept: List[str] = []
    for sentence in (part.strip() for part in SENT.split(str(text or "")) if part.strip()):
        if structural_issues(sentence):
            stats[key] = stats.get(key, 0) + 1
            continue
        kept.append(sentence)
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


# --------------------------------------------------------------------------- (ii) action
def leading(store: Dict, limit: int = 3) -> List[Dict]:
    """Printed leading deviations, sorted by |z| — the list every claim is checked against."""
    return sorted(store.get("sensors") or [], key=lambda s: -abs(float(s.get("z") or 0.0)))[:limit]


def subsystem_for_sensor(name: str) -> str:
    lowered = str(name).lower()
    for keyword, subsystem in SUBSYSTEM_KEYWORDS:
        if keyword in lowered:
            return subsystem
    return ""
def signal_strength(store: Dict) -> Dict:
    z = max((abs(float(sensor.get("z") or 0.0)) for sensor in store.get("sensors") or []),
            default=0.0)
    ratio = max((float(detector.get("ratio") or 0.0) for detector in store.get("detectors") or []),
                default=0.0)
    return {"max_abs_z": round(z, 2), "max_ratio": round(ratio, 2),
            "low": bool(z < 1.0 and ratio < 3.0)}


def check_action(store: Dict) -> str:
    lead = leading(store, 1)
    if not lead:
        return DEFAULT_ACTION
    sensor = lead[0]
    action = CHECK_ACTION.get(subsystem_for_sensor(str(sensor["name"])), DEFAULT_ACTION)
    return f"{action}, starting with {sensor['name']} ({sensor['tag']})"


def resolve_action(action: str, store: Dict, stats: Dict) -> str:
    """(ii) and (vi): the action names a printed leading deviation, and stays calm when the page is."""
    action = str(action or "").strip()
    if not action:
        return check_action(store)
    lead_tags = {sensor["tag"] for sensor in leading(store)}
    cited = set(iter_tags(action))
    reason = None
    if cited and not cited <= lead_tags:
        reason = "action names a sensor outside the printed leading deviations"
    elif signal_strength(store)["low"] and HARD_VERBS.search(action):
        reason = "low-signal event with emergency verb"
    if reason is None:
        return action
    stats["action_replaced"] = stats.get("action_replaced", 0) + 1
    stats.setdefault("action_replaced_reasons", {})
    stats["action_replaced_reasons"][reason] = stats["action_replaced_reasons"].get(reason, 0) + 1
    return check_action(store)


# --------------------------------------------------------------------------- (vii) trend words
def sign_of(value) -> int:
    number = float(value or 0.0)
    return 1 if number > 0 else (-1 if number < 0 else 0)


def trend_agrees(sensor: Dict) -> bool:
    """The printed trend is allowed only when it matches the sign of the printed deviation."""
    trend = str(sensor.get("trend") or "").strip().lower()
    dev = sign_of(sensor.get("dev"))
    if not trend:
        return False
    if trend.startswith("increas"):
        return dev > 0
    if trend.startswith("decreas"):
        return dev < 0
    if trend in ("stable", "flat", "unchanged", "steady"):
        return dev == 0
    return False


def sensor_phrase(sensor: Dict, stats: Dict) -> str:
    trend = f", {sensor['trend']}" if trend_agrees(sensor) else ""
    if trend:
        stats["trend_words_printed"] = stats.get("trend_words_printed", 0) + 1
    else:
        stats["trend_words_omitted"] = stats.get("trend_words_omitted", 0) + 1
    return f"{sensor['name']} ({sensor['tag']}, z={sensor['z']}{trend})"


def human_join(items: Sequence[str]) -> str:
    items = [str(item) for item in items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def evidence_sentence_v31(store: Dict, stats: Dict) -> str:
    """(iv): "largest" means largest |z|, in that order, and nothing else is called largest."""
    parts: List[str] = []
    ratios = [(d["name"], d["ratio"]) for d in store["detectors"] if d.get("ratio")]
    if ratios:
        parts.append("detector ratios are " + " and ".join(f"{name} {ratio}x"
                                                          for name, ratio in ratios[:2]))
    sensors = leading(store, 2)
    if sensors:
        parts.append("the largest deviations are " + " and ".join(sensor_phrase(s, stats)
                                                                 for s in sensors))
    onset = sorted(store["onset"], key=lambda entry: entry["rank"])
    if len(onset) >= 2:
        parts.append(f"onset starts with {onset[0]['name']} ({onset[0]['tag']}) at "
                     f"{onset[0]['at']} min and {onset[1]['name']} follows at {onset[1]['at']} min")
    elif onset:
        parts.append(f"onset starts with {onset[0]['name']} ({onset[0]['tag']}) at "
                     f"{onset[0]['at']} min")
    if store.get("lag"):
        parts.append(f"the dominant lag is {store['lag']['lag']} window(s) at a slow-drift ratio "
                     f"of {store['lag']['ratio']}")
    if not parts:
        return "No detector or sensor evidence was readable from this event."
    return "Evidence for this event: " + "; ".join(parts) + "."


def abstention_answer_v31(diagnosis: str, verdict: str, menu_ids: Sequence[str], store: Dict,
                          stats: Dict) -> Dict:
    """(i), (iii), (iv): closed sentences, candidates by id only, a narrative that is not a copy."""
    lead_sensors = leading(store)
    lead = lead_sensors[0] if lead_sensors else None
    sentences = [f"No listed fault is asserted for this event ({diagnosis})."]
    ratios = [(d["name"], d["ratio"]) for d in store["detectors"] if d.get("ratio")]
    if ratios:
        sentences.append("Detector ratios are " + " and ".join(f"{name} {ratio}x"
                                                              for name, ratio in ratios[:2]) + ".")
    if lead_sensors[:2]:
        sentences.append("The largest deviations are "
                         + " and ".join(sensor_phrase(s, stats) for s in lead_sensors[:2]) + ".")
    deviating = {sensor["tag"] for sensor in store["sensors"]}
    absent: List[str] = []
    for label in list(menu_ids or []):
        match = re.match(r"FAULT_(\d{1,2})", str(label))
        if not match:
            continue
        tags = initiating_tags(int(match.group(1)))
        if tags and not (set(tags) & deviating):
            absent.append(label)
        if len(absent) == 2:
            break
    if absent:
        # (i): ids only. The initiating tags are, by construction, tags the page does not print, so
        # naming them was exactly what the guard cut out of the sentence.
        verb = "would each start" if len(absent) > 1 else "would start"
        tail = ("so neither is supported by this page" if len(absent) > 1
                else "so it is not supported by this page")
        sentences.append(f"Of the listed faults, {human_join(absent)} {verb} at a tag that is not "
                         f"among these deviations, {tail}.")
    action = resolve_action(check_action(store), store, stats)
    sentences.append(f"Recommended: {action}.")
    reasoning = close_text(re.sub(r"\s+", " ", " ".join(sentences)).strip(), stats)

    narrative = f"The verdict is {diagnosis}."
    if lead:
        narrative += (f" {lead['name']} ({lead['tag']}) is the largest single deviation, at "
                      f"z={lead['z']}.")
    narrative += (f" {action[:1].upper() + action[1:]} before escalating." if action
                  else " Verify the leading deviation before escalating.")
    narrative = close_text(re.sub(r"\s+", " ", narrative).strip(), stats)
    if narrative.strip() == reasoning.strip():            # (iii): never the same sentence twice
        narrative += f" The verdict is unchanged: {diagnosis}."
        stats["narrative_rewritten"] = stats.get("narrative_rewritten", 0) + 1
    return {"fault_id": str(diagnosis), "verdict": str(verdict),
            "ranked_hypotheses": [str(diagnosis)],
            "physical_reasoning": reasoning, "narrative": narrative, "_repairs": []}


def classified_answer_v31(diagnosis: str, verdict: str, menu_ids: Sequence[str], store: Dict,
                          probs: Dict[str, float], overlay: Dict, stats: Dict) -> Dict:
    from llm.benchmark.menu import lookalikes

    supplied = supplied_values(probs, _top_two(probs, diagnosis))
    match = re.match(r"FAULT_(\d{1,2})", str(diagnosis))
    number = int(match.group(1)) if match else None
    entry = FAULT_KNOWLEDGE.get(number or -1) or {}
    mechanism, action, repairs = apply_overlay(number, entry, overlay)
    for pattern, replacement in overlay.get("global_text_repairs") or []:      # (v)
        if re.search(pattern, mechanism, re.I):
            mechanism = re.sub(pattern, replacement, mechanism, flags=re.I)
            repairs.append(f"overlay global repair: {pattern} -> {replacement}")

    mechanism, _ = scrub(mechanism, store)
    action, action_repairs = scrub(action, store)
    repairs.extend(action_repairs)

    sensors = store["sensors"]
    evidence_rule = {"top_anomalous_sensors": [
        {"display_name": sensor["name"], "name": sensor["name"], "z_score": float(sensor["z"]),
         "deviation_percent": float(sensor["dev"])} for sensor in sensors],
        "detector_evidence": {"triggered_by": (store["detectors"][0]["name"]
                                              if store["detectors"] else ""),
                              "change_type": "", "dynamic": {}},
        "anomaly_score": max((float(d.get("ratio") or 0.0) for d in store["detectors"]),
                             default=0.0)}
    severity = compute_deterministic_severity(evidence_rule)
    action, stripped = strip_alarm_wording(action, severity)
    repairs.extend(f"stripped alarm wording: {text}" for text in stripped)
    action = resolve_action(action, store, stats)                             # (ii), (vi)

    mechanism, contradicted = drop_contradicting_clauses(mechanism, store)
    repairs.extend(f"dropped contradicting clause: {text}" for text in contradicted)

    ranked = [str(diagnosis)]
    if number is not None:
        for other in lookalikes(number, limit=2):
            label = f"FAULT_{int(other):02d}"
            if label in set(menu_ids or []) and label not in ranked:
                ranked.append(label)
    if not mechanism:
        mechanism = ("the stored evidence is consistent with this diagnosis and no other listed "
                     "entry explains the pattern")
    reasoning = " ".join(part for part in (
        evidence_sentence_v31(store, stats),
        mechanism[:1].upper() + mechanism[1:],
        differential(str(diagnosis), probs, str(verdict), menu_ids),
    ) if part)
    reasoning, dropped = terminal_guard(reasoning, store, supplied)
    repairs.extend(f"terminal guard dropped: {text}" for text in dropped)
    reasoning = close_text(reasoning, stats)

    lead = leading(store, 1)
    narrative = f"Diagnosis {diagnosis} is the best fit for this event."
    if lead:
        naming = f" {sensor_phrase(lead[0], stats)} deviates most"
        onset = sorted(store["onset"], key=lambda entry: entry["rank"])[:1]
        if onset:
            naming += f", and it is first to move at onset {onset[0]['at']} min"
        narrative += naming + "."
    if action:
        narrative += f" Recommended: {action}"
    narrative = close_text(re.sub(r"\s+", " ", narrative).strip(), stats)
    return {"fault_id": str(diagnosis), "verdict": str(verdict), "ranked_hypotheses": ranked,
            "physical_reasoning": re.sub(r"\s+", " ", reasoning).strip(), "narrative": narrative,
            "_repairs": repairs, "_severity": severity}


def effective_verdict(diagnosis: str, verdict: str) -> str:
    """A non-fault diagnosis is an abstention whatever the menu rule called it.

    The stored rule verdict says "classified" for an on-menu UNCLASSIFIED_OPEN_SET prediction. That
    is right for scoring the menu rule and wrong for choosing a prose explainer: there is no fault
    to explain. Both columns of the comparison use this rule, so v3 and v3.1 are measured on the
    same rows.
    """
    if re.match(r"FAULT_\d", str(diagnosis)) and str(verdict) == "classified":
        return "classified"
    if str(verdict) in ("", "classified"):
        return "no_fault" if "NORMAL" in str(diagnosis) else "unclassified_open_set"
    return str(verdict)


def fallback_v31(diagnosis: str, verdict: str, menu_ids: Sequence[str], store: Dict,
                 probs: Optional[Dict[str, float]] = None, overlay: Optional[Dict] = None,
                 stats: Optional[Dict] = None) -> Dict:
    overlay = overlay or {"faults": {}, "collapse": []}
    probs = probs or {}
    stats = stats if stats is not None else {}
    verdict = effective_verdict(diagnosis, verdict)
    if str(verdict) != "classified":
        return abstention_answer_v31(diagnosis, verdict, menu_ids, store, stats)
    return classified_answer_v31(diagnosis, verdict, menu_ids, store, probs, overlay, stats)


# --------------------------------------------------------------------------- measurement
# --------------------------------------------------------------------------- measurement
def direction_conflicts(text: str, store: Dict) -> List[str]:
    """Clauses whose direction word fights the sign the page printed for the same variable.

    Parentheticals are held out of the split: `X (XMEAS_1, z=8.1, decreasing)` is one claim, and
    splitting it on its commas makes the next sensor's trend look like a conflict.
    """
    from scripts.fallback_v2 import PAREN
    from scripts.fallback_v3 import DOWN, UP
    bad: List[str] = []
    sensors = {sensor["name"].lower(): sensor for sensor in store["sensors"]}
    held: List[str] = []

    def hold(match) -> str:
        held.append(match.group(0))
        return f"<<PAREN{len(held) - 1}>>"

    body = PAREN.sub(hold, str(text or ""))
    for sentence in SENT.split(body):
        for clause in CLAUSE.split(sentence):
            restored = re.sub(r"<<PAREN(\d+)>>", lambda m: held[int(m.group(1))], clause)
            named = [sensor for name, sensor in sensors.items()
                     if name and (name in restored.lower() or sensor["tag"] in restored)]
            if len(named) != 1:
                continue          # a clause naming two sensors cannot attribute one trend word
            for sensor in named:
                trend = str(sensor.get("trend") or "").lower()
                down, up = bool(DOWN.search(restored)), bool(UP.search(restored))
                if (trend.startswith("increas") and down and not up) or \
                   (trend.startswith("decreas") and up and not down):
                    bad.append(restored.strip(" ,.;")[:100])
    return bad


def action_of(answer: Dict) -> str:
    match = re.search(r"Recommended:\s*(.+)$", str(answer.get("physical_reasoning") or ""))
    return match.group(1).strip() if match else ""


def summarise(rows: List[Dict], version: str) -> Dict:
    total = len(rows)
    return {
        "rows": total,
        "schema_valid": sum(1 for row in rows if row[version]["schema"]["valid"]),
        "rows_with_any_violation_prompt_only": sum(1 for row in rows
                                                   if row[version]["prompt_only"]["count"]),
        "violation_kinds_prompt_only": {kind: sum(len(row[version]["prompt_only"][kind])
                                                  for row in rows)
                                        for kind in ("numbers", "pairs", "ranges", "tags")},
        "rows_with_any_violation_prompt_plus_supplied": sum(
            1 for row in rows if row[version]["with_supplied"]["count"]),
        "violation_kinds_prompt_plus_supplied": {kind: sum(len(row[version]["with_supplied"][kind])
                                                          for row in rows)
                                                 for kind in ("numbers", "pairs", "ranges", "tags")},
        "rows_with_structural_issue": sum(1 for row in rows if row[version]["structure"]),
        "structural_issues": dict(Counter(re.split(r"[:，(]", issue)[0]
                                          for row in rows for issue in row[version]["structure"])),
        "rows_with_direction_conflict": sum(1 for row in rows if row[version]["direction"]),
        "direction_conflicts": sum(len(row[version]["direction"]) for row in rows),
        "empty_narrative": sum(1 for row in rows
                               if not str(row[version]["answer"]["narrative"]).strip()),
        "narrative_equals_reasoning": sum(
            1 for row in rows
            if str(row[version]["answer"]["narrative"]).strip()
            == str(row[version]["answer"]["physical_reasoning"]).strip()),
        "action_names_leading_sensor": sum(1 for row in rows if row[version]["action_bound"]),
        "action_rows_with_a_sensor": sum(1 for row in rows if row[version]["action_tags"]),
        "mean_words": {
            "physical_reasoning": round(sum(words(row[version]["answer"]["physical_reasoning"])
                                            for row in rows) / total, 1),
            "narrative": round(sum(words(row[version]["answer"]["narrative"])
                                   for row in rows) / total, 1)},
        "identical_prose": identical_prose(
            [{"diagnosis": row["diagnosis"], version: {"answer": row[version]["answer"]}}
             for row in rows], version),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", default=str(DEFAULT_OVERLAY))
    parser.add_argument("--out", default=str(FIT))
    args = parser.parse_args()

    overlay_path = Path(args.overlay)
    overlay = load_overlay(overlay_path)
    git_kb = subprocess.run(["git", "status", "--porcelain", "--", "scripts/fault_knowledge.py",
                             "outputs/benchmark_v2/dataset", "configs/menu*"],
                            cwd=ROOT, capture_output=True, text=True).stdout.strip()
    val = {row["item_id"]: row for row in load_jsonl(DATASET / "val.jsonl")}
    predictions = load_jsonl(HYBRID / "crossfit_val_rows.jsonl")
    rows: List[Dict] = []
    stats31: Dict = {}
    for prediction in predictions:
        item = val.get(prediction["item_id"])
        if item is None:
            continue
        store = parse_store(item["prompt"])
        probs = prediction.get("probs")
        if isinstance(probs, str):
            try:
                probs = ast.literal_eval(probs)
            except (ValueError, SyntaxError):
                probs = {}
        probs = {str(label): float(value) for label, value in (probs or {}).items()}
        diagnosis = str(prediction["pred"])
        verdict = effective_verdict(str(prediction["pred"]),
                                     str(prediction.get("verdict") or "classified"))
        menu_ids = item.get("menu_ids") or []
        supplied = supplied_values(probs, _top_two(probs, diagnosis))
        answers = {"v3": fallback_v3(diagnosis, verdict, menu_ids, store, probs, overlay),
                   "v31": fallback_v31(diagnosis, verdict, menu_ids, store, probs, overlay, stats31)}
        record: Dict = {"item_id": prediction["item_id"], "diagnosis": diagnosis,
                        "verdict": verdict, "family": item.get("variant")}
        for version, answer in answers.items():
            text = " ".join(str(answer.get(key) or "") for key in
                            ("physical_reasoning", "narrative", "fault_id", "verdict"))
            action = action_of(answer)
            tags = set(iter_tags(action))
            record[version] = {
                "answer": answer, "schema": schema_ok(answer),
                "prompt_only": strict_issues(text, store, {}),
                "with_supplied": strict_issues(text, store, supplied),
                "structure": structural_issues(str(answer.get("physical_reasoning") or ""))
                             + structural_issues(str(answer.get("narrative") or "")),
                "direction": direction_conflicts(text, store), "action_tags": sorted(tags),
                "action_bound": bool(tags) and tags <= {s["tag"] for s in leading(store)},
                "action": action}
        record["low_signal"] = signal_strength(store)["low"]
        rows.append(record)

    report = {
        "rows": len(rows),
        "overlay": {"path": str(overlay_path), "sha256": sha256(overlay_path),
                    "faults": sorted((overlay.get("faults") or {}).keys()),
                    "global_repairs": len(overlay.get("global_text_repairs") or [])},
        "kb": {"path": str(KB_FILE), "sha256": sha256(KB_FILE), "git_status": git_kb or "clean"},
        "v3": summarise(rows, "v3"), "v31": summarise(rows, "v31"),
        "v31_process": dict(sorted(stats31.items())),
        "low_signal_rows": sum(1 for row in rows if row["low_signal"]),
        "low_signal_abstention_rows": sum(1 for row in rows if row["low_signal"]
                                          and row["verdict"] != "classified"),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fallback_v31.json").write_text(json.dumps(report, indent=2, default=str),
                                           encoding="utf-8")
    with open(out / "fallback_v31_rows.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"item_id": row["item_id"], "diagnosis": row["diagnosis"],
                                     "verdict": row["verdict"], "v3": row["v3"]["answer"],
                                     "v31": row["v31"]["answer"],
                                     "v3_structure": row["v3"]["structure"],
                                     "v31_structure": row["v31"]["structure"]},
                                    default=str) + "\n")

    from csv import DictReader
    key = {row["item_id"]: row for row in DictReader(
        open(FIT / "audit_ab_100_v3_key.csv", encoding="utf-8-sig"))}
    sheet = {row["item_id"]: row for row in DictReader(
        open(FIT / "audit_ab_100_v3.csv", encoding="utf-8-sig"))}
    by_id = {row["item_id"]: row for row in predictions}
    before_after = []
    for item_id, entry in key.items():
        row = sheet.get(item_id)
        if row is None or row["verdict"] == "classified" or item_id not in val:
            continue
        prediction = by_id[item_id]
        probs = prediction.get("probs") or {}
        if isinstance(probs, str):
            probs = ast.literal_eval(probs)
        probs = {str(k): float(v) for k, v in probs.items()}
        answer = fallback_v31(str(prediction["pred"]), str(prediction.get("verdict") or ""),
                              val[item_id].get("menu_ids") or [], parse_store(val[item_id]["prompt"]),
                              probs, overlay, {})
        before_after.append({"item_id": item_id, "verdict": row["verdict"],
                             "v3_sheet_text": row["text_" + ("A" if entry["A"] == "fallback_v3"
                                                             else "B")],
                             "v31_reasoning": answer["physical_reasoning"],
                             "v31_narrative": answer["narrative"]})
    (out / "sheet_abstention_before_after.json").write_text(
        json.dumps(before_after, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({name: report[name] for name in ("rows", "v3", "v31", "v31_process",
                                                      "low_signal_rows")}, indent=2, default=str))
    print(f"written: {out / 'fallback_v31.json'} ({len(before_after)} sheet rows before/after)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
