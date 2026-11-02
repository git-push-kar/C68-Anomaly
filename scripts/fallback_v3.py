"""The deterministic explainer, third version: strict units, directions checked, severity-gated.

v2 removed the KB example's numbers (0 number and 0 tag violations against the prompt on 834
rows), and it left five things wrong that a reader of the text can see:

1. **the grounding check was too narrow.** It compared bare number tokens, so `219.91 %` passed
   when the page printed `219.91` with no unit, and a range assembled from two quotable numbers
   was never checked as a range. v3 checks **signed numbers, number+unit pairs and ranges**, and
   re-checks the finished string.
2. **a KB mechanism can point the wrong way.** "Cooling water flow suddenly drops by 35-60 %"
   printed for an event whose own page shows that flow rising. v3 drops any mechanism clause
   whose direction word contradicts the sign of the deviation printed for the same variable.
3. **the differential printed a 0.00 candidate.** v3 prints the second candidate only when it is
   on the menu and its calibrated probability is at least 0.05; otherwise the line is omitted.
4. **the action carried KB alarm words** ("CRITICAL EMERGENCY", "SAFETY INTERLOCK") regardless of
   the event. v3 asks `compute_deterministic_severity` first and strips that wording unless that
   rule returns its top severity.
5. **the "top" sensor was the first listed**, not the largest deviation. v3 takes max |z|.

For the abstention, `NORMAL_TRANSIENT` and unclassified strata v3 writes a different explainer:
detector ratios, the leading sensors, **which listed faults' initiating tags are absent** from
those sensors (descriptive, read from the KB signature, no fitted score), and one check action
derived from the leading sensor's subsystem. No "fault family" wording.

KB text defects (a mangled degree sign and doubled equipment phrases in the KB's own strings) are
handled through an **overlay file with its own hash**: the KB files are untouched, and the menu
text and dataset hashes are proven unchanged in the run output.

CPU only, deterministic, no model, no teacher call.

Usage::  python scripts/fallback_v3.py [--overlay configs/kb_overlay_v1.json] [--out DIR]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evidence.process_limits import compute_deterministic_severity  # noqa: E402
from scripts.fallback_v2 import (NUMBER, UNIT, _top_two, evidence_sentence,  # noqa: E402
                                 identical_prose, load_jsonl, parse_store, quotable, scrub,
                                 schema_ok, supplied_values, words)
from scripts.fallback_v2 import fallback_v2 as build_v2  # noqa: E402
from scripts.fault_knowledge import FAULT_KNOWLEDGE  # noqa: E402
from scripts.tags import iter_tags  # noqa: E402

DATASET = ROOT / "outputs/benchmark_v2/dataset"
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
OUT = ROOT / "outputs/analysis/phase3a/fit"
DEFAULT_OVERLAY = ROOT / "configs/kb_overlay_v1.json"
KB_FILE = ROOT / "scripts/fault_knowledge.py"

PAIR = re.compile(rf"(?<![A-Za-z0-9_.])(?P<sign>[+\-]?)(?P<n>\d+(?:\.\d+)?)\s*(?P<u>{UNIT})",
                  re.IGNORECASE)
RANGE_PRINT = re.compile(rf"(-?\d+(?:\.\d+)?)\s*(?:to|–|-|toward)\s*(-?\d+(?:\.\d+)?)\s*({UNIT})",
                         re.IGNORECASE)
SENT = re.compile(r"(?<=[.!?])\s+")
CLAUSE = re.compile(r"\s*;\s*|,\s+")
DOWN = re.compile(r"\b(drop|drops|dropped|fall|falls|fell|declin\w*|decreas\w*|downward|"
                  r"lower|reduc\w*|below)\b", re.I)
UP = re.compile(r"\b(rise|rises|rose|rising|increas\w*|climb\w*|higher|above|upward|spike\w*|"
                r"elevat\w*)\b", re.I)
ALARM = re.compile(r"\b(CRITICAL|EMERGENCY|SAFETY INTERLOCK|DANGER|IMMEDIATELY)\b")
TAG_PATTERN = re.compile(r"(?:XMEAS|XMV)_\d{1,2}")
#: one check action per KB subsystem, used by the abstention explainer
SUBSYSTEM_ACTION = {
    "feed_system": "check the feed header flow transmitter and its ratio controller",
    "reactor_cooling_system": "check the cooling water valve stroke and the jacket flow",
    "reactor_system": "check the reactor temperature controller and its jacket valve",
    "stripper_system": "check the stripper steam valve and the reboiler duty",
    "separation_system": "check the overhead condenser cooling water and the separator level",
    "recycle_system": "check the recycle compressor speed and the purge rate",
    "product_system": "check the product analyser and its sample line",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_overlay(path: Path) -> Dict:
    if not path.exists():
        return {"faults": {}, "collapse": []}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_overlay(number: Optional[int], entry: Dict, overlay: Dict) -> Tuple[str, str, List[str]]:
    """KB text after the overlay's own repairs, with the repairs listed."""
    repairs: List[str] = []
    mechanism = str(entry.get("physical_mechanism") or "")
    action = str(entry.get("recommended_action") or "")
    for pattern, replacement in overlay.get("collapse") or []:
        for field, value in (("mechanism", mechanism), ("action", action)):
            if re.search(pattern, value, re.I):
                new = re.sub(pattern, replacement, value, flags=re.I)
                repairs.append(f"overlay collapse {field}: {pattern} -> {replacement}")
                if field == "mechanism":
                    mechanism = new
                else:
                    action = new
    exact = (overlay.get("faults") or {}).get(str(number)) or {}
    if exact.get("recommended_action"):
        action = str(exact["recommended_action"])
        repairs.append(f"overlay action for FAULT_{number:02d}")
    for pattern, replacement in (exact.get("text_repairs") or []):
        mechanism = re.sub(pattern, replacement, mechanism, flags=re.I)
        repairs.append(f"overlay text repair: {pattern} -> {replacement}")
    return mechanism, action, repairs


# --------------------------------------------------------------------------- the guards
def strict_issues(text: str, store: Dict, supplied: Optional[Dict[str, str]] = None) -> Dict:
    """Numbers, number+unit pairs and ranges that the page does not print.

    ``supplied`` holds the classifier's own values, which §20.6 counts as a separate class.
    """
    supplied = supplied or {}
    prompt = store["text"]
    bare = [number for number in NUMBER.findall(text)
            if not quotable(number, store) and number not in supplied]
    pairs = []
    for match in PAIR.finditer(text):
        number, unit = match.group("n"), match.group("u")
        if number in supplied:
            continue
        if not quotable(number, store):
            continue                      # already counted as a bare number
        printed = re.search(rf"{re.escape(number)}\s*[^\d\s]{{0,3}}\s*{re.escape(unit)}", prompt, re.I)
        if not printed:
            pairs.append(match.group(0))
    ranges = []
    for match in RANGE_PRINT.finditer(text):
        first, second, unit = match.group(1), match.group(2), match.group(3)
        both = re.search(rf"{re.escape(first)}[^\d]{{0,12}}{re.escape(second)}", prompt)
        if not both:
            ranges.append(match.group(0))
    tags = [tag for tag in iter_tags(text) if tag not in store["tags"]]
    return {"numbers": sorted(set(bare)), "pairs": sorted(set(pairs)),
            "ranges": sorted(set(ranges)), "tags": sorted(set(tags)),
            "count": len(set(bare)) + len(set(pairs)) + len(set(ranges)) + len(set(tags))}


def drop_contradicting_clauses(text: str, store: Dict) -> Tuple[str, List[str]]:
    """Drop a clause whose direction word fights the sign the page printed for that variable."""
    dropped: List[str] = []
    sensors = {sensor["name"].lower(): sensor for sensor in store["sensors"]}
    kept_sentences = []
    for sentence in SENT.split(str(text or "")):
        clauses = []
        for clause in CLAUSE.split(sentence):
            clause = clause.strip(" ,.;")
            if not clause:
                continue
            named = [sensor for name, sensor in sensors.items()
                     if name and (name in clause.lower() or sensor["tag"] in clause)]
            bad = False
            for sensor in named:
                trend = str(sensor.get("trend") or "").lower()
                down, up = bool(DOWN.search(clause)), bool(UP.search(clause))
                if (trend.startswith("increas") and down and not up) or \
                   (trend.startswith("decreas") and up and not down):
                    bad = True
            if bad:
                dropped.append(clause)
                continue
            clauses.append(clause)
        if clauses:
            kept_sentences.append(", ".join(clauses) + ".")
    return " ".join(kept_sentences), dropped


def terminal_guard(text: str, store: Dict, supplied: Dict[str, str]) -> Tuple[str, List[str]]:
    """Last line of defence: nothing unquotable survives, at clause granularity."""
    kept, dropped = [], []
    for sentence in SENT.split(str(text or "")):
        clauses = []
        for clause in CLAUSE.split(sentence):
            clause = clause.strip(" ,.;")
            if not clause:
                continue
            issues = strict_issues(clause, store, supplied)
            if issues["count"]:
                dropped.append(f"{clause[:120]} ({issues})")
                continue
            clauses.append(clause)
        if clauses:
            kept.append(", ".join(clauses) + ".")
    body = " ".join(kept)
    body = re.sub(r"\s+", " ", body)
    body = re.sub(r"\s+([,.;:!?])", r"\1", body)
    return body.strip(), dropped


def strip_alarm_wording(action: str, severity: str) -> Tuple[str, List[str]]:
    if str(severity).lower() == "critical":
        return action, []
    cleaned = ALARM.sub("", action)
    cleaned = re.sub(r"^\s*[:\-–]\s*", "", cleaned)
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return re.sub(r"\s{2,}", " ", cleaned).strip(), ([] if cleaned == action else [action])


# --------------------------------------------------------------------------- the answer
def top_sensor(store: Dict) -> Optional[Dict]:
    """Largest |z|, not the first one listed."""
    sensors = store.get("sensors") or []
    if not sensors:
        return None
    return max(sensors, key=lambda sensor: abs(float(sensor.get("z") or 0.0)))


def differential(diagnosis: str, probs: Dict[str, float], verdict: str,
                 menu_ids: Sequence[str]) -> str:
    """Second candidate only when it is on the menu and worth at least 0.05."""
    if verdict != "classified":
        return ""
    ranked = sorted(((label, float(value)) for label, value in probs.items()
                     if label.startswith("FAULT")), key=lambda item: (-item[1], item[0]))
    if not ranked:
        return ""
    first, p1 = ranked[0]
    second = next((item for item in ranked[1:] if item[0] in set(menu_ids or [])
                   and item[1] >= 0.05), None)
    if second is None:
        return (f"Differential: the classifier gives {first} a calibrated probability of {p1:.2f}; "
                f"no other entry on the menu reaches 0.05.")
    return (f"Differential: the classifier keeps {first} at a calibrated probability of {p1:.2f} "
            f"and {second[0]} at {second[1]:.2f}; no other entry on the menu is ranked above "
            f"{second[0]}.")


def initiating_tags(number: int) -> List[str]:
    entry = FAULT_KNOWLEDGE.get(number) or {}
    tags: List[str] = []
    for item in entry.get("initiating") or []:
        tags.extend(TAG_PATTERN.findall(str(item.get("tag") or "")))
    return tags


def subsystem_for(name: str) -> str:
    """The KB subsystem whose equipment shares most tokens with this sensor's name."""
    tokens = {token for token in re.findall(r"[a-z0-9]+", str(name).lower()) if len(token) > 3}
    best, score = "", 0
    for _number, entry in FAULT_KNOWLEDGE.items():
        text = f"{entry.get('equipment')} {entry.get('name')}".lower()
        hits = sum(1 for token in tokens if token in text)
        if hits > score:
            best, score = str(entry.get("subsystem") or ""), hits
    return best


def abstention_answer(diagnosis: str, verdict: str, menu_ids: Sequence[str],
                      store: Dict) -> Dict:
    """The abstention / normal / unclassified explainer: descriptive, no fitted score."""
    lead = top_sensor(store)
    parts: List[str] = []
    ratios = [(d["name"], d["ratio"]) for d in store["detectors"] if d.get("ratio")]
    if ratios:
        parts.append("detector ratios are " + " and ".join(f"{name} {ratio}x"
                                                          for name, ratio in ratios[:2]))
    leading = store["sensors"][:3]
    if leading:
        parts.append("the leading deviations are " + " and ".join(
            f"{s['name']} ({s['tag']}, z={s['z']}, {s['trend']})" for s in leading))
    deviating = {sensor["tag"] for sensor in store["sensors"]}
    absent: List[str] = []
    for label in list(menu_ids or []):
        match = re.match(r"FAULT_(\d{1,2})", str(label))
        if not match:
            continue
        tags = initiating_tags(int(match.group(1)))
        if tags and not (set(tags) & deviating):
            absent.append(f"{label} (initiating {', '.join(sorted(set(tags)))})")
        if len(absent) == 2:
            break
    if absent:
        parts.append("of the listed faults, " + " and ".join(absent)
                     + " would start at a tag that is not among those deviations, so neither is "
                       "supported by this page")
    body = "No listed fault is asserted for this event (" + str(diagnosis) + ")."
    if parts:
        body += " Evidence: " + "; ".join(parts) + "."
    if lead:
        subsystem = subsystem_for(lead["name"])
        action = SUBSYSTEM_ACTION.get(subsystem,
                                      "check the instrument on the leading deviation and its loop")
        body += f" Recommended: {action}, starting with {lead['name']} ({lead['tag']})."
    return {"fault_id": str(diagnosis), "verdict": str(verdict),
            "ranked_hypotheses": [str(diagnosis)],
            "physical_reasoning": re.sub(r"\s+", " ", body).strip(),
            "narrative": re.sub(r"\s+", " ", body).strip(), "_repairs": []}


def fallback_v3(diagnosis: str, verdict: str, menu_ids: Sequence[str], store: Dict,
                probs: Optional[Dict[str, float]] = None,
                overlay: Optional[Dict] = None) -> Dict:
    from llm.benchmark.menu import lookalikes

    overlay = overlay or {"faults": {}, "collapse": []}
    probs = probs or {}
    supplied = supplied_values(probs, _top_two(probs, diagnosis))
    if str(verdict) != "classified":
        answer = abstention_answer(diagnosis, verdict, menu_ids, store)
        answer["physical_reasoning"], _ = terminal_guard(answer["physical_reasoning"], store,
                                                         supplied)
        answer["narrative"] = answer["physical_reasoning"]
        return answer

    match = re.match(r"FAULT_(\d{1,2})", str(diagnosis))
    number = int(match.group(1)) if match else None
    entry = FAULT_KNOWLEDGE.get(number or -1) or {}
    mechanism, action, repairs = apply_overlay(number, entry, overlay)

    mechanism, _ = scrub(mechanism, store)
    action, action_repairs = scrub(action, store)
    repairs.extend(action_repairs)

    sensors = store["sensors"]
    evidence_rule = {"top_anomalous_sensors": [
        {"display_name": sensor["name"], "name": sensor["name"],
         "z_score": float(sensor["z"]), "deviation_percent": float(sensor["dev"])}
        for sensor in sensors],
        "detector_evidence": {"triggered_by": (store["detectors"][0]["name"]
                                              if store["detectors"] else ""),
                              "change_type": "", "dynamic": {}},
        "anomaly_score": max((float(d.get("ratio") or 0.0) for d in store["detectors"]),
                             default=0.0)}
    severity = compute_deterministic_severity(evidence_rule)
    action, stripped = strip_alarm_wording(action, severity)
    repairs.extend(f"stripped alarm wording: {text}" for text in stripped)

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
        evidence_sentence(store),
        mechanism[:1].upper() + mechanism[1:],
        differential(str(diagnosis), probs, str(verdict), menu_ids),
    ) if part)
    reasoning, dropped = terminal_guard(reasoning, store, supplied)
    repairs.extend(f"terminal guard dropped: {text}" for text in dropped)

    lead = top_sensor(store) or {}
    onset = sorted(store["onset"], key=lambda item: item["rank"])[:1]
    naming = ""
    if lead:
        naming = (f" {lead['name']} ({lead['tag']}) deviates most, at z={lead['z']} and "
                  f"{lead['trend']}")
        if onset:
            naming += f", and it is first to move at onset {onset[0]['at']} min"
        naming += "."
    narrative = f"Diagnosis {diagnosis} is the best fit for this event." + naming
    if action:
        narrative += f" Recommended: {action}"
    narrative = re.sub(r"\s+", " ", narrative).strip()
    narrative, dropped_n = terminal_guard(narrative, store, supplied)
    repairs.extend(f"terminal guard dropped (narrative): {text}" for text in dropped_n)
    return {"fault_id": str(diagnosis), "verdict": str(verdict),
            "ranked_hypotheses": ranked,
            "physical_reasoning": re.sub(r"\s+", " ", reasoning).strip(),
            "narrative": narrative, "_repairs": repairs, "_severity": severity}


# --------------------------------------------------------------------------- measurement
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", default=str(DEFAULT_OVERLAY))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--known", default="EVT-val-f04-r317::menu,EVT-val-f11-r307::menu,"
                                          "EVT-val-f04-r306::menu,EVT-val-f04-r310::menu,"
                                          "EVT-val-f04-r307::menu")
    args = parser.parse_args()

    overlay_path = Path(args.overlay)
    overlay = load_overlay(overlay_path)
    kb_hash = sha256(KB_FILE)
    overlay_hash = sha256(overlay_path) if overlay_path.exists() else "missing"
    git_kb = subprocess.run(["git", "status", "--porcelain", "--", "scripts/fault_knowledge.py",
                             "outputs/benchmark_v2/dataset", "configs/menu*"],
                            cwd=ROOT, capture_output=True, text=True).stdout.strip()

    val = {row["item_id"]: row for row in load_jsonl(DATASET / "val.jsonl")}
    predictions = load_jsonl(HYBRID / "crossfit_val_rows.jsonl")
    rows: List[Dict] = []
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
        verdict = str(prediction.get("verdict") or "classified")
        answer = fallback_v3(diagnosis, verdict, item.get("menu_ids") or [], store, probs, overlay)
        top = _top_two(probs, diagnosis)
        supplied = supplied_values(probs, top)
        text = " ".join(str(answer.get(key) or "") for key in
                        ("physical_reasoning", "narrative", "fault_id", "verdict"))
        strict = strict_issues(text, store, supplied)
        strict_prompt_only = strict_issues(text, store, {})
        v2 = build_v2(diagnosis, verdict, item.get("menu_ids") or [], store, probs)
        v2_text = " ".join(str(v2.get(key) or "") for key in
                           ("physical_reasoning", "narrative", "fault_id", "verdict"))
        v2_strict = strict_issues(v2_text, store, {})
        rows.append({"item_id": prediction["item_id"], "diagnosis": diagnosis, "verdict": verdict,
                     "family": item.get("variant"), "answer": answer,
                     "schema": schema_ok(answer), "strict": strict,
                     "strict_prompt_only": strict_prompt_only,
                     "v2_strict_prompt_only": v2_strict,
                     "severity": answer.get("_severity")})

    known = [item.strip() for item in args.known.split(",") if item.strip()]
    known_rows = [{"item_id": row["item_id"],
                   "v2_prompt_only_count": row["v2_strict_prompt_only"]["count"],
                   "v2_prompt_only": row["v2_strict_prompt_only"],
                   "v3_strict": row["strict"]["count"],
                   "v3_strict_prompt_only": row["strict_prompt_only"]["count"]}
                  for row in rows if row["item_id"] in known]
    v2_total = {"rows_with_any_violation": sum(1 for row in rows
                                                if row["v2_strict_prompt_only"]["count"]),
                "kinds": {key: sum(len(row["v2_strict_prompt_only"][key]) for row in rows)
                          for key in ("numbers", "pairs", "ranges", "tags")}}

    report = {
        "rows": len(rows),
        "overlay": {"path": str(overlay_path), "sha256": overlay_hash,
                    "faults": sorted((overlay.get("faults") or {}).keys())},
        "kb": {"path": str(KB_FILE), "sha256": kb_hash, "git_status": git_kb or "clean"},
        "v3": {
            "schema_valid": sum(1 for row in rows if row["schema"]["valid"]),
            "rows_with_any_violation_strict": sum(1 for row in rows
                                                  if row["strict"]["count"]),
            "kinds": {key: sum(len(row["strict"][key]) for row in rows)
                      for key in ("numbers", "pairs", "ranges", "tags")},
            "mean_words": {
                "physical_reasoning": round(sum(words(row["answer"]["physical_reasoning"])
                                                for row in rows) / len(rows), 1),
                "narrative": round(sum(words(row["answer"]["narrative"])
                                       for row in rows) / len(rows), 1)},
            "default_mechanism_rows": sum(1 for row in rows
                                          if not str(row["diagnosis"]).startswith("FAULT")),
            "identical_prose_rate": identical_prose(
                [{"diagnosis": row["diagnosis"], "v2": {"answer": row["answer"]}}
                 for row in rows], "v2")["rate"],
            "severity_counts": {},
            "rows_where_a_clause_was_dropped_for_direction": sum(
                1 for row in rows
                if any("contradicting" in repair for repair in row["answer"].get("_repairs") or [])),
            "rows_with_alarm_wording_stripped": sum(
                1 for row in rows
                if any("stripped alarm" in repair for repair in row["answer"].get("_repairs") or [])),
            "differential_omitted_rows": sum(
                1 for row in rows if "Differential:" not in row["answer"]["physical_reasoning"]),
        },
        "known_rows": known_rows,
        "v2_under_the_same_strict_guard": v2_total,
        "examples": [],
    }
    counts: Dict[str, int] = {}
    for row in rows:
        severity = str(row["severity"])
        counts[severity] = counts.get(severity, 0) + 1
    report["v3"]["severity_counts"] = counts

    wanted = ["EVT-val-f01-r310::openset", "EVT-val-f01-r313::openset",
              "EVT-val-f01-r317::openset", "EVT-val-f03-r302::menu",
              "EVT-val-f03-r302::openset", "EVT-val-f01-r306::menu",
              "EVT-val-f01-r307::menu", "EVT-val-f01-r310::menu",
              "EVT-val-f01-r311::menu", "EVT-val-f01-r313::menu"]
    order = {item_id: index for index, item_id in enumerate(wanted)}
    chosen = sorted((row for row in rows if row["item_id"] in order),
                    key=lambda row: order[row["item_id"]])
    report["examples"] = [{"item_id": row["item_id"], "diagnosis": row["diagnosis"],
                           "verdict": row["verdict"], "answer": row["answer"],
                           "strict_prompt_only": row["strict_prompt_only"]}
                          for row in chosen]

    with open(ROOT / "outputs/analysis/phase3a/fit/fallback_v3_rows.jsonl", "w",
              encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"item_id": row["item_id"], "diagnosis": row["diagnosis"],
                                     "verdict": row["verdict"],
                                     "strict": row["strict_prompt_only"],
                                     "answer": row["answer"]}, default=str) + "\n")

    out = Path(args.out)
    (out / "fallback_v3.json").write_text(json.dumps(report, indent=2, default=str),
                                          encoding="utf-8")
    print(json.dumps({"rows": report["rows"], "v2_strict": v2_total, "v3": report["v3"],
                      "known_rows": known_rows, "overlay": report["overlay"],
                      "kb_git": report["kb"]["git_status"]}, indent=2, default=str))
    print(f"written: {out / 'fallback_v3.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
