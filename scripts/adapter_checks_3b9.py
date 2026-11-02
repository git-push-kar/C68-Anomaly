"""3B-9 step 2: what the stored adapter texts actually assert, checked against the store.

Four checks over the 242 main-batch answers, the 58 counterfactual ones and the 100 A/B
adapter texts, each split by verdict stratum:

``a`` ranked top    ``ranked_hypotheses[0]`` against the diagnosis that was supplied
``b`` numbers       a quoted number next to a detector's name, against that detector's own value,
                    threshold and ratio, with the comparison direction checked
``c`` tag binding   ``Name (XMEAS_n)`` / ``Name [TAG]`` against the canonical name for that tag
``d`` trend sign    a direction word next to a sensor, against the store's own trend

**These are heuristics, not measurements with a known precision.** Every check prints up to ten
flagged examples so a human can read what a flag means, and the rate of false flags is *not*
measured here — which is why nothing from this file enters the live diagnosis path (§21.4).

The ``a`` check also reconciles the "0 of 242" claim: the ranked-top rule was scoped to the
classified verdict (§5.2 of the transition report, item 1), which suppressed 88 flags on
abstention rows where ranking a candidate is honest. Both readings are reported.

Usage::  python scripts/adapter_checks_3b9.py [--out DIR]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.fallback_v2 import NUMBER, load_jsonl, parse_store  # noqa: E402
from scripts.tags import iter_tags  # noqa: E402

DATASET = ROOT / "outputs/benchmark_v2/dataset"
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
OUT = ROOT / "outputs/analysis/phase3a/fit"
TAG = r"(?:XMEAS|XMV)_\d{1,2}"
SECTION = re.compile(r"^(Diagnosis|Ranked|Reasoning|Narrative):\s*", re.M)
BIND_PAREN = re.compile(rf"([A-Za-z][A-Za-z_0-9 +\-]{{2,45}}?)\s*\(\s*({TAG})\s*\)")
BIND_BRACK = re.compile(rf"([A-Za-z][A-Za-z_0-9 +\-]{{2,45}}?)\s*\[\s*({TAG})\s*\]")
DOWN = re.compile(r"\b(drop|drops|dropped|fall|falls|fell|decline|declines|declined|decreas\w*|"
                  r"downward|lower|lowers|reduc\w*|below)\b", re.I)
UP = re.compile(r"\b(rise|rises|rose|rising|increas\w*|climb\w*|higher|higher-than|above|"
                r"upward|spike\w*|surge\w*|elevat\w*)\b", re.I)
SENT = re.compile(r"(?<=[.!?])\s+")
#: words that carry no identity, so a shared one does not make a name match
GENERIC = {"the", "a", "an", "and", "of", "flow", "temperature", "pressure", "level", "valve",
           "rate", "position", "stream", "reactor", "condenser", "cooling", "water", "feed",
           "column", "product", "compressor", "separator", "stripper", "reflux", "outlet",
           "inlet", "header", "ratio", "speed", "power", "current", "signal", "measurement",
           "transmitter", "sensor", "indicator", "setpoint", "tank", "drum", "purge"}


def sections(text: str) -> Dict[str, str]:
    """Split one A/B text into its labelled sections, tolerantly."""
    body = str(text or "")
    marks = [(m.start(), m.group(1), m.end()) for m in SECTION.finditer(body)]
    if not marks:
        return {"Reasoning": body}
    out: Dict[str, str] = {}
    for index, (_start, label, end) in enumerate(marks):
        stop = marks[index + 1][0] if index + 1 < len(marks) else len(body)
        out[label] = body[end:stop].strip()
    return out


def from_sheet(text: str) -> Dict:
    """A stored adapter answer, from the A/B text."""
    parts = sections(text)
    head = parts.get("Diagnosis", "")
    match = re.match(r"\s*(\S+)\s*\(verdict\s*(\w+)\)", head)
    fault = match.group(1) if match else (head.split()[0] if head.split() else "")
    verdict = match.group(2) if match else "classified"
    ranked = [token.strip(" ,.;") for token in re.split(r",", parts.get("Ranked", ""))
              if token.strip(" ,.;")]
    return {"fault_id": fault, "verdict": verdict, "ranked_hypotheses": ranked,
            "physical_reasoning": parts.get("Reasoning", ""),
            "narrative": parts.get("Narrative", "")}


def canonical_names() -> Dict[str, collections.Counter]:
    """Tag -> the names the validation prompts themselves use for it."""
    names: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for item in load_jsonl(DATASET / "val.jsonl"):
        for match in re.finditer(rf"- (?P<name>.+?) \[(?P<tag>{TAG}),", item.get("prompt") or ""):
            names[match.group("tag")][match.group("name").strip()] += 1
    return names


def identity_tokens(name: str) -> set:
    tokens = set(re.findall(r"[a-z0-9]+", str(name).lower().replace("_", " ")))
    return {token for token in tokens if token not in GENERIC and len(token) > 2}


def numbers_in(text: str) -> List[str]:
    return NUMBER.findall(str(text))


def variants(value: str) -> set:
    """The spellings a stored quantity may be printed in."""
    out = {str(value)}
    try:
        number = float(value)
    except (TypeError, ValueError):
        return out
    out.update({f"{number:.4f}", f"{number:.3f}", f"{number:.2f}", f"{number:.1f}",
                f"{number:.0f}", str(abs(number))})
    return out


def check_ranked(answer: Dict) -> Dict:
    ranked = answer.get("ranked_hypotheses") or []
    fault = str(answer.get("fault_id") or "")
    verdict = str(answer.get("verdict") or "")
    top = str(ranked[0]) if ranked else ""
    mismatch = bool(ranked) and top != fault
    return {"verdict": verdict, "fault_id": fault, "ranked_top": top,
            "ranked": ranked, "mismatch_any_verdict": mismatch,
            "mismatch_classified_only": mismatch and verdict == "classified",
            "abstention_label_ranked": any(str(entry).startswith(("UNCLASSIFIED", "NORMAL"))
                                           for entry in ranked)}


def check_numbers(answer: Dict, store: Dict) -> List[Dict]:
    """A number printed next to a detector's name must be one of that detector's quantities."""
    text = " ".join(str(answer.get(key) or "") for key in ("physical_reasoning", "narrative"))
    flags: List[Dict] = []
    for detector in store["detectors"]:
        label = str(detector["name"])
        key = "LSTM" if "lstm" in label.lower() else (
            "CVA" if "cva" in label.lower() else ("DPCA" if "dpca" in label.lower() else label))
        allowed = set()
        for field in ("value", "threshold", "ratio"):
            allowed |= variants(detector.get(field) or "")
        for sentence in re.split(r";|(?<=[.!?])\s+", text):
            if key.lower() not in sentence.lower():
                continue
            quoted = [number for number in numbers_in(sentence) if number not in allowed]
            if quoted:
                flags.append({"kind": "number_not_the_detectors", "detector": key,
                              "quoted": quoted[:4], "allowed": sorted(allowed)[:6],
                              "sentence": sentence[:220]})
            try:
                value = float(detector["value"]); threshold = float(detector["threshold"])
            except (TypeError, ValueError, KeyError):
                continue
            printed_above = bool(re.search(r"\b(above|exceeds|exceeded|over|greater)\b|>", sentence,
                                          re.I))
            printed_below = bool(re.search(r"\b(below|under|beneath|less)\b|<", sentence, re.I))
            if printed_above and value <= threshold:
                flags.append({"kind": "direction_above_but_value_below_threshold", "detector": key,
                              "value": value, "threshold": threshold, "sentence": sentence[:220]})
            if printed_below and value >= threshold:
                flags.append({"kind": "direction_below_but_value_above_threshold", "detector": key,
                              "value": value, "threshold": threshold, "sentence": sentence[:220]})
    return flags


def check_bindings(answer: Dict, canon: Dict[str, collections.Counter]) -> List[Dict]:
    text = " ".join(str(answer.get(key) or "") for key in ("physical_reasoning", "narrative"))
    flags: List[Dict] = []
    for pattern in (BIND_PAREN, BIND_BRACK):
        for match in pattern.finditer(text):
            claimed, tag = match.group(1).strip(), match.group(2)
            if tag not in canon:
                flags.append({"kind": "tag_unknown", "claimed": claimed, "tag": tag})
                continue
            best = canon[tag].most_common(1)[0][0]
            claimed_tokens, best_tokens = identity_tokens(claimed), identity_tokens(best)
            if not claimed_tokens or not best_tokens:
                flags.append({"kind": "name_unverifiable", "claimed": claimed, "tag": tag,
                              "canonical": best})
                continue
            if not claimed_tokens & best_tokens:
                flags.append({"kind": "name_does_not_match_tag", "claimed": claimed, "tag": tag,
                              "canonical": best, "context": text[max(0, match.start() - 60):
                                                                match.end() + 40]})
    return flags


def check_trends(answer: Dict, store: Dict) -> List[Dict]:
    text = " ".join(str(answer.get(key) or "") for key in ("physical_reasoning", "narrative"))
    flags: List[Dict] = []
    for sensor in store["sensors"]:
        for sentence in SENT.split(text):
            if sensor["name"].lower() not in sentence.lower() and sensor["tag"] not in sentence:
                continue
            trend = str(sensor.get("trend") or "").lower()
            down, up = bool(DOWN.search(sentence)), bool(UP.search(sentence))
            if trend.startswith("increas") and down and not up:
                flags.append({"kind": "text_says_down_store_says_up", "sensor": sensor["name"],
                              "z": sensor["z"], "sentence": sentence[:220]})
            if trend.startswith("decreas") and up and not down:
                flags.append({"kind": "text_says_up_store_says_down", "sensor": sensor["name"],
                              "z": sensor["z"], "sentence": sentence[:220]})
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    val = {row["item_id"]: row for row in load_jsonl(DATASET / "val.jsonl")}
    canon = canonical_names()
    key = {row["item_id"]: row for row in csv.DictReader(
        (OUT / "audit_ab_100_v2_key.csv").open(encoding="utf-8-sig"))}
    sheet = list(csv.DictReader((OUT / "audit_ab_100_v2.csv").open(encoding="utf-8-sig")))

    populations: Dict[str, List[Dict]] = {"main_242": [], "counterfactual_58": [], "ab_100": []}
    for row in load_jsonl(HYBRID / "generation_rows.jsonl"):
        answer = row["answer"] if isinstance(row["answer"], dict) else json.loads(row["answer"])
        item = val.get(row["item_id"]) or {}
        store = parse_store(item.get("prompt") or "")
        entry = {"item_id": row["item_id"], "family": row.get("family"), "store": store,
                 "answer": answer}
        populations["counterfactual_58" if row.get("kind") == "counterfactual"
                    else "main_242"].append(entry)

    for row in sheet:
        side = key.get(row["item_id"], {}).get("A", "fallback")
        other = "B" if side == "adapter" else "A"
        text = row.get(f"text_{other}") or ""
        item = val.get(row["item_id"]) or {}
        populations["ab_100"].append({"item_id": row["item_id"], "family": row["family"],
                                      "store": parse_store(item.get("prompt") or ""),
                                      "answer": from_sheet(text)})

    report: Dict[str, Dict] = {}
    for name, rows in populations.items():
        per: Dict[str, Dict] = {}
        for stratum, selector in (("all", lambda r: True),
                                  ("classified", lambda r: r["answer"].get("verdict") == "classified"),
                                  ("abstention", lambda r: r["answer"].get("verdict") != "classified")):
            subset = [row for row in rows if selector(row)]
            ranked = [check_ranked(row["answer"]) for row in subset]
            flags_b, flags_c, flags_d = [], [], []
            for row in subset:
                for flag in check_numbers(row["answer"], row["store"]):
                    flags_b.append({"item_id": row["item_id"], **flag})
                for flag in check_bindings(row["answer"], canon):
                    flags_c.append({"item_id": row["item_id"], **flag})
                for flag in check_trends(row["answer"], row["store"]):
                    flags_d.append({"item_id": row["item_id"], **flag})
            per[stratum] = {
                "rows": len(subset),
                "a_ranked_top_mismatch_any_verdict": sum(1 for r in ranked
                                                         if r["mismatch_any_verdict"]),
                "a_ranked_top_mismatch_classified_only": sum(1 for r in ranked
                                                             if r["mismatch_classified_only"]),
                "a_rows_ranking_an_abstention_label": sum(1 for r in ranked
                                                          if r["abstention_label_ranked"]),
                "b_flags": len(flags_b), "b_rows": len({f["item_id"] for f in flags_b}),
                "c_flags": len(flags_c), "c_rows": len({f["item_id"] for f in flags_c}),
                "d_flags": len(flags_d), "d_rows": len({f["item_id"] for f in flags_d}),
                "a_examples": ranked[:10] if stratum == "all" else None,
                "b_examples": flags_b[:10], "c_examples": flags_c[:10], "d_examples": flags_d[:10],
                "b_kinds": dict(collections.Counter(f["kind"] for f in flags_b)),
                "c_kinds": dict(collections.Counter(f["kind"] for f in flags_c)),
                "d_kinds": dict(collections.Counter(f["kind"] for f in flags_d)),
            }
        report[name] = per

    # the 13 rows: A/B classified rows whose ranked list does not start with the supplied diagnosis
    ranked_conflicts = [{"item_id": row["item_id"], "family": row["family"],
                         **check_ranked(row["answer"])}
                        for row in populations["ab_100"]
                        if check_ranked(row["answer"])["mismatch_classified_only"]]
    report["a_ranked_top_conflicts_in_ab"] = ranked_conflicts
    report["a_reconciliation"] = {
        "as_specified_any_verdict_flags": report["main_242"]["all"][
            "a_ranked_top_mismatch_any_verdict"],
        "of_which_abstention_rows": report["main_242"]["abstention"][
            "a_ranked_top_mismatch_any_verdict"],
        "classified_stratum_flags": report["main_242"]["classified"][
            "a_ranked_top_mismatch_classified_only"],
        "rule_that_dropped_the_check": "§16.4's ranked-top rule was scoped to the classified "
                                       "verdict in 3B-3 §5.2 item 1: on an abstention row the "
                                       "adapter ranks candidates by design, so ranking one is not "
                                       "a contradiction and the 88 flags it produced were false.",
        "note": "the two counts differ only by verdict scoping; both are printed so the "
                "'0 of 242' claim is reproducible on the reading that produced it.",
    }

    out = Path(args.out)
    (out / "adapter_checks_3b9.json").write_text(json.dumps(report, indent=2, default=str),
                                                 encoding="utf-8")
    summary = {name: {stratum: {key: value for key, value in per.items()
                                if not key.endswith("examples")}
                      for stratum, per in strata.items()}
               for name, strata in report.items() if isinstance(strata, dict)
               and name.startswith(("main", "counter", "ab_"))}
    summary["ab_conflicts"] = len(ranked_conflicts)
    summary["a_reconciliation"] = report["a_reconciliation"]
    print(json.dumps(summary, indent=2, default=str))
    print(f"written: {out / 'adapter_checks_3b9.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
