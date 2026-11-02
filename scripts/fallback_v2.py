"""The deterministic explainer, second version: grounded from this event's store only.

§19.2 makes the report path a choice between two prose sources. 3B-7 measured the first
candidate — `llm/hybrid/generate.py::fallback_answer` — and found that its `physical_reasoning`
quotes the knowledge base's mechanism text verbatim, which carries the KB **example** event's
numbers: 1,039 numbers over 268 / 834 rows. A comparator that fails §19.2's own grounding
criterion on a third of rows cannot decide a grounding contest, so PROTOCOL §20.1 replaced it
**before any score existed** with the version in this file.

Four changes (§20.6), all deterministic, CPU, no model, no teacher:

1. **the KB example's numbers and tags never reach the output.** The mechanism text is
   rewritten by rule: an equipment id or stream index becomes the role the KB itself gives it
   (`R-101` -> "the reactor", `Stream 4` -> "the combined feed header"), a numeric span this
   event's store does not contain becomes a qualitative phrase whose direction is read off the
   two numbers rather than asserted, and a clause that still holds an unquotable number or tag
   after that is **dropped**. The invariant is then re-checked on the finished answer, so
   "never quotes another event's numbers" is a property of the output, not a hope about the
   regexes.
2. **an item-specific evidence sentence** built from the store: detector ratios, the top
   sensors with z and trend, and the onset order.
3. **a differential line** from the classifier's top-2 and their calibrated probabilities.
   Those probabilities are *supplied* quantities and not prompt text, so grounding is counted
   two ways (§20.6) — against the prompt alone, which is the same rule that measured v1, and
   against the prompt plus the supplied classifier values — and the difference is reported as
   its own category instead of being netted out.
4. **the narrative names the top sensor and the onset.**

CPU only. No model, no generation, no test component.

Usage::

    python scripts/fallback_v2.py                 # the 834-row v1-vs-v2 comparison
    python scripts/fallback_v2.py --inspect       # print the 22 scrubbed KB mechanisms
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.hybrid.generate import fallback_answer  # noqa: E402
from scripts.tags import iter_tags  # noqa: E402

DATASET = ROOT / "outputs/benchmark_v2/dataset"
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
OUT = ROOT / "outputs/analysis/phase3a/fit"
SEED = 42

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
DETECTOR = re.compile(
    r"^- (?P<name>.+?) (?P<value>-?[\d.]+) \(threshold (?P<threshold>-?[\d.]+), "
    r"ratio (?P<ratio>-?[\d.]+)x, alarmed (?P<alarmed>\w+)\)$")
SENSOR = re.compile(
    r"^- (?P<name>.+?) \[(?P<tag>(?:XMEAS|XMV)_\d+), [^\]]*\]: z=(?P<z>-?[\d.]+), "
    r"deviation (?P<dev>-?[\d.]+)%, trend (?P<trend>\w+)$")
ONSET = re.compile(
    r"^(?P<rank>\d+)\. (?P<name>.+?) \[(?P<tag>(?:XMEAS|XMV)_\d+), [^\]]*\] "
    r"at (?P<at>[+-]?[\d.]+) min$")
RESIDUAL = re.compile(r"^- (?P<tag>(?:XMEAS|XMV)_\d+) \[[^\]]*\]: residual share (?P<share>[\d.]+)$")
LAG = re.compile(r"^Dominant lag (?P<lag>\d+) window\(s\), slow-drift ratio (?P<ratio>[\d.]+)$")
EVENT = re.compile(r"^Event: (?P<event>\S+)$")

#: Equipment ids the KB itself glosses, mapped to the role it gives them. Any other id is
#: dropped by clause rather than guessed at.
EQUIPMENT = {
    "R-101": "the reactor",
    "R-102": "the stripper reboiler",
    "E-102": "the overhead condenser",
    "V-102": "the separator",
    "K-101": "the recycle compressor",
}
#: Compound names first, so `condenser E-102` cannot become `condenser the overhead condenser`.
EQUIPMENT_COMPOUND = (
    (r"E-102\s+overhead condenser", "the overhead condenser"),
    (r"\bcondenser\s+E-102\b", "the overhead condenser"),
    (r"\bSeparator\s+V-102\b", "the separator"),
    (r"\bseparator\s+V-102\b", "the separator"),
    (r"\breactor\s+R-101\b", "the reactor"),
    (r"\bcompressor\s+K-101\b", "the recycle compressor"),
)
#: Stream indices the KB names by role (fault 6 "A feed loss (stream 1)", fault 3 "Stream 2
#: feed header ... feed D").
STREAMS = {1: "A feed line", 2: "D feed header", 4: "combined feed header", 6: "recycle header"}
UNIT = r"(?:kscfm|kscmh|kPa|kg/h|m3/h|mol%|ppm|rpm|°C|deg C|degC|%)(?![A-Za-z0-9_])"
#: A range of two numbers: "from 45.0 kscfm to ~38 kscfm", "+3 to +5 °C", ">50-80%".
#: The lookbehind keeps it out of a tag (`XMV_51`) and out of an equipment id (`R-101`).
RANGE = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<prefix>\bfrom\b|\bbetween\b)?\s*"
    rf"(?P<sign1>[+-><=~]*)\s*(?P<n1>-?\d+(?:\.\d+)?)\s*(?P<u1>{UNIT})?\s*"
    rf"(?:-|–|to|toward|towards)\s*(?:the\s+)?(?P<sign2>[+-><=~]*)\s*"
    rf"(?P<n2>-?\d+(?:\.\d+)?)\s*(?P<u2>{UNIT})?",
    re.IGNORECASE)
SINGLE = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<sign>[+-><=~]*)\s*(?P<n>-?\d+(?:\.\d+)?)(?![0-9])\s*"
    rf"(?P<u>{UNIT})?", re.IGNORECASE)
TAG_ALIAS = re.compile(r"\(\s*(?:XMV|XMEAS)[ _]\d{1,2}\s*\)", re.IGNORECASE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
CLAUSE_SPLIT = re.compile(r"\s*;\s*|,\s+")
PAREN = re.compile(r"\([^)]*\)")


# --------------------------------------------------------------------------- the store
def parse_store(prompt: str) -> Dict:
    """Everything the fallback may quote, read off the page it answers.

    Parsing rather than reading a stored event is deliberate: the grounding claim is about what
    a reader of the prompt can see, so the store *is* the prompt.
    """
    detectors, sensors, onset, residuals = [], [], [], []
    lag = None
    event = None
    for line in str(prompt or "").splitlines():
        line = line.rstrip()
        match = EVENT.match(line)
        if match:
            event = match.group("event")
            continue
        match = DETECTOR.match(line)
        if match:
            detectors.append({key: match.group(key) for key in
                              ("name", "value", "threshold", "ratio", "alarmed")})
            continue
        match = SENSOR.match(line)
        if match:
            sensors.append({key: match.group(key) for key in ("name", "tag", "z", "dev", "trend")})
            continue
        match = ONSET.match(line)
        if match:
            onset.append({"name": match.group("name"), "tag": match.group("tag"),
                          "at": match.group("at"), "rank": int(match.group("rank"))})
            continue
        match = RESIDUAL.match(line)
        if match:
            residuals.append({"tag": match.group("tag"), "share": match.group("share")})
            continue
        match = LAG.match(line)
        if match:
            lag = {"lag": match.group("lag"), "ratio": match.group("ratio")}
    numbers = set(NUMBER.findall(str(prompt or "")))
    return {"detectors": detectors, "sensors": sensors, "onset": onset,
            "residuals": residuals, "lag": lag, "event": event,
            "numbers": numbers, "tags": set(iter_tags(str(prompt or ""))),
            "text": str(prompt or "")}


def quotable(number: str, store: Dict) -> bool:
    """True when a number token appears in this event's store.

    The test is the one v1 was measured with — the token appears in the prompt text — so the
    two versions are compared on identical terms rather than on two different notions of
    "present". It is a substring test on purpose: `0.4` is quotable for an event whose page
    prints `0.40`, because it is the same quantity to the same precision, and the v1 measurement
    counted it that way.
    """
    return str(number) in store["text"]


# --------------------------------------------------------------------------- scrubbing
def _direction(first: str, second: str) -> str:
    """``lower``/``higher`` read off the two numbers, so no direction is asserted."""
    try:
        a, b = float(first), float(second)
    except ValueError:
        return "different"
    if b < a:
        return "lower"
    if b > a:
        return "higher"
    return "different"


def _range_phrase(match: "re.Match", before: str) -> str:
    """A qualitative phrase for one range, with its direction read off its own numbers."""
    direction = _direction(match.group("n1"), match.group("n2"))
    if before.rstrip().endswith("by") or (match.group("prefix") or "").lower() == "between":
        return f"a marked margin toward a {direction} level"
    return f"from its nominal level toward a {direction} level"


def _single_phrase(match: "re.Match", before: str) -> str:
    """A qualitative phrase for one bare quantity."""
    unit = (match.group("u") or "").lower()
    sign = (match.group("sign") or "").lower()
    if unit == "%":
        return "a marked margin" if before.rstrip().endswith("by") else ""
    if sign.startswith("-"):
        return "a lower level"
    if sign.startswith("+"):
        return "a raised level"
    if before.rstrip().endswith(("by", "at", "to", "toward", "towards", "near")):
        return "a new level"
    return "a nominal level"


def scrub(text: str, store: Dict) -> Tuple[str, List[str]]:
    """Rewrite ``text`` so that no number or tag outside ``store`` survives.

    Rules, in order, each deterministic:

    * a bracketed tag alias that only repeats the tag it follows is dropped
      (``XMV_51 (XMV 10)`` prints as ``XMV_51``);
    * a parenthetical holding an unquotable number is dropped whole;
    * an equipment id or stream index becomes the role the KB gives it;
    * a range or a bare quantity that is not in the store becomes a qualitative phrase;
    * a clause that *still* holds an unquotable number or tag is dropped.

    Returns the scrubbed text and the list of repairs, so the rewrite is auditable.
    """
    repairs: List[str] = []
    body = TAG_ALIAS.sub("", str(text or ""))

    def parenthetical(match: "re.Match") -> str:
        inner = match.group(0)
        if any(not quotable(number, store) for number in NUMBER.findall(inner)):
            repairs.append(f"dropped parenthetical: {inner.strip()}")
            return ""
        return inner

    body = PAREN.sub(parenthetical, body)

    for pattern, role in EQUIPMENT_COMPOUND:
        body = re.sub(pattern, role, body)
    for identifier, role in EQUIPMENT.items():
        if identifier in body:
            body = body.replace(identifier, role)
            repairs.append(f"{identifier} -> {role}")

    def stream_sub(match: "re.Match") -> str:
        role = STREAMS.get(int(match.group(1)), "feed header")
        repairs.append(f"stream {match.group(1)} -> {role}")
        return f"the {role}"

    body = re.sub(r"\b(?:the\s+)?Stream\s+(\d{1,2})(?:\s+feed header)?\b", stream_sub, body)
    for _ in range(3):
        body = re.sub(r"\bthe\s+the\b", "the", body, flags=re.IGNORECASE)
        body = re.sub(r"\b(?:reactor|separator|condenser|compressor)\s+the\s+"
                      r"(?:reactor|separator|overhead condenser|recycle compressor)\b",
                      lambda m: "the " + m.group(1) if False else re.sub(
                          r"^(?:reactor|separator|condenser|compressor)\s+", "", m.group(0)),
                      body, flags=re.IGNORECASE)

    for _ in range(3):
        body = re.sub(r"the overhead condenser\s+overhead condenser", "the overhead "
                      "condenser", body, flags=re.IGNORECASE)
        body = re.sub(r"the (reactor|separator|recycle compressor)\s+", r"the ", body,
                      flags=re.IGNORECASE)

    def range_sub(match: "re.Match") -> str:
        numbers = [match.group("n1"), match.group("n2")]
        if all(quotable(number, store) for number in numbers):
            return match.group(0)
        repairs.append(f"range -> phrase: {match.group(0).strip()}")
        return " " + _range_phrase(match, body[:match.start()])

    body = RANGE.sub(range_sub, body)

    def single_sub(match: "re.Match") -> str:
        if quotable(match.group("n"), store):
            return match.group(0)
        repairs.append(f"quantity -> phrase: {match.group(0).strip()}")
        return " " + _single_phrase(match, body[:match.start()])

    body = SINGLE.sub(single_sub, body)

    # clause-level safety net: nothing unquotable may survive. Parentheticals are held out of the
    # split, so a boundary inside `(A+C+D->G and A+C+E->H)` cannot strand a bracket.
    held: List[str] = []

    def hold(match: "re.Match") -> str:
        held.append(match.group(0))
        return "PAREN" + "A" * (len(held) - 1) + "Z"

    body = PAREN.sub(hold, body)
    kept: List[str] = []
    for sentence in SENTENCE_SPLIT.split(body):
        clauses, lost = [], False
        for clause in CLAUSE_SPLIT.split(sentence):
            clause = clause.strip(" ,.;")
            if not clause:
                continue
            bad = [number for number in NUMBER.findall(clause) if not quotable(number, store)]
            bad_tags = [tag for tag in iter_tags(clause) if tag not in store["tags"]]
            if bad or bad_tags:
                repairs.append(f"dropped clause: {clause}")
                lost = True
                continue
            if len(clause.split()) < 2:
                repairs.append(f"dropped stub clause: {clause}")
                lost = True
                continue
            clauses.append(clause)
        if not clauses:
            continue
        sentence_text = ", ".join(clauses)
        # a sentence that lost a clause and kept only a fragment ("With reduced reactant
        # inventory.") says less than nothing: it is dropped rather than printed as a stub.
        if lost and len(sentence_text.split()) < 6:
            repairs.append(f"dropped fragment: {sentence_text}")
            continue
        kept.append(sentence_text + ".")
    body = " ".join(kept)
    for index in reversed(range(len(held))):
        body = body.replace("PAREN" + "A" * index + "Z", held[index])
    body = re.sub(r"\s+", " ", body)
    body = re.sub(r"\s+([,.;:!?])", r"\1", body)
    body = re.sub(r"\b(from|toward|to|at|by|of)\s+\1\b", r"\1", body, flags=re.IGNORECASE)
    body = re.sub(r"\(\s*\)", "", body)
    body = re.sub(r"([.!?])\s*\.", r"\1", body)
    body = re.sub(r",\s*,", ",", body)
    body = re.sub(r"\s{2,}", " ", body).strip(" ,;")
    body = ". ".join(part[:1].upper() + part[1:] if part else part
                     for part in body.split(". "))
    if body and not body.endswith((".", "!")):
        body += "."
    return body, repairs


# --------------------------------------------------------------------------- the answer
def _top_two(probs: Dict[str, float], diagnosis: str) -> List[Tuple[str, float]]:
    """The classifier's top-2 named faults, with the supplied diagnosis first when it has one."""
    ranked = sorted(((label, float(value)) for label, value in probs.items()
                     if label.startswith("FAULT")), key=lambda item: (-item[1], item[0]))
    if not ranked:
        return []
    if diagnosis in probs:
        ranked = [(diagnosis, float(probs[diagnosis]))] + [item for item in ranked
                                                          if item[0] != diagnosis]
    return ranked[:2]


def evidence_sentence(store: Dict) -> str:
    """The item-specific sentence: detector ratios, top sensors with z and trend, onset order."""
    parts: List[str] = []
    ratios = [(d["name"], d["ratio"]) for d in store["detectors"] if d.get("ratio")]
    if ratios:
        parts.append("detector ratios are " + " and ".join(f"{name} {ratio}x"
                                                          for name, ratio in ratios[:2]))
    sensors = store["sensors"][:2]
    if sensors:
        parts.append("the largest deviations are " + " and ".join(
            f"{s['name']} ({s['tag']}, z={s['z']}, {s['trend']})" for s in sensors))
    onset = sorted(store["onset"], key=lambda entry: entry["rank"])
    if len(onset) >= 2:
        parts.append(f"onset starts with {onset[0]['name']} ({onset[0]['tag']}) at "
                     f"{onset[0]['at']} min and {onset[1]['name']} follows at "
                     f"{onset[1]['at']} min")
    elif onset:
        parts.append(f"onset starts with {onset[0]['name']} ({onset[0]['tag']}) at "
                     f"{onset[0]['at']} min")
    if store.get("lag"):
        parts.append(f"the dominant lag is {store['lag']['lag']} window(s) at a slow-drift "
                     f"ratio of {store['lag']['ratio']}")
    if not parts:
        return "No detector or sensor evidence was readable from this event."
    return "Evidence for this event: " + "; ".join(parts) + "."


def differential_sentence(diagnosis: str, probs: Dict[str, float], verdict: str) -> str:
    """Top-2 and their calibrated probabilities, read off the classifier's own output."""
    top = _top_two(probs, diagnosis)
    if verdict != "classified" or not top:
        return ("Differential: the classifier places no listed fault above its abstention "
                "threshold, so no alternative is ranked against a named diagnosis.")
    if len(top) == 1:
        return (f"Differential: the classifier gives {top[0][0]} a calibrated probability of "
                f"{top[0][1]:.2f} and no other listed fault above zero.")
    (first, p1), (second, p2) = top
    return (f"Differential: the classifier keeps {first} at a calibrated probability of {p1:.2f} "
            f"and {second} at {p2:.2f}; no other entry on the menu is ranked above {second}.")


def narrative_sentence(diagnosis: str, store: Dict, action: str) -> str:
    """Names the top sensor and the onset, then the KB's action (scrubbed)."""
    sensors = store["sensors"][:1]
    onset = sorted(store["onset"], key=lambda entry: entry["rank"])[:1]
    naming = ""
    if sensors:
        sensor = sensors[0]
        naming = (f" {sensor['name']} ({sensor['tag']}) deviates most, at z={sensor['z']} and "
                  f"{sensor['trend']}")
        if onset:
            naming += f", and it is first to move at onset {onset[0]['at']} min"
        naming += "."
    head = (f"Diagnosis {diagnosis} is the best fit for this event."
            if str(diagnosis).startswith("FAULT")
            else f"No listed fault is asserted for this event ({diagnosis}).")
    tail = f" Recommended: {action}" if action else ""
    return (head + naming + tail).strip()


def fallback_v2(diagnosis: str, verdict: str, menu_ids: Sequence[str], store: Dict,
                probs: Optional[Dict[str, float]] = None) -> Dict:
    """The whole answer, deterministic, from the store and the classifier's own numbers."""
    from llm.benchmark.menu import lookalikes
    from scripts.fault_knowledge import FAULT_KNOWLEDGE

    probs = probs or {}
    number = None
    match = re.match(r"FAULT_(\d{1,2})", str(diagnosis))
    if match:
        number = int(match.group(1))
    ranked: List[str] = [str(diagnosis)]
    if number is not None:
        for other in lookalikes(number, limit=2):
            label = f"FAULT_{int(other):02d}"
            if label in set(menu_ids or []) and label not in ranked:
                ranked.append(label)
    entry = FAULT_KNOWLEDGE.get(number or -1) or {}
    mechanism, repairs = scrub(str(entry.get("physical_mechanism") or ""), store)
    action, action_repairs = scrub(str(entry.get("recommended_action") or ""), store)
    if not mechanism:
        mechanism = ("the stored evidence is consistent with this fault family and no other "
                     "listed entry explains the pattern")
    mechanism = mechanism[:1].upper() + mechanism[1:]
    reasoning = " ".join(part for part in (
        evidence_sentence(store),
        mechanism.strip(),
        differential_sentence(str(diagnosis), probs, str(verdict)),
    ) if part)
    return {
        "fault_id": str(diagnosis),
        "verdict": str(verdict),
        "ranked_hypotheses": ranked,
        "physical_reasoning": re.sub(r"\s+", " ", reasoning).strip(),
        "narrative": re.sub(r"\s+", " ", narrative_sentence(str(diagnosis), store,
                                                            action.strip())).strip(),
        "_repairs": repairs + action_repairs,
    }


# --------------------------------------------------------------------------- measurement
def grounding_text(text: str, store: Dict, supplied: Dict[str, str]) -> Dict:
    """Numbers and tags not in the prompt, split by whether the classifier supplied them."""
    numbers = [number for number in NUMBER.findall(text) if not quotable(number, store)]
    from_supplied = [number for number in numbers if number in supplied]
    bad_tags = [tag for tag in iter_tags(text) if tag not in store["tags"]]
    return {"numbers_not_in_prompt": sorted(set(numbers)),
            "numbers_from_supplied_classifier": sorted(set(from_supplied)),
            "tags_not_in_prompt": sorted(set(bad_tags)),
            "violations": len(set(numbers)) + len(set(bad_tags))}


def schema_ok(answer: Dict) -> Dict:
    keys = ("fault_id", "verdict", "ranked_hypotheses", "physical_reasoning", "narrative")
    missing = [key for key in keys if key not in answer]
    types_ok = (isinstance(answer.get("fault_id"), str) and isinstance(answer.get("verdict"), str)
                and isinstance(answer.get("ranked_hypotheses"), list)
                and all(isinstance(item, str) for item in answer.get("ranked_hypotheses") or [])
                and isinstance(answer.get("physical_reasoning"), str)
                and isinstance(answer.get("narrative"), str))
    return {"valid": not missing and types_ok, "missing": missing}


def words(text: str) -> int:
    return len(str(text or "").split())


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def supplied_values(probs: Dict[str, float], top: Sequence[Tuple[str, float]]) -> Dict[str, str]:
    """The classifier's quantity tokens in the form the differential prints them."""
    values = {f"{float(value):.2f}" for value in probs.values()}
    values.update(f"{float(value):.2f}" for _label, value in top)
    return {value: value for value in values}


def identical_prose(rows: List[Dict], version: str) -> Dict:
    """How much of the reasoning is byte-identical inside a diagnosis group."""
    groups: Dict[str, List[str]] = {}
    for row in rows:
        groups.setdefault(str(row["diagnosis"]), []).append(
            row[version]["answer"]["physical_reasoning"])
    repeated = sum(1 for texts in groups.values() for text in texts
                   if texts.count(text) > 1)
    distinct = {label: len(set(texts)) for label, texts in groups.items()}
    return {"rows_whose_reasoning_repeats_inside_its_group": repeated,
            "rows": len(rows),
            "rate": round(repeated / len(rows), 4) if rows else 0.0,
            "distinct_texts_per_diagnosis": dict(sorted(distinct.items()))}


def compare(rows: List[Dict]) -> Dict:
    """v1 and v2 side by side, measured with one function each."""
    out: Dict[str, Dict] = {}
    for version in ("v1", "v2"):
        total = len(rows)
        out[version] = {
            "rows": total,
            "schema_valid": sum(1 for row in rows if row[version]["schema"]["valid"]),
            "rows_with_any_violation": sum(1 for row in rows
                                           if row[version]["grounding"]["violations"]),
            "violations_by_kind": {
                "numbers": sum(len(row[version]["grounding"]["numbers_not_in_prompt"])
                               for row in rows),
                "tags": sum(len(row[version]["grounding"]["tags_not_in_prompt"])
                            for row in rows),
            },
            "rows_whose_only_violations_are_supplied_numbers": sum(
                1 for row in rows
                if row[version]["grounding"]["numbers_from_supplied_classifier"]
                and not row[version]["grounding"]["tags_not_in_prompt"]
                and set(row[version]["grounding"]["numbers_from_supplied_classifier"])
                == set(row[version]["grounding"]["numbers_not_in_prompt"])),
            "mean_words": {
                "physical_reasoning": round(sum(
                    words(row[version]["answer"]["physical_reasoning"]) for row in rows) / total, 1),
                "narrative": round(sum(words(row[version]["answer"]["narrative"])
                                       for row in rows) / total, 1),
            },
            "default_mechanism_rows": sum(1 for row in rows
                                          if row[version]["default_mechanism"]),
            "identical_prose": identical_prose(rows, version),
        }
    return out


#: The ten items 3B-7 printed for v1 (3 abstention, 2 NORMAL, 5 faults), reused so the two
#: tables are the same ten events.
WANTED_EXAMPLES = (
    "EVT-val-f01-r310::openset", "EVT-val-f01-r313::openset", "EVT-val-f01-r317::openset",
    "EVT-val-f03-r302::menu", "EVT-val-f03-r302::openset",
    "EVT-val-f01-r306::menu", "EVT-val-f01-r307::menu", "EVT-val-f01-r310::menu",
    "EVT-val-f01-r311::menu", "EVT-val-f01-r313::menu",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--inspect", action="store_true",
                        help="print the scrubbed KB mechanism text and exit")
    args = parser.parse_args()

    if args.inspect:
        from scripts.fault_knowledge import FAULT_KNOWLEDGE
        empty = parse_store("")
        for number in sorted(FAULT_KNOWLEDGE):
            text, repairs = scrub(str(FAULT_KNOWLEDGE[number].get("physical_mechanism") or ""),
                                  empty)
            print(f"--- FAULT_{number:02d} ---\n{text}")
            if repairs:
                print("    repairs:", "; ".join(repairs))
        return 0

    val = {row["item_id"]: row for row in load_jsonl(DATASET / "val.jsonl")}
    predicted = load_jsonl(HYBRID / "crossfit_val_rows.jsonl")

    rows: List[Dict] = []
    leaks: List[Dict] = []
    for prediction in predicted:
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

        v1_evidence = {"top_anomalous_sensors": [
            {"display_name": sensor["name"], "name": sensor["name"],
             "z_score": float(sensor["z"]), "deviation_percent": float(sensor["dev"])}
            for sensor in store["sensors"]]}
        v1 = fallback_answer(diagnosis, verdict, item.get("menu_ids") or [], v1_evidence)
        v2 = fallback_v2(diagnosis, verdict, item.get("menu_ids") or [], store, probs)
        top = _top_two(probs, diagnosis)
        supplied = supplied_values(probs, top)

        entries = {}
        for version, answer in (("v1", v1), ("v2", v2)):
            text = " ".join(str(answer.get(key) or "") for key in
                            ("physical_reasoning", "narrative", "fault_id", "verdict"))
            grounding = grounding_text(text, store, supplied)
            entries[version] = {
                "answer": answer,
                "schema": schema_ok(answer),
                "grounding": grounding,
                "default_mechanism": not str(diagnosis).startswith("FAULT"),
            }
            if version == "v2" and grounding["tags_not_in_prompt"]:
                leaks.append({"item_id": prediction["item_id"],
                              "tags": grounding["tags_not_in_prompt"]})
        rows.append({"item_id": prediction["item_id"], "diagnosis": diagnosis,
                     "verdict": verdict, "family": item.get("variant"),
                     "unobservable": bool(item.get("unobservable")),
                     "v1": entries["v1"], "v2": entries["v2"], "store": {
                         "sensors": store["sensors"][:2], "onset": store["onset"][:2],
                         "detectors": store["detectors"]}})

    report = {
        "rows": len(rows),
        "v1_vs_v2": compare(rows),
        "v2_rows_with_no_violation_at_all": sum(
            1 for row in rows if not row["v2"]["grounding"]["violations"]),
        "v2_tag_leaks": leaks,
        "examples_of_numbers_not_in_prompt_v2": sorted({
            number for row in rows
            for number in row["v2"]["grounding"]["numbers_not_in_prompt"]})[:20],
    }

    chosen = [row for row in rows if row["item_id"] in WANTED_EXAMPLES]
    order = {item_id: index for index, item_id in enumerate(WANTED_EXAMPLES)}
    chosen.sort(key=lambda row: order[row["item_id"]])
    report["examples"] = [{
        "item_id": row["item_id"], "diagnosis": row["diagnosis"], "verdict": row["verdict"],
        "family": row["family"], "store": row["store"],
        "v1": row["v1"]["answer"], "v2": row["v2"]["answer"],
        "v1_violations": row["v1"]["grounding"], "v2_violations": row["v2"]["grounding"],
    } for row in chosen]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fallback_v2.json").write_text(json.dumps(report, indent=2, default=str),
                                          encoding="utf-8")
    summary = {version: {key: value for key, value in report["v1_vs_v2"][version].items()
                         if key != "identical_prose"}
               for version in ("v1", "v2")}
    summary["identical_prose_rate"] = {version: report["v1_vs_v2"][version]["identical_prose"]["rate"]
                                      for version in ("v1", "v2")}
    summary["v2_rows_with_no_violation_at_all"] = report["v2_rows_with_no_violation_at_all"]
    print(json.dumps(summary, indent=2, default=str))
    print(f"written: {out / 'fallback_v2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
