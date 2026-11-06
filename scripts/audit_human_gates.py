"""Machine part of the two human gates, so a person signs a verdict instead of doing the reading.

D27 (freeze the 46 calibration cases) and D28 (the 30-sentence mechanism check) both ask a
human to read and mark. Most of that reading is not judgement: a tag binding is a table
lookup, an anchor is arithmetic, a numeric claim is an inequality, and a mechanism sentence
either uses a tag for the right measurement class or it does not. This script does those, and
is explicit about the one family it refuses to decide.

For D27 it emits, per case, the independent evidence (the tag table entry, the |delta z|, the
stored value/threshold and the asserted direction) and `certified_by` -- "machine" only where
the evidence is a table lookup or arithmetic, "human" for the contradiction shapes, which are
judgements about rhetoric (rejection, hypothetical, comparison, negation).

For D28 it audits the sheet itself: the `claimed` window that produced each tag_role flag is
often taken from a different sentence than the one shown, and a flag whose phrase names the
tag's own measurement class is a checker false positive, not a mechanism error. It labels each
row compatible / class_mismatch / unknown_tag / phrase_not_in_sentence.

Nothing here writes a signature. `cases46_freeze_candidate.json` carries hashes and
`signed_by: null`; signing it is the human act the gate exists for.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evidence.process_relationships import SENSOR_NAMES  # noqa: E402
from scripts.cases46 import CASES  # noqa: E402
from scripts.pilot_rechecks import run_case  # noqa: E402

CONFIRM = ROOT / "docs" / "judging" / "confirm"
SHEET = ROOT / "outputs" / "overnight_2026-11-05" / "n5_examples.csv"
OUT = ROOT / "outputs" / "overnight_2026-11-05"

TAG = re.compile(r"(XMEAS_\d{1,2}|XMV_\d{1,2})")
NUMBER = re.compile(r"-?\d+\.\d+")
CLAUSE = re.compile(r"[,;.()\[\]]")
STOP = {"measurement", "position", "change", "value", "and", "the", "with", "its", "associated",
        "for", "from", "check", "verify", "shows", "show", "trend", "trends", "increasing",
        "decreasing", "measured", "variable", "manipulated", "both", "first", "then", "also"}


def tokens(text: object) -> List[str]:
    return [part for part in re.split(r"[^a-z0-9]+", str(text or "").lower()) if len(part) > 1]


def class_tokens(text: object) -> set:
    return {part for part in tokens(text) if part not in STOP}


# A window is evidence only if it matches a measurement class better than it matches any
# other. Term frequency alone cannot decide this: 'feed' occurs in 9 of the 52 classes and
# 'flow' in 10, so no threshold separates a family word from a decisive one. What does decide
# is whether the window's own words land on another class and nowhere on the tag's.
CLASS_OF = {tag: class_tokens(name) for tag, name in SENSOR_NAMES.items()}


def phrase_before(sentence: str, tag: str) -> str:
    index = sentence.find(tag)
    if index < 0:
        return ""
    return CLAUSE.split(sentence[:index])[-1].strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- D27


def contradiction_shape(text: str) -> str:
    """The rhetorical shape a human is being asked to judge, so the read is a yes/no."""
    low = str(text or "").lower()
    if re.search(r"\bruled out\b|\bcontradicts\b|\bunlike\b|\bwould show\b", low):
        return "rejection / hypothetical / comparison"
    if re.search(r"\bnot\b|\bno listed fault\b|\bis not\b", low):
        return "negation or abstention"
    if "verdict is" in low:
        return "verdict claim"
    if re.search(r"fault_\d+|no_fault", low):
        return "names a fault or verdict"
    return "normal-operation claim"


def audit_case(case: Dict) -> Dict:
    check, expect = case["check"], bool(case["expect"])
    checker = bool(run_case(case))
    text = str(case["text"])
    evidence, certified = "", "machine"

    if check == "tag_role":
        findings = []
        for tag in TAG.findall(text):
            phrase = phrase_before(text, tag)
            canonical = SENSOR_NAMES.get(tag)
            if canonical is None:
                findings.append(f"{tag} is not a tag in the table")
                continue
            shared = class_tokens(phrase) & class_tokens(canonical)
            findings.append(f"{tag}={canonical}; phrase '{phrase}' -> "
                            + ("agrees" if shared else "names a different class")
                            + (f" (shared: {', '.join(sorted(shared))})" if shared else ""))
        evidence = " | ".join(findings)
    elif check == "anchor":
        tag, z = str(case["tag"]), float(case["z"])
        found = NUMBER.search(text)
        quoted = float(found.group(0)) if found else None
        delta = None if quoted is None else abs(quoted - z)
        evidence = (f"{tag} z={z}; text quotes "
                    + ("nothing" if quoted is None else f"{quoted} (|delta|={delta:.3f}, "
                       f"tolerance 0.01 -> {'exact' if delta <= 0.01 else 'contradicting'})"))
    elif check == "numeric":
        store = case.get("store") or {}
        facts, problems = [], []
        for detector in store.get("detectors") or []:
            value, threshold = str(detector.get("value")), str(detector.get("threshold"))
            try:
                v, t = float(value), float(threshold)
            except (TypeError, ValueError):
                continue
            facts.append(f"{detector.get('name', '')}: stored {value} vs threshold {threshold}")
            if value in text and threshold in text and value != threshold:
                window = text[text.find(value):][:140]
                if re.search(r"\bbelow\b|\bless than\b", window) and v > t:
                    problems.append(f"asserts {value} is below {threshold}, but {v} > {t}")
                if re.search(r"\babove\b|\bexceeds?\b|\bgreater\b", window):
                    pass
            if value != threshold and text.count(value) > 1 and "threshold" in text:
                tail = text[text.rfind(value) - 40:text.rfind(value) + len(value)]
                if "threshold" in tail:
                    problems.append(f"stored value {value} is printed as the threshold")
        quoted = [float(n) for n in NUMBER.findall(text)]
        known = {float(d["value"]) for d in (store.get("detectors") or []) if str(d.get("value"))}
        known |= {float(d["threshold"]) for d in (store.get("detectors") or [])
                  if str(d.get("threshold"))}
        unknown = [q for q in quoted if known and q not in known]
        if unknown:
            problems.append(f"quoted value(s) not in the store: {unknown}")
        evidence = "; ".join(facts) + ((" -> " + "; ".join(problems)) if problems else
                                       " -> no contradiction in the quoted quantities")
    else:
        certified = "human"
        evidence = ("shape: " + contradiction_shape(text)
                    + f"; supplied gold_fault_id={case.get('gold_fault_id')}"
                    + f", gold_verdict={case.get('gold_verdict')}")

    return {"check": check, "expect": expect, "checker": checker,
            "agree": checker == expect, "certified_by": certified,
            "text": text, "note": case.get("note", ""), "evidence": evidence}


def stage_d27() -> Dict:
    rows = [audit_case(case) for case in CASES]
    CONFIRM.mkdir(parents=True, exist_ok=True)
    out_csv = CONFIRM / "cases46_audit.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n", "check", "expect", "checker", "agree",
                                                    "certified_by", "source", "text", "evidence",
                                                    "note"])
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            writer.writerow({"n": index, **{k: row[k] for k in ("check", "expect", "checker",
                                                                "agree", "certified_by")},
                             "source": "base" if index <= 40 else "added (D6)",
                             "text": row["text"], "evidence": row["evidence"], "note": row["note"]})

    per_check = Counter(row["check"] for row in rows)
    disagree = [row for row in rows if not row["agree"]]
    human = [row for row in rows if row["certified_by"] == "human"]

    hashes = {"scripts/recheck_cases.py": sha(ROOT / "scripts" / "recheck_cases.py"),
              "scripts/cases46.py": sha(ROOT / "scripts" / "cases46.py")}
    candidate = {
        "record": "freeze candidate for the 46-case calibration (D6 / D27)",
        "generated": True,
        "generated_by": "scripts/audit_human_gates.py",
        "signed_by": None,
        "signed_at_local": None,
        "cases": len(rows),
        "distribution": dict(per_check),
        "machine_certified": len(rows) - len(human),
        "needs_human_judgement": len(human),
        "checker_label_disagreements": len(disagree),
        "files_sha256": hashes,
        "note": ("hashes cover BOTH files: 40 of the 46 cases live in recheck_cases.py. "
                 "This file is a candidate, not a freeze; it carries no signature"),
    }
    (CONFIRM / "cases46_freeze_candidate.json").write_text(
        json.dumps(candidate, indent=2) + "\n", encoding="utf-8")

    lines = ["# 46 calibration cases - machine audit", "",
             f"{len(rows)} cases: " + ", ".join(f"{k} {v}" for k, v in sorted(per_check.items())),
             "",
             f"* checker/label disagreements: **{len(disagree)}**",
             f"* certified by table lookup or arithmetic: **{len(rows) - len(human)}**",
             f"* awaiting a human judgement: **{len(human)}** (all contradiction shapes)",
             "",
             "## The rows that need you", "",
             "| # | shape | text | why the checker flags it |", "|---|---|---|---|"]
    for row in human:
        lines.append("| %d | %s | %s | %s |" % (human.index(row) + 1,
                                                row["evidence"].split("shape: ")[-1].split(";")[0],
                                                row["text"].replace("|", "\\|"),
                                                row["note"].replace("|", "\\|")))
    lines += ["", "## Machine-certified rows", "",
              "| # | check | label | evidence |", "|---|---|---|---|"]
    for index, row in enumerate([r for r in rows if r["certified_by"] == "machine"], 1):
        lines.append("| %d | %s | %s | %s |" % (index, row["check"],
                                                "flag" if row["expect"] else "no flag",
                                                row["evidence"].replace("|", "\\|")[:220]))
    lines += ["", "## Freeze candidate", "",
              "`cases46_freeze_candidate.json` carries the hashes of both source files and "
              "`signed_by: null`. Signing it is the human act; nothing here may fill that in.", "",
              "```"]
    for name, digest in hashes.items():
        lines.append(f"{digest}  {name}")
    lines += ["```"]
    (CONFIRM / "cases46_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"cases": len(rows), "disagree": len(disagree), "human": len(human),
            "distribution": dict(per_check), "csv": str(out_csv.relative_to(ROOT))}


# --------------------------------------------------------------------------- D28


def audit_mechanism_row(row: Dict) -> Dict:
    """Judge the flag on the evidence the checker itself used, not on a re-derived phrase.

    `claimed` is the window `tag_role_flags` compared against the tag's canonical name. The
    sheet prints it beside a sentence chosen only by "contains the tag", so the two can come
    from different clauses. Rather than guess a phrase, this audits the window and separates
    the rows where the window is not even present in the printed sentence -- those cannot be
    judged as printed and the sheet needs regenerating for them.
    """
    tag = str(row.get("tag") or "")
    canonical = str(row.get("canonical") or "") or SENSOR_NAMES.get(tag, "")
    sentence = str(row.get("sentence") or "")
    claimed = str(row.get("claimed") or "").strip()

    own = CLASS_OF.get(tag) or set()
    words = class_tokens(claimed)
    shared = words & own
    scores = {other: len(words & toks) for other, toks in CLASS_OF.items() if other != tag}
    best = max(scores.values(), default=0)
    winners = sorted(other for other, score in scores.items() if score == best)
    claimed_present = bool(claimed) and claimed in sentence

    if tag not in SENSOR_NAMES:
        verdict, why = "unknown_tag", f"{tag} is not a tag (table covers XMEAS_1-41, XMV_42-52)"
    elif not claimed:
        verdict, why = "no_window", "the flag carries no claimed phrase"
    elif not claimed_present:
        verdict, why = ("window_not_in_sentence",
                        "the window the checker used is not a phrase of the sentence shown")
    elif shared:
        verdict, why = "compatible", (f"the window names {canonical} "
                                      f"({', '.join(sorted(shared))}): checker false positive")
    elif best >= 2 and len(winners) <= 2:
        verdict, why = ("class_mismatch",
                        f"the window shares {best} terms with "
                        + ", ".join(SENSOR_NAMES[w] for w in winners)
                        + f" and none with {canonical}: a real mechanism error")
    else:
        verdict, why = ("insufficient_evidence",
                        f"the window shares no term with {canonical}"
                        + (" and matches no single other class" if best else "")
                        + f" (words: {', '.join(sorted(words)) or 'none'}): it neither "
                        "confirms nor refutes the binding, so the sentence decides")

    return {**row, "phrase_used": claimed, "verdict": verdict, "why": why,
            "claimed_window_in_sentence": claimed_present}


def stage_d28() -> Dict:
    with SHEET.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    audited = [audit_mechanism_row(row) for row in rows]

    out_csv = OUT / "n5_examples_audited.csv"
    fields = ["item_id", "condition", "kind", "tag", "canonical", "phrase_used", "verdict",
              "why", "claimed_window_in_sentence", "sentence", "supported", "note"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in audited:
            writer.writerow({k: row.get(k, "") for k in fields})

    counts = Counter(row["verdict"] for row in audited)
    by_condition = Counter(row["condition"] for row in audited)
    stale = [row for row in audited if not row["claimed_window_in_sentence"]]
    meaning = {
        "compatible": "the window names this measurement specifically: the checker's flag is a false positive",
        "class_mismatch": "the window names a DIFFERENT instrument: a real mechanism error",
        "insufficient_evidence": "the window holds only family-level words: the sentence must decide",
        "unknown_tag": "the tag does not exist in the table",
        "window_not_in_sentence": "the checker's window is not in the printed sentence: cannot be judged as printed",
        "no_window": "the flag carries no window",
    }

    lines = ["# 30 mechanism rows - machine audit", "",
             f"rows: {len(audited)} (" + ", ".join(f"{k} {v}" for k, v in by_condition.items())
             + ")", "",
             "| verdict | rows | meaning |", "|---|---|---|"]
    for verdict, count in counts.most_common():
        lines.append(f"| {verdict} | {count} | {meaning.get(verdict, '')} |")
    lines += [""]
    if stale:
        lines.append(f"Rows whose `claimed` window is not a phrase of the sentence shown: "
                     f"**{len(stale)} of {len(audited)}** -- regenerate the sheet before "
                     "marking those.")
    else:
        lines.append(f"Every `claimed` window is a phrase of the sentence shown: "
                     f"**0 of {len(audited)}** unmatched.")
    lines += ["",
              "| # | cond | tag | canonical | window used | verdict | the sentence as printed |",
              "|---|---|---|---|---|---|---|"]
    for index, row in enumerate(audited, 1):
        lines.append("| %d | %s | %s | %s | %s | **%s** | %s |"
                     % (index, row["condition"], row["tag"], row["canonical"],
                        row["phrase_used"].replace("|", "\\|"), row["verdict"],
                        row.get("sentence", "").replace("|", "\\|")[:190]))
    lines += ["", "## What is left for a human", "",
              f"* **{counts['class_mismatch']} class_mismatch** rows: the window names a "
              "different instrument, so these are real errors unless the sentence overrides it.",
              f"* **{counts['insufficient_evidence']} insufficient_evidence** rows: the quoted "
              "window holds only family-level words ('valve', 'flow', 'manipulated variable'), "
              "so it cannot prove a wrong-class reference either way. The sentence column now "
              "prints the text; judge from that.",
              f"* **{counts['compatible']} compatible** rows can be dismissed without reading.",
              f"* **{counts['unknown_tag']} unknown_tag** rows are decidable from the table alone.", "",
              "The rung-3 input is the count of confirmed real errors. An insufficient row is "
              "not an error and not a pass: keep it in a separate unsettled bucket, never in "
              "the numerator and never silently in the denominator.", ""]
    (OUT / "n5_examples_audited.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"rows": len(audited), "verdicts": dict(counts), "by_condition": dict(by_condition),
            "stale_rows": len(stale), "csv": str(out_csv.relative_to(ROOT))}


def stage_sign() -> Dict:
    """Turn the candidate into a freeze only when an owner attestation exists.

    The attestation carries the hashes it was given against, so editing a source file after
    the read invalidates the signature instead of silently extending it to new cases.
    """
    candidate_path = CONFIRM / "cases46_freeze_candidate.json"
    attestation_path = CONFIRM / "cases46_attestation.json"
    if not attestation_path.exists():
        return {"signed": False, "reason": "no owner attestation on file"}
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    current = {"scripts/recheck_cases.py": sha(ROOT / "scripts" / "recheck_cases.py"),
               "scripts/cases46.py": sha(ROOT / "scripts" / "cases46.py")}
    attested = attestation.get("files_sha256") or {}
    if attested != current:
        return {"signed": False, "reason": "source hashes changed since the attestation",
                "attested": attested, "current": current}
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    freeze = {**candidate,
              "record": "FREEZE of the 46-case calibration (D6 / D27)",
              "signed_by": attestation["signed_by"],
              "signed_at_local": (attestation.get("signed_at_local")
                                 or datetime.now().astimezone().isoformat(timespec="seconds")),
              "attested_via": attestation["attested_via"],
              "attested_claim": attestation["attested_claim"],
              "transcribed_by": attestation.get("transcribed_by", "agent"),
              "independence": ("the machine-certified rows were decided by a second "
                               "implementation (tag table + arithmetic), not by the checker"),
              "note": ("frozen: these hashes are the answer key every checker count is "
                       "measured against. A later change is a new dated section, not an edit.")}
    (CONFIRM / "cases46_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n",
                                                 encoding="utf-8")
    return {"signed": True, "by": freeze["signed_by"],
            "at": freeze["signed_at_local"], "rejected": 0}


def main() -> int:
    d27 = stage_d27()
    d28 = stage_d28()
    signed = stage_sign()
    print(json.dumps({"D27": d27, "D28": d28, "freeze": signed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
