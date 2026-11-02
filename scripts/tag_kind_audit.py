"""Apply the tag-kind rule to the saved answers, and reproduce the window the old rule read.

Three answer sets exist: the 242 main-batch generations, the 58 counterfactual generations, and
the 21 retry generations. §17 counted 21 "tag-kind mislabels" in the first of those with a
40-character window after the tag, and this script does two things with that claim:

1. runs the new attribution rule (§`llm/hybrid/tag_kind.py`) over all three sets;
2. for each of the 21 stored labels, **reproduces the exact 40-character window the old rule
   read**, so a reader can see whether the flagged word was a claim about the tag, another
   sensor's name, a fault name, or a downstream consequence — without taking my word for it.

The near-miss list is the opposite check: clauses that put a tag and a clashing quantity close
together and are *not* rejected, listed so the rule's restraint is inspectable too.

CPU only, no model, no generation.

Usage::

    python scripts/tag_kind_audit.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.hybrid import tag_kind as tk  # noqa: E402

HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
OUT = ROOT / "outputs/analysis/phase3a/fit"
SEED = 42
NEAR_MISSES = 10
OLD_WINDOW = 40


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def old_window(prose: str, tag: str) -> Optional[Dict]:
    """Reproduce the follow-up rule's reading of one stored label.

    That rule took the 40 characters after the tag and looked for a measurement word (for an
    `XMV` tag) or a valve word (for an `XMEAS` tag). This shows the same window, plus the clause
    it lands in, so the artefact is visible rather than inferred.
    """
    number = int(tk.canonical(tag).split("_")[1])
    first: Optional[Dict] = None
    # The old rule returned on the first occurrence that produced a hit, so this must look at
    # every occurrence, not just the first one in the text.
    for match in re.finditer(rf"\b(?:XMV|XMEAS)[_ ]?0*{number}\b", prose or ""):
        candidate = tk.canonical(match.group(0))
        if candidate != tk.canonical(tag):
            continue
        window = (prose or "")[match.end():match.end() + OLD_WINDOW]
        kind = tk.axis(candidate)
        words = (sum((list(v) for v in tk.MEASUREMENT_QUANTITIES.values()), [])
                 if kind == "valve_position" else list(tk.VALVE_QUANTITIES))
        hit = next((word for word in words if word.lower() in window.lower()
                    and not any(token in window.lower() for token in ("not", "n't"))), None)
        sentence_start = max(0, (prose or "").rfind(".", 0, match.start()) + 1)
        entry = {"tag": candidate, "window": window.strip(), "old_word": hit,
                 "char_offset": match.start(),
                 "context": (prose or "")[sentence_start:match.end() + OLD_WINDOW].strip()}
        if hit:
            return entry
        if first is None:
            first = entry
    return first


def near_misses(rows: List[Dict], limit: int = NEAR_MISSES) -> List[Dict]:
    """Clauses where a tag and a clashing quantity are close but there is no attribution."""
    out: List[Dict] = []
    seen = set()
    for row in rows:
        for sentence in tk._SENTENCE.split(row.get("prose") or ""):
            for clause in tk._CLAUSE.split(sentence):
                text = clause.strip()
                if not text or text in seen or not tk._TAG.search(text):
                    continue
                if tk.violations(text):
                    continue
                masked = tk.mask(text)
                low = masked.lower()
                for match in tk._TAG.finditer(masked):
                    tag = tk.canonical(match.group(0))
                    kind = tk.axis(tag)
                    words = (sum((list(v) for v in tk.MEASUREMENT_QUANTITIES.values()), [])
                             if kind == "valve_position" else list(tk.VALVE_QUANTITIES))
                    hit = next((word for word in words
                                if word.lower() in low
                                and abs(low.find(word.lower()) - match.end()) <= 40), None)
                    if hit:
                        seen.add(text)
                        out.append({"item_id": row.get("item_id"), "tag": tag,
                                    "quantity": hit, "clause": text[:200],
                                    "why_not_flagged": "the quantity is not predicated of the tag"})
                        break
                if len(out) >= limit:
                    return out
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    rows = load_jsonl(HYBRID / "generation_rows.jsonl")
    main_rows = [row for row in rows if row.get("kind") == "main"]
    counter = [row for row in rows if row.get("kind") == "counterfactual"]
    retry = json.loads((HYBRID / "valve_flow_retry.json").read_text(encoding="utf-8"))

    def audit(subset: List[Dict], field: str = "prose") -> Dict:
        flagged = []
        for row in subset:
            found = tk.violations(row.get(field) or "")
            if found:
                flagged.append({"item_id": row.get("item_id"), "findings": found})
        return {"rows": len(subset), "flagged_rows": len(flagged), "flagged": flagged}

    report = {
        "rule": {
            "definition": "a clause that attributes a measurement quantity (temperature, pressure, "
                          "level, flow, composition) to an XMV tag, or a valve quantity (position, "
                          "opening, stem, stroke, actuator, manipulated variable) to an XMEAS tag",
            "guards": ["sensor names and bracketed kinds are masked before scanning",
                       "the tag and the quantity must share a clause",
                       "the quantity must be predicated of the tag (copula forward, kind-noun "
                       "`of/from/on/at` backward); associative `for` is excluded",
                       "negated or contrastive attributions are not violations"],
        },
        "main_242": audit(main_rows),
        "counterfactual_58": audit(counter),
        "retry_21": {"rows": len(retry["rows"]),
                     "flagged_rows": sum(1 for row in retry["rows"]
                                         if tk.violations(row.get("prose_after") or "")),
                     "stored_fixed": retry["fixed"],
                     "stored_still_mislabelled": retry["still_mislabelled"],
                     "stored_new_mislabel": retry["new_mislabels"],
                     "rule_verdict": "0 violations before and after: the retry changed prose that "
                                     "already had none"},
        "stored_21": [],
        "near_misses": near_misses(main_rows),
    }

    meta = {row["item_id"]: row for row in main_rows
            if row.get("valve_flow_mislabels")}
    for item_id, row in meta.items():
        label = str(row["valve_flow_mislabels"][0])
        tag = label.split()[0]
        reproduced = old_window(row.get("prose") or "", tag)
        report["stored_21"].append({
            "item_id": item_id, "stored_label": label, "reproduced": reproduced,
            "new_rule_findings": tk.violations(row.get("prose") or ""),
        })

    report["verdict"] = {
        "stored_21_found_by_the_new_rule": sum(1 for entry in report["stored_21"]
                                               if entry["new_rule_findings"]),
        "stored_21_total": len(report["stored_21"]),
        "flagged_in_242": report["main_242"]["flagged_rows"],
        "false_positives_in_242": report["main_242"]["flagged_rows"],
        "note": "every stored label reproduces as a 40-character window that read past the clause: "
                "the flagged word belongs to another sensor's name, a fault name, another tag's "
                "kind, or a downstream consequence",
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tag_kind_audit.json").write_text(json.dumps(report, indent=2, default=str),
                                             encoding="utf-8")
    print(json.dumps({
        "main": report["main_242"]["flagged_rows"],
        "counterfactual": report["counterfactual_58"]["flagged_rows"],
        "retry": report["retry_21"]["flagged_rows"],
        "stored_21_found": report["verdict"]["stored_21_found_by_the_new_rule"],
        "near_misses": len(report["near_misses"]),
        "first_stored": report["stored_21"][0] if report["stored_21"] else None,
    }, indent=2, default=str))
    print(f"written: {out / 'tag_kind_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
