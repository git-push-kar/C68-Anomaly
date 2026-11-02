"""A tag and the kind of quantity it carries, checked against each other.

The v2 prompt prints every tag with its kind — ``A_Feed_Flow [XMV_44, valve position, manipulated
variable]``, ``Reactor_Pressure [XMEAS_7, measurement]`` — so an answer that calls a valve position
a temperature has contradicted the page it was reading, not merely worded something loosely.

**What went wrong before this module existed.** §17 counted 21 such contradictions in 242 answers
with a 40-character window after the tag, and this round hand-read all 21: **none is a
mislabel.** Twenty came from a *neighbouring sensor's name* (``Stripper_Steam_Valve (XMV_50) and
Stripper_Temperature (XMEAS_18)`` — "temperature" belongs to `XMEAS_18`), and the rest from a fault
name or a downstream consequence (``XMV_45 is increasing, causing pressure rises``). A window that
reads past the end of a clause cannot tell a claim about a tag from the next tag's name.

So this rule is built the other way round — it requires **attribution**, and it masks the text that
is not the model's claim:

1. sensor names and bracketed tag groups are replaced before scanning, so ``A_Feed_Flow`` and
   ``[XMV_51, valve position, manipulated variable]`` cannot contribute a quantity word;
2. the tag and the quantity must be in the **same clause**;
3. the quantity must be **predicated of the tag** — either ``<tag> … is/are/reads/shows/… a
   <quantity>``, or ``<quantity> (reading|signal|measurement|value|sensor) of/for/from <tag>``;
4. a negated or contrastive attribution ("XMV_45 is **not** a pressure") is correct usage and is
   not a violation.

What it can police is the axis confusion — a valve position given a measurement quantity, or a
measurement called a valve position. It says nothing about *which* measurement an `XMEAS` tag
carries; that needs the plant tag table and is a different rule.

Not wired into the live path: `llm/response/` is untouched (§18.7). This module exists for the
report path to adopt and for the audit that decides whether it should.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

#: Quantities only a measurement can carry.
MEASUREMENT_QUANTITIES: Dict[str, Sequence[str]] = {
    "temperature": ("temperature", "temp", "thermal", "deg c", "°c", "celsius"),
    "pressure": ("pressure", "psig", "psia"),
    "level": ("level", "liquid height"),
    "flow": ("flow", "flow rate", "flowrate"),
    "composition": ("composition", "concentration", "mole fraction", "mole %", "mass fraction"),
}

#: Words that assert valve-position semantics.
VALVE_QUANTITIES: Sequence[str] = ("valve position", "valve opening", "valve stem", "stroke",
                                   "actuator", "manipulated variable")

_TAG = re.compile(r"\b(XMV|XMEAS)[_ ]?(\d{1,2})\b")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])|\n+")
_CLAUSE = re.compile(r",\s+|\s+(?:while|whereas|but|and)\s+|;", re.IGNORECASE)
#: A sensor display name (`A_Feed_Flow`) is context, not a claim. Tags are exempt: they are the
#: thing being talked about, so they survive masking.
_NAME = re.compile(r"\b(?!XMV_|XMEAS_)[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
_BRACKET = re.compile(r"[\[(][^\])]{0,80}[\])]")
_COPULA = (r"(?:is|are|was|were|reads|read|shows|showed|indicates|indicated|represents|"
           r"behaves as|acts as|corresponds to|moves as|registers)")
_KIND_NOUN = r"(?:reading|signal|measurement|value|sensor|transmitter|gauge)"
_NEGATIONS = ("not ", "n't", "rather than", "instead of", "no ", "never ", "neither ")
WINDOW = 60


def canonical(tag: str) -> str:
    """``XMV 5`` / ``xmv_05`` → ``XMV_05``; the same normalisation the prompt prints."""
    match = _TAG.search(str(tag))
    if not match:
        return str(tag).upper()
    family, number = match.group(1).upper(), int(match.group(2))
    return f"{family}_{number:02d}"


def axis(tag: str) -> str:
    """What kind of variable a tag is: ``valve_position`` or ``measurement``."""
    return "valve_position" if canonical(tag).startswith("XMV_") else "measurement"


def mask(clause: str) -> str:
    """Reduce a clause to the model's own words about the tags.

    A bracketed group that names a tag is replaced by that tag alone — so
    ``[XMV_51, valve position, manipulated variable]`` contributes no quantity word — and a
    display name such as ``A_Feed_Flow`` is replaced by ``NAME``. What remains is what the model
    asserted, which is the only part the rule should read.
    """

    def bracket(match: "re.Match") -> str:
        tag = _TAG.search(match.group(0))
        return f" {canonical(tag.group(0))} " if tag else " "

    return _NAME.sub("NAME", _BRACKET.sub(bracket, clause))


def _attribution(clause: str, tag: str, quantity: str, words: Sequence[str]) -> bool:
    """True when the clause predicates ``quantity`` of ``tag``."""
    low = clause.lower()
    tag_low = tag.lower()
    for word in words:
        quantity_low = word.lower()
        forward = re.search(
            rf"{re.escape(tag_low)}[^.;]{{0,{WINDOW}}}\b{_COPULA}\b[^.;]{{0,30}}"
            rf"\b{re.escape(quantity_low)}\b", low)
        # "the temperature reading **of** XMV_45" attributes the quantity to the tag.
        # "the flow measurement **for** XMV_52" does not: `for` is associative — it means the loop
        # the valve sits in — and hand-reading the one answer that used it showed a correct
        # statement about the valve's controlled variable, so `for` is deliberately not here.
        backward = re.search(
            rf"\b{re.escape(quantity_low)}\b\s*{_KIND_NOUN}?\s*(?:of|from|on|at)\s*"
            rf"{re.escape(tag_low)}", low)
        if forward or backward:
            return True
    return False


def violations(text: str) -> List[Dict]:
    """Every clause where a tag is *said to be* a quantity of the wrong kind."""
    found: List[Dict] = []
    for sentence in _SENTENCE.split(str(text or "")):
        for clause in _CLAUSE.split(sentence):
            if not clause.strip() or not _TAG.search(clause):
                continue
            masked = mask(clause)
            low = masked.lower()
            for match in _TAG.finditer(masked):
                tag = canonical(match.group(0))
                kind = axis(tag)
                terms: List[Tuple[str, Sequence[str]]] = (
                    list(MEASUREMENT_QUANTITIES.items()) if kind == "valve_position"
                    else [("valve_position", VALVE_QUANTITIES)])
                for quantity, words in terms:
                    for word in words:
                        if word.lower() not in low:
                            continue
                        if not _attribution(low, tag, quantity, [word]):
                            continue
                        at = low.find(word.lower())
                        if any(token in low[max(0, at - 20):at + 14] for token in _NEGATIONS):
                            continue
                        entry = {"tag": tag, "axis": kind, "quantity": quantity,
                                 "rule": f"{tag} attributed a {quantity} quantity",
                                 "clause": clause.strip()}
                        if entry not in found:
                            found.append(entry)
    return found


def rejects(text: str) -> bool:
    """The adoptable predicate: ``True`` when the text must be rejected and rewritten."""
    return bool(violations(text))


def summarise(text: str) -> Dict:
    """A compact form for audit output."""
    found = violations(text)
    return {"violations": len(found),
            "tags": sorted({entry["tag"] for entry in found}),
            "quantities": sorted({entry["quantity"] for entry in found}),
            "rejected": bool(found),
            "first": found[0]["clause"] if found else None}


__all__ = ["MEASUREMENT_QUANTITIES", "VALVE_QUANTITIES", "axis", "canonical", "mask", "rejects",
           "summarise", "violations"]
