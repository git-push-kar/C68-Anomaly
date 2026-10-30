"""Prose that talks about the teacher's instructions instead of the plant.

The teacher sees more than the student ever will: a system prompt telling it the
diagnosis is already decided, a ``DECIDED DIAGNOSIS`` block, a candidate menu, and
an instruction to return two named JSON fields. When that framing bleeds into the
prose -- "the decided diagnosis FAULT_04 ...", "the affected area is the feed
system", "the candidate menu does not contain ..." -- the sentence is only true
inside the teacher's wrapper. A deployed student has no decided diagnosis to
refer to, so the sentence is unlearnable and it teaches the model to write about
its own prompt.

The phrase list is **derived from the template**, not written here. This module
builds the same teacher prompt the builder builds, keeps only the wrapper (the
part after the evidence and menu, plus the teacher system prompt), and reads out
of it:

* **labels** -- lines of the wrapper that name a field, i.e. ``LABEL:``;
* **identifiers** -- quoted key names, e.g. ``"physical_reasoning"``;
* **menu references** -- the two-word phrases the wrapper uses for the menu, e.g.
  ``candidate menu``, ``menu entry``, ``listed candidate``.

A template edit therefore moves the rule with it. Nothing in this list was typed
from memory: each phrase is a substring of the template at the place the regex
found it.

One exemption, stated because it looks arbitrary otherwise: the abstention
contract *requires* an open-set answer to name the nearest listed candidates it
rules out, so ``listed candidate`` / ``menu candidates`` cannot be a rejection on
an abstention item. It stays a rejection everywhere else.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

#: A wrapper label is only scaffolding when it is used *as a label* -- followed by
#: a colon, or by a word that points back at the wrapper ("given", "above", ...).
#: "the disturbance is a sticking valve" is ordinary process prose and passes.
_LABEL_FOLLOWERS = (":", "given", "stated", "listed", "above", "provided",
                    "specified", "as decided")

_LABEL_RE = re.compile(r"^\s*(?:-\s*)?([A-Za-z][A-Za-z ()/]{2,40}):", re.MULTILINE)
#: Quoted tokens, but only where the template is naming an output key (a line that
#: says "keys"), or anywhere as a snake_case identifier. A bare quoted word like
#: "flow" in the system prompt's valve example is English, not a field name.
_IDENTIFIER_RE = re.compile(r"\"([a-z][a-z_]{3,})\"")
_KEY_LINE = "keys"
_MENU_BEFORE_RE = re.compile(r"\b(\w+)\s+menu\b", re.IGNORECASE)
_MENU_AFTER_RE = re.compile(r"\bmenu\s+(entry|entries|item|items)\b", re.IGNORECASE)
_LISTED_CANDIDATE_RE = re.compile(r"\blisted\s+candidate", re.IGNORECASE)

#: Phrases the abstention contract requires an open-set answer to use.
ABSTAIN_REQUIRED = ("listed candidate", "menu candidate", "listed menu candidate")


def teacher_wrapper(full_prompt: bool = False) -> str:
    """The teacher prompt, built the way the builder builds it.

    A stub event and menu are enough: the wording does not depend on which event
    it wraps, only the numbers in the evidence block above it do. With
    ``full_prompt`` the menu block is included as well, because the menu is
    teacher-side framing too -- prose that says "the candidate menu" is talking
    about the prompt.
    """
    from llm.benchmark.menu import build_menu
    from llm.benchmark.teacher import SYSTEM_PROMPT_TEACHER, build_teacher_prompt

    evidence = {
        "event_id": "stub",
        "anomaly_score": 1.0,
        "detector_evidence": {"lstm_ae": {"score": 1.0, "threshold": 0.5, "alarmed": True}},
        "top_anomalous_sensors": [{"name": "XMEAS_9", "display_name": "Reactor_Temperature",
                                   "z_score": 1.0, "deviation_percent": 1.0,
                                   "trend": "increasing"}],
        "temporal_sequence": [{"sensor": "XMEAS_9", "display_name": "Reactor_Temperature",
                               "relative_time_minutes": 0.0}],
    }
    target = {"fault_id": "FAULT_01", "truth_fault_number": 1,
              "primary_subsystem": "feed_system",
              "ranked_hypotheses": [{"fault_id": "FAULT_01"},
                                    {"fault_id": "FAULT_06"}]}
    prompt = build_teacher_prompt(evidence, build_menu(), target)
    if full_prompt:
        return SYSTEM_PROMPT_TEACHER + "\n\n" + prompt
    marker = prompt.find("DECIDED DIAGNOSIS")
    wrapper = prompt[marker:] if marker >= 0 else prompt
    return SYSTEM_PROMPT_TEACHER + "\n\n" + wrapper


def derived_phrases(wrapper: Optional[str] = None) -> List[Dict[str, str]]:
    """The scaffolding phrases, read out of the template."""
    text = wrapper if wrapper is not None else teacher_wrapper()
    found: List[Dict[str, str]] = []
    seen: set = set()

    def add(phrase: str, kind: str) -> None:
        key = phrase.strip().lower()
        if len(key) < 3 or key in seen:
            return
        seen.add(key)
        found.append({"phrase": key, "kind": kind})

    for line in text.splitlines():
        match = _LABEL_RE.match(line)
        if not match:
            continue
        label = " ".join(re.sub(r"\s*\(.*?\)", "", match.group(1)).split())
        if len(label.split()) > 6:
            continue
        # A block heading (not a "- field:" bullet) that is more than one word
        # names the wrapper itself -- "DECIDED DIAGNOSIS" can only come from the
        # prompt -- so it is scaffolding wherever it appears. A one-word heading
        # or an inline field label is only scaffolding when used as a label, which
        # is what the follower check below decides.
        is_heading = not line.lstrip().startswith("-") and len(label.split()) > 1
        add(label, "heading" if is_heading else "label")
    for line in text.splitlines():
        names_keys = _KEY_LINE in line.lower()
        for match in _IDENTIFIER_RE.finditer(line):
            token = match.group(1)
            if names_keys or "_" in token:
                add(token, "identifier")
    for line in (text + "\n" + teacher_wrapper(full_prompt=True)).splitlines():
        low = line.lower()
        if "menu" not in low and "candidate" not in low:
            continue
        # "candidate menu", "the menu" -> the two-word phrase ending at "menu".
        for match in _MENU_BEFORE_RE.finditer(low):
            head = match.group(1)
            if head not in ("the", "a", "this", "that", "one", "each", "no", "per"):
                add(f"{head} menu", "menu_reference")
        # "menu entry/entries" -> the phrase starting at "menu".
        for match in _MENU_AFTER_RE.finditer(low):
            add(f"menu {match.group(1)}", "menu_reference")
        # "listed candidate(s)" -> the refusal wording the contract requires.
        if _LISTED_CANDIDATE_RE.search(low):
            add("listed candidate", "menu_reference")
    return found


def _label_hit(text: str, phrase: str) -> bool:
    for match in re.finditer(re.escape(phrase), text):
        tail = text[match.end():match.end() + 14].lstrip()
        if any(tail.lower().startswith(follower) for follower in _LABEL_FOLLOWERS):
            return True
    return False


def scaffolding_hits(text: object, abstain: bool = False,
                     phrases: Optional[Sequence[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    """Scaffolding found in ``text``, each hit naming the phrase and its kind."""
    body = " ".join(str(text or "").lower().split())
    if not body:
        return []
    hits: List[Dict[str, str]] = []
    for entry in phrases if phrases is not None else derived_phrases():
        phrase, kind = entry["phrase"], entry["kind"]
        if kind == "label":
            matched = _label_hit(body, phrase)
        else:
            matched = phrase in body
        if not matched:
            continue
        if abstain and any(required in phrase for required in ABSTAIN_REQUIRED):
            continue
        hits.append({"phrase": phrase, "kind": kind})
    return hits


__all__ = ["ABSTAIN_REQUIRED", "derived_phrases", "scaffolding_hits", "teacher_wrapper"]
