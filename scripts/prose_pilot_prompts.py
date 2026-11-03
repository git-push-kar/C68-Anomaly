"""Prompts and gate readings for the prose pilot evaluation (PROTOCOL 25.3-25.5).

The three menu orders are rebuilt here with the *training* prompt builder rather than read verbatim
from the export: the committed KB-order export carries the protocol-A tail (asks for fault_id, no
supplied block) while the shuffled exports carry the prose tail, so comparing the stored files would
compare two different prompts. Only the menu line order differs between the prompts built here.
"""
from __future__ import annotations

import ast as _ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_protocolA import prompt_numbers, prompt_tags  # noqa: E402
from scripts.adapter_checks_3b9 import canonical_names, check_bindings, check_numbers  # noqa: E402
from scripts.build_prose_dataset import load_jsonl, shuffled  # noqa: E402
from scripts.fallback_v2 import parse_store  # noqa: E402
from llm.benchmark import evaluate_v2 as ev  # noqa: E402

EXPORT = ROOT / "outputs/benchmark_v2/eval/prose_pilot"
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
DATASET = ROOT / "outputs/benchmark_v2/dataset"
ORDERS = (("kb", None), ("shuffled-1", 1), ("shuffled-2", 2))
MENU_LINE = re.compile("^- (?:FAULT_[0-9][0-9]?|UNCLASSIFIED_OPEN_SET): .+$", re.M)
FAULT_ID = re.compile("FAULT_[0-9][0-9]?")
JSON_OBJECT = re.compile("[{].*[}]", re.S)
MENU_HEADER = "CANDIDATE MENU (order varies by event)"
TAIL = ('Return one JSON object with exactly these keys: "physical_reasoning", "narrative". '
        "Do not restate the supplied fields.")


def assemble(prompt: str, supplied: Dict, top2: Sequence, order: Sequence[str]) -> str:
    """The training prompt shape, with the menu lines in the given order."""
    head, _marker, _tail = prompt.partition("CANDIDATE MENU")
    block = [MENU_HEADER, *order, "", "SUPPLIED CLASSIFICATION",
             "The following are fixed for this answer. Do not restate them.",
             "- fault_id: " + str(supplied["fault_id"]),
             "- verdict: " + str(supplied["verdict"]),
             "- ranked_hypotheses: " + ", ".join(str(x) for x in supplied["ranked_hypotheses"])]
    if top2:
        block.append("- classifier top-2: " + ", ".join(
            str(name) + " " + ("%.2f" % float(value)) for name, value in top2))
    block += ["", TAIL, ""]
    return head + chr(10).join(block)


def supplied_of(item: Dict) -> Dict:
    target = item.get("target") or {}
    ranked = [entry.get("fault_id") for entry in target.get("ranked_hypotheses") or []]
    return {"fault_id": target.get("fault_id"), "verdict": target.get("verdict"),
            "ranked_hypotheses": ranked or [target.get("fault_id")]}


def top2_of(raw) -> List:
    if isinstance(raw, str):
        raw = _ast.literal_eval(raw)
    return sorted(((str(k), float(v)) for k, v in (raw or {}).items()), key=lambda x: -x[1])[:2]


def build_orders() -> List[Dict]:
    """Every 520-item row with its three rebuilt prompts and the order-only check."""
    items = {row["item_id"]: row for row in load_jsonl(DATASET / "val.jsonl")}
    probs = {row["item_id"]: row.get("probs")
             for row in load_jsonl(HYBRID / "crossfit_val_rows.jsonl")}
    built: List[Dict] = []
    for row in load_jsonl(EXPORT / "prose_eval_val_kb.jsonl"):
        item = items.get(row["item_id"]) or {}
        supplied = supplied_of(item)
        raw = str(item.get("prompt") or "")
        field, _marker, rest = raw.partition("CANDIDATE MENU")
        lines = MENU_LINE.findall(rest)
        prompts = {}
        for name, seed in ORDERS:
            order = lines if seed is None else shuffled(lines, seed, str(row["item_id"]))
            prompts[name] = assemble(raw, supplied, top2_of(probs.get(row["item_id"])), order)
        built.append({"item_id": row["item_id"], "gold_fault_id": supplied["fault_id"],
                      "gold_verdict": supplied["verdict"], "variant": row.get("variant"),
                      "menu_ids": item.get("menu_ids") or [], "menu_lines": len(lines),
                      "prompts": prompts, "order_only": _order_only(prompts)})
    return built


def _strip_menu(text: str) -> str:
    text = text.replace(MENU_HEADER + chr(10), "")
    return MENU_LINE.sub("", text)


def _order_only(prompts: Dict) -> bool:
    return len({_strip_menu(text) for text in prompts.values()}) == 1


def parse_prose(text: str, cap: int = 512) -> Dict:
    body = (text or "").strip()
    match = JSON_OBJECT.search(body)
    parsed, error = None, None
    if match:
        try:
            parsed = json.loads(match.group(0))
        except ValueError as exc:
            error = "invalid_json: " + str(exc)
    elif body:
        error = "no_json_object"
    else:
        error = "empty"
    if error or not isinstance(parsed, dict):
        return {"schema_valid": False, "error": error or "not_object", "prose": "",
                "reasoning": "", "narrative": "", "extra_keys": []}
    reasoning, narrative = parsed.get("physical_reasoning"), parsed.get("narrative")
    valid = isinstance(reasoning, str) and isinstance(narrative, str)
    prose = " ".join(str(value or "") for value in (reasoning, narrative)).strip()
    return {"schema_valid": valid, "error": None if valid else "keys_missing_or_not_string",
            "reasoning": reasoning if isinstance(reasoning, str) else "",
            "narrative": narrative if isinstance(narrative, str) else "",
            "prose": prose,
            "extra_keys": sorted(set(parsed) - {"physical_reasoning", "narrative"})}


def contradiction(text: str, gold_fault_id: str) -> List[str]:
    """A printed fault id that is not the supplied one; any id when the verdict withholds one."""
    found = sorted(set(FAULT_ID.findall(text or "")))
    if str(gold_fault_id).startswith("FAULT"):
        return [entry for entry in found if entry != gold_fault_id]
    return found


_CANON: Dict = {}


def read_gates(prompt: str, text: str, tokens: int, gold_fault_id: str, cap: int = 512) -> Dict:
    """Every 23.4 reading that can be computed from one generation, plus its codes."""
    parsed = parse_prose(text, cap)
    if not _CANON:
        _CANON.update(canonical_names())
    answer = {"physical_reasoning": parsed["reasoning"], "narrative": parsed["narrative"]}
    store = parse_store(prompt)
    return {"schema_valid": parsed["schema_valid"], "error": parsed.get("error"),
            "complete": bool(parsed["schema_valid"] and tokens < cap),
            "empty_narrative": parsed["narrative"].strip() == "",
            "grounding": ev.grounding_violations(parsed["prose"], prompt_tags(prompt),
                                                 prompt_numbers(prompt)),
            "numeric": check_numbers(answer, store),
            "tag_role": check_bindings(answer, _CANON),
            "contradiction": contradiction(parsed["prose"], gold_fault_id),
            "extra_keys": parsed.get("extra_keys") or [],
            "prose_chars": len(parsed["prose"])}
