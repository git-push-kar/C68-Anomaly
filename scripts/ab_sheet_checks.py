"""Checks that an A/B sheet pairs one event with itself (PROTOCOL 26.3, 26.4).

Both sides of a row must come from the same item: the adapter's stored generation for that item
and the deterministic explainer on the same prompt. The v4 sheet failed exactly here, so these
checks are shared by the diagnosis, the v5 build and the regression tests.
"""
from __future__ import annotations

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

from scripts.build_prose_dataset import load_jsonl  # noqa: E402
from scripts.fallback_v31 import fallback_v31  # noqa: E402
from scripts.fallback_v3 import load_overlay  # noqa: E402
from scripts.fallback_v2 import parse_store  # noqa: E402
from scripts.prose_pilot_prompts import EXPORT, build_orders  # noqa: E402

FIT = ROOT / "outputs/analysis/phase3a/fit"
DATASET = ROOT / "outputs/benchmark_v2/dataset"
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
SHEET_SCORES = ("correct_A", "grounded_A", "consistent_A", "useful_A",
                "correct_B", "grounded_B", "consistent_B", "useful_B", "preference", "notes")
#: Labels only one generator writes. Neither may survive on one side of a v5 row.
LEAK_LABELS = ("Differential:", "Evidence for this event:", "Recommended:",
               "physical_reasoning", "candidate_supplied")


def read_rows(path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def v4_render(answer: Dict) -> str:
    reasoning = str((answer or {}).get("physical_reasoning") or "").strip()
    narrative = str((answer or {}).get("narrative") or "").strip()
    return "Reasoning: " + reasoning + chr(10) + chr(10) + "Narrative: " + narrative


def style_of(text: str) -> str:
    body = str(text or "").lstrip()
    if body.startswith("{"):
        return "json"
    if body.startswith("Reasoning:"):
        return "labelled"
    if body.startswith("Diagnosis:"):
        return "layout"
    return "other"


def top_sensor(prompt: str) -> Tuple[str, float]:
    store = parse_store(str(prompt or ""))
    sensors = sorted(store.get("sensors") or [],
                     key=lambda sensor: -abs(float(sensor.get("z") or 0.0)))
    if not sensors:
        return "", 0.0
    return str(sensors[0]["tag"]), round(float(sensors[0]["z"]), 2)


class Sources:
    """Every stored text mapped back to the item it was generated for."""

    def __init__(self) -> None:
        orders = {row["item_id"]: row for row in build_orders()}
        self.val = {row["item_id"]: row for row in load_jsonl(DATASET / "val.jsonl")}
        probs = {row["item_id"]: row.get("probs")
                 for row in load_jsonl(HYBRID / "crossfit_val_rows.jsonl")}
        overlay = load_overlay(ROOT / "configs/kb_overlay_v1.json")
        self.items = orders
        self.adapter: Dict[str, str] = {}
        self.adapter_texts: Dict[str, str] = {}
        for row in load_jsonl(EXPORT / "raw_pilot_kb.jsonl"):
            self.adapter.setdefault(row["text"], row["item_id"])
            self.adapter_texts.setdefault(row["item_id"], row["text"])
        self.fallback: Dict[str, str] = {}
        self.fallback_answer: Dict[str, Dict] = {}
        for item_id, row in orders.items():
            item = self.val.get(item_id) or {}
            store = parse_store(str(item.get("prompt") or ""))
            raw_probs = probs.get(item_id) or {}
            if isinstance(raw_probs, str):
                import ast
                raw_probs = ast.literal_eval(raw_probs)
            answer = fallback_v31(str(row["gold_fault_id"]), str(row["gold_verdict"]),
                                 row["menu_ids"], store, raw_probs, overlay)
            self.fallback_answer[item_id] = answer
            self.fallback.setdefault(v4_render(answer), item_id)

    def lookup(self, text: str) -> Tuple[Optional[str], str]:
        if text in self.adapter:
            return self.adapter[text], "adapter"
        if text in self.fallback:
            return self.fallback[text], "fallback"
        return None, "unknown"

    def adapter_text(self, item_id: str) -> str:
        return self.adapter_texts[str(item_id)]

    def sensor_of(self, item_id: Optional[str]) -> Tuple[str, float]:
        if not item_id:
            return "", 0.0
        item = self.val.get(item_id) or {}
        return top_sensor(str(item.get("prompt") or ""))


def audit_sheet(sheet_path, key_path, sources: Optional[Sources] = None,
                provenance: Optional[Dict[str, Dict]] = None) -> Dict:
    """Every defect the v4 review named, measured row by row.

    A sheet whose texts were re-rendered (v5) cannot be anchored by text lookup, so its build
    records which item each side was generated for; that record is passed in as `provenance`.
    """
    sources = sources or Sources()
    provenance = provenance or {}
    sheet = read_rows(sheet_path)
    key = {row["item_id"]: row for row in read_rows(key_path)}
    rows: List[Dict] = []
    for row in sheet:
        entry = key.get(row["item_id"])
        if entry is None:
            rows.append({"item_id": row["item_id"], "defect": ["no key row"]})
            continue
        a_item, a_source = sources.lookup(row["text_A"])
        b_item, b_source = sources.lookup(row["text_B"])
        prov = provenance.get(row["item_id"]) or {}
        if prov:
            a_item = prov.get("adapter_item") or a_item
            b_item = prov.get("fallback_item") or b_item
            a_source = a_source if a_source != "unknown" else "adapter"
            b_source = b_source if b_source != "unknown" else "fallback"
            tag_a, z_a = str(prov.get("tag") or ""), round(float(prov.get("z") or 0.0), 2)
            tag_b, z_b = tag_a, z_a
        else:
            tag_a, z_a = sources.sensor_of(a_item)
            tag_b, z_b = sources.sensor_of(b_item)
        same_item = bool(a_item) and a_item == b_item
        same_event = same_item and tag_a == tag_b and abs(z_a - z_b) < 0.005
        style_a, style_b = style_of(row["text_A"]), style_of(row["text_B"])
        expected = {"A": entry.get("A"), "B": entry.get("B")}
        style_matches = {"A": _matches(style_a, expected["A"]),
                         "B": _matches(style_b, expected["B"])}
        diagnosis, verdict = str(entry.get("diagnosis")), str(entry.get("verdict"))
        if diagnosis.startswith("FAULT"):
            stratum = "classified_fault"
        elif verdict == "no_fault":
            stratum = "normal"
        elif entry.get("family") == "removed":
            stratum = "abstention_removed"
        else:
            stratum = "abstention_menu"
        defects = []
        if not same_item:
            defects.append("different item")
        elif not same_event:
            defects.append("same item, different event")
        if not (style_matches["A"] and style_matches["B"]):
            defects.append("style does not match key")
        if stratum == "classified_fault":
            defects.append("classified row inside a non-classified stratum")
        rows.append({"item_id": row["item_id"], "key_family": entry.get("family"),
                     "key_diagnosis": diagnosis, "key_verdict": verdict,
                     "key_A": expected["A"], "key_B": expected["B"],
                     "text_A_item": a_item or "", "text_A_source": a_source,
                     "text_A_style": style_a, "text_B_item": b_item or "",
                     "text_B_source": b_source, "text_B_style": style_b,
                     "same_item": same_item, "tag_A": tag_a, "z_A": z_a,
                     "tag_B": tag_b, "z_B": z_b, "same_event": same_event,
                     "style_matches_key": bool(style_matches["A"] and style_matches["B"]),
                     "stratum_of_row": stratum, "defect": defects})
    return {"rows": len(rows),
            "same_item_rows": sum(1 for row in rows if row.get("same_item")),
            "rows_with_different_items": sum(1 for row in rows
                                             if "different item" in row.get("defect", [])),
            "rows_with_style_mismatch": sum(1 for row in rows
                                            if "style does not match key" in row.get("defect", [])),
            "classified_rows": sum(1 for row in rows
                                   if row.get("stratum_of_row") == "classified_fault"),
            "per_row": rows}


def _matches(style: str, label: str) -> bool:
    if style == "layout":
        return True
    if label == "adapter":
        return style == "json"
    if label == "fallback":
        return style == "labelled"
    if label == "fallback_v3":
        return style == "labelled"
    return False


def format_features(text: str) -> Dict:
    """Structural layout only: length and prose content are deliberately not features (26.3)."""
    body = str(text or "")
    labels = re.findall(r"^([A-Z][A-Za-z ]{2,20}):", body, re.M)
    return {"labels": tuple(labels), "lines": body.count(chr(10)) + 1,
            "braces": body.count("{") + body.count("}"),
            "bullets": len(re.findall(r"^[ \t]*[-*\u2022][ \t]", body, re.M)),
            "leak_labels": tuple(label for label in LEAK_LABELS if label in body)}
