"""Audit `audit_ab_100_v5.csv` and score it (PROTOCOL 29.2, 29.3).

Two stages, deliberately separate so the sheet can be scored blind:

  structure  (no key, no scores)  -- row count, columns, layout identity, and a decisive
              pairing check that recovers the item each side was generated for from the stored
              pilot generations.
  join       (key + scores)       -- stratum composition, A/B balance, and the preference
              statistic that the decision rule reads.

Why this script exists beside `scripts/ab_sheet_checks.py`: that module's `Sources` needs
`outputs/benchmark_v2/dataset/val.jsonl` and `outputs/analysis/phase3a/hybrid/
crossfit_val_rows.jsonl`, which this checkout does not carry, so it cannot re-derive the
fallback answers. Everything here runs on the sheet, the key, the provenance file, the stored
pilot generations and the exported prompts, which are all present.

Usage:
  python scripts/audit_v5_sheet.py structure
  python scripts/audit_v5_sheet.py join --scored <scored.csv>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FIT = ROOT / "outputs/analysis/phase3a/fit"
EXPORT = ROOT / "outputs/benchmark_v2/eval/prose_pilot"
SHEET = FIT / "audit_ab_100_v5.csv"
KEY = FIT / "audit_ab_100_v5_key.csv"
PROVENANCE = FIT / "audit_ab_100_v5_provenance.json"
RAW_PILOT = EXPORT / "raw_pilot_kb.jsonl"
PROMPTS = EXPORT / "prose_eval_val_kb.jsonl"

SCORE_COLUMNS = ("correct_A", "grounded_A", "consistent_A", "useful_A",
                 "correct_B", "grounded_B", "consistent_B", "useful_B", "preference", "notes")
FORBIDDEN_COLUMNS = ("family", "diagnosis", "verdict", "source", "side_A_source",
                     "side_B_source", "adapter", "fallback", "stratum", "tag", "z")
LEAK_LABELS = ("Differential:", "Evidence for this event:", "Recommended:",
               "physical_reasoning", "candidate_supplied")
LABELS = ("Diagnosis: ", "Reasoning: ", "Narrative: ")
TAG = re.compile(r"\b(XMEAS_\d{1,2}|XMV_\d{1,2})\b")
NUMBER = re.compile(r"-?\d+\.\d+")


def read_rows(path: Path):
    """Rows as strings, with embedded newlines normalised to LF (the writer emits CRLF)."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for column, value in list(row.items()):
            if isinstance(value, str) and "\r" in value:
                row[column] = value.replace("\r\n", "\n").replace("\r", "\n")
    return rows


def load_jsonl(path: Path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def normalise(text: str) -> str:
    """Inverse of the v5 renderer's clean-up, applied identically to both sides."""
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    body = re.sub(r"\bDifferential:[^.]*\.\s*", "", body)
    body = re.sub(r"\bRecommended:\s*", "Next step: ", body)
    body = re.sub(r"^The verdict is [A-Z_0-9]+\.\s*", "", body)
    body = re.sub(r"^Evidence for this event:\s*", "", body)
    return body.strip()


def layout(diagnosis: str, reasoning: str, narrative: str) -> str:
    return ("Diagnosis: " + diagnosis + "\n\nReasoning: " + normalise(reasoning)
            + "\n\nNarrative: " + normalise(narrative))


def sections(text: str):
    """Split a rendered side into its three labelled parts, keyed without the colon."""
    marks = []
    for label in LABELS:
        index = text.find(label)
        if index < 0:
            return None
        marks.append((label.rstrip().rstrip(":"), index, len(label)))
    marks.sort(key=lambda mark: mark[1])
    parts = {}
    for position, (name, start, width) in enumerate(marks):
        end = marks[position + 1][1] if position + 1 < len(marks) else len(text)
        parts[name] = text[start + width:end].strip()
    return parts


def layout_features(text: str) -> dict:
    """Everything a rater could use to tell the two generators apart by format alone."""
    labels = tuple(label.strip() for label in LABELS if label in text)
    prefixes = tuple(sorted({line.split(":")[0].strip() for line in text.splitlines()
                             if ":" in line and len(line.split(":")[0]) < 24}))
    punctuation = tuple(sorted({character for character in text
                                if character in "{}[]\"*•-–:|;"}))
    return {"labels": labels, "prefixes": prefixes, "punctuation": punctuation,
            "lines": len(text.splitlines()),
            "leaks": tuple(label for label in LEAK_LABELS if label in text)}


def adapter_layouts():
    """Rebuild the rendered adapter text for every stored pilot generation, keyed by item."""
    from scripts.build_v5_sheet import adapter_parts  # noqa: E402  (repo renderer, reused exactly)
    rendered = {}
    for row in load_jsonl(RAW_PILOT):
        reasoning, narrative = adapter_parts(row["text"])
        rendered.setdefault(layout(str(row["gold_fault_id"]), reasoning, narrative),
                            row["item_id"])
    return rendered


def prompt_index():
    """Item -> (top tag, top |z|) straight from the exported prompt."""
    from scripts.fallback_v2 import parse_store  # noqa: E402
    index = {}
    for row in load_jsonl(PROMPTS):
        store = parse_store(str(row.get("prompt") or ""))
        sensors = sorted(store.get("sensors") or [],
                         key=lambda sensor: -abs(float(sensor.get("z") or 0.0)))
        index[row["item_id"]] = ("", 0.0) if not sensors else (
            str(sensors[0]["tag"]), round(float(sensors[0]["z"]), 2))
    return index


def cited(text: str):
    """(tag, z) pairs a side states, in order of appearance."""
    pairs = []
    for match in TAG.finditer(text):
        window = text[match.end():match.end() + 90]
        number = NUMBER.search(window)
        if number:
            pairs.append((match.group(1), round(float(number.group(0)), 2)))
    return pairs


def stage_structure() -> dict:
    rows = read_rows(SHEET)
    columns = list(rows[0].keys()) if rows else []
    report = {"rows": len(rows), "columns": columns,
              "forbidden_columns": [c for c in columns if c in FORBIDDEN_COLUMNS],
              "unique_item_ids": len({row["item_id"] for row in rows}),
              "expected_item_ids": [f"v5-{index:02d}" for index in range(1, len(rows) + 1)],
              "filled_score_cells": sum(1 for row in rows for column in SCORE_COLUMNS
                                        if str(row.get(column) or "").strip()),
              "empty_sides": 0, "identical_sides": 0, "layout_mismatch": 0,
              "leaks": 0, "diagnosis_line_differs": 0, "sections_missing": 0,
              "sections_parsed": 0,
              "layout_structural_mismatch": 0, "layout_punctuation_only": 0,
              "pair_recovered": 0, "pair_wrong_item": 0, "pair_both_adapter": 0,
              "pair_no_adapter": 0, "fallback_side_not_recoverable": 0,
              "anchor_top_sensor_both_sides": 0, "anchor_top_sensor_one_side": 0,
              "anchor_top_sensor_no_side": 0,
              "per_row": []}
    rendered = adapter_layouts()
    prompts = prompt_index()
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    for row in rows:
        item_id = row["item_id"]
        text_a, text_b = row["text_A"], row["text_B"]
        if not text_a.strip() or not text_b.strip():
            report["empty_sides"] += 1
        if text_a.strip() == text_b.strip():
            report["identical_sides"] += 1
        left, right = layout_features(text_a), layout_features(text_b)
        structural = (left["labels"] != right["labels"]
                      or left["prefixes"] != right["prefixes"])
        punctuation_only = (not structural and left["punctuation"] != right["punctuation"])
        differ = structural or punctuation_only
        report["layout_mismatch"] += int(differ)
        report["layout_structural_mismatch"] += int(structural)
        report["layout_punctuation_only"] += int(punctuation_only)
        report["leaks"] += len(left["leaks"]) + len(right["leaks"])
        parts_a, parts_b = sections(text_a), sections(text_b)
        if parts_a is None or parts_b is None or set(parts_a) != {"Diagnosis", "Reasoning", "Narrative"}:
            report["sections_missing"] += 1
        elif parts_a["Diagnosis"] != parts_b["Diagnosis"] or not parts_a["Diagnosis"]:
            report["diagnosis_line_differs"] += 1
        else:
            report["sections_parsed"] += 1
        # Decisive pairing: exactly one side must be the stored adapter output for this item.
        matched = [name for name, text in (("A", text_a), ("B", text_b)) if text in rendered]
        unknown = [name for name, text in (("A", text_a), ("B", text_b))
                   if text not in rendered]
        expected = str((provenance.get(item_id) or {}).get("corpus_item_id") or "")
        if len(matched) == 1:
            # The other side is the fallback, which cannot be regenerated in this checkout
            # (it needs the crossfit probabilities); that is expected and is not a defect.
            if unknown:
                report["fallback_side_not_recoverable"] += 1
            if rendered[text_a if text_a in rendered else text_b] == expected:
                report["pair_recovered"] += 1
            else:
                report["pair_wrong_item"] += 1
        elif len(matched) == 2:
            report["pair_both_adapter"] += 1
        else:
            report["pair_no_adapter"] += 1
        # Both sides should cite the item's own top sensor with its z.
        tag, z = prompts.get(expected, ("", 0.0))
        prov_tag = str((provenance.get(item_id) or {}).get("tag") or "")
        prov_z = (provenance.get(item_id) or {}).get("z")
        cites_a = [pair for pair in cited(text_a) if pair[0] == tag]
        cites_b = [pair for pair in cited(text_b) if pair[0] == tag]
        hits = sum(1 for cites in (cites_a, cites_b)
                   if tag and any(abs(value - float(prov_z)) <= 0.01 for _t, value in cites))
        if hits == 2:
            report["anchor_top_sensor_both_sides"] += 1
        elif hits == 1:
            report["anchor_top_sensor_one_side"] += 1
        else:
            report["anchor_top_sensor_no_side"] += 1
        # No per-row field here may say which side is which: this stage runs before the sheet is
        # scored, and the pairing evidence it needs is aggregate.
        report["per_row"].append({
            "item_id": item_id,
            "provenance_item": expected, "prompt_top_tag": tag, "prompt_top_z": z,
            "provenance_tag": prov_tag, "provenance_z": prov_z,
            "top_sensor_cited_both_sides": hits == 2, "layout_differs": bool(differ),
            "length_A": len(text_a), "length_B": len(text_b)})
    report["provenance_tag_matches_prompt"] = sum(
        1 for entry in report["per_row"]
        if entry["provenance_tag"] == entry["prompt_top_tag"]
        and abs(float(entry["provenance_z"] or 0) - float(entry["prompt_top_z"] or 0)) <= 0.01)
    lengths = sorted(entry["length_A"] + entry["length_B"] for entry in report["per_row"])
    report["layout_detail"] = [
        {"item_id": entry["item_id"], "A": layout_features(dict_a["text_A"]),
         "B": layout_features(dict_a["text_B"])}
        for entry, dict_a in zip(report["per_row"], rows) if entry["layout_differs"]]
    report["length_total_min_median_max"] = [lengths[0], lengths[len(lengths) // 2], lengths[-1]]
    report["passed"] = (report["rows"] == 40 and not report["forbidden_columns"]
                        and report["unique_item_ids"] == report["rows"]
                        and report["filled_score_cells"] == 0
                        and report["empty_sides"] == 0 and report["identical_sides"] == 0
                        and report["layout_structural_mismatch"] == 0 and report["leaks"] == 0
                        and report["sections_missing"] == 0
                        and report["sections_parsed"] == report["rows"]
                        and report["pair_recovered"] == report["rows"])
    # Content, not format: how often both sides state the item's own top sensor with its z.
    # Low counts are the adapter's measured anchor weakness (PROTOCOL 28.9), not a sheet defect,
    # so this is reported and does not gate.
    report["top_sensor_cited_on_both_sides"] = report["anchor_top_sensor_both_sides"]
    return report


def stage_join(scored_path: Path) -> dict:
    key = {row["item_id"]: row for row in read_rows(KEY)}
    scored = {row["item_id"]: row for row in read_rows(scored_path)}
    if set(key) != set(scored):
        raise SystemExit("scored sheet does not cover the key: "
                         + str(sorted(set(key) ^ set(scored))[:5]))
    per_stratum = defaultdict(Counter)
    cells = Counter()
    flagged = {"adapter": [], "fallback": []}
    ties = []
    disagreements = []
    for item_id, entry in key.items():
        row = scored[item_id]
        if entry["A"] == "adapter":
            adapter_side, fallback_side = "A", "B"
        elif entry["B"] == "adapter":
            adapter_side, fallback_side = "B", "A"
        else:
            raise SystemExit("key names no adapter side on " + item_id)
        preference = str(row.get("preference") or "").strip().upper()
        if preference not in ("A", "B", "TIE"):
            raise SystemExit("unscored or illegal preference on " + item_id + ": " + preference)
        stratum = entry["family"]
        per_stratum[stratum]["rows"] += 1
        if preference == "TIE":
            per_stratum[stratum]["tie"] += 1
        else:
            wanted = "adapter" if preference == adapter_side else "fallback"
            per_stratum[stratum][wanted] += 1
        for base in ("correct", "grounded", "consistent", "useful"):
            cells[base + "_adapter"] += int(str(row.get(base + "_" + adapter_side) or "0").strip() == "1")
            cells[base + "_fallback"] += int(str(row.get(base + "_" + fallback_side) or "0").strip() == "1")
        if str(row.get("correct_" + adapter_side) or "") != str(row.get("correct_" + fallback_side) or ""):
            disagreements.append(item_id)
        if preference == "TIE":
            ties.append(item_id)
        for base in ("grounded", "consistent"):
            if str(row.get(base + "_" + adapter_side) or "0").strip() != "1":
                flagged["adapter"].append(item_id + ":" + base)
            if str(row.get(base + "_" + fallback_side) or "0").strip() != "1":
                flagged["fallback"].append(item_id + ":" + base)
    total = Counter()
    for counter in per_stratum.values():
        total.update(counter)
    non_tie = total["adapter"] + total["fallback"]
    share = (total["adapter"] / non_tie) if non_tie else float("nan")
    strata = {}
    for name, counter in per_stratum.items():
        denominator = counter["adapter"] + counter["fallback"]
        strata[name] = {"rows": counter["rows"], "adapter": counter["adapter"],
                        "fallback": counter["fallback"], "tie": counter["tie"],
                        "adapter_share_non_tie": round(counter["adapter"] / denominator, 4)
                        if denominator else None}
    return {"rows": len(key), "per_stratum": strata,
            "pooled": {"adapter": total["adapter"], "fallback": total["fallback"],
                       "tie": total["tie"], "non_tie": non_tie,
                       "adapter_share_non_tie": round(share, 4)},
            "correct_cell_counts": dict(cells),
            "flagged_cells": flagged, "tie_rows": ties,
            "rows_where_correct_differs": disagreements,
            "decision": "adapter_preferred" if share >= 0.60 else "fallback_v31",
            "rule": "adapter preferred when the pooled non-tie share is at least 0.60 (PROTOCOL 29.3)"}


def stage_packet(out: Path) -> dict:
    """Write the scoring packet: per row the supplied diagnosis, the evidence panel, both texts.

    Carries nothing that identifies a generator: no key, no provenance, no source column.
    """
    from scripts.fallback_v2 import parse_store  # noqa: E402
    rows = read_rows(SHEET)
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    prompts = {row["item_id"]: row for row in load_jsonl(PROMPTS)}
    blocks, covered = [], 0
    for row in rows:
        corpus = str((provenance.get(row["item_id"]) or {}).get("corpus_item_id") or "")
        prompt = str((prompts.get(corpus) or {}).get("prompt") or "")
        store = parse_store(prompt) if prompt else {}
        covered += int(bool(store))
        diagnosis = ""
        for line in row["text_A"].splitlines():
            if line.startswith("Diagnosis: "):
                diagnosis = line[len("Diagnosis: "):].strip()
                break
        detectors = "; ".join(
            f"{entry.get('name')} value {entry.get('value')} threshold {entry.get('threshold')} "
            f"ratio {entry.get('ratio')}" for entry in (store.get("detectors") or []))
        sensors = sorted(store.get("sensors") or [], key=lambda s: -abs(float(s.get("z") or 0)))
        panel = "; ".join(f"{entry['tag']} {entry.get('name')} z {entry.get('z')} "
                          f"({entry.get('trend')})" for entry in sensors[:6])
        residuals = "; ".join(f"{entry['tag']} {entry.get('share')}"
                              for entry in (store.get("residuals") or [])[:3])
        blocks.append(
            f"### {row['item_id']}\n\n"
            f"Diagnosis supplied to both: {diagnosis}\n\n"
            f"Detectors: {detectors}\n\n"
            f"Top sensors by |z|: {panel}\n\n"
            f"Residual shares: {residuals}\n\n"
            f"Numbers present in the evidence store: {len(store.get('numbers') or [])}; "
            f"tags: {len(store.get('tags') or [])}\n\n"
            f"--- text_A ---\n\n{row['text_A']}\n\n"
            f"--- text_B ---\n\n{row['text_B']}\n\n")
    out.write_text("\n".join(blocks), encoding="utf-8")
    numbers = out.with_name("audit_v5_evidence_numbers.tsv")
    lines = ["item_id\tdiagnosis\tstore_numbers\tstore_tags"]
    for row in rows:
        corpus = str((provenance.get(row["item_id"]) or {}).get("corpus_item_id") or "")
        store = parse_store(str((prompts.get(corpus) or {}).get("prompt") or ""))
        diagnosis = next((line[len("Diagnosis: "):].strip() for line in row["text_A"].splitlines()
                          if line.startswith("Diagnosis: ")), "")
        lines.append("\t".join([row["item_id"], diagnosis,
                                " ".join(sorted(store.get("numbers") or [])),
                                " ".join(sorted(store.get("tags") or []))]))
    numbers.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": len(rows), "rows_with_evidence": covered, "out": str(out),
            "chars": len(out.read_text(encoding="utf-8")), "numbers_out": str(numbers)}


def stage_worksheet(out: Path) -> dict:
    """One CSV to fill in: evidence and both texts per row, score cells blank, no side labels."""
    from scripts.fallback_v2 import parse_store  # noqa: E402
    rows = read_rows(SHEET)
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    prompts = {row["item_id"]: row for row in load_jsonl(PROMPTS)}
    columns = ["item_id", "supplied_diagnosis", "detectors", "top_sensors", "residual_shares"] \
        + list(SCORE_COLUMNS[:-1]) + ["text_A", "text_B"]
    import csv as _csv
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            corpus = str((provenance.get(row["item_id"]) or {}).get("corpus_item_id") or "")
            store = parse_store(str((prompts.get(corpus) or {}).get("prompt") or ""))
            diagnosis = next((line[len("Diagnosis: "):].strip()
                              for line in row["text_A"].splitlines()
                              if line.startswith("Diagnosis: ")), "")
            sensors = sorted(store.get("sensors") or [],
                             key=lambda s: -abs(float(s.get("z") or 0)))
            writer.writerow({
                "item_id": row["item_id"], "supplied_diagnosis": diagnosis,
                "detectors": "; ".join(f"{entry.get('name')} {entry.get('value')} "
                                        f"(thr {entry.get('threshold')})"
                                        for entry in (store.get("detectors") or [])),
                "top_sensors": "; ".join(f"{entry['tag']} z {entry.get('z')}"
                                          for entry in sensors[:5]),
                "residual_shares": "; ".join(f"{entry['tag']} {entry.get('share')}"
                                             for entry in (store.get("residuals") or [])[:3]),
                "text_A": row["text_A"], "text_B": row["text_B"]})
    return {"rows": len(rows), "out": str(out), "columns": columns}


def stage_adjudicate(out: Path) -> dict:
    """Machine adjudication of both texts of every v5 row, by the four checks of PROTOCOL 28.

    This is the factual ground truth for a rater: which side's claims survive the checkers. It
    reads the key only to label the two sides in the companion file, so `out` (side A/B only) is
    safe to read while scoring and `out.with_name(..._keyed.csv)` is not.
    """
    import csv as _csv
    from scripts.fallback_v2 import parse_store  # noqa: E402
    from scripts.pilot_rechecks import measure  # noqa: E402

    rows = read_rows(SHEET)
    key = {row["item_id"]: row for row in read_rows(KEY)}
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    prompts = {row["item_id"]: row for row in load_jsonl(PROMPTS)}
    plain, keyed, per_side = [], [], Counter()
    for row in rows:
        item_id = row["item_id"]
        corpus = str((provenance.get(item_id) or {}).get("corpus_item_id") or "")
        record = prompts.get(corpus) or {}
        prompt = str(record.get("prompt") or "")
        store = parse_store(prompt)
        verdicts = {}
        for side in ("A", "B"):
            parts = sections(row["text_" + side]) or {}
            prose = " ".join([parts.get("Reasoning", ""), parts.get("Narrative", "")]).strip()
            result = measure(prose, prompt, str(record.get("gold_fault_id") or ""),
                             str(record.get("gold_verdict") or ""), store)
            entry = {
                "item_id": item_id, "side": side,
                "contradiction": len(result["contradiction"]),
                "tag_role": len(result["tag_role"]),
                "numeric": len(result["numeric"]),
                "grounding": len(result["grounding"]),
                "anchor": result["anchor"]["status"],
                "top_tag": result["anchor"]["tag"],
                "top_z": result["anchor"]["z"],
                "flags": " ".join(filter(None, [
                    "contradiction:" + str(len(result["contradiction"]))
                    if result["contradiction"] else "",
                    "tag_role:" + str(len(result["tag_role"])) if result["tag_role"] else "",
                    "numeric:" + str(len(result["numeric"])) if result["numeric"] else "",
                    "grounding:" + str(len(result["grounding"])) if result["grounding"] else ""])) or "clean",
                "anchor_context": str(result["anchor"]["context"])[:160],
            }
            plain.append(entry)
            verdicts[side] = entry
            per_side[side + "_" + ("clean" if entry["flags"] == "clean" else "flagged")] += 1
        entry_row = key[item_id]
        keyed.append({"item_id": item_id, "family": entry_row["family"],
                      "A_source": entry_row["A"], "B_source": entry_row["B"],
                      "A_flags": verdicts["A"]["flags"], "A_anchor": verdicts["A"]["anchor"],
                      "B_flags": verdicts["B"]["flags"], "B_anchor": verdicts["B"]["anchor"],
                      "clean_sides": ",".join(sorted(s for s in ("A", "B")
                                                      if verdicts[s]["flags"] == "clean"))})
    columns = list(plain[0])
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(plain)
    keyed_path = out.with_name(out.stem + "_keyed.csv")
    with keyed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(keyed[0]))
        writer.writeheader()
        writer.writerows(keyed)
    return {"rows": len(rows), "out": str(out), "keyed_out": str(keyed_path),
            "per_side": dict(per_side),
            "clean_both": sum(1 for entry in keyed if entry["clean_sides"] == "A,B"),
            "clean_neither": sum(1 for entry in keyed if not entry["clean_sides"])}


def stage_reveal(scored_path: Path, out: Path) -> dict:
    """Join scores to the key with the source labels visible, for checking a scoring pass.

    This is the file to open **after** scoring, never before: it names which side is which.
    """
    key = {row["item_id"]: row for row in read_rows(KEY)}
    scored = {row["item_id"]: row for row in read_rows(scored_path)}
    lines = ["item_id,family,verdict,A_source,B_source,preferred_source,preference,"
             "adapter_correct,adapter_grounded,adapter_consistent,adapter_useful,"
             "fallback_correct,fallback_grounded,fallback_consistent,fallback_useful,notes"]
    tally = Counter()
    for item_id in sorted(key):
        entry, row = key[item_id], scored[item_id]
        adapter_side = "A" if entry["A"] == "adapter" else "B"
        fallback_side = "B" if adapter_side == "A" else "A"
        preference = str(row["preference"]).strip().upper()
        if preference == "TIE":
            preferred = "tie"
        else:
            preferred = "adapter" if preference == adapter_side else "fallback"
        tally[preferred] += 1
        cells = [str(row.get(base + "_" + side) or "").strip()
                 for side, base in ((adapter_side, "correct"), (adapter_side, "grounded"),
                                    (adapter_side, "consistent"), (adapter_side, "useful"),
                                    (fallback_side, "correct"), (fallback_side, "grounded"),
                                    (fallback_side, "consistent"), (fallback_side, "useful"))]
        notes = str(row.get("notes") or "").replace(",", ";")
        lines.append(",".join([item_id, entry["family"], entry["verdict"], entry["A"],
                               entry["B"], preferred, preference, *cells, notes]))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": len(key), "tally": dict(tally), "out": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("structure", "join", "packet", "reveal", "worksheet",
                                        "adjudicate"))
    parser.add_argument("--scored", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.stage == "packet":
        target = Path(args.out or (FIT / "audit_v5_scoring_packet.md"))
        print(json.dumps(stage_packet(target), indent=1))
        return 0
    if args.stage == "adjudicate":
        target = Path(args.out or (FIT / "audit_v5_adjudication.csv"))
        print(json.dumps(stage_adjudicate(target), indent=1))
        return 0
    if args.stage == "worksheet":
        target = Path(args.out or (FIT / "v5_scoring_worksheet.csv"))
        print(json.dumps(stage_worksheet(target), indent=1))
        return 0
    if args.stage == "reveal":
        target = Path(args.out or (FIT / "audit_v5_reveal.csv"))
        print(json.dumps(stage_reveal(Path(args.scored), target), indent=1))
        return 0
    report = stage_structure() if args.stage == "structure" else stage_join(Path(args.scored))
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.stage == "structure":
        return 0 if report["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
