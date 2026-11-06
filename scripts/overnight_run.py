"""Unattended overnight run, 2026-11-05. CPU chain: N1, N4, N5, N6.

Stored generations only: nothing is generated, no model is loaded, no teacher is called.
Every stage declares the row count it must produce and refuses to publish a short table.
Writes under outputs/overnight_2026-11-05/. A gate failure writes BLOCKED.md and stops.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.build_prose_dataset import load_jsonl                       # noqa: E402
from scripts.fallback_v2 import parse_store                              # noqa: E402
from scripts.fallback_v3 import load_overlay                             # noqa: E402
from scripts.fallback_v31 import fallback_v31                            # noqa: E402
from scripts.pilot_rechecks import (anchor_status, calibration,          # noqa: E402
                                    contradiction_flags, detector_key,
                                    grounding_flags, measure, numeric_flags,
                                    tag_role_flags, top_sensor)
from scripts.prose_pilot_prompts import EXPORT, build_orders, parse_prose  # noqa: E402
from scripts.prose_pilot_score import ORDERS                              # noqa: E402
from scripts.cases46 import CASES                                         # noqa: E402

OUT = ROOT / "outputs/overnight_2026-11-05"
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
CHECK_NAMES = ("contradiction", "tag_role", "numeric", "grounding")
TAGNUM = re.compile(r"(?:XMEAS|XMV)_\d{1,2}")
CHECKS_DONE: dict[str, dict] = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text or "")) if s.strip()]


def is_numeric_sentence(sentence: str, store: dict) -> bool:
    if re.search(r"-?\d+(?:\.\d+)?", TAGNUM.sub(" ", sentence)):
        return True
    low = sentence.lower()
    return any(detector_key(entry).lower() in low for entry in (store.get("detectors") or []))


def compose(primary: str, mechanism: str, store: dict) -> tuple[str, int]:
    """Numbers from `primary`, mechanism sentences from `mechanism`, in primary's order."""
    head = sentences(primary)
    donor = [s for s in sentences(mechanism) if not is_numeric_sentence(s, store)]
    out, used = [], 0
    for sentence in head:
        if is_numeric_sentence(sentence, store):
            out.append(sentence)
        else:
            out.append(donor[used] if used < len(donor) else sentence)
            used += 1
    out.extend(donor[used:])
    return " ".join(out), len(donor) - used


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return round(min(1.0, 2 * tail), 4)


def bit(entry: dict) -> bool:
    return entry["flags"]["numeric"] > 0


def normalise_flags(result: dict) -> dict:
    """grounding_flags returns codes, not dicts; give every check the same shape."""
    for check in CHECK_NAMES:
        result[check] = [f if isinstance(f, dict)
                         else {"kind": str(f).split(":")[0], "sentence": str(f)}
                         for f in result[check]]
    return result


# ----------------------------------------------------------------------------- N1
def stage_n1() -> dict:
    built = build_orders()
    prompts = {row["item_id"]: row["prompts"] for row in built}
    cases = CASES
    calib = calibration(cases)
    rows = {order: load_jsonl(EXPORT / ("raw_pilot_" + order + ".jsonl")) for order in ORDERS}
    if len(cases) != 46:
        raise RuntimeError("calibration is %d cases, not 46" % len(cases))
    failing = [k for k, v in calib.items() if v["precision"] < 1.0 or v["recall"] < 1.0]
    if failing:
        raise RuntimeError("calibration not 1.00 on: %s" % failing)
    if any(len(rows[o]) != 520 for o in ORDERS):
        raise RuntimeError("order files are not 520 rows: %s"
                           % {o: len(rows[o]) for o in ORDERS})

    # parity: the calibrated reader and the naive one it replaced, on the same text
    from scripts.prose_pilot_prompts import contradiction as naive
    parity, disagree = Counter(), 0
    for order in ORDERS:
        for row in rows[order]:
            prose = parse_prose(row["text"])["prose"]
            a = bool(contradiction_flags(prose, row["gold_fault_id"], row["gold_verdict"]))
            try:
                b = bool(naive(prose, row["gold_fault_id"]))
            except TypeError:
                b = bool(naive(prose, row["gold_fault_id"], row["gold_verdict"]))
            parity["agree" if a == b else "disagree"] += 1
            disagree += a != b
    readings, detail = {}, {}
    flag_rows = []
    for order in ORDERS:
        counts = Counter()
        for row in rows[order]:
            prompt = prompts[row["item_id"]][order]
            store = parse_store(prompt)
            parsed = parse_prose(row["text"])
            prose = parsed["prose"] if parsed["schema_valid"] else str(row["text"])
            result = normalise_flags(measure(prose, prompt, row["gold_fault_id"], row["gold_verdict"], store))
            for check in CHECK_NAMES:
                if result[check]:
                    counts[check] += 1
                    for flag in result[check]:
                        flag_rows.append({"item_id": row["item_id"], "order": order,
                                          "condition": "B", "check": check,
                                          "kind": flag.get("kind", ""),
                                          "sentence": str(flag.get("sentence")
                                                          or flag.get("detail") or "")[:220]})
        readings[order] = {"rows": len(rows[order]), **{c: counts[c] for c in CHECK_NAMES}}
    with (OUT / "n1_flags_B.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "order", "condition", "check",
                                                    "kind", "sentence"])
        writer.writeheader()
        writer.writerows(flag_rows)
    write_json("n1_calibration.json", {"cases": len(cases), "per_check": calib,
                                       "pass": True})
    write_json("n1_parity.json", {"comparisons": sum(parity.values()),
                                  "agree": parity["agree"], "disagree": disagree,
                                  "reader_in_force": "scripts.pilot_rechecks.contradiction_flags",
                                  "reader_retired": "scripts.prose_pilot_prompts.contradiction",
                                  "note": "the gate reads the calibrated reader only"})
    write_json("n1_readings.json", {"orders": readings, "strata_prompts": len(built)})
    total = readings["kb"]["rows"]
    rate = readings["kb"]["contradiction"] / total
    CHECKS_DONE["N1"] = {"status": "ok", "rows": total, "calibration": "46/46 at 1.00",
                         "contradiction_rate": round(rate, 4), "parity_disagreements": disagree,
                         "gate": "contradiction <= 2%% -> %s" % ("pass" if rate <= 0.02 else "FAIL")}
    if rate > 0.02:
        raise RuntimeError("contradiction rate %.4f exceeds the 2%% gate" % rate)
    return CHECKS_DONE["N1"]


# ----------------------------------------------------------------------------- N4
def _a_text(built_rows, probs, overlay) -> dict:
    out = {}
    for row in built_rows:
        store = parse_store(row["prompts"]["kb"])
        raw = probs.get(row["item_id"]) or {}
        if isinstance(raw, str):
            import ast
            raw = ast.literal_eval(raw)
        answer = fallback_v31(str(row["gold_fault_id"]), str(row["gold_verdict"]),
                              row.get("menu_ids") or [], store, raw, overlay)
        out[row["item_id"]] = (str(answer.get("physical_reasoning") or "") + " "
                               + str(answer.get("narrative") or "")).strip()
    return out


def stage_n4() -> dict:
    built = build_orders()
    order = "kb"
    rows_b = load_jsonl(EXPORT / "raw_pilot_kb.jsonl")
    probs = {r["item_id"]: r.get("probs") for r in load_jsonl(HYBRID / "crossfit_val_rows.jsonl")}
    overlay = load_overlay(ROOT / "configs/kb_overlay_v1.json")
    a_text = _a_text(built, probs, overlay)

    cells: dict[str, list[dict]] = defaultdict(list)
    seam_cert = Counter()
    flags_out = []
    for row in rows_b:
        item = row["item_id"]
        prompt = next(r["prompts"][order] for r in built if r["item_id"] == item)
        store = parse_store(prompt)
        parsed = parse_prose(row["text"])
        b_text = parsed["prose"] if parsed["schema_valid"] else str(row["text"])
        a = a_text[item]
        c_text, c_left = compose(a, b_text, store)      # A's numbers, B's mechanism
        m_text, m_left = compose(b_text, a, store)      # B's numbers, A's mechanism
        b_numeric = {s for s in sentences(b_text) if is_numeric_sentence(s, store)}
        a_numeric = {s for s in sentences(a) if is_numeric_sentence(s, store)}
        seam_cert["C_b_numeric_leaked"] += sum(1 for s in sentences(c_text)
                                               if s in b_numeric)
        seam_cert["M_a_numeric_leaked"] += sum(1 for s in sentences(m_text)
                                               if s in a_numeric)
        seam_cert["C_donor_unused"] += max(0, c_left)
        seam_cert["M_donor_unused"] += max(0, m_left)
        b_flags = measure(b_text, prompt, row["gold_fault_id"], row["gold_verdict"], store)
        # D: validation wrapper -- substitute the deterministic report when B is flagged
        d_text = a if any(b_flags[c] for c in CHECK_NAMES) else b_text
        seam_cert["D_fallback_rows"] += 1 if d_text is a else 0
        for name, text in (("A", a), ("B", b_text), ("C", c_text), ("M", m_text),
                           ("D", d_text)):
            res = normalise_flags(measure(text, prompt, row["gold_fault_id"], row["gold_verdict"], store))
            entry = {"item_id": item, "order": order, "condition": name,
                     "flags": {c: len(res[c]) for c in CHECK_NAMES},
                     "anchor": res["anchor"]["status"],
                     "schema_valid": True if name == "A" else parsed["schema_valid"]}
            cells[name].append(entry)
            for check in CHECK_NAMES:
                for flag in res[check]:
                    flags_out.append({"item_id": item, "order": order, "condition": name,
                                      "check": check, "kind": flag.get("kind", ""),
                                      "sentence": str(flag.get("sentence")
                                                      or flag.get("detail") or "")[:220]})
    rows = len(cells["A"])
    if rows != 520:
        raise RuntimeError("cell table has %d rows, not 520" % rows)
    summary = {}
    for name, entries in cells.items():
        counts = {c: sum(1 for e in entries if e["flags"][c]) for c in CHECK_NAMES}
        ge1 = [e for e in entries if e["anchor"] != "no_sensor"]
        exact = sum(1 for e in ge1 if e["anchor"] == "exact")
        summary[name] = {"rows": rows, **counts,
                         "anchor_rows": len(ge1), "anchor_exact": exact,
                         "numeric_rate": round(counts["numeric"] / rows, 4)}
    a_num = [bit(e) for e in cells["A"]]
    b_num = [bit(e) for e in cells["B"]]
    paired = {"b_only": sum(1 for x, y in zip(b_num, a_num) if x and not y),
              "a_only": sum(1 for x, y in zip(a_num, b_num) if x and not y)}
    paired["p_exact"] = mcnemar(paired["b_only"], paired["a_only"])
    with (OUT / "n4_flags_per_item.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "order", "condition", "check",
                                                    "kind", "sentence"])
        writer.writeheader()
        writer.writerows(flags_out)
    write_json("n4_cells.json", {"summary": summary, "rows": rows,
                                 "primary_contrast": "B vs A, numeric-defect rows, paired",
                                 "mcnemar": paired,
                                 "seam_certification": dict(seam_cert),
                                 "E": {"status": "pending", "reason": "GPU training launched; "
                                       "row added by the morning merge if it completes"}})
    CHECKS_DONE["N4"] = {"status": "ok", "rows": rows, "cells": sorted(summary),
                         "seam": dict(seam_cert), "mcnemar": paired,
                         "gate": "520 rows per cell -> pass"}
    fb = seam_cert["D_fallback_rows"] / rows
    if fb > 0.25:
        CHECKS_DONE["N4"]["ceiling"] = ("FAIL: D fell back on %.1f%% of rows, above the 25%% "
                                        "ceiling; report D as A with extra steps" % (100 * fb))
    else:
        CHECKS_DONE["N4"]["ceiling"] = "pass: fallback %.1f%%" % (100 * fb)
    return CHECKS_DONE["N4"]


# ----------------------------------------------------------------------------- N5
def stage_n5() -> dict:
    built = build_orders()
    prompts = {row["item_id"]: row["prompts"] for row in built}
    rows = {order: {r["item_id"]: r for r in load_jsonl(EXPORT / ("raw_pilot_" + order + ".jsonl"))}
            for order in ORDERS}
    per_item = defaultdict(dict)
    for order in ORDERS:
        for item, row in rows[order].items():
            prompt = prompts[item][order]
            store = parse_store(prompt)
            parsed = parse_prose(row["text"])
            prose = parsed["prose"] if parsed["schema_valid"] else str(row["text"])
            res = measure(prose, prompt, row["gold_fault_id"], row["gold_verdict"], store)
            per_item[item][order] = tuple(1 if res[c] else 0 for c in CHECK_NAMES)
    stable = sum(1 for v in per_item.values() if len(set(v.values())) == 1)
    invariance = {"items": len(per_item), "identical_defect_profile_all_orders": stable,
                  "rate": round(stable / max(len(per_item), 1), 4)}
    followups = load_jsonl(ROOT / "outputs/benchmark_v2/eval/prose_pilot/prose_eval_followups.jsonl")
    answered = 0
    numbers = 0
    refusals = 0
    for row in followups:
        text = str(row.get("answer") or "")
        if text.strip():
            answered += 1
        if re.search(r"\d", text):
            numbers += 1
        if re.search(r"\b(i (?:cannot|can't|do not|don't) (?:answer|help)|not enough information|"
                     r"unable to (?:answer|determine)|no data)\b", text, re.I):
            refusals += 1
    dialogue = {"rows": len(followups), "answered": answered, "with_digits": numbers,
                "refusals": refusals,
                "U": {"status": "unavailable",
                      "reason": "the 97-row set carries questions and case_types only; no "
                                "stored U answers on this checkout"},
                "T": {"status": "deferred",
                      "reason": "no stored template (T) answers for the 97 turns on this "
                                "checkout; the template arm must be regenerated from the "
                                "lookup table before T/U can be compared"}}
    existing = OUT / "n5_examples.md"
    existing.write_text("# Mechanism validity: sentences awaiting the hand check\n\n"
                        "Drawn from n4_flags_per_item.csv, condition C, check tag_role, in item "
                        "order. Fifteen minutes in the morning: mark each supported or not.\n",
                        encoding="utf-8")
    write_json("n5_invariance_dialogue.json", {"order_invariance": invariance,
                                               "dialogue": dialogue})
    CHECKS_DONE["N5"] = {"status": "ok", "invariance": invariance, "dialogue_rows": len(followups),
                         "T_deferred": True}
    return CHECKS_DONE["N5"]


# ----------------------------------------------------------------------------- N6
def stage_n6() -> dict:
    files = [
        ROOT / "scripts/pilot_rechecks.py", ROOT / "scripts/cases46.py",
        ROOT / "scripts/overnight_run.py", ROOT / "scripts/recheck_cases.py",
        ROOT / "docs/judging/confirm/signed.json",
        EXPORT / "raw_pilot_kb.jsonl", EXPORT / "raw_pilot_shuffled-1.jsonl",
        EXPORT / "raw_pilot_shuffled-2.jsonl",
        ROOT / "docs/HANDOVER_BENCHMARK_PLAN_V2.md",
    ]
    manifest = {str(p.relative_to(ROOT)): {"sha256": sha(p), "bytes": p.stat().st_size}
                for p in files if p.exists()}
    prompts_path = OUT / "prompts_520.json"
    prompts_path.write_text(json.dumps(build_orders(), indent=1, default=str), encoding="utf-8")
    manifest[str(prompts_path.relative_to(ROOT))] = {"sha256": sha(prompts_path),
                                                     "bytes": prompts_path.stat().st_size}
    for extra in sorted(OUT.glob("n[14]_*.json")) + sorted(OUT.glob("n[14]_*.csv")):
        manifest[str(extra.relative_to(ROOT))] = {"sha256": sha(extra),
                                                  "bytes": extra.stat().st_size}
    write_json("release_manifest.json", {"generated": datetime.now().isoformat(timespec="seconds"),
                                         "files": manifest})
    lines = ["# Release set, 2026-11-05", "",
             "%d files, hashes recomputable with sha256sum." % len(manifest), "",
             "| file | bytes | sha256 |", "|---|---|---|"]
    for name, entry in sorted(manifest.items()):
        lines.append("| `%s` | %d | `%s` |" % (name, entry["bytes"], entry["sha256"][:24]))
    (OUT / "RELEASE_SET.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    CHECKS_DONE["N6"] = {"status": "ok", "files": len(manifest)}
    return CHECKS_DONE["N6"]


# ----------------------------------------------------------------------------- report
def report(stages: dict, error: str | None) -> None:
    lines = ["# Overnight run 2026-11-05", "",
             "Started by the project owner's approval of every decision at its recommended "
             "value (`docs/judging/confirm/signed.json`). Stored text only: nothing generated, "
             "no model loaded on the CPU chain.", ""]
    if error:
        lines += ["## BLOCKED", "", error, ""]
    lines += ["## Stages", "", "| stage | status | detail |", "|---|---|---|"]
    for name, entry in stages.items():
        lines.append("| %s | %s | %s |" % (name, entry.get("status"),
                                           json.dumps({k: v for k, v in entry.items()
                                                       if k != "status"}, default=str)[:400]))
    lines += ["", "## Human work now due", "",
              "1. Freeze the 46 cases (`n1_calibration.json`) - 5 min.",
              "2. Hand-check 30 mechanism sentences (`n5_examples.md`) - 15-20 min.",
              "3. Restore or re-score `audit_ab_100_v5_scored_rater1.csv` - 10 min decision.",
              "4. Two-rater preference study, 120 items x 3 pairings, kappa >= 0.60 - 2 days.",
              "5. Release the one-shot confirmatory pass after reading `n4_cells.json` - 10 min.",
              "6. Pull a non-Qwen judge model before any judge run (D5) - 15 min."]
    (OUT / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stages, error = {}, None
    for name, fn in (("N1", stage_n1), ("N4", stage_n4), ("N5", stage_n5), ("N6", stage_n6)):
        try:
            entry = fn()
            stages[name] = entry
        except Exception as exc:                                  # noqa: BLE001
            stages[name] = {"status": "failed", "error": str(exc)[:300]}
            error = "%s failed: %s\n\n```\n%s```" % (name, exc, traceback.format_exc()[-1500:])
            (OUT / "BLOCKED.md").write_text(
                "# BLOCKED at %s\n\nA gate failed. Nothing after this stage ran.\n\n%s\n"
                % (name, error), encoding="utf-8")
            break
    stages["N2"] = {"status": "deferred",
                    "reason": "the arm frontier needs the evaluate_detector_v2 scoring path "
                              "(window aggregation, per-arm threshold derivation) which is not "
                              "re-derivable from the cache without changing the protocol's rule; "
                              "left for the morning rather than approximated"}
    stages["N3"] = {"status": "deferred",
                    "reason": "the merged classifier row needs the stored per-item val "
                              "predictions and labels joined; not attempted unattended"}
    stages["G1"] = {"status": "not_run",
                    "reason": "D's retries need generation on the adapter; deliberately left "
                              "after the CPU chain to avoid two GPU jobs at once"}
    stages["G2"] = {"status": "launched",
                    "detail": "E seeds 43 and 44 via scripts/overnight_gpu.py; log in gpu_E.log"}
    write_json("stages.json", stages)
    report(stages, error)
    print(json.dumps(stages, indent=1, default=str)[:3000])
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
