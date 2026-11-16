"""Validate a model judge against the frozen D30 sheet (D31).

The frozen reading is D30's strict-grounded second rater (section 20): a number that is not a
panel value is not grounded, derived or not. A judge is only usable as the corroboration tier if
it is scored against that sheet and its agreement is reported, never assumed.

The judge here is an instruction model outside the SFT teacher's family (D5), called through the
local Ollama server. It sees exactly what a rater saw: the four panel fields and the two answers
under the same side labels, with no key and no provenance. Its four cells are the ones the
`d30_score_confirmation_prompt.md` defines, and the strict rule is stated in the prompt so the
comparison is to rater 2's rule rather than to an unnamed one.

Timing first, because an 8B judge on this device has no measured throughput:

    python scripts/judge_validate.py --limit 1 --timing
    python scripts/judge_validate.py            # full sheet, resuming any saved rows

Exit status is 0 when the sheet was fully scored, 2 when the judge or the sheet is unusable, and
1 when the pass completed but too few items were valid to report agreement.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/judging/confirm/evidence_2026-11-06"
SHEET_R2 = EVIDENCE / "audit_ab_100_v5_scored_rater2_strict_grounded.csv"
SHEET_R1 = EVIDENCE / "audit_ab_100_v5_scored_rater1.csv"
OUT_JSON = EVIDENCE / "d31_judge_validate.json"
OUT_RAW = EVIDENCE / "d31_judge_raw.jsonl"

CELLS = ("correct", "grounded", "consistent", "useful")
SIDES = ("A", "B")

PROMPT = """You audit two candidate operator answers written about the same plant alarm.

Evidence panel for this event:
- Diagnosis supplied to both answers: {supplied}
- Detectors: {detectors}
- Top sensors by absolute z: {sensors}
- Residual shares: {residuals}

Answer A:
{text_a}

Answer B:
{text_b}

Score each answer on four cells, each 1 or 0, then state a preference. Definitions:
correct - measured against the supplied diagnosis and the detectors: 1 if the answer asserts
nothing the panel does not support, else 0.
grounded - measured against the detectors, top sensors and residual shares: 1 only if every number
and tag in the answer appears in the panel and is used in the right direction, else 0. STRICT
RULE: a derived number - a ratio, a percentage, a multiple, or a difference computed from panel
values - is NOT panel content, so an answer that reports one is 0 on this cell.
consistent - measured against the answer alone: 1 if it does not contradict itself, else 0.
useful - measured against the operator's next action: 1 if it names the top sensor, its
direction, and what to check, else 0.
preference - which answer you would rather an operator read: "A", "B" or "TIE".

Reply with JSON only, no prose outside it:
{{"correct_A":0,"grounded_A":0,"consistent_A":0,"useful_A":0,"correct_B":0,"grounded_B":0,
"consistent_B":0,"useful_B":0,"preference":"A","reason":"at most twenty words"}}"""


def read_rows(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for column, value in list(row.items()):
            if isinstance(value, str) and "\r" in value:
                row[column] = value.replace("\r\n", "\n").replace("\r", "\n")
    return rows


def panel_prompt(row: dict) -> str:
    return PROMPT.format(
        supplied=row["supplied_diagnosis"].strip(),
        detectors=row["detectors"].strip(),
        sensors=row["top_sensors"].strip(),
        residuals=row["residual_shares"].strip(),
        text_a=row["text_A"].strip(),
        text_b=row["text_B"].strip(),
    )


def post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return json.loads(handle.read().decode("utf-8"))


def server_model(host: str, model: str) -> dict:
    """Fail loudly on a missing server or model; return the tag record for provenance."""
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=15) as handle:
            tags = json.loads(handle.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise SystemExit("cannot reach the Ollama server at %s: %s" % (host, error))
    for entry in tags.get("models") or []:
        if entry.get("name") == model or entry.get("name") == model + ":latest":
            return entry
    raise SystemExit("model %r is not installed; have: %s"
                     % (model, ", ".join(sorted(e.get("name", "") for e in
                                                tags.get("models") or []))))


def parse_scores(text: str) -> dict | None:
    """Coerce a reply into the nine cells, or None when the reply cannot carry them."""
    body = text.strip()
    if not body:
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", body, re.S)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    scores = {}
    for cell in CELLS:
        for side in SIDES:
            value = payload.get("%s_%s" % (cell, side))
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, str) and value.strip() in ("0", "1"):
                value = int(value.strip())
            if value not in (0, 1):
                return None
            scores["%s_%s" % (cell, side)] = int(value)
    preference = str(payload.get("preference") or "").strip().upper()
    if preference not in ("A", "B", "TIE"):
        return None
    scores["preference"] = preference
    scores["reason"] = str(payload.get("reason") or "")[:200]
    return scores


def flip(scores: dict) -> dict:
    """Undo a side swap: the cells and the preference come back to the sheet's A/B order."""
    out = {}
    for cell in CELLS:
        out["%s_A" % cell] = scores["%s_B" % cell]
        out["%s_B" % cell] = scores["%s_A" % cell]
    out["preference"] = {"A": "B", "B": "A", "TIE": "TIE"}[scores["preference"]]
    out["reason"] = scores.get("reason", "")
    return out


def kappa(pairs: list[tuple[int, int]]) -> dict:
    n = len(pairs)
    a = sum(1 for x, y in pairs if x and y)
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    d = n - a - b - c
    po = (a + d) / n
    pe = ((a + b) / n) * ((a + c) / n) + ((c + d) / n) * ((b + d) / n)
    return {"pairs": n, "agree": a + d, "po": round(po, 4), "pe": round(pe, 4),
            "kappa": None if pe >= 1.0 else round((po - pe) / (1 - pe), 4),
            "judge_ones": a + b, "reader_ones": a + c}


def validate(judge: dict, reader: dict, label: str) -> dict:
    # Sheet cells are strings; compare integers, or "0" is truthy and every reader cell counts as 1.
    per = {cell: kappa([(int(judge[item]["%s_A" % cell]), int(reader[item]["%s_A" % cell]))
                        for item in judge] +
                       [(int(judge[item]["%s_B" % cell]), int(reader[item]["%s_B" % cell]))
                        for item in judge]) for cell in CELLS}
    same = tie = opposite = 0
    opposite_rows = []
    for item in judge:
        chosen = judge[item]["preference"]
        seen = str(reader[item]["preference"]).strip().upper()
        if chosen == seen:
            same += 1
        elif chosen == "TIE" or seen == "TIE":
            tie += 1
        else:
            opposite += 1
            opposite_rows.append(item)
    cells_total = sum(per[cell]["pairs"] for cell in CELLS)
    cells_agree = sum(per[cell]["agree"] for cell in CELLS)
    return {"against": label, "per_metric": per,
            "cells": {"total": cells_total, "agree": cells_agree,
                      "share": round(cells_agree / cells_total, 4)},
            "preference": {"rows": len(judge), "same": same, "one_tie": tie,
                           "opposite": opposite, "opposite_rows": opposite_rows}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", default=str(SHEET_R2), help="frozen sheet the judge is scored on")
    ap.add_argument("--compare", default=str(SHEET_R1), help="the other reader, for a second column")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=0, help="score at most N rows")
    ap.add_argument("--ids", default="", help="comma-separated item ids to score")
    ap.add_argument("--timing", action="store_true", help="report per-row wall time and token counts")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--raw", default=str(OUT_RAW))
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--fresh", action="store_true", help="drop saved rows and score again")
    ap.add_argument("--swap", action="store_true",
                    help="present text_B as Answer A and text_A as Answer B, then un-swap: "
                         "the order control for a position-biased judge")
    args = ap.parse_args()

    rows = read_rows(pathlib.Path(args.sheet))
    if not rows:
        print("refusing: the sheet is empty", file=sys.stderr)
        return 2
    by_id = {row["item_id"]: row for row in rows}
    wanted = [item for item in by_id if not args.ids or item in args.ids.split(",")]
    wanted.sort()
    if args.limit:
        wanted = wanted[:args.limit]

    tag = server_model(args.host, args.model)
    print("judge %s (digest %s) on %d of %d rows of %s"
          % (args.model, str(tag.get("digest") or "")[:16], len(wanted), len(rows),
             pathlib.Path(args.sheet).name))

    raw_path = pathlib.Path(args.raw)
    saved: dict[str, dict] = {}
    if raw_path.exists() and not args.fresh:
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                saved[record["item_id"]] = record
    elif args.fresh and raw_path.exists():
        raw_path.unlink()

    times = []
    handle = raw_path.open("a", encoding="utf-8")
    try:
        for index, item in enumerate(wanted, 1):
            if (item in saved and saved[item].get("parsed")
                    and bool(saved[item].get("swap")) == bool(args.swap)):
                continue
            row = by_id[item]
            if args.swap:
                row = dict(row, text_A=row["text_B"], text_B=row["text_A"])
            prompt = panel_prompt(row)
            started = time.perf_counter()
            reply = post(args.host + "/api/chat", {
                "model": args.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "seed": 42, "num_ctx": 8192,
                            "num_predict": 400},
                "messages": [{"role": "user", "content": prompt}],
            }, args.timeout)
            elapsed = time.perf_counter() - started
            content = str((reply.get("message") or {}).get("content") or "")
            parsed = parse_scores(content)
            if parsed is not None and args.swap:
                parsed = flip(parsed)
            if parsed is None:
                # One stricter retry, same prompt plus the schema, because an unparsed row is a
                # missing row rather than a low score.
                retry = post(args.host + "/api/chat", {
                    "model": args.model,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0, "seed": 43, "num_ctx": 8192,
                                "num_predict": 400},
                    "messages": [{"role": "user", "content": prompt +
                                  "\n\nReturn the JSON object and nothing else. Every one of the "
                                  "eight cells is the integer 0 or 1; preference is A, B or TIE."}],
                }, args.timeout)
                content = str((retry.get("message") or {}).get("content") or "")
                parsed = parse_scores(content)
                if parsed is not None and args.swap:
                    parsed = flip(parsed)
                reply = retry
                elapsed = time.perf_counter() - started
            record = {"item_id": item, "model": args.model, "swap": bool(args.swap),
                      "digest": tag.get("digest"), "parsed": parsed,
                      "raw": content[:4000], "seconds": round(elapsed, 2),
                      "eval_count": reply.get("eval_count"),
                      "prompt_eval_count": reply.get("prompt_eval_count")}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            saved[item] = record
            times.append(elapsed)
            if args.timing:
                print("  %-7s %6.2fs  prompt %4d tok  out %3s tok  %s"
                      % (item, elapsed, reply.get("prompt_eval_count") or 0,
                         reply.get("eval_count"), "parsed" if parsed else "UNPARSED"))
    finally:
        handle.close()

    judged = {item: saved[item]["parsed"] for item in wanted
              if item in saved and saved[item].get("parsed")}
    invalid = [item for item in wanted if item not in judged]
    if not judged:
        print("no parsed rows: the judge produced nothing usable", file=sys.stderr)
        return 2

    frozen = {row["item_id"]: row for row in read_rows(pathlib.Path(args.sheet))}
    other = {row["item_id"]: row for row in read_rows(pathlib.Path(args.compare))}
    report = {
        "date": time.strftime("%Y-%m-%d"),
        "gate": "D31",
        "question": "does an out-of-family judge reproduce the frozen D30 reading",
        "model": args.model, "digest": tag.get("digest"),
        "sheet": pathlib.Path(args.sheet).name,
        "rule": "strict grounded: a derived number is not panel content",
        "order": "text_B shown as Answer A (swap control)" if args.swap else "text_A shown as Answer A",
        "rows_scored": len(judged), "rows_invalid": invalid,
        "timing": {"rows": len(times),
                   "mean_seconds": round(sum(times) / len(times), 2) if times else None,
                   "max_seconds": round(max(times), 2) if times else None},
        "against_frozen": validate(judged, frozen, "rater2 strict (frozen)"),
        "against_other": validate(judged, other, "rater1"),
    }
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nrows parsed %d, invalid %d" % (len(judged), len(invalid)))
    if invalid:
        print("  invalid:", ", ".join(invalid))
    for block in (report["against_frozen"], report["against_other"]):
        print("against %s: cells %d/%d (%.1f%%)"
              % (block["against"], block["cells"]["agree"], block["cells"]["total"],
                 100 * block["cells"]["share"]))
        for cell in CELLS:
            entry = block["per_metric"][cell]
            kappa_text = "n/a" if entry["kappa"] is None else "%+.4f" % entry["kappa"]
            print("  %-10s agree %2d/%d  po %.4f  kappa %s  judge %d  reader %d"
                  % (cell, entry["agree"], entry["pairs"], entry["po"], kappa_text,
                     entry["judge_ones"], entry["reader_ones"]))
        pref = block["preference"]
        print("  preference same %d, one tie %d, opposite %d%s"
              % (pref["same"], pref["one_tie"], pref["opposite"],
                 " (%s)" % ", ".join(pref["opposite_rows"]) if pref["opposite_rows"] else ""))
    print("\nwrote", args.out, "and", args.raw)
    return 0 if not invalid else 1


if __name__ == "__main__":
    sys.exit(main())
