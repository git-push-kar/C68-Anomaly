"""The six-rung selection rule, with a receipt and a refusal.

Refuses to emit a receipt unless `m` is supplied and the rule is marked committed.
Rung 2 (preference) is skipped when no preference input exists, and the receipt says so.
A choice made only by rung 6 is marked provisional: "A wins ties" is a default, not a
measured win, and a receipt that hid that would be a claim the data cannot carry.

Rung 3 follows the D32 amendment of 2026-11-16, option A:

    `--mechanism` is keyed by cell and carries {"supported", "marked", "items"} - the count of
    knowledge-base-validated claims the hand-marked sheet (`scripts/mechanism_sheet.py`) marks
    supported, out of the claims marked at all, over a stated number of distinct items. A cell
    enters the rung at 15 marked claims from 10 items; a cell below either floor is excluded and
    the exclusion is written into the receipt. Two cells are adjudicated only when their exact
    Clopper-Pearson 95 % intervals do not overlap; when the best cell's interval overlaps another
    cell's, no cell is chosen at rung 3 and the selection proceeds to rung 4 with the overlap
    recorded.

A bare scalar is refused rather than interpreted, because under a raw count the sheet's sampling -
19 sentences from C and 11 from B on this checkout - would decide the rung.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

MARKED_FLOOR = 15
ITEMS_FLOOR = 10
AMENDMENT = "D32 amendment, 2026-11-16, option A (supported-claim rate)"


def sha(path: str) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:24]


def refuse(message: str) -> None:
    """A refusal is exit 2 with the reason on stderr, never a silent default."""
    print(message, file=sys.stderr)
    raise SystemExit(2)


def clopper_pearson(supported: int, marked: int, alpha: float = 0.05) -> list[float]:
    """Exact 95 % interval for a binomial proportion, at this sheet's small denominators."""
    try:
        from scipy.stats import beta
    except ImportError:  # pragma: no cover - scipy is installed on this device
        refuse("refusing: rung 3 needs scipy for the exact interval")
    low = 0.0 if supported == 0 else float(beta.ppf(alpha / 2.0, supported, marked - supported + 1))
    high = (1.0 if supported == marked
            else float(beta.ppf(1.0 - alpha / 2.0, supported + 1, marked - supported)))
    return [round(low, 4), round(high, 4)]


def load_mechanism(path: str) -> dict:
    """Read and validate the rung-3 file; a scalar, a short entry or a bad count is a refusal."""
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        refuse("refusing: the mechanism file must be an object keyed by cell")
    for cell, entry in data.items():
        if isinstance(entry, bool) or not isinstance(entry, dict):
            refuse(
                "refusing: %s carries %r; the %s requires "
                "{'supported': int, 'marked': int, 'items': int}, never a bare count"
                % (cell, entry, AMENDMENT))
        missing = {"supported", "marked", "items"} - set(entry)
        if missing:
            refuse("refusing: %s is missing %s" % (cell, ", ".join(sorted(missing))))
        for key in ("supported", "marked", "items"):
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                refuse("refusing: %s.%s=%r is not a non-negative integer" % (cell, key, value))
        if entry["marked"] <= 0:
            refuse("refusing: %s marks no claims, so it has no rate" % cell)
        if entry["supported"] > entry["marked"]:
            refuse("refusing: %s supports more claims than it marks" % cell)
    return data


def overlaps(one: list[float], two: list[float]) -> bool:
    return not (one[1] < two[0] or two[1] < one[0])


def rung3(passers: list[str], data: dict) -> dict:
    """Apply the amendment to the gate-passers and say what it did, decided or not."""
    per_cell, excluded, eligible = {}, {}, []
    for cell in passers:
        entry = data.get(cell)
        if entry is None:
            continue
        rate = entry["supported"] / entry["marked"]
        per_cell[cell] = {"supported": entry["supported"], "marked": entry["marked"],
                          "items": entry["items"], "rate": round(rate, 4),
                          "interval": clopper_pearson(entry["supported"], entry["marked"])}
        if entry["marked"] < MARKED_FLOOR:
            excluded[cell] = "marked %d < floor %d" % (entry["marked"], MARKED_FLOOR)
        elif entry["items"] < ITEMS_FLOOR:
            excluded[cell] = "items %d < floor %d" % (entry["items"], ITEMS_FLOOR)
        else:
            eligible.append(cell)

    block = {"rule": AMENDMENT, "floors": {"marked": MARKED_FLOOR, "items": ITEMS_FLOOR},
             "per_cell": per_cell, "excluded": excluded, "eligible": sorted(eligible),
             "overlaps": [], "decided": False, "chosen": None,
             "passers": list(passers), "skipped": []}

    if len(eligible) < 2:
        block["skipped"].append(
            "rung 3: %d eligible cell(s); the amendment needs two, so rungs 2 and 3 are "
            "recorded as unavailable" % len(eligible))
        return block

    best = max(eligible, key=lambda cell: (per_cell[cell]["rate"], per_cell[cell]["supported"]))
    contenders = [cell for cell in eligible
                  if overlaps(per_cell[cell]["interval"], per_cell[best]["interval"])]
    block["best_point_estimate"] = best
    block["overlaps"] = [[best, cell] for cell in contenders if cell != best]
    if len(contenders) == 1:
        block.update({"decided": True, "chosen": best, "passers": [best]})
        return block
    block["passers"] = contenders
    block["skipped"].append(
        "rung 3: interval of %s overlaps %s, so no cell is chosen at this rung"
        % (best, ", ".join(cell for cell in contenders if cell != best)))
    return block


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--m", type=float, default=None)
    ap.add_argument("--rule-committed", action="store_true")
    ap.add_argument("--preference", default=None, help="json: {condition: {wins, n}}")
    ap.add_argument("--mechanism", default=None,
                    help="json: {condition: {supported, marked, items}} - D32 amendment, option A")
    ap.add_argument("--invariance", default=None, help="json: {condition: rate}")
    ap.add_argument("--cost", default=None,
                    help="json: {condition: value} or {unit, basis, values: {condition: value}}; "
                         "the unit is recorded in the receipt verbatim")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.m is None:
        print("refusing: m is unset (D3 requires a declared margin)", file=sys.stderr)
        return 2
    if not args.rule_committed:
        print("refusing: the six-rung rule is not marked committed", file=sys.stderr)
        return 2

    summary = json.loads(pathlib.Path(args.cells).read_text(encoding="utf-8"))["summary"]
    base = summary.get("A")
    if not base:
        print("refusing: no A row to anchor the gate", file=sys.stderr)
        return 2

    ceiling = base["numeric_rate"] + args.m
    passers = sorted(c for c, v in summary.items()
                     if v["numeric_rate"] <= ceiling + 1e-9)
    skipped, chosen, rung = [], None, None

    if args.preference:
        pref = json.loads(pathlib.Path(args.preference).read_text(encoding="utf-8"))
        scored = [(c, pref[c]["wins"] / max(pref[c]["n"], 1))
                  for c in passers if c in pref and pref[c].get("n")]
        if scored:
            best = max(v for _, v in scored)
            top = [c for c, v in scored if abs(v - best) < 1e-12]
            if len(top) == 1:
                chosen, rung = top[0], 2
            else:
                passers = top
                skipped.append("rung 2: tie among passers -> next rung")
        else:
            skipped.append("rung 2: preference file names no passer")
    else:
        skipped.append("rung 2: no preference data; tier removed per the plan")

    mechanism = None
    if chosen is None:
        if args.mechanism:
            mechanism = rung3(passers, load_mechanism(args.mechanism))
            skipped.extend(mechanism["skipped"])
            if mechanism["decided"]:
                chosen, rung = mechanism["chosen"], 3
            else:
                passers = mechanism["passers"]
        else:
            skipped.append("rung 3: no mechanism data; the amendment's floors were not applied")

    measured: list[int] = []
    tie_break = None
    cost_meta = None
    best = None
    for key, source, better, rung_id in (
            ("invariance", args.invariance, max, 4),
            ("cost", args.cost, min, 5)):
        if chosen is not None or not source:
            continue
        data = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "values" in data:
            sig = data.get("signature") or {}
            cost_meta = {"unit": data.get("unit"), "basis": data.get("basis"),
                         "signed": data.get("signed", sig.get("signed")),
                         "signed_by": sig.get("signed_by"), "signed_date": sig.get("date")}
            data = data["values"]
        scored = [(c, data[c]) for c in passers if c in data]
        if not scored:
            continue
        measured.append(rung_id)
        best = better(v for _, v in scored)
        top = [c for c, v in scored if v == best]
        if len(top) == 1:
            chosen, rung = top[0], rung_id
        else:
            passers = top

    if chosen is None:
        chosen = "A" if "A" in passers else passers[0]
        rung = 6
        if measured:
            tie_break = ("rung %d carried input and left %s tied at %s; %s is chosen by the "
                         "declared tie-break, not by default"
                         % (measured[-1], sorted(passers), best, chosen))

    receipt = {
        "rule": "six rungs: numeric gate, preference, mechanism validity, order "
                "invariance, cost, A wins ties",
        "amendments": [AMENDMENT],
        "m": args.m,
        "gate_ceiling": round(ceiling, 6),
        "candidates": sorted(summary),
        "passers": passers,
        "chosen": chosen,
        "decided_by_rung": rung,
        "measured_rungs_with_input": measured,
        "tie_break": tie_break,
        "provisional": rung == 6 and not measured,
        "skipped": skipped,
        "mechanism": mechanism,
        "cost": cost_meta if args.cost else None,
        "inputs": {k: {"path": v, "sha256_24": sha(v)} for k, v in
                   (("cells", args.cells), ("preference", args.preference),
                    ("mechanism", args.mechanism), ("invariance", args.invariance),
                    ("cost", args.cost)) if v},
        "note": ("rung 6 is the deterministic default, not a measured win; this receipt "
                 "stays provisional until rungs 2 and 3 carry data" if not measured and rung == 6 else
                 "decided by a measured rung" if rung != 6 else
                 "no measured rung chose a single cell; the declared tie-break resolved the tie "
                 "a measured rung produced"),
    }
    pathlib.Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
