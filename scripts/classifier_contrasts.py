"""P2 - the merged classifier row: menu rule vs hybrid vs the adapter's own label.

Three stored systems, compared on the items all three scored, with exact McNemar on correctness
and the abstention gates reported per system:

  menu rule        `classifier_label` in `hybrid/preds_protocolA_val.jsonl` - the label the menu
                   classifier produced before any rule overrode it
  hybrid           `pred` in the same file - the label after the threshold, menu and no-fault rules
  adapter          `pred` in `evidence_classifier/preds_adapter_val503.jsonl` - the 2B model's own
                   decoded label

Abstention is the label `UNCLASSIFIED_OPEN_SET` on either side: a system abstains when it predicts
it, and an abstention is *correct* when the gold label is the same. That gives the four rates the
plan names - abstention precision, abstention recall, false abstention and missed abstention.

The plan asks for root cause, subsystem and severity as one merged row. The stored predictions carry
root cause only, so this script reports the merged row as root cause plus a statement of what is
missing; subsystem and severity are functions of the predicted fault id in the knowledge base, not
independent outputs of any of the three systems, and inventing them here would be a result the
artefacts do not contain.

Usage:
    python scripts/classifier_contrasts.py
    python scripts/classifier_contrasts.py --out-json ... --out-md ...
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid/preds_protocolA_val.jsonl"
ADAPTER = ROOT / "outputs/analysis/phase3a/evidence_classifier/preds_adapter_val503.jsonl"
ABSTAIN = "UNCLASSIFIED_OPEN_SET"
SYSTEMS = ("menu_rule", "hybrid", "adapter")


def load(path: pathlib.Path) -> dict[str, dict]:
    rows = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            rows[record["item_id"]] = record
    return rows


def wilson(correct: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = correct / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [round(centre - half, 4), round(centre + half, 4)]


def mcnemar(b: int, c: int) -> dict:
    """Exact two-sided McNemar: b and c are the discordant pairs."""
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_exact": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2.0 ** n)
    return {"b": b, "c": c, "p_exact": round(min(1.0, 2.0 * tail), 6)}


def abstention(labels: list[tuple[str, str]]) -> dict:
    abstain_support = sum(1 for pred, _gold in labels if pred == ABSTAIN)
    gold_abstain = sum(1 for _pred, gold in labels if gold == ABSTAIN)
    right = sum(1 for pred, gold in labels if pred == ABSTAIN and gold == ABSTAIN)
    named = len(labels) - gold_abstain
    false_abstention = sum(1 for pred, gold in labels
                           if pred == ABSTAIN and gold != ABSTAIN)
    missed = sum(1 for pred, gold in labels if pred != ABSTAIN and gold == ABSTAIN)
    return {
        "predicted_abstains": abstain_support,
        "gold_abstains": gold_abstain,
        "abstention_precision": round(right / abstain_support, 4) if abstain_support else None,
        "abstention_recall": round(right / gold_abstain, 4) if gold_abstain else None,
        "false_abstention_rate": round(false_abstention / named, 4) if named else None,
        "missed_abstention_rate": round(missed / gold_abstain, 4) if gold_abstain else None,
        "false_abstention_rows": false_abstention,
        "missed_abstention_rows": missed,
    }


def fmt_p(p: float) -> str:
    return "p<1e-06" if p < 1e-06 else "p=%.4g" % p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", default=str(ROOT / "outputs/analysis/phase3a/p2/classifier_contrasts.json"))
    ap.add_argument("--out-md", default=str(ROOT / "outputs/analysis/phase3a/p2/classifier_contrasts.md"))
    ap.add_argument("--false-abstention-budget", type=float, default=0.05)
    args = ap.parse_args()

    hybrid, adapter = load(HYBRID), load(ADAPTER)
    shared = sorted(set(hybrid) & set(adapter))
    if not shared:
        raise SystemExit("no items shared between the hybrid and adapter predictions")

    labels = {system: [] for system in SYSTEMS}
    for item in shared:
        row, ada = hybrid[item], adapter[item]
        gold = str(row.get("gold") or ada.get("gold") or "").strip()
        if gold != str(ada.get("gold") or gold).strip():
            raise SystemExit("gold label disagrees between sources on " + item)
        labels["menu_rule"].append((str(row.get("classifier_label") or "").strip(), gold))
        labels["hybrid"].append((str(row.get("pred") or "").strip(), gold))
        labels["adapter"].append((str(ada.get("pred") or "").strip(), gold))

    report = {"gate": "P2", "items": len(shared),
              "systems": {"menu_rule": "classifier_label before rules (hybrid preds)",
                          "hybrid": "pred after the threshold, menu and no-fault rules",
                          "adapter": "the 2B model's own decoded label"},
              "abstain_label": ABSTAIN, "per_system": {}, "contrasts": {},
              "merged_row": {"outputs_present": ["root_cause"],
                             "outputs_absent": ["subsystem", "severity"],
                             "note": "subsystem and severity are knowledge-base functions of the "
                                     "predicted fault id, not independent outputs of these three "
                                     "systems, so their accuracy is arithmetically the root-cause "
                                     "accuracy and is not reported as a separate measurement"}}

    correct = {}
    for system in SYSTEMS:
        pairs = labels[system]
        hits = [pred == gold for pred, gold in pairs]
        correct[system] = hits
        accuracy = sum(hits) / len(pairs)
        report["per_system"][system] = {
            "accuracy": round(accuracy, 4),
            "accuracy_ci95": wilson(sum(hits), len(pairs)),
            "correct": sum(hits), "items": len(pairs),
            "abstention": abstention(pairs),
            "predicted_labels": dict(Counter(pred for pred, _gold in pairs).most_common(8)),
        }
        gate = report["per_system"][system]["abstention"]["false_abstention_rate"]
        report["per_system"][system]["false_abstention_gate"] = (
            "pass" if gate is not None and gate <= args.false_abstention_budget else "fail")

    for one, two in (("menu_rule", "hybrid"), ("menu_rule", "adapter"), ("hybrid", "adapter")):
        b = sum(1 for i in range(len(shared)) if not correct[one][i] and correct[two][i])
        c = sum(1 for i in range(len(shared)) if correct[one][i] and not correct[two][i])
        report["contrasts"]["%s_vs_%s" % (one, two)] = mcnemar(b, c)

    for system, block in report["per_system"].items():
        print("%-10s accuracy %.4f %s  abstains %d of %d gold (precision %s, recall %s)  "
              "false-abstention %s %s"
              % (system, block["accuracy"], block["accuracy_ci95"],
                 block["abstention"]["predicted_abstains"],
                 block["abstention"]["gold_abstains"],
                 block["abstention"]["abstention_precision"],
                 block["abstention"]["abstention_recall"],
                 block["abstention"]["false_abstention_rate"],
                 block["false_abstention_gate"]))
    for name, result in report["contrasts"].items():
        print("  %-22s b=%d c=%d exact %s" % (name, result["b"], result["c"],
                                               fmt_p(result["p_exact"])))

    out_json = pathlib.Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = ["# P2 - merged classifier row", "",
             "Items scored by all three systems: **%d**." % len(shared), "",
             "| system | root-cause accuracy (95 % CI) | abstains / gold abstains | precision | recall | false-abstention | gate |",
             "|---|---|---|---|---|---|---|"]
    for system, block in report["per_system"].items():
        a = block["abstention"]
        lines.append("| %s | %.4f (%.4f-%.4f) | %d / %d | %s | %s | %s | %s |"
                     % (system, block["accuracy"], block["accuracy_ci95"][0],
                        block["accuracy_ci95"][1], a["predicted_abstains"],
                        a["gold_abstains"], a["abstention_precision"],
                        a["abstention_recall"], a["false_abstention_rate"],
                        block["false_abstention_gate"]))
    lines += ["", "| contrast | discordant for first | discordant for second | exact |",
              "|---|---|---|---|"]
    for name, result in report["contrasts"].items():
        lines.append("| %s | %d | %d | %s |" % (name, result["b"], result["c"],
                                                 fmt_p(result["p_exact"])))
    lines += ["", "Subsystem and severity are not reported: they are knowledge-base functions of the "
                  "predicted fault id, so a separate accuracy for them would restate the root-cause "
                  "number rather than measure anything new."]
    pathlib.Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nwrote", out_json, "and", args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
