"""§17.3-17.4: merge the surrogate shards, average the signals, and pick one.

The choice is the pre-registered one: highest mean AUROC across the faults, ties (within 0.005)
to the simpler signal, in the order margin, entropy, max_prob, novelty. Nothing is re-weighted,
re-scaled or re-thresholded here — the shards already contain the numbers.

Usage::

    python scripts/hybrid_surrogate_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.hybrid import calibration as cal  # noqa: E402

OUT = ROOT / "outputs/analysis/phase3a/hybrid"
SHARDS = ("surrogate_shard1.json", "surrogate_shard2.json")


def main() -> int:
    merged: Dict[str, Dict] = {}
    for name in SHARDS:
        path = OUT / name
        if not path.exists():
            raise SystemExit(f"missing {path}")
        merged.update(json.loads(path.read_text(encoding="utf-8")))

    faults = sorted(merged)
    signals: List[str] = sorted(merged[faults[0]]["auroc"])
    # The shards store the raw rank comparison P(held-out score > seen score). For max_prob
    # and margin that is the *wrong* direction -- an unseen class gets a lower maximum -- so
    # 1 - AUROC is the §17.3 number. Recovering it here is exact (AUROC with ties half is
    # symmetric), and the wide script now orients before ranking, so both routes agree.
    def oriented(raw: float, name: str) -> float:
        return raw if cal.ORIENTATION[name] > 0 else 1.0 - raw

    mean_auroc = {name: float(np.mean([oriented(merged[fault]["auroc"][name], name)
                                       for fault in faults])) for name in signals}
    mean_raw = {name: float(np.mean([merged[fault]["auroc"][name] for fault in faults]))
                for name in signals}
    mean_auroc_menu = {name: float(np.mean([oriented(
        merged[fault].get("auroc_menu_only", {}).get(name, np.nan), name)
        for fault in faults])) for name in signals}
    chosen = cal.choose_signal(mean_auroc)

    absorption = {}
    for fault in faults:
        named = merged[fault]["named_as"].get("menu", {})
        if named:
            absorption[fault] = {
                "declined_rate_menu": merged[fault]["by_variant"]["menu"]["declined_rate"],
                "absorbed_by": named,
            }

    outcome = {
        "faults": faults,
        "signal_choice": {
            "rule": "highest mean AUROC across faults; ties within 0.005 to the simpler signal",
            "orientation": "higher score = more likely the unseen class (PROTOCOL §17.3); "
                           "max_prob and margin are negated before ranking",
            "mean_auroc": {name: round(value, 4) for name, value in mean_auroc.items()},
            "mean_auroc_raw": {name: round(value, 4) for name, value in mean_raw.items()},
            "mean_auroc_menu_only": {name: round(value, 4) for name, value
                                     in mean_auroc_menu.items()},
            "simplicity_order": list(cal.SIMPLICITY),
            "tie_tolerance": cal.TIE_TOLERANCE,
            "chosen": chosen,
        },
        "per_fault": merged,
        "lookalike_absorption": absorption,
        "declined_rate_menu": {fault: merged[fault]["by_variant"]["menu"]["declined_rate"]
                               for fault in faults},
        "false_abstention_on_others": {fault: merged[fault]["false_abstention_on_others"]
                                       for fault in faults},
    }
    (OUT / "surrogate_wide.json").write_text(json.dumps(outcome, indent=2, default=str),
                                             encoding="utf-8")
    print(json.dumps({"mean_auroc": outcome["signal_choice"]["mean_auroc"],
                      "mean_auroc_menu_only": outcome["signal_choice"]["mean_auroc_menu_only"],
                      "chosen": chosen}, indent=2))
    print(json.dumps({"declined_rate_menu": outcome["declined_rate_menu"],
                      "absorbers": absorption}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
