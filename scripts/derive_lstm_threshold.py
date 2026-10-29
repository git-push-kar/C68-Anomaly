"""A1: re-derive the LSTM AE window threshold from NORMAL VALIDATION windows only.

The frozen evaluation threshold (0.687) is not stored with the artifacts:
`outputs/anomaly_detector/threshold.json` was deleted in commit 49b006b, and the
value in history (1.047) is a 69-score smoke calibration. This script recomputes
the statistic from the run-level validation partition recorded in
`data/processed/manifests/detector_split.json` (75 normal runs of
TEP_FaultFree_Training), so the number has a provenance that can be checked
instead of a comment that cannot.

It reports window-level percentiles 99.0 / 99.5 / 99.9 and compares them with
0.687. Nothing is retrained and nothing is written into the live detector: the
output is a JSON record plus stdout.

    python scripts/derive_lstm_threshold.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate_detector_v2 import DetectorBundle, iter_runs  # noqa: E402
from utils import ensure_dir, load_config, save_json  # noqa: E402

logger = logging.getLogger("derive_lstm_threshold")

FROZEN_THRESHOLD = 0.687
PERCENTILES = (99.0, 99.5, 99.9)
OUT = ROOT / "outputs" / "evaluation" / "detector_v2" / "lstm_threshold_derivation.json"


def main() -> None:
    logging.basicConfig(level="INFO", format="%(levelname)-7s %(message)s")
    config = load_config("configs/config_a5000.yaml")
    window_size = int(config["windowing"]["window_size"])
    stride = int(config["windowing"]["stride"])

    man = Path(config["paths"]["processed_data_path"]) / "manifests" / "detector_split.json"
    split = json.loads(man.read_text(encoding="utf-8"))
    val_runs = sorted(int(r) for r in split["validation_runs"])
    logger.info("normal validation runs: %d (from %s)", len(val_runs), man)

    # The threshold is irrelevant here (only raw scores are used); pass the
    # frozen value so the bundle is constructed through the evaluator's path.
    bundle = DetectorBundle(
        config,
        {"lstm_ae": {"threshold": FROZEN_THRESHOLD}, "dynamic": {"cva": {}, "dpca": {}}},
        device="cpu",
        window_size=window_size,
        stride=stride,
    )

    normal_csv = Path(config["paths"]["normal_data_path"]) / "TEP_FaultFree_Training.csv"
    wanted = set(val_runs)
    t0 = time.time()
    chunks = []
    per_run_windows = []
    for _fault, run, raw in iter_runs(
        normal_csv, faults=None, run_range=(min(val_runs), max(val_runs)), samples_per_run=500
    ):
        if run not in wanted:
            continue
        scaled = bundle.scale(raw)
        _starts, scores = bundle.score_lstm(scaled)
        chunks.append(np.asarray(scores, dtype=np.float64))
        per_run_windows.append(int(scores.size))
    scores = np.concatenate(chunks)
    elapsed = time.time() - t0

    record = {
        "method": "window-level percentile of LSTM AE reconstruction error",
        "data": str(normal_csv),
        "split": "detector_split.json validation_runs",
        "n_runs": len(chunks),
        "window_size": window_size,
        "stride": stride,
        "n_windows": int(scores.size),
        "windows_per_run": int(np.median(per_run_windows)) if per_run_windows else 0,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "max": float(scores.max()),
        "percentiles": {f"p{p}": float(np.percentile(scores, p)) for p in PERCENTILES},
        "frozen_threshold": FROZEN_THRESHOLD,
        "runtime_seconds": round(elapsed, 1),
    }
    record["delta_vs_frozen"] = {
        f"p{p}": round(float(np.percentile(scores, p)) - FROZEN_THRESHOLD, 6) for p in PERCENTILES
    }

    ensure_dir(OUT.parent)
    save_json(OUT, record)

    print(f"\nnormal validation windows: {record['n_windows']} "
          f"({record['n_runs']} runs x ~{record['windows_per_run']} windows)")
    print(f"score distribution: mean {record['mean']:.5f}  std {record['std']:.5f}  "
          f"max {record['max']:.5f}")
    print(f"{'percentile':>12} {'threshold':>10} {'- 0.687':>10}")
    for p in PERCENTILES:
        v = record["percentiles"][f"p{p}"]
        print(f"{'p'+format(p,'.1f'):>12} {v:>10.5f} {v-FROZEN_THRESHOLD:>+10.5f}")
    print(f"{'frozen':>12} {FROZEN_THRESHOLD:>10.5f}")
    print(f"\nwrote {OUT} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
