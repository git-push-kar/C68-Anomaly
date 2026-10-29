"""Reproduce the published per-fault LSTM numbers from the frozen artifacts.

Target: outputs/evaluation/all_faults_detector_summary.json, produced by
scripts/evaluate_all_faults_full.py on TEP_Faulty_Training.csv (500 samples per
run, 89 windows per run) with the frozen threshold 0.687.

The comparison checks score-level quantities (mean/median/std of the per-run
maximum, delay in windows, pre-onset false alarms), so a match proves the
model weights, the scaler and the windowing are the same ones that produced
the published numbers. The dynamic detectors have no comparable published
table (data/processed/anomaly_detector_eval.json was written on 800-sample
runs and older thresholds), which is reported as a limitation instead.

Usage:
    python scripts/verify_detector_reproduction.py --faults 1 3 4 9 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.windowing import to_windows  # noqa: E402
from scripts.evaluate_detector_v2 import iter_runs  # noqa: E402
from utils import ensure_dir, load_config, save_json  # noqa: E402


def max_consecutive(arr: np.ndarray) -> int:
    best = cur = 0
    for v in arr:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_a5000.yaml")
    ap.add_argument("--thresholds", default="configs/detector_thresholds_v2.json")
    ap.add_argument("--faults", type=int, nargs="+", default=[1, 3, 4, 9, 15])
    ap.add_argument("--summary", default="outputs/evaluation/all_faults_detector_summary.json")
    ap.add_argument("--out", default="outputs/evaluation/detector_v2/reproduction.json")
    args = ap.parse_args()

    import torch

    from anomaly_detection.inference import AnomalyDetector, build_autoencoder
    from anomaly_detection.threshold import ThresholdConfig
    from preprocessing.scaler import load_scaler
    from preprocessing.tep_loader import CANONICAL_NAMES, _normalize_rieth_columns

    config = load_config(args.config)
    thr_cfg = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    threshold = float(thr_cfg["lstm_ae"]["threshold"])
    scaler = load_scaler(config["preprocessing"]["scaler_dir"])
    model_dir = Path(config["anomaly_detector"]["model_dir"])
    state = torch.load(model_dir / "model.pt", map_location="cpu")
    dec_ih = state["model_state_dict"]["decoder.lstm.weight_ih_l0"]
    model = build_autoencoder(
        {"anomaly_detector": {"lstm": config["anomaly_detector"]["lstm"]}},
        int(state["num_features"]),
        int(state["sequence_length"]),
        use_cell_proj="decoder.cell_proj.weight" in state["model_state_dict"],
        decoder_input_dim=int(dec_ih.shape[1]),
    )
    model.load_state_dict(state["model_state_dict"])
    detector = AnomalyDetector(model, scaler, ThresholdConfig(method="manual", threshold=threshold), device="cpu")

    csv = Path(config["paths"]["fault_data_path"]) / "TEP_Faulty_Training.csv"
    rows = []
    for fault, run, raw in iter_runs(csv, faults=args.faults, run_range=None, samples_per_run=500):
        windows = to_windows(raw, 60, 5)
        scores, _ = detector.score_windows(windows)
        above = scores > threshold
        n_above = int(above.sum())
        max_c = max_consecutive(above)
        starts = np.arange(len(scores)) * 5
        pre = (starts + 60) <= 160
        rows.append(
            {
                "fault": int(fault),
                "run": int(run),
                "max": float(scores.max()),
                "n_above": n_above,
                "max_consec": int(max_c),
                "first_win": int(np.argmax(above)) if n_above else -1,
                "event": int(max_c >= 3),
                "false_before_onset": int((above & pre).sum()),
            }
        )
    df = pd.DataFrame(rows)
    published = json.loads(Path(args.summary).read_text(encoding="utf-8"))

    comparison = {}
    for fault in args.faults:
        sub = df[df["fault"] == fault]
        if sub.empty:
            continue
        delays = [int(r["first_win"]) for r in sub.to_dict("records") if r["first_win"] >= 0]
        computed = {
            "runs": int(len(sub)),
            "mean_max": float(sub["max"].mean()),
            "median_max": float(sub["max"].median()),
            "std_max": float(sub["max"].std(ddof=0)),
            "window_rate": float((sub["n_above"] > 0).mean()),
            "event_rate": float(sub["event"].mean()),
            "mean_delay_windows": float(np.mean(delays)) if delays else None,
            "mean_false_before": float(sub["false_before_onset"].mean()),
        }
        pub = published.get(str(fault), {})
        deltas = {}
        for key, val in computed.items():
            ref = pub.get(key)
            if isinstance(val, float) and isinstance(ref, (int, float)):
                deltas[key] = float(val - ref)
        comparison[str(fault)] = {"computed": computed, "published": {k: pub.get(k) for k in computed}, "delta": deltas}

    ensure_dir(Path(args.out).parent)
    save_json(Path(args.out), {"threshold": threshold, "source": str(csv), "faults": args.faults, "comparison": comparison})

    print(f"threshold={threshold}  target={args.summary}")
    print(f"{'fault':>5} {'runs':>5} {'mean_max':>10} {'pub':>10} {'delta':>10} {'event_rate':>10} {'pub':>7} {'delay':>7} {'pub':>7}")
    worst = 0.0
    for fault, c in comparison.items():
        comp, pub, delta = c["computed"], c["published"], c["delta"]
        worst = max(worst, abs(delta.get("mean_max", 0.0)))
        print(
            f"{fault:>5} {comp['runs']:>5} {comp['mean_max']:>10.4f} {pub['mean_max']:>10.4f} "
            f"{delta.get('mean_max', 0):>10.2e} {comp['event_rate']:>10.3f} {pub['event_rate']:>7.3f} "
            f"{comp['mean_delay_windows'] if comp['mean_delay_windows'] is not None else -1:>7.2f} "
            f"{pub['mean_delay_windows'] if pub['mean_delay_windows'] is not None else -1:>7.2f}"
        )
    print(f"max |mean_max delta| = {worst:.3e}")


if __name__ == "__main__":
    main()
