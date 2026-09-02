"""Evaluate predictor vs reconstruction on 75 normal test + 500 Fault 3,9,15.
Uses frozen predictor and frozen reconstruction detector, p99 thresholds from val only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
from utils import load_config, load_json
from anomaly_detection.inference import AnomalyDetector
from anomaly_detection.lstm_predictor import LSTMPredictor
from preprocessing.windowing import to_windows
import torch
from preprocessing.scaler import load_scaler

SENSORS = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(1, 12)]
SENSOR_COLS = [s.lower().replace("XMEAS","xmeas").replace("XMV","xmv") for s in SENSORS]

def max_consecutive(arr):
    m=c=0
    for v in arr:
        if v:
            c+=1
            m=max(m,c)
        else:
            c=0
    return m

def evaluate_predictor_on_run(arr_scaled, predictor, device):
    # arr_scaled: [500,52] scaled
    # Create windows for predictor: 60 -> next 1, stride 5
    # For each window at start s, target is arr_scaled[s+60]
    windows = []
    targets = []
    for start in range(0, len(arr_scaled)-60, 5):
        windows.append(arr_scaled[start:start+60])
        targets.append(arr_scaled[start+60])
    if not windows:
        return np.array([]), 0
    windows = np.stack(windows)  # [N,60,52]
    targets = np.stack(targets)  # [N,52]
    # Score via predictor
    predictor.eval()
    with torch.no_grad():
        w_t = torch.from_numpy(windows).float().to(device)
        t_t = torch.from_numpy(targets).float().to(device)
        preds = predictor(w_t)
        errs = ((preds - t_t)**2).mean(dim=1).cpu().numpy()  # [N]
    return errs, len(windows)

def main():
    config = load_config("configs/config_a5000.yaml")
    # Load detectors
    recon_detector = AnomalyDetector.from_artifacts(
        config["anomaly_detector"]["model_dir"],
        config["preprocessing"]["scaler_dir"],
        config["anomaly_detector"]["model_dir"]
    )
    recon_threshold = float(recon_detector.threshold.threshold)
    print(f"Recon threshold: {recon_threshold:.4f}")

    # Load predictor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_state = torch.load("outputs/prediction_detector/model.pt", map_location=device)
    predictor = LSTMPredictor(num_features=52, hidden_size=64, num_layers=1).to(device)
    predictor.load_state_dict(pred_state["model_state_dict"])
    predictor.eval()
    pred_thresholds = load_json(Path("outputs/prediction_detector/threshold.json"))
    pred_p99 = float(pred_thresholds["p99"])
    print(f"Pred thresholds: p99 {pred_p99:.4f}, p99.5 {pred_thresholds['p99.5']:.4f}")

    # Load manifests
    det_manifest = load_json(Path("data/processed/manifests/detector_split.json"))
    normal_test_runs = det_manifest["test_runs"]
    print(f"Normal test runs: {len(normal_test_runs)}")

    # Pre-load
    normal_all = pd.read_csv("data/raw/normal/TEP_FaultFree_Training.csv")
    fault_all = pd.read_csv("data/raw/faults/TEP_Faulty_Training.csv")
    scaler = load_scaler(config["preprocessing"]["scaler_dir"])

    out_rows = []
    # Evaluate normal test
    print("\nScoring normal test...")
    for run in normal_test_runs:
        df_run = normal_all[normal_all["simulationRun"]==run].sort_values("sample")
        arr = df_run[SENSOR_COLS].to_numpy(dtype=np.float32)
        arr_scaled = scaler.transform(arr)
        # Recon - pass RAW windows, detector scales internally (do NOT double-scale)
        windows_raw = to_windows(arr, 60, 5)
        recon_scores, _ = recon_detector.score_windows(windows_raw)
        # Pred - pass SCALED windows (predictor was trained on scaled)
        pred_scores, _ = evaluate_predictor_on_run(arr_scaled, predictor, device)
        out_rows.append({
            "type": "normal", "run": run, "fault_id": 0,
            "recon_max": float(recon_scores.max()) if len(recon_scores) else 0,
            "recon_mean": float(recon_scores.mean()) if len(recon_scores) else 0,
            "recon_p95": float(np.percentile(recon_scores,95)) if len(recon_scores) else 0,
            "recon_p99": float(np.percentile(recon_scores,99)) if len(recon_scores) else 0,
            "recon_n_above": int((recon_scores>recon_threshold).sum()),
            "recon_max_consec": int(max_consecutive(recon_scores>recon_threshold)),
            "recon_event": int(max_consecutive(recon_scores>recon_threshold)>=3),
            "pred_max": float(pred_scores.max()) if len(pred_scores) else 0,
            "pred_mean": float(pred_scores.mean()) if len(pred_scores) else 0,
            "pred_p95": float(np.percentile(pred_scores,95)) if len(pred_scores) else 0,
            "pred_p99": float(np.percentile(pred_scores,99)) if len(pred_scores) else 0,
            "pred_n_above": int((pred_scores>pred_p99).sum()) if len(pred_scores) else 0,
            "pred_max_consec": int(max_consecutive(pred_scores>pred_p99)) if len(pred_scores) else 0,
            "pred_event": int(max_consecutive(pred_scores>pred_p99)>=3) if len(pred_scores) else 0,
        })

    # Faults 3,9,15
    for fid in [3,9,15]:
        fault_runs = sorted(fault_all[fault_all["faultNumber"]==fid]["simulationRun"].unique())
        print(f"\nFault {fid} ({len(fault_runs)} runs)...")
        for run in fault_runs:
            df_run = fault_all[(fault_all["faultNumber"]==fid) & (fault_all["simulationRun"]==run)].sort_values("sample")
            arr = df_run[SENSOR_COLS].to_numpy(dtype=np.float32)
            arr_scaled = scaler.transform(arr)
            windows_raw = to_windows(arr, 60, 5)
            recon_scores, _ = recon_detector.score_windows(windows_raw)
            pred_scores, _ = evaluate_predictor_on_run(arr_scaled, predictor, device)
            out_rows.append({
                "type": f"fault{fid}", "run": run, "fault_id": fid,
                "recon_max": float(recon_scores.max()) if len(recon_scores) else 0,
                "recon_mean": float(recon_scores.mean()) if len(recon_scores) else 0,
                "recon_p95": float(np.percentile(recon_scores,95)) if len(recon_scores) else 0,
                "recon_p99": float(np.percentile(recon_scores,99)) if len(recon_scores) else 0,
                "recon_n_above": int((recon_scores>recon_threshold).sum()),
                "recon_max_consec": int(max_consecutive(recon_scores>recon_threshold)),
                "recon_event": int(max_consecutive(recon_scores>recon_threshold)>=3),
                "pred_max": float(pred_scores.max()) if len(pred_scores) else 0,
                "pred_mean": float(pred_scores.mean()) if len(pred_scores) else 0,
                "pred_p95": float(np.percentile(pred_scores,95)) if len(pred_scores) else 0,
                "pred_p99": float(np.percentile(pred_scores,99)) if len(pred_scores) else 0,
                "pred_n_above": int((pred_scores>pred_p99).sum()) if len(pred_scores) else 0,
                "pred_max_consec": int(max_consecutive(pred_scores>pred_p99)) if len(pred_scores) else 0,
                "pred_event": int(max_consecutive(pred_scores>pred_p99)>=3) if len(pred_scores) else 0,
            })

    # Save per-run
    import csv
    out_path = Path("outputs/evaluation/prediction_vs_reconstruction_per_run.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved {out_path} ({len(out_rows)} rows)")

    # Summary per fault
    df = pd.DataFrame(out_rows)
    summary = []
    for label in ["normal", "fault3", "fault9", "fault15"]:
        sub = df[df["type"]==label] if label!="normal" else df[df["type"]=="normal"]
        # For fault labels, type is fault3 etc.
        if label.startswith("fault"):
            fid = int(label[5:])
            sub = df[df["fault_id"]==fid]
        else:
            sub = df[df["type"]=="normal"]
        summary.append({
            "condition": label,
            "runs": len(sub),
            "recon_mean_max": float(sub["recon_max"].mean()),
            "recon_median_max": float(sub["recon_max"].median()),
            "recon_p95_max": float(np.percentile(sub["recon_max"],95)),
            "recon_event_rate": float(sub["recon_event"].mean()),
            "pred_mean_max": float(sub["pred_max"].mean()),
            "pred_median_max": float(sub["pred_max"].median()),
            "pred_p95_max": float(np.percentile(sub["pred_max"],95)),
            "pred_event_rate": float(sub["pred_event"].mean()),
        })

    # Save summary
    with open("outputs/evaluation/prediction_vs_reconstruction_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved summary")

    # Print table
    print("\ncondition | recon_mean | recon_p95 | recon_event | pred_mean | pred_p95 | pred_event")
    for s in summary:
        print(f"{s['condition']:8s} | {s['recon_mean_max']:.3f} | {s['recon_median_max']:.3f} | {s['recon_event_rate']:.3f} | {s['pred_mean_max']:.3f} | {s['pred_median_max']:.3f} | {s['pred_event_rate']:.3f}")

    # Also check temporal: for fault15, does pred error increase after onset?
    # For each fault15 run, compute pred error pre vs post
    print("\n=== Temporal: Fault 15 pred error pre vs post ===")
    # Sample 5 runs
    for run in [1,2,3,4,5]:
        df_run = fault_all[(fault_all["faultNumber"]==15) & (fault_all["simulationRun"]==run)].sort_values("sample")
        arr = df_run[SENSOR_COLS].to_numpy(dtype=np.float32)
        arr_scaled = scaler.transform(arr)
        pred_scores, _ = evaluate_predictor_on_run(arr_scaled, predictor, device)
        starts = np.arange(len(pred_scores))*5
        is_post = (starts + 60) > 160
        pre = pred_scores[~is_post] if (~is_post).any() else np.array([0])
        post = pred_scores[is_post] if is_post.any() else np.array([0])
        print(f"Run {run}: pre mean {pre.mean():.3f} max {pre.max():.3f} | post mean {post.mean():.3f} max {post.max():.3f} | post/pre ratio {post.mean()/(pre.mean()+1e-9):.2f}")

    # Save report
    with open("outputs/evaluation/prediction_detector_report.md", "w") as f:
        f.write("# Prediction vs Reconstruction - Faults 3,9,15\n\n")
        f.write(f"Recon threshold: {recon_threshold:.4f} (p99 val), Pred p99: {pred_p99:.4f}\n\n")
        f.write("| condition | recon_mean_max | recon_p95 | recon_event | pred_mean_max | pred_p95 | pred_event |\n")
        f.write("|-----------|--------------|-----------|-------------|-------------|----------|------------|\n")
        for s in summary:
            f.write(f"| {s['condition']} | {s['recon_mean_max']:.3f} | {s['recon_median_max']:.3f} | {s['recon_event_rate']:.3f} | {s['pred_mean_max']:.3f} | {s['pred_median_max']:.3f} | {s['pred_event_rate']:.3f} |\n")
        f.write("\nQuestions:\n")
        # Answer based on results
        for fid in [3,9,15]:
            recon_ok = summary[[x["condition"] for x in summary].index(f"fault{fid}")]["recon_event_rate"] > 0.5
            pred_ok = summary[[x["condition"] for x in summary].index(f"fault{fid}")]["pred_event_rate"] > 0.5
            f.write(f"- Fault {fid}: recon {'separates' if recon_ok else 'not'}, pred {'separates' if pred_ok else 'not'}\n")
    print("Saved report")

if __name__ == "__main__":
    main()
