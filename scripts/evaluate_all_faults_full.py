"""Full detector evaluation: 75 normal test + 20 faults x 500 runs.
Frozen detector, no retraining, threshold 0.687.
Saves per-run and per-fault summaries.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
from utils import ensure_dir, load_config, load_json
from anomaly_detection.inference import AnomalyDetector
from preprocessing.windowing import to_windows

SENSORS = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(1, 12)]
SENSOR_COLS = [s.lower().replace("XMEAS","xmeas").replace("XMV","xmv") for s in SENSORS]

def max_consecutive(arr):
    m = c = 0
    for v in arr:
        if v:
            c+=1
            m=max(m,c)
        else:
            c=0
    return m

def main():
    config = load_config("configs/config_a5000.yaml")
    detector = AnomalyDetector.from_artifacts(
        config["anomaly_detector"]["model_dir"],
        config["preprocessing"]["scaler_dir"],
        config["anomaly_detector"]["model_dir"]
    )
    threshold = float(detector.threshold.threshold)
    print(f"Threshold: {threshold:.4f}")

    det_manifest = load_json(Path("data/processed/manifests/detector_split.json"))
    normal_test_runs = det_manifest["test_runs"]
    print(f"Normal test runs: {len(normal_test_runs)}")

    print("Pre-loading data...")
    normal_all = pd.read_csv("data/raw/normal/TEP_FaultFree_Training.csv")
    fault_all = pd.read_csv("data/raw/faults/TEP_Faulty_Training.csv")
    print(f"Normal rows: {len(normal_all)}, Fault rows: {len(fault_all)}")

    out_dir = ensure_dir(Path("outputs/evaluation"))
    per_run_rows = []
    per_fault_stats = {}

    # Normal
    print("\nScoring normal test runs...")
    for run in normal_test_runs:
        df_run = normal_all[normal_all["simulationRun"]==run]
        df_run = df_run.sort_values("sample")
        arr = df_run[SENSOR_COLS].to_numpy(dtype=np.float32)
        windows = to_windows(arr, 60, 5)
        scores, _ = detector.score_windows(windows)
        n_above = int((scores>threshold).sum())
        max_c = max_consecutive(scores>threshold)
        first_win = int(np.argmax(scores>threshold)) if n_above>0 else -1
        per_run_rows.append({
            "type": "normal", "fault_id": 0, "simulationRun": run,
            "max": float(scores.max()), "mean": float(scores.mean()),
            "p95": float(np.percentile(scores,95)), "p99": float(np.percentile(scores,99)),
            "n_above": n_above, "max_consec": int(max_c),
            "first_win": first_win, "first_sample": int(first_win*5) if first_win>=0 else -1,
            "n_windows": len(scores), "event": int(max_c>=3),
            "false_before_onset": 0
        })

    # Faults 1..20 x 500
    for fid in range(1, 21):
        print(f"\nFault {fid}...")
        fault_runs = sorted(fault_all[fault_all["faultNumber"]==fid]["simulationRun"].unique())
        print(f"  {len(fault_runs)} runs")
        for run in fault_runs:
            df_run = fault_all[(fault_all["faultNumber"]==fid) & (fault_all["simulationRun"]==run)]
            df_run = df_run.sort_values("sample")
            arr = df_run[SENSOR_COLS].to_numpy(dtype=np.float32)
            windows = to_windows(arr, 60, 5)
            scores, _ = detector.score_windows(windows)
            n_above = int((scores>threshold).sum())
            max_c = max_consecutive(scores>threshold)
            first_win = int(np.argmax(scores>threshold)) if n_above>0 else -1
            # false before onset: windows with start <160 and score>th
            n_windows = len(scores)
            starts = np.arange(n_windows)*5
            is_pre = starts < 100  # window start <100 means window [0,60) to [95,155) is pre-onset (onset 160, window end 60, so start 100 is last pre)
            # More accurate: window end >160 is post, so pre is start+60 <=160
            is_pre = (starts + 60) <= 160
            false_before = int(((scores>threshold) & is_pre).sum())
            per_run_rows.append({
                "type": f"fault{fid}", "fault_id": fid, "simulationRun": run,
                "max": float(scores.max()), "mean": float(scores.mean()),
                "p95": float(np.percentile(scores,95)), "p99": float(np.percentile(scores,99)),
                "n_above": n_above, "max_consec": int(max_c),
                "first_win": first_win, "first_sample": int(first_win*5) if first_win>=0 else -1,
                "n_windows": len(scores), "event": int(max_c>=3),
                "false_before_onset": int(false_before),
            })
        # Per-fault stats
        sub = [r for r in per_run_rows if r["fault_id"]==fid]
        max_vals = np.array([r["max"] for r in sub])
        p95_vals = np.array([r["p95"] for r in sub])
        p99_vals = np.array([r["p99"] for r in sub])
        n_above_vals = np.array([r["n_above"] for r in sub])
        max_c_vals = np.array([r["max_consec"] for r in sub])
        first_vals = np.array([r["first_win"] for r in sub if r["first_win"]>=0])
        per_fault_stats[fid] = {
            "fault_id": fid,
            "runs": len(sub),
            "window_rate": float((n_above_vals>0).mean() if len(n_above_vals) else 0),  # actually per window, but we have per run n_above
            "event_rate": float(np.mean([r["event"] for r in sub])),
            "mean_max": float(max_vals.mean()),
            "median_max": float(np.median(max_vals)),
            "std_max": float(max_vals.std()),
            "p95_max": float(np.percentile(max_vals,95)),
            "p99_max": float(np.percentile(max_vals,99)),
            "mean_p95": float(p95_vals.mean()),
            "mean_p99": float(p99_vals.mean()),
            "mean_n_above": float(n_above_vals.mean()),
            "mean_max_consec": float(max_c_vals.mean()),
            "pct_1_consec": float((max_c_vals>=1).mean()),
            "pct_2_consec": float((max_c_vals>=2).mean()),
            "pct_3_consec": float((max_c_vals>=3).mean()),
            "mean_delay_windows": float(first_vals.mean()) if len(first_vals) else -1,
            "median_delay_windows": float(np.median(first_vals)) if len(first_vals) else -1,
            "mean false_before": float(np.mean([r["false_before_onset"] for r in sub])),
        }
        print(f"  Fault {fid}: event_rate {per_fault_stats[fid]['event_rate']:.3f} mean_max {per_fault_stats[fid]['mean_max']:.3f} mean_n_above {per_fault_stats[fid]['mean_n_above']:.1f}")

    # Normal per-fault stats for comparison
    normal_sub = [r for r in per_run_rows if r["fault_id"]==0]
    per_fault_stats[0] = {
        "fault_id": 0,
        "runs": len(normal_sub),
        "event_rate": float(np.mean([r["event"] for r in normal_sub])),
        "mean_max": float(np.mean([r["max"] for r in normal_sub])),
        "median_max": float(np.median([r["max"] for r in normal_sub])),
        "std_max": float(np.std([r["max"] for r in normal_sub])),
        "p95_max": float(np.percentile([r["max"] for r in normal_sub],95)),
        "p99_max": float(np.percentile([r["max"] for r in normal_sub],99)),
        "mean_n_above": float(np.mean([r["n_above"] for r in normal_sub])),
        "mean_max_consec": float(np.mean([r["max_consec"] for r in normal_sub])),
    }
    print(f"\nNormal: event_rate {per_fault_stats[0]['event_rate']:.3f} mean_max {per_fault_stats[0]['mean_max']:.3f}")

    # Save per-run
    import csv
    with open(out_dir / "all_faults_detector_per_run.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_run_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_run_rows)
    print(f"Saved {out_dir/'all_faults_detector_per_run.csv'} ({len(per_run_rows)} rows)")

    # Save per-fault summary
    with open(out_dir / "all_faults_detector_summary.csv", "w", newline="") as f:
        # Use keys from first fault
        keys = list(per_fault_stats[1].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for fid in sorted(per_fault_stats.keys()):
            writer.writerow(per_fault_stats[fid])
    print(f"Saved {out_dir/'all_faults_detector_summary.csv'}")

    with open(out_dir / "all_faults_detector_summary.json", "w") as f:
        json.dump(per_fault_stats, f, indent=2)
    print(f"Saved {out_dir/'all_faults_detector_summary.json'}")

    # Human-readable report
    with open(out_dir / "all_faults_detector_report.md", "w") as f:
        f.write("# All Faults Detector Report (Frozen, Threshold 0.687)\n\n")
        f.write(f"Threshold: {threshold:.4f} (mean {detector.threshold.mean:.4f} std {detector.threshold.std:.4f})\n\n")
        f.write(f"Normal test: 75 runs, {len([r for r in normal_sub if r['event']==1])}/75 events\n\n")
        f.write("| fault_id | runs | event_rate | mean_max | median_max | mean_n_above | pct_3_consec | mean_delay |\n")
        f.write("|----------|------|------------|----------|------------|--------------|--------------|------------|\n")
        for fid in range(0, 21):
            s = per_fault_stats.get(fid)
            if not s:
                continue
            fid_str = "Normal" if fid==0 else str(fid)
            f.write(f"| {fid_str} | {s['runs']} | {s['event_rate']:.3f} | {s['mean_max']:.3f} | {s['median_max']:.3f} | {s['mean_n_above']:.1f} | {s.get('pct_3_consec',0):.3f} | {s.get('mean_delay_windows',-1):.1f} |\n")
        f.write("\nRankings:\n")
        # Easiest/hardest
        sorted_by_event = sorted([v for k,v in per_fault_stats.items() if k!=0], key=lambda x: x["event_rate"], reverse=True)
        f.write(f"\nEasiest (highest event_rate): {', '.join(str(x['fault_id']) for x in sorted_by_event[:3])}\n")
        f.write(f"Hardest (lowest event_rate): {', '.join(str(x['fault_id']) for x in sorted_by_event[-3:])}\n")
        # Most overlap
        normal_mean_max = per_fault_stats[0]["mean_max"]
        overlap = sorted([v for k,v in per_fault_stats.items() if k!=0], key=lambda x: abs(x["mean_max"]-normal_mean_max))
        f.write(f"Most overlap with normal (mean_max closest to {normal_mean_max:.3f}): {', '.join(str(x['fault_id']) for x in overlap[:3])}\n")
        # Strongest persistence
        persist = sorted([v for k,v in per_fault_stats.items() if k!=0], key=lambda x: x["mean_max_consec"], reverse=True)
        f.write(f"Strongest persistence (max_consec): {', '.join(str(x['fault_id']) for x in persist[:3])}\n")
        # Weakest
        f.write(f"Weakest persistence: {', '.join(str(x['fault_id']) for x in persist[-3:])}\n")
    print(f"Saved {out_dir/'all_faults_detector_report.md'}")

if __name__ == "__main__":
    main()
