"""Relationship anomaly diagnostic for Faults 3,9,15 vs Normal.
Uses 18 defined pairs from evidence/process_relationships.py, no invented relationships.
Baseline from 350 normal train, threshold p99 from 75 normal val, evaluate on 75 normal test + 500 each fault.
Scores: actuator->process linear residual, paired correlation deviation, lagged deviation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
from utils import ensure_dir, load_config, load_json

# 18 defined pairs - all from KNOWN_PROCESS_RELATIONSHIPS or SUBSYSTEM_GROUPS, no invented
# Actuator->Process (linear residual) - 9 pairs from KNOWN_PROCESS_RELATIONSHIPS
ACTUATOR_PROCESS_PAIRS = [
    ("Reactor_Cooling_Water_Flow", "Reactor_Temperature"),
    ("Reactor_Cooling_Water_Outlet_Temperature", "Reactor_Temperature"),
    ("Reactor_Temperature", "Reactor_Pressure"),
    ("A_C_Feed_Flow", "Reactor_Feed_Rate"),
    ("A_Feed_Flow", "Reactor_Feed_Rate"),
    ("D_Feed_Flow", "Reactor_Feed_Rate"),
    ("Condenser_Cooling_Water_Flow", "Separator_Temperature"),
    ("Condenser_Cooling_Water_Outlet_Temperature", "Separator_Temperature"),
    ("Separator_Temperature", "Separator_Pressure"),
]

# Paired sensor relationships (correlation) - within same SUBSYSTEM_GROUPS, 3 pairs
PAIRED_PAIRS = [
    ("Reactor_Feed_Rate", "Reactor_Temperature"),
    ("Separator_Temperature", "Separator_Pressure"),
    ("Reactor_Temperature", "Reactor_Pressure"),
]

# Lagged - same 3 actuator pairs with lag 5 and 10 samples (1 and 2 windows), within same run only
LAGGED_PAIRS = [
    ("Reactor_Cooling_Water_Flow", "Reactor_Temperature", 5),
    ("A_C_Feed_Flow", "Reactor_Feed_Rate", 5),
    ("Condenser_Cooling_Water_Flow", "Separator_Temperature", 5),
    ("Reactor_Cooling_Water_Flow", "Reactor_Temperature", 10),
    ("A_C_Feed_Flow", "Reactor_Feed_Rate", 10),
    ("Condenser_Cooling_Water_Flow", "Separator_Temperature", 10),
]

SENSORS = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(1, 12)]
# Map display names to xmeas/xmv lower case for CSV
DISPLAY_TO_COL = {
    "A_Feed_Stream1": "xmeas_1", "D_Feed_Stream2": "xmeas_2", "E_Feed_Stream3": "xmeas_3",
    "A_C_Feed_Stream4": "xmeas_4", "Recycle_Flow_Stream8": "xmeas_5", "Reactor_Feed_Rate": "xmeas_6",
    "Reactor_Pressure": "xmeas_7", "Reactor_Level": "xmeas_8", "Reactor_Temperature": "xmeas_9",
    "Purge_Rate": "xmeas_10", "Separator_Temperature": "xmeas_11", "Separator_Level": "xmeas_12",
    "Separator_Pressure": "xmeas_13", "Separator_Underflow_Stream10": "xmeas_14",
    "Stripper_Level": "xmeas_15", "Stripper_Pressure": "xmeas_16", "Stripper_Underflow_Stream11": "xmeas_17",
    "Stripper_Temperature": "xmeas_18", "Stripper_Steam_Flow": "xmeas_19", "Compressor_Work": "xmeas_20",
    "Reactor_Cooling_Water_Outlet_Temperature": "xmeas_21", "Condenser_Cooling_Water_Outlet_Temperature": "xmeas_22",
    "Component_A_Stream6": "xmeas_23", "Component_B_Stream6": "xmeas_24", "Component_C_Stream6": "xmeas_25",
    "Component_D_Stream6": "xmeas_26", "Component_E_Stream6": "xmeas_27", "Component_F_Stream6": "xmeas_28",
    "Component_A_Stream9": "xmeas_29", "Component_B_Stream9": "xmeas_30", "Component_C_Stream9": "xmeas_31",
    "Component_D_Stream9": "xmeas_32", "Component_E_Stream9": "xmeas_33", "Component_F_Stream9": "xmeas_34",
    "Component_G_Stream9": "xmeas_35", "Component_H_Stream9": "xmeas_36", "Component_D_Stream11": "xmeas_37",
    "Component_E_Stream11": "xmeas_38", "Component_F_Stream11": "xmeas_39", "Component_G_Stream11": "xmeas_40",
    "Component_H_Stream11": "xmeas_41", "D_Feed_Flow": "xmv_1", "E_Feed_Flow": "xmv_2",
    "A_Feed_Flow": "xmv_3", "A_C_Feed_Flow": "xmv_4", "Compressor_Recycle_Valve": "xmv_5",
    "Purge_Valve": "xmv_6", "Separator_Pot_Liquid_Flow": "xmv_7", "Stripper_Liquid_Product_Flow": "xmv_8",
    "Stripper_Steam_Valve": "xmv_9", "Reactor_Cooling_Water_Flow": "xmv_10", "Condenser_Cooling_Water_Flow": "xmv_11",
}

def fit_linear_residual(train_runs_data, a_col, b_col):
    # Collect all normal train samples for A and B
    all_a = np.concatenate([df[a_col].values for df in train_runs_data])
    all_b = np.concatenate([df[b_col].values for df in train_runs_data])
    # Fit B = a*A + b via least squares
    A = np.column_stack([all_a, np.ones_like(all_a)])
    coeffs, _, _, _ = np.linalg.lstsq(A, all_b, rcond=None)
    a, b = coeffs[0], coeffs[1]
    # Also compute residual std for z-like
    pred = a * all_a + b
    resid = all_b - pred
    std = float(np.std(resid) + 1e-9)
    return a, b, std

def main():
    config = load_config("configs/config_a5000.yaml")
    det_manifest = load_json(Path("data/processed/manifests/detector_split.json"))
    train_runs = det_manifest["train_runs"]
    val_runs = det_manifest["validation_runs"]
    test_runs = det_manifest["test_runs"]
    print(f"Normal train {len(train_runs)}, val {len(val_runs)}, test {len(test_runs)}")

    print("Pre-loading normal data...")
    normal_all = pd.read_csv("data/raw/normal/TEP_FaultFree_Training.csv")
    fault_all = pd.read_csv("data/raw/faults/TEP_Faulty_Training.csv")

    # Build baselines from 350 train runs
    print("Building baselines from 350 train runs...")
    # Collect train runs data for actuator->process linear fits
    train_runs_data = []
    for run in train_runs:
        df_run = normal_all[normal_all["simulationRun"]==run].sort_values("sample")
        train_runs_data.append(df_run)

    # Fit linear models for 9 actuator pairs
    linear_models = {}
    for a_disp, b_disp in ACTUATOR_PROCESS_PAIRS:
        a_col = DISPLAY_TO_COL[a_disp]
        b_col = DISPLAY_TO_COL[b_disp]
        a, b, std = fit_linear_residual(train_runs_data, a_col, b_col)
        linear_models[(a_disp, b_disp)] = (a, b, std)
        print(f"  {a_disp} -> {b_disp}: B = {a:.4f}*A + {b:.4f}, resid std {std:.4f}")

    # For paired correlation, compute baseline mean/std from train
    paired_baselines = {}
    for a_disp, b_disp in PAIRED_PAIRS:
        a_col = DISPLAY_TO_COL[a_disp]
        b_col = DISPLAY_TO_COL[b_disp]
        corrs = []
        for run in train_runs:
            df_run = normal_all[normal_all["simulationRun"]==run]
            # Use 60-window correlation as in detector, but for relationship we use whole run correlation
            # For baseline, use per-window correlation mean
            arr_a = df_run[a_col].values
            arr_b = df_run[b_col].values
            # Compute windowed correlation (60, stride 5)
            for start in range(0, len(arr_a)-60, 5):
                win_a = arr_a[start:start+60]
                win_b = arr_b[start:start+60]
                if np.std(win_a) < 1e-9 or np.std(win_b) < 1e-9:
                    continue
                corr = np.corrcoef(win_a, win_b)[0,1]
                corrs.append(corr)
        paired_baselines[(a_disp, b_disp)] = (float(np.mean(corrs)), float(np.std(corrs)))
        print(f"  Paired {a_disp} vs {b_disp}: corr mean {np.mean(corrs):.4f} std {np.std(corrs):.4f}")

    # For lagged, similar but with lag
    lagged_baselines = {}
    for a_disp, b_disp, lag in LAGGED_PAIRS:
        a_col = DISPLAY_TO_COL[a_disp]
        b_col = DISPLAY_TO_COL[b_disp]
        corrs = []
        for run in train_runs:
            df_run = normal_all[normal_all["simulationRun"]==run]
            arr_a = df_run[a_col].values
            arr_b = df_run[b_col].values
            for start in range(0, len(arr_a)-60-lag, 5):
                win_a = arr_a[start:start+60]
                win_b = arr_b[start+lag:start+60+lag]
                if np.std(win_a) < 1e-9 or np.std(win_b) < 1e-9:
                    continue
                corr = np.corrcoef(win_a, win_b)[0,1]
                corrs.append(corr)
        lagged_baselines[(a_disp, b_disp, lag)] = (float(np.mean(corrs)), float(np.std(corrs)))
        print(f"  Lagged {a_disp}->{b_disp} lag {lag}: corr mean {np.mean(corrs):.4f} std {np.std(corrs):.4f}")

    # Now compute thresholds from 75 val runs (p99 per relationship)
    print("\nComputing p99 thresholds from 75 val runs...")
    # For each relationship, compute per-window scores for val runs, then p99
    val_scores = {f"act_{a}->{b}": [] for a,b in ACTUATOR_PROCESS_PAIRS}
    val_scores.update({f"pair_{a}_vs_{b}": [] for a,b in PAIRED_PAIRS})
    val_scores.update({f"lag_{a}->{b}_lag{lag}": [] for a,b,lag in LAGGED_PAIRS})

    for run in val_runs:
        df_run = normal_all[normal_all["simulationRun"]==run].sort_values("sample")
        for (a_disp, b_disp), (a,b,std) in linear_models.items():
            a_col = DISPLAY_TO_COL[a_disp]
            b_col = DISPLAY_TO_COL[b_disp]
            arr_a = df_run[a_col].values
            arr_b = df_run[b_col].values
            # Per-window residual |B - (a*A + b)| / std
            for start in range(0, len(arr_a)-60, 5):
                win_a = arr_a[start:start+60]
                win_b = arr_b[start:start+60]
                # Predict B from A per sample, then mean residual over window
                pred_b = a * win_a + b
                resid = np.abs(win_b - pred_b) / (std + 1e-9)
                score = float(resid.mean())
                val_scores[f"act_{a_disp}->{b_disp}"].append(score)
        for (a_disp, b_disp) in PAIRED_PAIRS:
            a_col = DISPLAY_TO_COL[a_disp]
            b_col = DISPLAY_TO_COL[b_disp]
            arr_a = df_run[a_col].values
            arr_b = df_run[b_col].values
            for start in range(0, len(arr_a)-60, 5):
                win_a = arr_a[start:start+60]
                win_b = arr_b[start:start+60]
                if np.std(win_a) < 1e-9 or np.std(win_b) < 1e-9:
                    score = 0
                else:
                    corr = np.corrcoef(win_a, win_b)[0,1]
                    mean, std = paired_baselines[(a_disp, b_disp)]
                    score = abs(corr - mean) / (std + 1e-9)
                val_scores[f"pair_{a_disp}_vs_{b_disp}"].append(float(score))
        for (a_disp, b_disp, lag) in LAGGED_PAIRS:
            a_col = DISPLAY_TO_COL[a_disp]
            b_col = DISPLAY_TO_COL[b_disp]
            arr_a = df_run[a_col].values
            arr_b = df_run[b_col].values
            for start in range(0, len(arr_a)-60-lag, 5):
                win_a = arr_a[start:start+60]
                win_b = arr_b[start+lag:start+60+lag]
                if np.std(win_a) < 1e-9 or np.std(win_b) < 1e-9:
                    score = 0
                else:
                    corr = np.corrcoef(win_a, win_b)[0,1]
                    mean, std = lagged_baselines[(a_disp, b_disp, lag)]
                    score = abs(corr - mean) / (std + 1e-9)
                val_scores[f"lag_{a_disp}->{b_disp}_lag{lag}"].append(float(score))

    thresholds = {}
    for key, vals in val_scores.items():
        arr = np.array(vals)
        thresholds[key] = {
            "p99": float(np.percentile(arr, 99)),
            "p99.5": float(np.percentile(arr, 99.5)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
        print(f"  {key}: p99 {thresholds[key]['p99']:.3f} mean {thresholds[key]['mean']:.3f}")

    # Evaluate on 75 test + 500 each fault
    print("\nEvaluating on test...")
    out_rows = []
    # Normal test
    for run in test_runs:
        df_run = normal_all[normal_all["simulationRun"]==run].sort_values("sample")
        for (a_disp, b_disp) in ACTUATOR_PROCESS_PAIRS:
            key = f"act_{a_disp}->{b_disp}"
            a_col = DISPLAY_TO_COL[a_disp]
            b_col = DISPLAY_TO_COL[b_disp]
            a, b, std = linear_models[(a_disp, b_disp)]
            arr_a = df_run[a_col].values
            arr_b = df_run[b_col].values
            scores = []
            for start in range(0, len(arr_a)-60, 5):
                win_a = arr_a[start:start+60]
                win_b = arr_b[start:start+60]
                pred_b = a * win_a + b
                resid = np.abs(win_b - pred_b) / (std + 1e-9)
                scores.append(float(resid.mean()))
            scores = np.array(scores)
            p99 = thresholds[key]["p99"]
            out_rows.append({
                "type": "normal", "simulationRun": run, "relationship": key,
                "score_type": "actuator_process_residual",
                "mean_score": float(scores.mean()), "max_score": float(scores.max()),
                "p95_score": float(np.percentile(scores,95)), "n_above_p99": int((scores>p99).sum()),
                "max_consec_p99": int(max_consecutive(scores>p99)), "event_p99": int(max_consecutive(scores>p99)>=3),
            })
        for (a_disp, b_disp) in PAIRED_PAIRS:
            key = f"pair_{a_disp}_vs_{b_disp}"
            a_col = DISPLAY_TO_COL[a_disp]
            b_col = DISPLAY_TO_COL[b_disp]
            mean, std = paired_baselines[(a_disp, b_disp)]
            arr_a = df_run[a_col].values
            arr_b = df_run[b_col].values
            scores = []
            for start in range(0, len(arr_a)-60, 5):
                win_a = arr_a[start:start+60]
                win_b = arr_b[start:start+60]
                if np.std(win_a) < 1e-9 or np.std(win_b) < 1e-9:
                    scores.append(0)
                else:
                    corr = np.corrcoef(win_a, win_b)[0,1]
                    scores.append(float(abs(corr - mean) / (std+1e-9)))
            scores = np.array(scores)
            p99 = thresholds[key]["p99"]
            out_rows.append({
                "type": "normal", "simulationRun": run, "relationship": key,
                "score_type": "pair_correlation_deviation",
                "mean_score": float(scores.mean()), "max_score": float(scores.max()),
                "p95_score": float(np.percentile(scores,95)), "n_above_p99": int((scores>p99).sum()),
                "max_consec_p99": int(max_consecutive(scores>p99)), "event_p99": int(max_consecutive(scores>p99)>=3),
            })
        for (a_disp, b_disp, lag) in LAGGED_PAIRS:
            key = f"lag_{a_disp}->{b_disp}_lag{lag}"
            a_col = DISPLAY_TO_COL[a_disp]
            b_col = DISPLAY_TO_COL[b_disp]
            mean, std = lagged_baselines[(a_disp, b_disp, lag)]
            arr_a = df_run[a_col].values
            arr_b = df_run[b_col].values
            scores = []
            for start in range(0, len(arr_a)-60-lag, 5):
                win_a = arr_a[start:start+60]
                win_b = arr_b[start+lag:start+60+lag]
                if np.std(win_a) < 1e-9 or np.std(win_b) < 1e-9:
                    scores.append(0)
                else:
                    corr = np.corrcoef(win_a, win_b)[0,1]
                    scores.append(float(abs(corr - mean) / (std+1e-9)))
            scores = np.array(scores)
            p99 = thresholds[key]["p99"]
            out_rows.append({
                "type": "normal", "simulationRun": run, "relationship": key,
                "score_type": "lagged_correlation_deviation",
                "mean_score": float(scores.mean()), "max_score": float(scores.max()),
                "p95_score": float(np.percentile(scores,95)), "n_above_p99": int((scores>p99).sum()),
                "max_consec_p99": int(max_consecutive(scores>p99)), "event_p99": int(max_consecutive(scores>p99)>=3),
            })

    # Faults 3,9,15 (500 each)
    for fid in [3,9,15]:
        fault_runs = sorted(fault_all[fault_all["faultNumber"]==fid]["simulationRun"].unique())
        print(f"\nFault {fid} ({len(fault_runs)} runs)...")
        for run in fault_runs:
            df_run = fault_all[(fault_all["faultNumber"]==fid) & (fault_all["simulationRun"]==run)].sort_values("sample")
            for (a_disp, b_disp) in ACTUATOR_PROCESS_PAIRS:
                key = f"act_{a_disp}->{b_disp}"
                a_col = DISPLAY_TO_COL[a_disp]
                b_col = DISPLAY_TO_COL[b_disp]
                a, b, std = linear_models[(a_disp, b_disp)]
                arr_a = df_run[a_col].values
                arr_b = df_run[b_col].values
                scores = []
                for start in range(0, len(arr_a)-60, 5):
                    win_a = arr_a[start:start+60]
                    win_b = arr_b[start:start+60]
                    pred_b = a * win_a + b
                    resid = np.abs(win_b - pred_b) / (std + 1e-9)
                    scores.append(float(resid.mean()))
                scores = np.array(scores)
                # For fault, check if any window post-onset is above
                # Use same p99 from val
                p99 = thresholds[key]["p99"]
                # Only count post-onset windows for event
                n_windows = len(scores)
                starts = np.arange(n_windows)*5
                is_post = (starts + 60) > 160
                # For event, need consecutive in post
                post_scores = scores[is_post] if is_post.any() else np.array([0])
                out_rows.append({
                    "type": f"fault{fid}", "simulationRun": run, "relationship": key,
                    "score_type": "actuator_process_residual",
                    "mean_score": float(scores.mean()), "max_score": float(scores.max()),
                    "p95_score": float(np.percentile(scores,95)), "n_above_p99": int((scores>p99).sum()),
                    "max_consec_p99": int(max_consecutive(scores[is_post]>p99)) if is_post.any() else 0,
                    "event_p99": int(max_consecutive(scores[is_post]>p99)>=3) if is_post.any() else 0,
                })
            for (a_disp, b_disp) in PAIRED_PAIRS:
                key = f"pair_{a_disp}_vs_{b_disp}"
                a_col = DISPLAY_TO_COL[a_disp]
                b_col = DISPLAY_TO_COL[b_disp]
                mean, std = paired_baselines[(a_disp, b_disp)]
                arr_a = df_run[a_col].values
                arr_b = df_run[b_col].values
                scores = []
                for start in range(0, len(arr_a)-60, 5):
                    win_a = arr_a[start:start+60]
                    win_b = arr_b[start:start+60]
                    if np.std(win_a) < 1e-9 or np.std(win_b) < 1e-9:
                        scores.append(0)
                    else:
                        corr = np.corrcoef(win_a, win_b)[0,1]
                        scores.append(float(abs(corr - mean) / (std+1e-9)))
                scores = np.array(scores)
                p99 = thresholds[key]["p99"]
                starts = np.arange(len(scores))*5
                is_post = (starts + 60) > 160
                out_rows.append({
                    "type": f"fault{fid}", "simulationRun": run, "relationship": key,
                    "score_type": "pair_correlation_deviation",
                    "mean_score": float(scores.mean()), "max_score": float(scores.max()),
                    "p95_score": float(np.percentile(scores,95)), "n_above_p99": int((scores>p99).sum()),
                    "max_consec_p99": int(max_consecutive(scores[is_post]>p99)) if is_post.any() else 0,
                    "event_p99": int(max_consecutive(scores[is_post]>p99)>=3) if is_post.any() else 0,
                })
            for (a_disp, b_disp, lag) in LAGGED_PAIRS:
                key = f"lag_{a_disp}->{b_disp}_lag{lag}"
                a_col = DISPLAY_TO_COL[a_disp]
                b_col = DISPLAY_TO_COL[b_disp]
                mean, std = lagged_baselines[(a_disp, b_disp, lag)]
                arr_a = df_run[a_col].values
                arr_b = df_run[b_col].values
                scores = []
                for start in range(0, len(arr_a)-60-lag, 5):
                    win_a = arr_a[start:start+60]
                    win_b = arr_b[start+lag:start+60+lag]
                    if np.std(win_a) < 1e-9 or np.std(win_b) < 1e-9:
                        scores.append(0)
                    else:
                        corr = np.corrcoef(win_a, win_b)[0,1]
                        scores.append(float(abs(corr - mean) / (std+1e-9)))
                scores = np.array(scores)
                p99 = thresholds[key]["p99"]
                starts = np.arange(len(scores))*5
                is_post = (starts + 60) > 160
                out_rows.append({
                    "type": f"fault{fid}", "simulationRun": run, "relationship": key,
                    "score_type": "lagged_correlation_deviation",
                    "mean_score": float(scores.mean()), "max_score": float(scores.max()),
                    "p95_score": float(np.percentile(scores,95)), "n_above_p99": int((scores>p99).sum()),
                    "max_consec_p99": int(max_consecutive(scores[is_post]>p99)) if is_post.any() else 0,
                    "event_p99": int(max_consecutive(scores[is_post]>p99)>=3) if is_post.any() else 0,
                })
        print(f"  Fault {fid} done: {len([r for r in out_rows if r['type']==f'fault{fid}'])} rows")

    # Save
    import csv
    out_path = Path("outputs/evaluation/relationship_detector_per_run.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nSaved {out_path} ({len(out_rows)} rows)")

    # Summary per relationship
    df = pd.DataFrame(out_rows)
    summary = []
    for rel in sorted(df["relationship"].unique()):
        for typ in ["normal", "fault3", "fault9", "fault15"]:
            sub = df[(df["relationship"]==rel) & (df["type"]==typ)]
            if len(sub)==0:
                continue
            summary.append({
                "relationship": rel,
                "type": typ,
                "runs": len(sub),
                "mean_max": float(sub["max_score"].mean()),
                "p95_max": float(np.percentile(sub["max_score"],95)),
                "event_rate_p99": float(sub["event_p99"].mean()),
                "mean_n_above": float(sub["n_above_p99"].mean()),
            })
    with open("outputs/evaluation/relationship_detector_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved summary")

    # Print comparison: for each relationship, normal vs fault event rates
    print("\n=== Relationship Event Rates (p99, post-onset, >=3 consec) ===")
    rels = sorted(df["relationship"].unique())
    for rel in rels[:5]:
        n = df[(df["relationship"]==rel) & (df["type"]=="normal")]["event_p99"].mean()
        f3 = df[(df["relationship"]==rel) & (df["type"]=="fault3")]["event_p99"].mean() if len(df[(df["relationship"]==rel) & (df["type"]=="fault3")]) else 0
        f9 = df[(df["relationship"]==rel) & (df["type"]=="fault9")]["event_p99"].mean() if len(df[(df["relationship"]==rel) & (df["type"]=="fault9")]) else 0
        f15 = df[(df["relationship"]==rel) & (df["type"]=="fault15")]["event_p99"].mean() if len(df[(df["relationship"]==rel) & (df["type"]=="fault15")]) else 0
        print(f"{rel:35s} normal {n:.3f} fault3 {f3:.3f} fault9 {f9:.3f} fault15 {f15:.3f}")

    # Save report
    with open("outputs/evaluation/relationship_detector_report.md", "w") as f:
        f.write("# Relationship Anomaly Detection - Faults 3,9,15\n\n")
        f.write("18 defined pairs from KNOWN_PROCESS_RELATIONSHIPS and SUBSYSTEM_GROUPS, no invented\n\n")
        f.write("Baselines from 350 normal train, thresholds p99 from 75 normal val (same methodology as 0.687, not same value)\n\n")
        f.write("| relationship | normal_event | fault3 | fault9 | fault15 |\n")
        f.write("|--------------|--------------|--------|--------|---------|\n")
        for rel in sorted(df["relationship"].unique()):
            n = df[(df["relationship"]==rel) & (df["type"]=="normal")]["event_p99"].mean()
            f3 = df[(df["relationship"]==rel) & (df["type"]=="fault3")]["event_p99"].mean() if len(df[(df["relationship"]==rel) & (df["type"]=="fault3")]) else 0
            f9 = df[(df["relationship"]==rel) & (df["type"]=="fault9")]["event_p99"].mean() if len(df[(df["relationship"]==rel) & (df["type"]=="fault9")]) else 0
            f15 = df[(df["relationship"]==rel) & (df["type"]=="fault15")]["event_p99"].mean() if len(df[(df["relationship"]==rel) & (df["type"]=="fault15")]) else 0
            f.write(f"| {rel} | {n:.3f} | {f3:.3f} | {f9:.3f} | {f15:.3f} |\n")
    print("Saved report")

def max_consecutive(arr):
    m=c=0
    for v in arr:
        if v:
            c+=1
            m=max(m,c)
        else:
            c=0
    return m

if __name__ == "__main__":
    main()
