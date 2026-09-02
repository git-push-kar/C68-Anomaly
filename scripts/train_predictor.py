"""Train minimal LSTM predictor on normal training runs only.
Uses 60x52 -> 52 next-timestep, MSE across 52 sensors.
Threshold p99 from 75 normal VAL only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import ensure_dir, load_config, load_json, save_json, set_seed
from anomaly_detection.lstm_predictor import LSTMPredictor
from preprocessing.scaler import load_scaler

SENSORS = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(1, 12)]
SENSOR_COLS = [s.lower().replace("XMEAS","xmeas").replace("XMV","xmv") for s in SENSORS]

class PredictionDataset(Dataset):
    def __init__(self, runs_data, window_size=60, stride=5):
        # runs_data: list of arrays [500,52] per run
        self.samples = []
        for arr in runs_data:
            # arr: [500,52]
            for start in range(0, len(arr) - window_size, stride):
                window = arr[start:start+window_size]
                target = arr[start+window_size]
                self.samples.append((window, target))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        w, t = self.samples[idx]
        return torch.from_numpy(w).float(), torch.from_numpy(t).float()

def load_runs_for_ids(ids, file_path, sensor_cols):
    df_all = pd.read_csv(file_path)
    runs_data = []
    for run in ids:
        df_run = df_all[df_all["simulationRun"]==run].sort_values("sample")
        arr = df_run[sensor_cols].to_numpy(dtype=np.float32)
        runs_data.append(arr)
    return runs_data

def main():
    config = load_config("configs/config_a5000.yaml")
    set_seed(42)
    manifest = load_json(Path("data/processed/manifests/detector_split.json"))
    train_runs = manifest["train_runs"]
    val_runs = manifest["validation_runs"]
    
    print(f"Train runs: {len(train_runs)}, Val runs: {len(val_runs)}")
    
    # Load scaler
    scaler = load_scaler(config["preprocessing"]["scaler_dir"])
    print("Scaler loaded")
    
    # Load data
    normal_path = "data/raw/normal/TEP_FaultFree_Training.csv"
    # Need to get sensor cols in correct order
    # Use the same sensor_cols as before
    df_sample = pd.read_csv(normal_path, nrows=5)
    sensor_cols = [c for c in df_sample.columns if c.startswith("xmeas") or c.startswith("xmv")]
    # Reorder to SENSORS order
    sensor_cols_ordered = [s.lower().replace("XMEAS","xmeas").replace("XMV","xmv") for s in SENSORS]
    sensor_cols_ordered = [c for c in sensor_cols_ordered if c in sensor_cols]
    
    print("Loading train runs...")
    train_runs_data = []
    for run in train_runs:
        df_run = pd.read_csv(normal_path)
        df_run = df_run[df_run["simulationRun"]==run].sort_values("sample")
        arr = df_run[sensor_cols_ordered].to_numpy(dtype=np.float32)
        # Scale
        arr_scaled = scaler.transform(arr)
        train_runs_data.append(arr_scaled)
    
    print("Loading val runs...")
    val_runs_data = []
    for run in val_runs:
        df_run = pd.read_csv(normal_path)
        df_run = df_run[df_run["simulationRun"]==run].sort_values("sample")
        arr = df_run[sensor_cols_ordered].to_numpy(dtype=np.float32)
        arr_scaled = scaler.transform(arr)
        val_runs_data.append(arr_scaled)
    
    # Create datasets
    train_ds = PredictionDataset(train_runs_data, window_size=60, stride=5)
    val_ds = PredictionDataset(val_runs_data, window_size=60, stride=5)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    
    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = LSTMPredictor(num_features=52, hidden_size=64, num_layers=1, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    best_val = float('inf')
    best_state = None
    patience = 15
    stale = 0
    
    for epoch in range(50):
        model.train()
        train_loss = 0
        for w, t in tqdm(train_loader, desc=f"Epoch {epoch+1}/50", leave=False):
            w, t = w.to(device), t.to(device)
            pred = model(w)
            loss = nn.MSELoss()(pred, t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(w)
        train_loss /= len(train_ds)
        
        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for w, t in val_loader:
                w, t = w.to(device), t.to(device)
                pred = model(w)
                loss = nn.MSELoss()(pred, t)
                val_loss += loss.item() * len(w)
        val_loss /= len(val_ds)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: train {train_loss:.5f} val {val_loss:.5f}")
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stop at epoch {epoch+1}")
                break
    
    # Save
    out_dir = ensure_dir(Path("outputs/prediction_detector"))
    torch.save({
        "model_state_dict": best_state,
        "val_mse": best_val,
        "config": {"hidden_size": 64, "num_layers": 1}
    }, out_dir / "model.pt")
    print(f"Saved to {out_dir / 'model.pt'} best val {best_val:.5f}")
    
    # Threshold from val only (p99)
    # Compute val scores (MSE per sample)
    model.load_state_dict(best_state)
    model.eval()
    val_scores = []
    with torch.no_grad():
        for w, t in val_loader:
            w, t = w.to(device), t.to(device)
            pred = model(w)
            err = ((pred - t) ** 2).mean(dim=1).cpu().numpy()
            val_scores.extend(err.tolist())
    val_scores = np.array(val_scores)
    thresholds = {
        "p99": float(np.percentile(val_scores, 99)),
        "p99.5": float(np.percentile(val_scores, 99.5)),
        "p99.9": float(np.percentile(val_scores, 99.9)),
        "max": float(val_scores.max()),
        "mean": float(val_scores.mean()),
        "std": float(val_scores.std()),
    }
    save_json(out_dir / "threshold.json", thresholds)
    print(f"Thresholds: {thresholds}")
    # Also save p99 as primary
    save_json(out_dir / "threshold_p99.json", {"threshold": thresholds["p99"], "method": "p99_val"})

if __name__ == "__main__":
    main()
