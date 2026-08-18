"""Prepare the raw TEP data: validate, scale-fit (normal only), window, split.

Produces:
  * data/processed/normal_values.npy            (T x F, raw scale)
  * data/processed/normal_feature_names.json
  * data/processed/fault_values/fault_XX.npy     (raw scale per fault)
  * data/processed/fault_metadata.json           (onset index per fault)
  * outputs/preprocessing/scaler.pkl + config.json + baseline_stats.json
  * data/processed/normal_windows_train.npy / _val.npy / _test.npy
  * data/processed/window_metadata.json

The scaler and baseline are fitted on normal-operation data ONLY.

Usage:
    python scripts/prepare_tep.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.scaler import build_and_save_scaler  # noqa: E402
from preprocessing.tep_loader import load_tep_data  # noqa: E402
from preprocessing.windowing import (  # noqa: E402
    segment_ids_for_runs,
    split_windows_by_segment,
    to_windows,
)
from utils import ensure_dir, load_config, save_json, set_seed  # noqa: E402

logger = logging.getLogger(__name__)


def prepare_tep(config: Dict) -> Dict:
    set_seed(config.get("seed", 42))
    processed_dir = ensure_dir(config["paths"]["processed_data_path"])
    scaler_dir = ensure_dir(config["preprocessing"]["scaler_dir"])

    data = load_tep_data(config)
    if data.normal.empty:
        raise RuntimeError(
            "No normal-operation data found. Place normal CSV(s) in "
            f"{config['paths']['normal_data_path']} matching "
            f"{config['dataset']['normal_file_pattern']}."
        )
    if not data.faults:
        logger.warning("No fault data found; only normal preprocessing will run.")

    num_features = data.normal.shape[1]
    normal_values = data.normal.astype(np.float32).to_numpy()

    # ---- scaler + baseline (fitted on normal only) --------------------------
    build_and_save_scaler(data.normal, config, scaler_dir)

    np.save(processed_dir / "normal_values.npy", normal_values)
    save_json(processed_dir / "normal_feature_names.json",
              {"feature_names": list(data.normal.columns),
               "num_features": num_features})

    # ---- faults -------------------------------------------------------------
    fault_dir = ensure_dir(processed_dir / "fault_values")
    fault_metadata: Dict[str, dict] = {}
    for fault_id, (frame, onset) in sorted(data.faults.items()):
        np.save(fault_dir / f"fault_{fault_id:02d}.npy",
                frame.astype(np.float32).to_numpy())
        fault_metadata[f"fault_{fault_id:02d}"] = {
            "fault_id": fault_id,
            "onset_index": onset,
            "n_samples": len(frame),
        }
    save_json(processed_dir / "fault_metadata.json", fault_metadata)

    # ---- windows + leakage-free split ----------------------------------------
    window_size = int(config["windowing"]["window_size"])
    stride = int(config["windowing"]["stride"])
    block_samples = int(config["windowing"].get("block_samples", 500))

    # Split the (single) normal run into contiguous temporal blocks. Each block
    # is one segment; the split assigns whole blocks to train/val/test, which is
    # leakage-free (a window never spans two segments) while still allowing a
    # validation set from a single long run.
    blocks: List[np.ndarray] = []
    segment_ids: List[int] = []
    block_starts = np.arange(0, len(normal_values), block_samples)
    for block_id, start in enumerate(block_starts):
        block = normal_values[start:start + block_samples]
        if len(block) >= window_size:
            blocks.append(block)
            segment_ids.extend([block_id] * max(0, (len(block) - window_size) // stride + 1))
    normal_windows = np.concatenate([to_windows(b, window_size, stride) for b in blocks], axis=0)

    train_idx, val_idx, test_idx = split_windows_by_segment(
        normal_windows, segment_ids,
        val_ratio=float(config["anomaly_detector"]["train"].get("val_ratio", 0.15)),
        seed=int(config.get("seed", 42)),
        test_ratio=float(config["windowing"].get("test_ratio", 0.2)),
    )
    np.save(processed_dir / "normal_windows_train.npy", normal_windows[train_idx])
    np.save(processed_dir / "normal_windows_val.npy", normal_windows[val_idx])
    if test_idx is not None and len(test_idx):
        np.save(processed_dir / "normal_windows_test.npy", normal_windows[test_idx])

    save_json(processed_dir / "window_metadata.json", {
        "window_size": window_size,
        "stride": stride,
        "block_samples": block_samples,
        "num_features": num_features,
        "n_windows": int(len(normal_windows)),
        "split": {
            "method": "by_temporal_segment_blocks",
            "n_blocks": len(blocks),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)) if test_idx is not None else 0,
            "val_ratio": float(config["anomaly_detector"]["train"].get("val_ratio", 0.15)),
        },
    })

    summary = {
        "num_features": num_features,
        "normal_samples": int(len(normal_values)),
        "n_fault_scenarios": len(data.faults),
        "faults": fault_metadata,
        "normal_windows": int(len(normal_windows)),
        "processed_dir": str(processed_dir),
        "scaler_dir": str(scaler_dir),
    }
    save_json(processed_dir / "prepare_summary.json", summary)
    logger.info("TEP preparation complete: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TEP data.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")
    config = load_config(args.config)
    prepare_tep(config)


if __name__ == "__main__":
    main()