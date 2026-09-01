"""Train the LSTM autoencoder anomaly detector on normal windows.

Computes the anomaly threshold from normal validation scores and persists the
detector + threshold.

Usage:
    python scripts/train_anomaly_detector.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anomaly_detection import (  # noqa: E402
    AnomalyDetector,
    compute_threshold,
    load_threshold,
    save_threshold,
    train_autoencoder,
)
from anomaly_detection.inference import MODEL_FILE  # noqa: E402
from anomaly_detection.threshold import THRESHOLD_FILE  # noqa: E402
from utils import (  # noqa: E402
    ensure_dir,
    load_config,
    load_json,
    save_json,
    set_seed,
)

logger = logging.getLogger(__name__)


def train_anomaly_detector(config: Dict, resume: bool = False, epochs: int = None) -> dict:
    set_seed(config.get("seed", 42))
    processed = Path(config["paths"]["processed_data_path"])
    model_dir = ensure_dir(config["anomaly_detector"]["model_dir"])

    train_windows = np.load(processed / "normal_windows_train.npy")
    val_windows = np.load(processed / "normal_windows_val.npy")
    logger.info("Train windows: %s | Val windows: %s",
                train_windows.shape, val_windows.shape)

    if epochs:
        config["anomaly_detector"]["train"]["epochs"] = epochs

    # Train in SCALED space (the detector re-scales at inference). Scaling is
    # applied here because the raw TEP magnitudes (reactor pressure ~2500 kPa,
    # temperatures ~100 C) make an unnormalized LSTM unstable.
    from preprocessing.scaler import load_scaler

    scaler = load_scaler(config["preprocessing"]["scaler_dir"])
    num_features = train_windows.shape[2]
    train_scaled = scaler.transform(
        train_windows.reshape(-1, num_features)
    ).reshape(train_windows.shape)
    val_scaled = scaler.transform(
        val_windows.reshape(-1, num_features)
    ).reshape(val_windows.shape)

    model = train_autoencoder(
        config,
        train_scaled,
        segment_ids=None,
        output_dir=model_dir,
        resume=resume,
        validation_windows=val_scaled,
    )
    model.eval()

    # ---- threshold from normal validation scores ----------------------------
    detector = AnomalyDetector(model, scaler, None)  # threshold set below
    val_scores, _ = detector.score_windows(val_windows)
    threshold = compute_threshold(val_scores, config)
    save_threshold(threshold, model_dir)

    # Also record normal-score distribution for reporting.
    save_json(model_dir / "normal_val_scores.json",
              {"mean": float(val_scores.mean()), "std": float(val_scores.std()),
               "max": float(val_scores.max()), "p99": float(np.percentile(val_scores, 99))})

    logger.info("Anomaly detector + threshold saved under %s", model_dir)
    return {
        "model": str(model_dir / MODEL_FILE),
        "threshold_file": str(model_dir / THRESHOLD_FILE),
        "threshold": threshold.threshold,
        "val_scores_mean": float(val_scores.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM autoencoder anomaly detector.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")
    config = load_config(args.config)
    train_anomaly_detector(config, resume=args.resume, epochs=args.epochs)


if __name__ == "__main__":
    main()