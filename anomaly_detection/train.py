"""Training for the LSTM autoencoder (normal-operation windows only)."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly_detection.dataset import WindowedTensorDataset, make_train_val_sets
from anomaly_detection.lstm_autoencoder import LSTMAutoencoder, build_autoencoder
from utils import ensure_dir, get_device, save_json, set_seed

logger = logging.getLogger(__name__)

MODEL_FILE = "model.pt"
CONFIG_FILE = "config.json"
CHECKPOINT_FILE = "checkpoint_last.pt"


def _mse_loss(
    model: nn.Module, batch: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = batch.to(device)
    recon = model(batch)
    loss = nn.functional.mse_loss(recon, batch, reduction="none")
    per_sample = loss.mean(dim=(1, 2))
    return per_sample.mean(), per_sample


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, np.ndarray]:
    """Return (mean MSE, per-sample MSE array)."""
    model.eval()
    per_sample: list = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            _, losses = _mse_loss(model, batch, device)
            per_sample.append(losses.cpu().numpy())
    if not per_sample:
        return float("nan"), np.array([])
    scores = np.concatenate(per_sample)
    return float(scores.mean()), scores


def train_autoencoder(
    config: Dict,
    normal_windows: np.ndarray,
    segment_ids,
    output_dir,
    resume: bool = False,
    device=None,
) -> LSTMAutoencoder:
    """Train the LSTM AE on normal windows; returns the best model.

    Supports checkpointing every N epochs, resume-from-last-checkpoint, early
    stopping on validation MSE, CPU fallback and GPU detection.
    """
    set_seed(config.get("seed", 42))
    output_dir = ensure_dir(output_dir)
    device = device or get_device()
    logger.info("Training device: %s", device)

    train_cfg = config["anomaly_detector"]["train"]
    window_cfg = config["windowing"]
    num_features = int(normal_windows.shape[2])
    sequence_length = int(normal_windows.shape[1])

    model = build_autoencoder(config, num_features, sequence_length).to(device)
    logger.info("LSTM AE parameters: %d", sum(p.numel() for p in model.parameters()))

    train_set, val_set = make_train_val_sets(
        normal_windows,
        segment_ids,
        float(train_cfg.get("val_ratio", 0.15)),
        int(config.get("seed", 42)),
    )
    batch_size = int(train_cfg.get("batch_size", 64))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    scheduler_name = train_cfg.get("lr_scheduler", "cosine").lower()
    epochs = int(train_cfg.get("epochs", 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    ) if scheduler_name == "cosine" else None

    patience = int(train_cfg.get("early_stopping_patience", 8))
    clip_norm = float(train_cfg.get("gradient_clip_norm", 1.0))
    start_epoch = 0
    best_val: Optional[float] = None
    best_epoch = -1
    stale = 0

    checkpoint_path = output_dir / CHECKPOINT_FILE
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_val = state.get("best_val")
        logger.info("Resumed from checkpoint %s (epoch %d).", checkpoint_path, start_epoch - 1)

    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            optimizer.zero_grad()
            loss, _ = _mse_loss(model, batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.5f}")
        if scheduler is not None:
            scheduler.step()

        train_mse = running / max(n_batches, 1)
        val_mse, _ = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d/%d | train MSE %.5f | val MSE %.5f",
            epoch + 1, epochs, train_mse, val_mse,
        )

        if best_val is None or val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            stale = 0
            _save_model(model, config, output_dir, num_features, sequence_length, val_mse)
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stopping at epoch %d (best val MSE %.5f @ epoch %d).",
                            epoch + 1, best_val, best_epoch + 1)
                break

        if (epoch + 1) % int(train_cfg.get("checkpoint_every", 5)) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val": best_val,
                },
                checkpoint_path,
            )

    model.eval()
    logger.info("Anomaly detector training complete. Best val MSE %.5f @ epoch %d.",
                best_val, best_epoch + 1)
    return model


def _save_model(
    model: nn.Module,
    config: Dict,
    output_dir: Path,
    num_features: int,
    sequence_length: int,
    val_mse: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_features": num_features,
            "sequence_length": sequence_length,
            "val_mse": float(val_mse),
        },
        output_dir / MODEL_FILE,
    )
    save_json(
        output_dir / CONFIG_FILE,
        {
            "num_features": num_features,
            "sequence_length": sequence_length,
            "val_mse": float(val_mse),
            "lstm": config["anomaly_detector"]["lstm"],
        },
    )
    logger.info("Saved best model to %s", output_dir / MODEL_FILE)