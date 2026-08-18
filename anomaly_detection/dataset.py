"""PyTorch datasets for the LSTM autoencoder."""
from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.windowing import split_windows_by_segment

logger = logging.getLogger(__name__)


class WindowedTensorDataset(Dataset):
    """Dataset over [N, W, F] windows (optionally labelled)."""

    def __init__(
        self,
        windows: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        if windows.ndim != 3:
            raise ValueError(f"Expected 3D windows [N, W, F], got {windows.shape}")
        self.windows = torch.as_tensor(np.asarray(windows, dtype=np.float32))
        self.labels = None
        if labels is not None:
            self.labels = torch.as_tensor(np.asarray(labels, dtype=np.float32))
            if self.labels.shape[0] != self.windows.shape[0]:
                raise ValueError("labels and windows have different lengths")

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, index: int):
        if self.labels is not None:
            return self.windows[index], self.labels[index]
        return self.windows[index]


def make_train_val_sets(
    windows: np.ndarray,
    segment_ids: Sequence[int],
    val_ratio: float,
    seed: int,
    labels: Optional[np.ndarray] = None,
) -> Tuple[WindowedTensorDataset, WindowedTensorDataset]:
    """Build train/validation datasets using a leakage-free segment split.

    Windows that belong to the same contiguous run are never split across
    train and validation (see preprocessing/windowing.split_windows_by_segment).
    """
    train_idx, val_idx, _ = split_windows_by_segment(windows, segment_ids, val_ratio, seed)
    train_labels = labels[train_idx] if labels is not None else None
    val_labels = labels[val_idx] if labels is not None else None
    return (
        WindowedTensorDataset(windows[train_idx], train_labels),
        WindowedTensorDataset(windows[val_idx], val_labels),
    )