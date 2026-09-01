"""Continuous sensor stream simulator.

The TEP dataset is historical, so this module replays records sequentially as
if they were arriving in real time::

    t1 -> sensor_vector
    t2 -> sensor_vector
    ...

Usage::

    stream = SensorStream(df, window_size=60, stride=5, replay_rate=0.0)
    for window, meta in stream.iter_windows():
        process_window(window)
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from preprocessing.tep_loader import load_single_csv
from utils import load_config

logger = logging.getLogger(__name__)


class SensorStream:
    """Replay a T x F sensor matrix one record at a time.

    A rolling buffer emits a new [W, F] window every ``stride`` records once at
    least ``window_size`` records are buffered. An optional fault frame can be
    spliced in at a chosen sample index to simulate an injected disturbance.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        window_size: int = 60,
        stride: int = 5,
        replay_rate: float = 0.0,
        inject_fault_frame: Optional[pd.DataFrame] = None,
        inject_fault_at: Optional[int] = None,
        fault_label: Optional[int] = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.replay_rate = float(replay_rate)   # seconds per record; 0 = fast
        self.fault_frame = inject_fault_frame
        self.inject_fault_at = inject_fault_at
        self.fault_label = fault_label
        self.feature_names: List[str] = list(frame.columns)

    # ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        path: str,
        config: Optional[dict] = None,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
        replay_rate: Optional[float] = None,
        inject_fault_frame: Optional[pd.DataFrame] = None,
        inject_fault_at: Optional[int] = None,
        fault_label: Optional[int] = None,
        fault_number: Optional[int] = None,
        simulation_run: Optional[int] = None,
    ) -> "SensorStream":
        config = config or load_config()
        df = load_single_csv(
            path, config, fault_number=fault_number, simulation_run=simulation_run
        )
        ws = window_size or int(config["streaming"]["window_size"])
        st = stride or int(config["streaming"]["stride"])
        rr = replay_rate if replay_rate is not None else float(config["streaming"].get("replay_rate", 0.0))
        return cls(df, ws, st, rr, inject_fault_frame, inject_fault_at, fault_label)

    # ------------------------------------------------------------------
    def iter_records(self) -> Iterator[Dict]:
        """Yield each record as {sample_index, values[F], is_fault, fault_id}."""
        total = len(self.frame)
        for i, (_, row) in enumerate(self.frame.iterrows()):
            if self._should_delay():
                time.sleep(self.replay_rate)
            injected = False
            label = None
            if self.inject_fault_at is not None and i >= self.inject_fault_at and self.fault_frame is not None:
                injected = True
                label = self.fault_label
            if injected:
                offset = i - self.inject_fault_at
                row = self.fault_frame.iloc[offset % len(self.fault_frame)]
            yield {
                "sample_index": int(i),
                "values": np.asarray(row.to_numpy(dtype=np.float32), dtype=np.float32),
                "is_fault": bool(injected),
                "fault_label": label,
            }

    def _should_delay(self) -> bool:
        return self.replay_rate > 0

    def iter_windows(self) -> Iterator[Tuple[np.ndarray, Dict]]:
        """Yield (window[W, F], metadata) every ``stride`` records."""
        buffer: List[np.ndarray] = []
        counter = 0
        for record in self.iter_records():
            buffer.append(record["values"])
            counter += 1
            if len(buffer) >= self.window_size:
                window = np.stack(buffer[-self.window_size:], axis=0)
                meta = {
                    "sample_index": record["sample_index"],
                    "is_fault": record["is_fault"],
                    "fault_label": record["fault_label"],
                    "window_id": counter // self.stride,
                }
                yield window, meta
                # keep overlap for the next window
                keep = self.window_size - self.stride
                buffer = buffer[-keep:] if keep > 0 else []

    def run(self, process_window) -> int:
        """Run the stream, calling ``process_window(window, meta)`` per window.

        Returns the number of windows processed.
        """
        n = 0
        for window, meta in self.iter_windows():
            process_window(window, meta)
            n += 1
        return n

    @property
    def n_records(self) -> int:
        return len(self.frame)