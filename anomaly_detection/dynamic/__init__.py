"""Dynamic detector package: DPCA + CVA for temporal/second-order faults."""

from .lag_builder import build_augmented_matrix, build_past_future
from .runner import DynamicDetectorRunner

__all__ = ["build_past_future", "build_augmented_matrix", "DynamicDetectorRunner"]
