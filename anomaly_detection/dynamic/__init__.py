"""Dynamic detector package: DPCA + CVA for temporal/second-order faults."""

from .lag_builder import build_past_future

__all__ = ["build_past_future"]
