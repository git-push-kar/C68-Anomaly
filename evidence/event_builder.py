"""Anomaly event aggregation and structured evidence generation.

Multiple consecutive anomalous windows are grouped into ONE anomaly event. The
event builder converts raw numerical evidence into the structured JSON object
that is later handed to InternVL2-2B.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np

from evidence.process_relationships import RelationshipKind, suggest_subsystems
from evidence.sensor_contribution import (
    compute_percent_deviations,
    event_level_deviations,
    top_k_sensors,
)
from evidence.temporal_analysis import (
    analyze_temporal_sequence,
    pre_post_context,
    sensor_trend,
)
from preprocessing.scaler import BaselineStats
from utils import to_native

logger = logging.getLogger(__name__)


@dataclass
class EventEvidence:
    """Structured evidence handed to the LLM (JSON-serializable)."""

    event_id: str
    anomaly_score: float
    severity: str
    pre_anomaly_context: Dict
    top_anomalous_sensors: List[Dict]
    temporal_sequence: List[Dict]
    candidate_subsystem: str
    candidate_subsystem_score: float
    reasoning_notes: List[str]
    evidence_type: str = "model_derived_evidence"
    uncertainty: str = ""

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "anomaly_score": float(self.anomaly_score),
            "severity": self.severity,
            "pre_anomaly_context": self.pre_anomaly_context,
            "top_anomalous_sensors": self.top_anomalous_sensors,
            "temporal_sequence": self.temporal_sequence,
            "candidate_subsystem": self.candidate_subsystem,
            "candidate_subsystem_score": float(self.candidate_subsystem_score),
            "reasoning_notes": self.reasoning_notes,
            "evidence_type": self.evidence_type,
            "uncertainty": self.uncertainty,
        }


@dataclass
class AnomalyEvent:
    """The full runtime record of one anomaly event."""

    event_id: str
    start_time: datetime
    detection_time: datetime
    end_time: Optional[datetime]
    start_sample: int
    end_sample: Optional[int]
    n_windows: int
    max_anomaly_score: float
    mean_anomaly_score: float
    evidence: EventEvidence
    sensor_time_series: Optional[Dict] = field(default=None)  # light summaries only
    report: Optional[Dict] = None
    fault_label: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "start_time": self.start_time.isoformat(),
            "detection_time": self.detection_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "n_windows": self.n_windows,
            "max_anomaly_score": float(self.max_anomaly_score),
            "mean_anomaly_score": float(self.mean_anomaly_score),
            "evidence": self.evidence.to_dict(),
            "sensor_time_series": self.sensor_time_series,
            "report": self.report,
            "fault_label": self.fault_label,
        }


def _severity_from_score(score: float, threshold: float) -> str:
    ratio = score / max(threshold, 1e-9)
    if ratio >= 3.0:
        return "critical"
    if ratio >= 1.8:
        return "high"
    if ratio >= 1.2:
        return "medium"
    return "low"


def build_event(
    event_id: str,
    anomaly_scores: Sequence[float],
    per_sensor_errors: np.ndarray,
    event_windows: np.ndarray,
    baseline: BaselineStats,
    feature_names: Sequence[str],
    threshold: float,
    start_sample: int,
    config: Dict,
    start_time: Optional[datetime] = None,
    detection_time: Optional[datetime] = None,
    fault_label: Optional[int] = None,
) -> AnomalyEvent:
    """Assemble one anomaly event from a run of anomalous windows.

    Args:
        anomaly_scores: per-window anomaly scores (len == N).
        per_sensor_errors: [N, F] per-sensor reconstruction errors.
        event_windows: [N, W, F] event windows in ORIGINAL (unscaled) units.
        baseline: normal baseline statistics (original units).
        feature_names: sensor column names.
        threshold: anomaly threshold (for severity scaling).
        start_sample: sample index of the first anomalous window.
        config: the ``evidence`` configuration section.
    """
    ev = config["evidence"]
    top_k = int(ev.get("top_k_sensors", 5))
    min_dev = float(ev.get("min_deviation_percent", 3.0))
    baseline_windows = int(ev.get("baseline_windows", 20))
    trend_smoothing = int(ev.get("trend_smoothing_window", 8))
    onset_sensitivity = float(ev.get("onset_sensitivity", 1.5))
    max_temporal = int(ev.get("max_temporal_events", 6))

    scores = np.asarray(anomaly_scores, dtype=np.float32)
    errors = np.asarray(per_sensor_errors, dtype=np.float32)
    windows = np.asarray(event_windows, dtype=np.float32)

    max_score = float(scores.max())
    mean_score = float(scores.mean())
    severity = _severity_from_score(max_score, threshold)

    if errors.ndim == 1:
        errors = errors[None, :]
    agg_errors = errors.max(axis=0)  # worst per-sensor error over the event

    # ---- top sensors with % deviation and trend ---------------------------
    event_mean = windows.mean(axis=(0, 1))
    top_sensors = event_level_deviations(
        agg_errors, event_mean, baseline.mean, baseline.std, feature_names,
        top_k=top_k, min_deviation_percent=min_dev,
    )
    for entry in top_sensors:
        idx = entry["index"]
        series = windows[:, :, idx].mean(axis=1)
        entry["trend"] = sensor_trend(series, smoothing=trend_smoothing)

    # ---- temporal sequence ------------------------------------------------
    temporal = analyze_temporal_sequence(
        errors, feature_names,
        window_size=int(config["windowing"]["window_size"]),
        stride=int(config["windowing"]["stride"]),
        onset_sensitivity=onset_sensitivity,
        max_events=max_temporal,
    )
    sequence = temporal.get("sequence", [])

    # ---- candidate subsystem (evidence-based, NOT causation) --------------
    candidates = suggest_subsystems(
        [s["display_name"] for s in top_sensors], top_k=3
    )
    candidate_subsystem = candidates[0]["subsystem"] if candidates else "unknown"
    candidate_score = float(candidates[0]["matched_sensors"]) / max(top_k, 1) if candidates else 0.0

    # ---- pre/post context -------------------------------------------------
    context = pre_post_context(
        windows, baseline.mean, n_baseline=baseline_windows
    )

    reasoning_notes = [
        "Deviations are measured against the normal-operation baseline "
        "(fitted on normal data only).",
        "Sensor onset ordering is temporal evidence; it does not by itself "
        "prove causation.",
        f"Candidate subsystem '{candidate_subsystem}' is inferred from which "
        "process variables deviate; verify before acting.",
    ]

    evidence = EventEvidence(
        event_id=event_id,
        anomaly_score=max_score,
        severity=severity,
        pre_anomaly_context={
            "duration_minutes": round(float(baseline_windows * config["windowing"]["stride"] / 60.0), 2),
            "status": context["pre_status"],
        },
        top_anomalous_sensors=to_native(top_sensors),
        temporal_sequence=to_native(sequence),
        candidate_subsystem=candidate_subsystem,
        candidate_subsystem_score=candidate_score,
        reasoning_notes=reasoning_notes,
        evidence_type=RelationshipKind.MODEL_DERIVED.value,
        uncertainty=(
            "Correlation between sensor deviations and the candidate subsystem "
            "does not prove causation; on-site inspection is required to confirm."
        ),
    )

    sensor_time_series = {
        "per_sensor_mean_error": to_native(errors.mean(axis=0)),
        "post_mean": to_native(context["post_mean"]),
        "delta": to_native(context["delta"]),
    }

    return AnomalyEvent(
        event_id=event_id,
        start_time=start_time or datetime.now(),
        detection_time=detection_time or datetime.now(),
        end_time=datetime.now(),
        start_sample=start_sample,
        end_sample=start_sample + len(scores) * int(config["windowing"]["stride"]),
        n_windows=int(len(scores)),
        max_anomaly_score=max_score,
        mean_anomaly_score=mean_score,
        evidence=evidence,
        sensor_time_series=sensor_time_series,
        fault_label=fault_label,
    )


def make_event_id(counter: int) -> str:
    return f"ANOM-{counter:04d}"