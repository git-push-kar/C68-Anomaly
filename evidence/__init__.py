"""Evidence extraction: the numeric bridge between the detector and InternVL."""
from .event_builder import AnomalyEvent, EventEvidence, build_event
from .process_relationships import (
    KNOWN_PROCESS_RELATIONSHIPS,
    SENSOR_NAMES,
    SUBSYSTEM_GROUPS,
    RelationshipKind,
    suggest_subsystems,
)
from .sensor_contribution import (
    compute_percent_deviations,
    compute_sensor_contributions,
    top_k_sensors,
)
from .temporal_analysis import (
    analyze_temporal_sequence,
    detect_onsets,
    sensor_trend,
)

__all__ = [
    "AnomalyEvent",
    "EventEvidence",
    "build_event",
    "KNOWN_PROCESS_RELATIONSHIPS",
    "SENSOR_NAMES",
    "SUBSYSTEM_GROUPS",
    "RelationshipKind",
    "suggest_subsystems",
    "compute_percent_deviations",
    "compute_sensor_contributions",
    "top_k_sensors",
    "analyze_temporal_sequence",
    "detect_onsets",
    "sensor_trend",
]