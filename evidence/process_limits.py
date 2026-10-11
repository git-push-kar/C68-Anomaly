"""Tennessee Eastman Process (TEP) standard operating limits and deterministic severity rules.

Provides domain grounding for process safety limits and an authoritative rule-based
severity evaluation engine to avoid arbitrary LLM guessing.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Standard operating baselines and safe operating envelopes for key TEP measurements
TEP_OPERATING_LIMITS: Dict[str, Dict[str, Any]] = {
    "Reactor_Pressure": {
        "unit": "kPa",
        "nominal": 2705.0,
        "warning_high": 2850.0,
        "trip_critical": 2950.0,
        "safety_description": "Reactor vessel overpressure / relief valve trip hazard",
    },
    "Reactor_Temperature": {
        "unit": "°C",
        "nominal": 120.4,
        "warning_high": 125.0,
        "trip_critical": 129.0,
        "safety_description": "Exothermic runaway reaction hazard",
    },
    "Separator_Level": {
        "unit": "%",
        "nominal": 50.0,
        "warning_low": 30.0,
        "warning_high": 75.0,
        "safety_description": "Liquid carryover into compressor or loss of seal",
    },
    "Stripper_Level": {
        "unit": "%",
        "nominal": 50.0,
        "warning_low": 30.0,
        "warning_high": 75.0,
        "safety_description": "Bottoms pump cavitation or column flooding",
    },
    "Compressor_Work": {
        "unit": "kW",
        "nominal": 341.0,
        "warning_high": 370.0,
        "safety_description": "Compressor motor overload / surge",
    },
    "Reactor_Cooling_Water_Flow": {
        "unit": "m3/h",
        "nominal": 38.2,
        "warning_high": 45.0,
        "safety_description": "Reactor thermal runaway risk if cooling capacity saturates",
    },
    "Condenser_Cooling_Water_Flow": {
        "unit": "m3/h",
        "nominal": 15.0,
        "warning_high": 22.0,
        "safety_description": "Condenser heat transfer degradation / pressure escalation",
    },
}


def get_sensor_limit_context(sensor_name: str) -> Optional[str]:
    """Return a short human-readable safety limit string for a sensor if known."""
    for key, spec in TEP_OPERATING_LIMITS.items():
        if key.lower() in sensor_name.lower() or sensor_name.lower() in key.lower():
            nom = spec["nominal"]
            unit = spec["unit"]
            desc = spec["safety_description"]
            if "warning_high" in spec:
                return f"nominal {nom}{unit}, safe max ~{spec['warning_high']}{unit} ({desc})"
            return f"nominal {nom}{unit} ({desc})"
    return None


def compute_deterministic_severity(evidence: Dict[str, Any]) -> str:
    """Compute an authoritative process safety severity rating from evidence metrics.

    Returns one of: 'critical', 'high', 'medium', 'low'.
    """
    top_sensors = evidence.get("top_anomalous_sensors", [])
    sensor_names = [str(s.get("display_name", s.get("name", ""))) for s in top_sensors]
    sensor_text = " ".join(sensor_names).lower()

    det_ev = evidence.get("detector_evidence", {}) or {}
    change_type = det_ev.get("change_type", "")
    triggered_by = det_ev.get("triggered_by", "")
    dyn = det_ev.get("dynamic", {})
    cva = (dyn.get("cva") if isinstance(dyn, dict) else None) or {}

    max_z = 0.0
    max_dev = 0.0
    for s in top_sensors:
        try:
            z = abs(float(s.get("z_score", s.get("zScore", 0.0))))
            max_z = max(max_z, z)
        except (ValueError, TypeError):
            pass
        try:
            dev = abs(float(s.get("deviation_percent", 0.0)))
            max_dev = max(max_dev, dev)
        except (ValueError, TypeError):
            pass

    anomaly_score = float(evidence.get("anomaly_score", 0.0))

    # --- 1. CRITICAL RULES ---
    # Valve stiction / persistent mechanical hysteresis in cooling loops (Fault 14, 15)
    is_stiction = (
        change_type == "dynamics" or
        (cva.get("is_anomalous") and ("condenser" in sensor_text or "reactor_cooling" in sensor_text)) or
        (cva.get("slow_drift_ratio") is not None and float(cva.get("slow_drift_ratio", 0)) > 0.45)
    )
    if is_stiction and ("cooling" in sensor_text or "condenser" in sensor_text):
        return "critical"

    # Severe vessel overpressure or thermal runaway excursion
    for s in top_sensors:
        s_name = str(s.get("display_name", s.get("name", ""))).lower()
        z = abs(float(s.get("z_score", s.get("zScore", 0.0))))
        if "reactor_pressure" in s_name and (z >= 3.5 or max_dev >= 12.0):
            return "critical"
        if "reactor_temp" in s_name and (z >= 3.5 or max_dev >= 10.0):
            return "critical"

    # Extreme score ratio
    if anomaly_score >= 3.0:
        return "critical"

    # --- 2. HIGH RULES ---
    # Major disturbances: reactor cooling water step (4, 11), condenser water step (5, 12),
    # A feed loss (6), C header pressure loss (7), valve position stuck (21, 22)
    high_keywords = ["cooling_water", "feed_flow", "c_header", "valve_position", "purgetemp"]
    if any(k in sensor_text for k in high_keywords) and (max_z >= 2.2 or max_dev >= 15.0):
        return "high"
    if anomaly_score >= 1.7 or max_z >= 3.0 or max_dev >= 20.0:
        return "high"

    # --- 3. LOW RULES ---
    # Slow kinetics drift (Fault 13) or mild transient deviations
    if anomaly_score < 1.15 and max_z < 1.8 and max_dev < 6.0:
        return "low"

    # --- 4. MEDIUM (Default operational disturbances contained by controllers) ---
    return "medium"
