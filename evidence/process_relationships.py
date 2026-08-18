"""Known TEP process relationships and sensor/subsystem taxonomies.

This module encodes (a) the human-readable English names of the 52 TEP
variables, (b) which subsystem each variable belongs to, and (c) the *known*
process couplings (e.g. cooling water -> reactor temperature -> reactor
pressure). These are supporting evidence only. The system never fabricates a
causal graph; anything derived statistically is labelled separately by the
evidence builder.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)

# Standard TEP 52-variable English names (XMEAS 1..41, XMV 42..52)
SENSOR_NAMES: Dict[str, str] = {
    "XMEAS_1": "A_Feed_Stream1",
    "XMEAS_2": "D_Feed_Stream2",
    "XMEAS_3": "E_Feed_Stream3",
    "XMEAS_4": "A_C_Feed_Stream4",
    "XMEAS_5": "Recycle_Flow_Stream8",
    "XMEAS_6": "Reactor_Feed_Rate",
    "XMEAS_7": "Reactor_Pressure",
    "XMEAS_8": "Reactor_Level",
    "XMEAS_9": "Reactor_Temperature",
    "XMEAS_10": "Purge_Rate",
    "XMEAS_11": "Separator_Temperature",
    "XMEAS_12": "Separator_Level",
    "XMEAS_13": "Separator_Pressure",
    "XMEAS_14": "Separator_Underflow_Stream10",
    "XMEAS_15": "Stripper_Level",
    "XMEAS_16": "Stripper_Pressure",
    "XMEAS_17": "Stripper_Underflow_Stream11",
    "XMEAS_18": "Stripper_Temperature",
    "XMEAS_19": "Stripper_Steam_Flow",
    "XMEAS_20": "Compressor_Work",
    "XMEAS_21": "Reactor_Cooling_Water_Outlet_Temperature",
    "XMEAS_22": "Condenser_Cooling_Water_Outlet_Temperature",
    "XMEAS_23": "Component_A_Stream6",
    "XMEAS_24": "Component_B_Stream6",
    "XMEAS_25": "Component_C_Stream6",
    "XMEAS_26": "Component_D_Stream6",
    "XMEAS_27": "Component_E_Stream6",
    "XMEAS_28": "Component_F_Stream6",
    "XMEAS_29": "Component_A_Stream9",
    "XMEAS_30": "Component_B_Stream9",
    "XMEAS_31": "Component_C_Stream9",
    "XMEAS_32": "Component_D_Stream9",
    "XMEAS_33": "Component_E_Stream9",
    "XMEAS_34": "Component_F_Stream9",
    "XMEAS_35": "Component_G_Stream9",
    "XMEAS_36": "Component_H_Stream9",
    "XMEAS_37": "Component_D_Stream11",
    "XMEAS_38": "Component_E_Stream11",
    "XMEAS_39": "Component_F_Stream11",
    "XMEAS_40": "Component_G_Stream11",
    "XMEAS_41": "Component_H_Stream11",
    "XMV_42": "D_Feed_Flow",
    "XMV_43": "E_Feed_Flow",
    "XMV_44": "A_Feed_Flow",
    "XMV_45": "A_C_Feed_Flow",
    "XMV_46": "Compressor_Recycle_Valve",
    "XMV_47": "Purge_Valve",
    "XMV_48": "Separator_Pot_Liquid_Flow",
    "XMV_49": "Stripper_Liquid_Product_Flow",
    "XMV_50": "Stripper_Steam_Valve",
    "XMV_51": "Reactor_Cooling_Water_Flow",
    "XMV_52": "Condenser_Cooling_Water_Flow",
}

SUBSYSTEM_GROUPS: Dict[str, List[str]] = {
    "feed_system": [
        "A_Feed_Stream1", "D_Feed_Stream2", "E_Feed_Stream3", "A_C_Feed_Stream4",
        "D_Feed_Flow", "E_Feed_Flow", "A_Feed_Flow", "A_C_Feed_Flow",
        "Component_A_Stream6", "Component_B_Stream6", "Component_C_Stream6",
        "Component_D_Stream6", "Component_E_Stream6", "Component_F_Stream6",
    ],
    "reactor_system": [
        "Reactor_Feed_Rate", "Reactor_Pressure", "Reactor_Level",
        "Reactor_Temperature", "Reactor_Cooling_Water_Outlet_Temperature",
        "Reactor_Cooling_Water_Flow",
    ],
    "reactor_cooling_system": [
        "Reactor_Cooling_Water_Flow", "Reactor_Cooling_Water_Outlet_Temperature",
        "Reactor_Temperature", "Reactor_Pressure",
    ],
    "separation_system": [
        "Separator_Temperature", "Separator_Level", "Separator_Pressure",
        "Separator_Underflow_Stream10", "Separator_Pot_Liquid_Flow",
        "Recycle_Flow_Stream8", "Compressor_Work",
    ],
    "condenser_cooling_system": [
        "Condenser_Cooling_Water_Flow", "Condenser_Cooling_Water_Outlet_Temperature",
        "Separator_Temperature", "Separator_Pressure",
    ],
    "stripper_system": [
        "Stripper_Level", "Stripper_Pressure", "Stripper_Underflow_Stream11",
        "Stripper_Temperature", "Stripper_Steam_Flow", "Stripper_Steam_Valve",
        "Stripper_Liquid_Product_Flow",
    ],
    "purge_compressor_system": [
        "Purge_Rate", "Purge_Valve", "Compressor_Work", "Compressor_Recycle_Valve",
    ],
    "product_composition_system": [
        "Component_A_Stream9", "Component_B_Stream9", "Component_C_Stream9",
        "Component_D_Stream9", "Component_E_Stream9", "Component_F_Stream9",
        "Component_G_Stream9", "Component_H_Stream9",
        "Component_D_Stream11", "Component_E_Stream11", "Component_F_Stream11",
        "Component_G_Stream11", "Component_H_Stream11",
    ],
}

# Known process couplings, expressed as (upstream_effect, downstream_effect).
# These come from the published TEP process description (Downs & Vogel 1993),
# not from the training data.
KNOWN_PROCESS_RELATIONSHIPS: List[Tuple[str, str]] = [
    ("Reactor_Cooling_Water_Flow", "Reactor_Temperature"),
    ("Reactor_Cooling_Water_Outlet_Temperature", "Reactor_Temperature"),
    ("Reactor_Temperature", "Reactor_Pressure"),
    ("Reactor_Temperature", "Component_A_Stream6"),
    ("Reactor_Feed_Rate", "Reactor_Temperature"),
    ("A_C_Feed_Flow", "Reactor_Feed_Rate"),
    ("A_Feed_Flow", "Reactor_Feed_Rate"),
    ("D_Feed_Flow", "Reactor_Feed_Rate"),
    ("E_Feed_Flow", "Reactor_Feed_Rate"),
    ("Reactor_Level", "Reactor_Feed_Rate"),
    ("Reactor_Pressure", "Purge_Rate"),
    ("Separator_Underflow_Stream10", "Recycle_Flow_Stream8"),
    ("Recycle_Flow_Stream8", "Reactor_Feed_Rate"),
    ("Condenser_Cooling_Water_Flow", "Separator_Temperature"),
    ("Condenser_Cooling_Water_Outlet_Temperature", "Separator_Temperature"),
    ("Separator_Temperature", "Separator_Pressure"),
    ("Separator_Pressure", "Compressor_Work"),
    ("Separator_Underflow_Stream10", "Stripper_Level"),
    ("Stripper_Steam_Flow", "Stripper_Temperature"),
    ("Stripper_Temperature", "Stripper_Underflow_Stream11"),
    ("Stripper_Steam_Valve", "Stripper_Steam_Flow"),
]


class RelationshipKind(str, Enum):
    KNOWN_PROCESS = "known_process_relationship"
    STATISTICAL = "statistical_relationship"
    TEMPORAL = "temporal_relationship"
    MODEL_DERIVED = "model_derived_evidence"


def sensor_display_name(raw_name: str) -> str:
    """Map an XMEAS_x / XMV_y column to its English display name."""
    if raw_name in SENSOR_NAMES:
        return SENSOR_NAMES[raw_name]
    return raw_name


def _invert_subsystem_groups() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for subsystem, sensors in SUBSYSTEM_GROUPS.items():
        for sensor in sensors:
            mapping.setdefault(sensor, subsystem)
    return mapping


_SENSOR_TO_SUBSYSTEM: Dict[str, str] = _invert_subsystem_groups()


def subsystem_for_sensor(display_name: str) -> str:
    return _SENSOR_TO_SUBSYSTEM.get(display_name, "unknown")


def suggest_subsystems(
    top_sensors: Sequence[str], top_k: int = 3
) -> List[Dict]:
    """Rank candidate subsystems by how many top sensors they contain.

    The result is a *candidate*, not a verified cause.
    """
    counts: Dict[str, int] = {}
    for sensor in top_sensors:
        subs = subsystem_for_sensor(sensor)
        counts[subs] = counts.get(subs, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        {
            "subsystem": name,
            "matched_sensors": counts[name],
            "sensors": [s for s in top_sensors if subsystem_for_sensor(s) == name],
        }
        for name, _ in ranked[:top_k]
    ]


def known_downstream_effects(sensor_display_name: str) -> List[str]:
    """List variables that are downstream of ``sensor`` per the known graph."""
    return [dst for src, dst in KNOWN_PROCESS_RELATIONSHIPS if src == sensor_display_name]