"""Ground-truth knowledge of the 22 TEP fault scenarios.

Used to build the supervised LLM dataset. For each known fault we encode the
initiating variable, the cascading variables with typical delays, magnitude
ranges, severity, reasoning text, and a recommended action. Unknown/disturbance
faults (16-20) are labelled as such rather than fabricated.

This is *reference* knowledge for supervision; the runtime evidence builder
does not consult this table, so the model cannot "cheat" by looking it up.
"""
from __future__ import annotations

from typing import Dict, List

FAULT_KNOWLEDGE: Dict[int, dict] = {
    1: {
        "id": 1,
        "name": "A/C feed ratio step change",
        "subsystem": "feed_system",
        "initiating": [{"sensor": "A_C_Feed_Flow", "direction": "decrease", "dev_range": (25, 45)}],
        "cascading": [
            {"sensor": "Reactor_Feed_Rate", "direction": "decrease", "delay_min": (2, 5), "dev_range": (8, 20)},
            {"sensor": "Reactor_Pressure", "direction": "decrease", "delay_min": (4, 8), "dev_range": (4, 12)},
            {"sensor": "Component_A_Stream6", "direction": "decrease", "delay_min": (3, 7), "dev_range": (5, 15)},
        ],
        "severity": "medium",
        "reasoning": "The A/C feed ratio step changes the composition delivered to the reactor. "
                     "The A/C feed flow changed first, followed by the reactor feed rate and then "
                     "reactor pressure. This ordering is consistent with a feed-system disturbance.",
        "action": "Inspect the A/C feed ratio controller setpoint and the feed control valves.",
    },
    2: {
        "id": 2,
        "name": "B composition step change",
        "subsystem": "feed_system",
        "initiating": [{"sensor": "Component_B_Stream6", "direction": "increase", "dev_range": (10, 25)}],
        "cascading": [
            {"sensor": "Reactor_Feed_Rate", "direction": "decrease", "delay_min": (2, 6), "dev_range": (5, 15)},
            {"sensor": "Reactor_Pressure", "direction": "decrease", "delay_min": (5, 10), "dev_range": (3, 10)},
        ],
        "severity": "medium",
        "reasoning": "The step in B composition alters the reactor feed chemistry. The B component "
                     "changed first, followed by the reactor feed rate and reactor pressure, "
                     "consistent with a feed-composition disturbance.",
        "action": "Check the B feed supply and composition analyzer.",
    },
    3: {
        "id": 3,
        "name": "D feed temperature step change",
        "subsystem": "reactor_system",
        "initiating": [{"sensor": "D_Feed_Stream2", "direction": "decrease", "dev_range": (10, 30)}],
        "cascading": [
            {"sensor": "Reactor_Temperature", "direction": "decrease", "delay_min": (2, 5), "dev_range": (3, 10)},
            {"sensor": "Reactor_Pressure", "direction": "decrease", "delay_min": (5, 9), "dev_range": (2, 8)},
        ],
        "severity": "medium",
        "reasoning": "The D feed temperature step shifts reactor thermal balance. D feed temperature "
                     "changed first, followed by reactor temperature and pressure, consistent with a "
                     "feed-temperature disturbance.",
        "action": "Verify the D feed preheater and its temperature control.",
    },
    4: {
        "id": 4,
        "name": "Reactor cooling water inlet temperature step",
        "subsystem": "reactor_cooling_system",
        "initiating": [{"sensor": "Reactor_Cooling_Water_Flow", "direction": "increase", "dev_range": (15, 35)}],
        "cascading": [
            {"sensor": "Reactor_Temperature", "direction": "increase", "delay_min": (2, 5), "dev_range": (4, 12)},
            {"sensor": "Reactor_Pressure", "direction": "increase", "delay_min": (4, 8), "dev_range": (3, 10)},
        ],
        "severity": "high",
        "reasoning": "The reactor cooling water inlet temperature step reduces cooling effectiveness; "
                     "the control system increased cooling-water flow first, then reactor temperature "
                     "and pressure rose. This sequence is consistent with a reactor cooling-system abnormality.",
        "action": "Inspect the reactor cooling water supply, exchanger fouling, and inlet temperature control.",
    },
    5: {
        "id": 5,
        "name": "Condenser cooling water inlet temperature step",
        "subsystem": "condenser_cooling_system",
        "initiating": [{"sensor": "Condenser_Cooling_Water_Flow", "direction": "increase", "dev_range": (15, 35)}],
        "cascading": [
            {"sensor": "Separator_Temperature", "direction": "increase", "delay_min": (2, 5), "dev_range": (3, 9)},
            {"sensor": "Separator_Pressure", "direction": "increase", "delay_min": (4, 8), "dev_range": (2, 8)},
        ],
        "severity": "high",
        "reasoning": "The condenser cooling water inlet temperature step reduces condenser duty. The "
                     "condenser cooling-water flow increased first, followed by separator temperature "
                     "and pressure, consistent with a condenser cooling-system abnormality.",
        "action": "Inspect the condenser cooling water supply and heat-exchanger performance.",
    },
    6: {
        "id": 6,
        "name": "A feed loss (stream 1)",
        "subsystem": "feed_system",
        "initiating": [{"sensor": "A_Feed_Flow", "direction": "decrease", "dev_range": (30, 60)}],
        "cascading": [
            {"sensor": "Reactor_Feed_Rate", "direction": "decrease", "delay_min": (1, 4), "dev_range": (10, 25)},
            {"sensor": "Reactor_Pressure", "direction": "decrease", "delay_min": (3, 7), "dev_range": (5, 15)},
        ],
        "severity": "high",
        "reasoning": "A loss of A feed reduces the total reactor feed. The A feed flow dropped first, "
                     "followed by reactor feed rate and pressure, consistent with a feed-supply loss.",
        "action": "Check the A feed supply (stream 1) pumps, valves, and inventory.",
    },
    7: {
        "id": 7,
        "name": "C header pressure loss (reduced availability)",
        "subsystem": "feed_system",
        "initiating": [{"sensor": "A_C_Feed_Flow", "direction": "decrease", "dev_range": (15, 40)}],
        "cascading": [
            {"sensor": "Purge_Rate", "direction": "decrease", "delay_min": (2, 6), "dev_range": (5, 15)},
            {"sensor": "Reactor_Feed_Rate", "direction": "decrease", "delay_min": (3, 7), "dev_range": (6, 16)},
        ],
        "severity": "high",
        "reasoning": "C header pressure loss reduces C availability. The A/C feed flow dropped first, "
                     "followed by purge and reactor feed rate, consistent with a feed-supply pressure loss.",
        "action": "Check the C header pressure, supply compressors, and isolation valves.",
    },
    8: {
        "id": 8,
        "name": "A, B, C feed composition random variation",
        "subsystem": "feed_system",
        "initiating": [{"sensor": "Component_A_Stream6", "direction": "increase", "dev_range": (5, 15)}],
        "cascading": [
            {"sensor": "Component_B_Stream6", "direction": "increase", "delay_min": (1, 4), "dev_range": (4, 12)},
            {"sensor": "Reactor_Feed_Rate", "direction": "decrease", "delay_min": (3, 8), "dev_range": (3, 10)},
        ],
        "severity": "medium",
        "reasoning": "Random variation in the A/B/C feed composition propagates through the reactor "
                     "feed. The composition variables fluctuated first, followed by reactor feed rate, "
                     "consistent with a feed-composition disturbance.",
        "action": "Inspect the feed composition analyzers and upstream blending.",
    },
    9: {
        "id": 9,
        "name": "D feed temperature random variation",
        "subsystem": "reactor_system",
        "initiating": [{"sensor": "D_Feed_Stream2", "direction": "decrease", "dev_range": (8, 20)}],
        "cascading": [
            {"sensor": "Reactor_Temperature", "direction": "decrease", "delay_min": (2, 6), "dev_range": (3, 9)},
        ],
        "severity": "medium",
        "reasoning": "Random variation in D feed temperature causes reactor temperature swings. D feed "
                     "temperature changed first, followed by reactor temperature, consistent with a "
                     "feed-temperature disturbance.",
        "action": "Inspect the D feed preheater controls and temperature sensor.",
    },
    10: {
        "id": 10,
        "name": "C feed temperature random variation",
        "subsystem": "feed_system",
        "initiating": [{"sensor": "Reactor_Temperature", "direction": "increase", "dev_range": (3, 9)}],
        "cascading": [
            {"sensor": "Reactor_Pressure", "direction": "increase", "delay_min": (2, 6), "dev_range": (2, 7)},
        ],
        "severity": "medium",
        "reasoning": "Random variation in C feed temperature disturbs reactor thermal balance. Reactor "
                     "temperature changed first, followed by reactor pressure, consistent with a "
                     "feed-temperature disturbance.",
        "action": "Inspect C feed heat-exchange and temperature controls.",
    },
    11: {
        "id": 11,
        "name": "Reactor cooling water inlet temperature random variation",
        "subsystem": "reactor_cooling_system",
        "initiating": [{"sensor": "Reactor_Cooling_Water_Flow", "direction": "increase", "dev_range": (12, 30)}],
        "cascading": [
            {"sensor": "Reactor_Temperature", "direction": "increase", "delay_min": (2, 5), "dev_range": (3, 8)},
        ],
        "severity": "high",
        "reasoning": "Random variation of the reactor cooling-water inlet temperature causes the "
                     "cooling-water flow to fluctuate first, followed by reactor temperature, "
                     "consistent with a reactor cooling-system disturbance.",
        "action": "Inspect reactor cooling-water inlet temperature stability and exchanger fouling.",
    },
    12: {
        "id": 12,
        "name": "Condenser cooling water inlet temperature random variation",
        "subsystem": "condenser_cooling_system",
        "initiating": [{"sensor": "Condenser_Cooling_Water_Flow", "direction": "increase", "dev_range": (12, 30)}],
        "cascading": [
            {"sensor": "Separator_Temperature", "direction": "increase", "delay_min": (2, 5), "dev_range": (2, 7)},
        ],
        "severity": "high",
        "reasoning": "Random variation of the condenser cooling-water inlet temperature disturbs the "
                     "condenser first, followed by separator temperature, consistent with a condenser "
                     "cooling-system disturbance.",
        "action": "Inspect condenser cooling-water inlet temperature stability and condenser duty.",
    },
    13: {
        "id": 13,
        "name": "Reaction kinetics slow drift",
        "subsystem": "reactor_system",
        "initiating": [{"sensor": "Reactor_Pressure", "direction": "decrease", "dev_range": (3, 10)}],
        "cascading": [
            {"sensor": "Reactor_Temperature", "direction": "decrease", "delay_min": (3, 8), "dev_range": (2, 7)},
            {"sensor": "Component_B_Stream6", "direction": "increase", "delay_min": (5, 12), "dev_range": (2, 6)},
        ],
        "severity": "low",
        "reasoning": "A slow drift in reaction kinetics reduces conversion. Reactor pressure drifted "
                     "first, followed by reactor temperature and unreacted B, consistent with a slow "
                     "reactor-kinetics drift.",
        "action": "Check catalyst/reactant quality; schedule a kinetics review and reactor performance test.",
    },
    14: {
        "id": 14,
        "name": "Reactor cooling water valve sticking",
        "subsystem": "reactor_cooling_system",
        "initiating": [{"sensor": "Reactor_Cooling_Water_Flow", "direction": "decrease", "dev_range": (35, 55)}],
        "cascading": [
            {"sensor": "Reactor_Temperature", "direction": "increase", "delay_min": (2, 6), "dev_range": (15, 35)},
            {"sensor": "Reactor_Pressure", "direction": "increase", "delay_min": (4, 9), "dev_range": (8, 22)},
        ],
        "severity": "critical",
        "reasoning": "The reactor cooling-water valve sticking reduced cooling-water flow first; the "
                     "reactor then heated and pressurized. This temporal sequence makes a reactor "
                     "cooling-system abnormality the most likely initiating event.",
        "action": "Inspect the reactor cooling-water control valve for sticking/stroking, and check "
                  "cooling-water supply pressure. Emergency: monitor reactor temperature closely.",
    },
    15: {
        "id": 15,
        "name": "Condenser cooling water valve sticking",
        "subsystem": "condenser_cooling_system",
        "initiating": [{"sensor": "Condenser_Cooling_Water_Flow", "direction": "decrease", "dev_range": (30, 50)}],
        "cascading": [
            {"sensor": "Separator_Pressure", "direction": "increase", "delay_min": (2, 6), "dev_range": (10, 25)},
            {"sensor": "Separator_Temperature", "direction": "increase", "delay_min": (3, 8), "dev_range": (6, 15)},
        ],
        "severity": "critical",
        "reasoning": "The condenser cooling-water valve sticking reduced condenser duty first; separator "
                     "pressure then temperature rose. This sequence is consistent with a condenser "
                     "cooling-system abnormality.",
        "action": "Inspect the condenser cooling-water control valve for sticking, and check condenser "
                  "duty and separator pressure safety interlocks.",
    },
    16: {
        "id": 16,
        "name": "Unknown disturbance",
        "subsystem": "unknown",
        "initiating": [{"sensor": "Reactor_Pressure", "direction": "increase", "dev_range": (5, 12)}],
        "cascading": [
            {"sensor": "Separator_Pressure", "direction": "increase", "delay_min": (3, 8), "dev_range": (3, 8)},
        ],
        "severity": "medium",
        "reasoning": "The evidence is not sufficient to isolate a single known fault. Reactor and "
                     "separator pressures both increased, but no initiating variable dominates. "
                     "The root cause remains unclassified.",
        "action": "Review historian trends across the reactor and separator; run additional diagnostic "
                  "tests before acting.",
    },
    17: {
        "id": 17,
        "name": "Unknown disturbance",
        "subsystem": "unknown",
        "initiating": [{"sensor": "Reactor_Pressure", "direction": "increase", "dev_range": (5, 12)}],
        "cascading": [
            {"sensor": "Separator_Pressure", "direction": "increase", "delay_min": (3, 8), "dev_range": (3, 8)},
        ],
        "severity": "medium",
        "reasoning": "The evidence is not sufficient to isolate a single known fault. No clear "
                     "initiating variable was detected; the root cause remains unclassified.",
        "action": "Review historian trends and run additional diagnostic tests before acting.",
    },
    18: {
        "id": 18,
        "name": "Unknown disturbance",
        "subsystem": "unknown",
        "initiating": [{"sensor": "Reactor_Temperature", "direction": "increase", "dev_range": (4, 10)}],
        "cascading": [
            {"sensor": "Reactor_Pressure", "direction": "increase", "delay_min": (3, 8), "dev_range": (3, 8)},
        ],
        "severity": "medium",
        "reasoning": "The evidence is not sufficient to isolate a single known fault. Reactor "
                     "temperature changed but no single initiating variable dominates; the root cause "
                     "remains unclassified.",
        "action": "Review historian trends and run additional diagnostic tests before acting.",
    },
    19: {
        "id": 19,
        "name": "Unknown disturbance",
        "subsystem": "unknown",
        "initiating": [{"sensor": "Separator_Pressure", "direction": "increase", "dev_range": (4, 10)}],
        "cascading": [
            {"sensor": "Separator_Temperature", "direction": "increase", "delay_min": (3, 8), "dev_range": (2, 7)},
        ],
        "severity": "medium",
        "reasoning": "The evidence is not sufficient to isolate a single known fault. Separator-side "
                     "variables changed without a clear initiating variable; the root cause remains "
                     "unclassified.",
        "action": "Review separator/condenser trends and run additional diagnostic tests before acting.",
    },
    20: {
        "id": 20,
        "name": "Unknown disturbance",
        "subsystem": "unknown",
        "initiating": [{"sensor": "Component_B_Stream6", "direction": "increase", "dev_range": (4, 10)}],
        "cascading": [
            {"sensor": "Reactor_Pressure", "direction": "increase", "delay_min": (3, 8), "dev_range": (3, 8)},
        ],
        "severity": "medium",
        "reasoning": "The evidence is not sufficient to isolate a single known fault. Feed composition "
                     "changed but the initiating variable is unclear; the root cause remains unclassified.",
        "action": "Review feed and reactor trends and run additional diagnostic tests before acting.",
    },
    21: {
        "id": 21,
        "name": "Valve position stuck (stream 4)",
        "subsystem": "feed_system",
        "initiating": [{"sensor": "A_C_Feed_Flow", "direction": "decrease", "dev_range": (20, 40)}],
        "cascading": [
            {"sensor": "Reactor_Feed_Rate", "direction": "decrease", "delay_min": (2, 6), "dev_range": (8, 18)},
            {"sensor": "Reactor_Pressure", "direction": "decrease", "delay_min": (4, 9), "dev_range": (4, 12)},
        ],
        "severity": "high",
        "reasoning": "A valve in stream 4 stuck, reducing the A/C feed flow first, followed by reactor "
                     "feed rate and pressure. This sequence is consistent with a feed-valve fault.",
        "action": "Inspect the stream-4 feed valve position, actuator, and positioner.",
    },
    22: {
        "id": 22,
        "name": "Valve position stuck",
        "subsystem": "feed_system",
        "initiating": [{"sensor": "D_Feed_Flow", "direction": "decrease", "dev_range": (15, 35)}],
        "cascading": [
            {"sensor": "Reactor_Feed_Rate", "direction": "decrease", "delay_min": (2, 6), "dev_range": (6, 15)},
            {"sensor": "Reactor_Pressure", "direction": "decrease", "delay_min": (4, 9), "dev_range": (3, 10)},
        ],
        "severity": "high",
        "reasoning": "A valve stuck in the feed path reduced D feed flow first, followed by reactor "
                     "feed rate and pressure. This sequence is consistent with a feed-valve fault.",
        "action": "Inspect the D feed valve position, actuator, and positioner.",
    },
}


def knowledge_for_fault(fault_id: int) -> dict:
    if fault_id not in FAULT_KNOWLEDGE:
        raise KeyError(f"Fault {fault_id} not present in knowledge base.")
    return FAULT_KNOWLEDGE[fault_id]


def all_fault_ids() -> List[int]:
    return sorted(FAULT_KNOWLEDGE.keys())