"""Comprehensive ground-truth knowledge base for all 22 TEP fault scenarios.

Encodes deep industrial domain knowledge:
- Specific physical equipment (R-101, E-102, V-102, C-101, K-101, Streams 1..11)
- Specific manipulated variables and actuators (XMV_42..XMV_52)
- Specific sensor tags (XMEAS_1..XMEAS_41)
- Thermodynamic and chemical propagation mechanisms
- Differential diagnostic reasoning to distinguish lookalike faults
- Concrete control room and field technician actions with safety limits

Used by the distillation pipeline to synthesize realistic, non-generic,
sensor-grounded reasoning and QA pairs for student LLM training.
"""
from __future__ import annotations

from typing import Any, Dict, List

FAULT_KNOWLEDGE: Dict[int, Dict[str, Any]] = {
    1: {
        "id": 1,
        "name": "A/C feed ratio step change",
        "subsystem": "feed_system",
        "equipment": ["Stream 4 Feed Header", "R-101 Continuous Stirred Tank Reactor"],
        "valves": ["XMV_45 (A and C Feed Flow valve / XMV 4)"],
        "initiating": [{"sensor": "A_C_Feed_Flow", "tag": "XMV_45 / XMEAS_4", "direction": "decrease", "dev_range": (25, 45)}],
        "cascading": [
            {"sensor": "Reactor_Feed_Rate", "tag": "XMEAS_6", "direction": "decrease", "delay_min": (2, 5), "dev_range": (8, 20)},
            {"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "decrease", "delay_min": (4, 8), "dev_range": (4, 12)},
            {"sensor": "Component_A_Stream6", "tag": "XMEAS_23", "direction": "decrease", "delay_min": (3, 7), "dev_range": (5, 15)},
        ],
        "severity": "medium",
        "physical_mechanism": (
            "A step disturbance in the A/C feed ratio controller alters the stoichiometric proportion of "
            "reactants delivered through Stream 4. As A_C_Feed_Flow drops, the total molar feed rate to R-101 "
            "decreases (Reactor_Feed_Rate drops from nominal 45.0 kscfm to ~38 kscfm). With reduced reactant inventory, "
            "the gas-phase consumption rate drops and Reactor_Pressure declines from 2705 kPa toward 2550 kPa."
        ),
        "differential_diagnosis": (
            "Distinguished from Fault 6 (A feed loss on Stream 1) because Stream 1 flow (A_Feed_Flow) is completely "
            "normal; the disturbance is isolated strictly to Stream 4. Distinguished from Fault 7 (C header pressure loss) "
            "because the Stream 4 valve (XMV_45) is not saturated at 100% open, indicating a ratio/setpoint shift "
            "rather than total supply starvation."
        ),
        "recommended_action": (
            "Inspect the Stream 4 A/C ratio controller setpoint in the DCS. Stroke control valve XMV_45 to confirm "
            "responsive modulation, and verify Stream 4 mass flow meter calibration."
        ),
        "reasoning": (
            "A/C feed ratio disturbance on Stream 4 altered reactor stoichiometry. Initiated at A_C_Feed_Flow at t=0.0 min, "
            "propagating to Reactor_Feed_Rate and Reactor_Pressure."
        ),
        "action": "Inspect A/C feed ratio controller setpoint and calibrate Stream 4 flow control valve XMV_45.",
    },
    2: {
        "id": 2,
        "name": "B composition step change",
        "subsystem": "feed_system",
        "equipment": ["Stream 4 Feed Supply", "R-101 Reactor", "Stream 9 Purge Header"],
        "valves": ["XMV_47 (Purge valve / XMV 6)"],
        "initiating": [{"sensor": "Component_B_Stream6", "tag": "XMEAS_24", "direction": "increase", "dev_range": (12, 28)}],
        "cascading": [
            {"sensor": "Purge_Rate", "tag": "XMEAS_10", "direction": "increase", "delay_min": (2, 5), "dev_range": (10, 25)},
            {"sensor": "Reactor_Feed_Rate", "tag": "XMEAS_6", "direction": "decrease", "delay_min": (3, 7), "dev_range": (4, 12)},
            {"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "decrease", "delay_min": (5, 10), "dev_range": (3, 9)},
        ],
        "severity": "medium",
        "physical_mechanism": (
            "An upstream feed supplier step change increases inert component B concentration in Stream 4. "
            "Component B does not react in R-101 and accumulates in the recycle loop. To prevent inert buildup and "
            "maintain total pressure, the purge controller opens valve XMV_47, causing Purge_Rate to increase. "
            "The dilution of active reactants slightly slows reaction kinetics, lowering Reactor_Pressure."
        ),
        "differential_diagnosis": (
            "Differentiated from Fault 1 by analyzer data: Component_B_Stream6 (XMEAS_24) exhibits a clear upward step "
            "and Purge_Rate increases, whereas Fault 1 shows pure A/C reactant shifts without abnormal B accumulation."
        ),
        "recommended_action": (
            "Review upstream raw material delivery certificates for Stream 4 feed tanks. Adjust purge flow setpoint "
            "to stabilize recycle gas purity and inspect online gas chromatograph XMEAS_24."
        ),
        "reasoning": (
            "Inert component B composition step in Stream 4 accumulated in the recycle loop, forcing higher purge rates "
            "and diluting reactor active reactants."
        ),
        "action": "Check Stream 4 feed supplier purity, monitor gas chromatograph XMEAS_24, and verify purge valve XMV_47.",
    },
    3: {
        "id": 3,
        "name": "D feed temperature step change",
        "subsystem": "reactor_system",
        "equipment": ["Stream 2 D Preheater", "R-101 Reactor"],
        "valves": ["XMV_42 (D Feed Flow valve / XMV 1)"],
        "initiating": [{"sensor": "D_Feed_Stream2", "tag": "XMEAS_2", "direction": "decrease", "dev_range": (12, 32)}],
        "cascading": [
            {"sensor": "Reactor_Temperature", "tag": "XMEAS_9", "direction": "decrease", "delay_min": (2, 5), "dev_range": (3, 10)},
            {"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "decrease", "delay_min": (4, 8), "dev_range": (2, 8)},
        ],
        "severity": "medium",
        "physical_mechanism": (
            "A sudden temperature drop in the Stream 2 feed header introduces subcooled liquid D into R-101. "
            "The sensible heat deficit quenches the reactor liquid inventory, pulling Reactor_Temperature down "
            "from 120.4 °C to ~116 °C. Because the reaction rate follows Arrhenius kinetics (k = A*exp(-E/RT)), "
            "the cooling reduces reaction rate, lowering gas consumption and reactor pressure."
        ),
        "differential_diagnosis": (
            "Distinguished from cooling water faults (Faults 4, 11, 14) because cooling water outlet temperatures "
            "(XMEAS_21, XMEAS_22) are steady; the disturbance enters exclusively through feed Stream 2 thermowell."
        ),
        "recommended_action": (
            "Inspect the steam supply and thermostatic control valve on the Stream 2 feed preheater. "
            "Verify the temperature transmitter calibration on Stream 2."
        ),
        "reasoning": (
            "Subcooled D feed in Stream 2 quenched reactor thermal balance, reducing Arrhenius reaction rates and "
            "depressing reactor temperature and pressure."
        ),
        "action": "Inspect Stream 2 preheater steam supply, temperature controller, and associated steam trap.",
    },
    4: {
        "id": 4,
        "name": "Reactor cooling water inlet temperature step",
        "subsystem": "reactor_cooling_system",
        "equipment": ["R-101 Reactor Cooling Jacket", "Plant Cooling Tower Utility"],
        "valves": ["XMV_51 (Reactor Cooling Water Flow valve / XMV 10)"],
        "initiating": [{"sensor": "Reactor_Cooling_Water_Flow", "tag": "XMV_51 / XMEAS_21", "direction": "increase", "dev_range": (18, 38)}],
        "cascading": [
            {"sensor": "Reactor_Cooling_Water_Outlet_Temperature", "tag": "XMEAS_21", "direction": "increase", "delay_min": (1, 4), "dev_range": (8, 20)},
            {"sensor": "Reactor_Temperature", "tag": "XMEAS_9", "direction": "increase", "delay_min": (2, 6), "dev_range": (4, 12)},
            {"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "increase", "delay_min": (4, 9), "dev_range": (3, 10)},
        ],
        "severity": "high",
        "physical_mechanism": (
            "The supply temperature of cooling water to the reactor jacket steps up (+3 to +5 °C) due to cooling tower upset. "
            "With degraded temperature difference (LMTD), the jacket heat removal capacity drops. The reactor temperature "
            "controller aggressively opens valve XMV_51 (+25% flow) to compensate, but heat duty remains constrained, "
            "causing Reactor_Temperature to rise from 120.4 °C toward 124 °C and increasing reactor vapor pressure."
        ),
        "differential_diagnosis": (
            "CRITICAL DIFFERENTIAL VS FAULT 14: In Fault 4, the cooling water control valve XMV_51 is fully functional and "
            "DRIVES OPEN (+20-38% flow) trying to reject heat. In Fault 14 (valve sticking), valve XMV_51 is mechanically "
            "STUCK and flow DROPS or freezes, causing unmitigated thermal runaway."
        ),
        "recommended_action": (
            "Verify cooling tower supply water temperature and header pressure. If cooling water flow saturates near 100% "
            "on valve XMV_51, reduce reactor feed rates to prevent exceeding the 125 °C reactor thermal trip limit."
        ),
        "reasoning": (
            "Elevated cooling water supply temperature degraded reactor jacket LMTD. Reactor cooling water flow increased "
            "first to compensate, followed by rising reactor temperature and pressure."
        ),
        "action": "Check cooling tower supply temperature, clean jacket heat exchanger, and monitor reactor temperature margins.",
    },
    5: {
        "id": 5,
        "name": "Condenser cooling water inlet temperature step",
        "subsystem": "condenser_cooling_system",
        "equipment": ["E-102 Overhead Condenser", "V-102 Vapor-Liquid Separator"],
        "valves": ["XMV_52 (Condenser Cooling Water Flow valve / XMV 11)"],
        "initiating": [{"sensor": "Condenser_Cooling_Water_Flow", "tag": "XMV_52 / XMEAS_22", "direction": "increase", "dev_range": (18, 38)}],
        "cascading": [
            {"sensor": "Condenser_Cooling_Water_Outlet_Temperature", "tag": "XMEAS_22", "direction": "increase", "delay_min": (1, 4), "dev_range": (8, 22)},
            {"sensor": "Separator_Temperature", "tag": "XMEAS_11", "direction": "increase", "delay_min": (2, 5), "dev_range": (4, 12)},
            {"sensor": "Separator_Pressure", "tag": "XMEAS_13", "direction": "increase", "delay_min": (4, 8), "dev_range": (3, 10)},
        ],
        "severity": "high",
        "physical_mechanism": (
            "Warmer cooling water entering E-102 overhead condenser reduces condensation of volatile components from reactor "
            "overhead vapor. The condenser flow controller opens valve XMV_52 (+20-38%) to sustain heat removal. Despite "
            "maximum flow, uncondensed vapor enthalpy carries over into Separator V-102, causing Separator_Temperature "
            "to climb from 80.1 °C to 86 °C and Separator_Pressure to increase from 2633 kPa toward 2800 kPa."
        ),
        "differential_diagnosis": (
            "CRITICAL DIFFERENTIAL VS FAULT 15: In Fault 5, the condenser cooling water valve XMV_52 is fully active and "
            "OPENS (+25% flow) to reject heat. In Fault 15 (condenser valve sticking), valve XMV_52 is physically FROZEN or "
            "restricted, causing cooling water flow to DROP by 30-50% while separator pressure spikes."
        ),
        "recommended_action": (
            "Inspect condenser cooling water supply header temperature. Verify E-102 tube-side fouling and prepare to "
            "increase purge venting if separator pressure approaches the 2850 kPa alarm threshold."
        ),
        "reasoning": (
            "Condenser cooling water inlet temperature step degraded E-102 condensation duty. Cooling water flow increased "
            "first, followed by elevated separator temperature and pressure."
        ),
        "action": "Check condenser cooling water supply temperature, backflush E-102 tubes, and verify valve XMV_52 authority.",
    },
    6: {
        "id": 6,
        "name": "A feed loss (stream 1)",
        "subsystem": "feed_system",
        "equipment": ["Stream 1 Supply Pump P-101", "Stream 1 Feed Line", "R-101 Reactor"],
        "valves": ["XMV_44 (A Feed Flow valve / XMV 3)"],
        "initiating": [{"sensor": "A_Feed_Flow", "tag": "XMV_44 / XMEAS_1", "direction": "decrease", "dev_range": (45, 80)}],
        "cascading": [
            {"sensor": "Reactor_Feed_Rate", "tag": "XMEAS_6", "direction": "decrease", "delay_min": (1, 4), "dev_range": (12, 28)},
            {"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "decrease", "delay_min": (3, 7), "dev_range": (6, 18)},
            {"sensor": "Component_A_Stream6", "tag": "XMEAS_23", "direction": "decrease", "delay_min": (2, 6), "dev_range": (15, 35)},
        ],
        "severity": "high",
        "physical_mechanism": (
            "A sudden loss of Stream 1 supply (pump trip, suction strainer blockage, or upstream valve closure) causes "
            "A_Feed_Flow to plunge by >50-80%. Reactant A is an essential precursor for both product reactions "
            "(A+C+D->G and A+C+E->H). Starved of reactant A, reactor feed rate collapses, stoichiometric conversion stops, "
            "and unreacted D and E accumulate while Reactor_Pressure drops sharply."
        ),
        "differential_diagnosis": (
            "Distinguished from Fault 1 (A/C ratio step) because Stream 1 flow collapses catastrophically rather than "
            "undergoing a mild setpoint adjustment on Stream 4. Distinguished from Fault 7 because Stream 4 flow is steady; "
            "the failure is strictly isolated to Stream 1 pump/valve train."
        ),
        "recommended_action": (
            "Immediately inspect Stream 1 feed pump status, check suction strainer differential pressure, and verify "
            "upstream raw material tank level. If pump tripped, restart standby pump P-101B."
        ),
        "reasoning": (
            "Complete or major loss of Stream 1 feed flow starved reactant A from the reactor. Initiated at A_Feed_Flow "
            "at t=0.0 min, causing sharp collapse in reactor feed rate and reactor pressure."
        ),
        "action": "Check Stream 1 feed pump P-101, inspect suction strainers, and restart standby feed pump immediately.",
    },
    7: {
        "id": 7,
        "name": "C header pressure loss (reduced availability)",
        "subsystem": "feed_system",
        "equipment": ["Stream 4 C Supply Header", "Compressor C-Header", "R-101 Reactor"],
        "valves": ["XMV_45 (A and C Feed Flow valve / XMV 4)"],
        "initiating": [{"sensor": "A_C_Feed_Flow", "tag": "XMV_45 / XMEAS_4", "direction": "decrease", "dev_range": (20, 45)}],
        "cascading": [
            {"sensor": "Purge_Rate", "tag": "XMEAS_10", "direction": "decrease", "delay_min": (2, 6), "dev_range": (8, 20)},
            {"sensor": "Reactor_Feed_Rate", "tag": "XMEAS_6", "direction": "decrease", "delay_min": (3, 7), "dev_range": (6, 16)},
            {"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "decrease", "delay_min": (4, 9), "dev_range": (4, 14)},
        ],
        "severity": "high",
        "physical_mechanism": (
            "Loss of upstream C header supply pressure reduces available differential pressure across control valve XMV_45. "
            "Despite the valve opening to 100% saturation, actual A_C_Feed_Flow drops significantly. The deficiency of "
            "gaseous C slows the formation of products G and H, reducing recycle gas generation and pulling down purge flow "
            "and reactor pressure."
        ),
        "differential_diagnosis": (
            "Distinguished from Fault 1 (ratio change) and Fault 21 (valve stuck) by valve actuator behavior: in Fault 7, "
            "valve controller output XMV_45 is driven to 100% full open attempting to fight the loss of header pressure, "
            "whereas in Fault 1 the valve is in mid-range control, and in Fault 21 the stem is mechanically jammed."
        ),
        "recommended_action": (
            "Contact utility plant operators to verify C header supply compressors and header pressure regulators. "
            "Inspect upstream C battery limit isolation valves."
        ),
        "reasoning": (
            "Upstream C supply header pressure loss starved Stream 4 delivery. A_C_Feed_Flow dropped first while valve "
            "XMV_45 saturated open, followed by declining purge rate and reactor pressure."
        ),
        "action": "Check C header supply pressure, verify compressor station, and inspect valve XMV_45 opening percentage.",
    },
    8: {
        "id": 8,
        "name": "A, B, C feed composition random variation",
        "subsystem": "feed_system",
        "equipment": ["Stream 4 Feed Mixer", "R-101 Reactor"],
        "valves": ["XMV_45 (A and C Feed Flow valve / XMV 4)"],
        "initiating": [{"sensor": "Component_A_Stream6", "tag": "XMEAS_23", "direction": "increase", "dev_range": (6, 18)}],
        "cascading": [
            {"sensor": "Component_B_Stream6", "tag": "XMEAS_24", "direction": "increase", "delay_min": (1, 4), "dev_range": (5, 14)},
            {"sensor": "Reactor_Feed_Rate", "tag": "XMEAS_6", "direction": "decrease", "delay_min": (3, 8), "dev_range": (3, 10)},
        ],
        "severity": "medium",
        "physical_mechanism": (
            "Stochastic composition disturbances in upstream feed blending cause simultaneous, erratic fluctuations in "
            "components A, B, and C entering reactor R-101 via Stream 6. The varying reactant concentrations cause "
            "transient kinetic perturbations in reaction conversion without a sustained directional step."
        ),
        "differential_diagnosis": (
            "Distinguished from Fault 1 and Fault 2 by variance characteristics: multi-component analyzers (XMEAS_23, "
            "XMEAS_24, XMEAS_25) display continuous stochastic drift and high variance rather than a clean step shift."
        ),
        "recommended_action": (
            "Inspect upstream raw feed storage tanks for stratification or inadequate mixing. Check online composition "
            "analyzer calibration."
        ),
        "reasoning": (
            "Random composition variations in A/B/C feed streams caused stochastic fluctuations in reactor feed chemistry "
            "and reaction dynamics."
        ),
        "action": "Inspect upstream feed tank agitation, check feed blending controls, and calibrate Stream 6 analyzers.",
    },
    9: {
        "id": 9,
        "name": "D feed temperature random variation",
        "subsystem": "reactor_system",
        "equipment": ["Stream 2 D Preheater", "R-101 Reactor"],
        "valves": ["XMV_42 (D Feed Flow valve / XMV 1)"],
        "initiating": [{"sensor": "D_Feed_Stream2", "tag": "XMEAS_2", "direction": "decrease", "dev_range": (8, 22)}],
        "cascading": [
            {"sensor": "Reactor_Temperature", "tag": "XMEAS_9", "direction": "decrease", "delay_min": (2, 6), "dev_range": (3, 9)},
        ],
        "severity": "medium",
        "physical_mechanism": (
            "Thermal hunting in the Stream 2 preheater temperature control loop induces continuous fluctuations in the "
            "temperature of feed D. This thermal noise enters R-101, producing correlated swings in Reactor_Temperature."
        ),
        "differential_diagnosis": (
            "Distinguished from Fault 3 (step change) by signal pattern: temperature exhibits periodic or random oscillations "
            "around nominal rather than a sharp sustained downward step."
        ),
        "recommended_action": (
            "Tune PID parameters on the Stream 2 preheater temperature controller to eliminate loop hunting. Inspect steam "
            "condensate trap for intermittent cycling."
        ),
        "reasoning": (
            "Thermal hunting on Stream 2 preheater caused random temperature variations in D feed, inducing reactor thermal fluctuations."
        ),
        "action": "Retune Stream 2 preheater temperature controller and check steam trap operation.",
    },
    10: {
        "id": 10,
        "name": "C feed temperature random variation",
        "subsystem": "feed_system",
        "equipment": ["Stream 4 Heat Exchanger", "R-101 Reactor"],
        "valves": ["XMV_45 (A and C Feed Flow valve / XMV 4)"],
        "initiating": [{"sensor": "Reactor_Temperature", "tag": "XMEAS_9", "direction": "increase", "dev_range": (4, 11)}],
        "cascading": [
            {"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "increase", "delay_min": (2, 6), "dev_range": (2, 8)},
        ],
        "severity": "medium",
        "physical_mechanism": (
            "Random temperature fluctuations in feed stream C introduce thermal perturbations into the reactor feed mixture, "
            "causing minor swings in reactor reaction rate and equilibrium vapor pressure."
        ),
        "differential_diagnosis": (
            "Distinguished from reactor jacket cooling faults (Faults 4, 11) because reactor cooling water flow and outlet "
            "temperature do not show initiating lead; the excursion is driven from feed stream thermal noise."
        ),
        "recommended_action": (
            "Inspect Stream 4 heat exchanger temperature control loop and upstream ambient cooling conditions."
        ),
        "reasoning": (
            "Thermal noise in C feed stream produced modest transient fluctuations in reactor temperature and pressure."
        ),
        "action": "Check Stream 4 feed heat exchanger controls and temperature transmitter.",
    },
    11: {
        "id": 11,
        "name": "Reactor cooling water inlet temperature random variation",
        "subsystem": "reactor_cooling_system",
        "equipment": ["R-101 Reactor Jacket", "Cooling Tower Utility"],
        "valves": ["XMV_51 (Reactor Cooling Water Flow valve / XMV 10)"],
        "initiating": [{"sensor": "Reactor_Cooling_Water_Flow", "tag": "XMV_51 / XMEAS_21", "direction": "increase", "dev_range": (14, 32)}],
        "cascading": [
            {"sensor": "Reactor_Temperature", "tag": "XMEAS_9", "direction": "increase", "delay_min": (2, 5), "dev_range": (3, 8)},
        ],
        "severity": "high",
        "physical_mechanism": (
            "Cooling tower fan cycling or utility supply pressure hunting causes random fluctuations in reactor cooling water "
            "inlet temperature. The jacket flow controller modulates valve XMV_51 continuously to fight the thermal noise, "
            "producing persistent fluctuations in reactor temperature."
        ),
        "differential_diagnosis": (
            "Distinguished from Fault 4 (step change) by continuous oscillatory behavior, and from Fault 14 (stiction) because "
            "valve XMV_51 is actively tracking and modulating rather than frozen at a fixed position."
        ),
        "recommended_action": (
            "Inspect cooling tower water supply header pressure stability and cooling tower fan sequencing."
        ),
        "reasoning": (
            "Cooling tower utility fluctuations caused random variations in reactor cooling water inlet temperature, "
            "inducing continuous modulation in valve XMV_51."
        ),
        "action": "Stabilize cooling tower supply header and inspect reactor jacket temperature controller tuning.",
    },
    12: {
        "id": 12,
        "name": "Condenser cooling water inlet temperature random variation",
        "subsystem": "condenser_cooling_system",
        "equipment": ["E-102 Overhead Condenser", "V-102 Separator"],
        "valves": ["XMV_52 (Condenser Cooling Water Flow valve / XMV 11)"],
        "initiating": [{"sensor": "Condenser_Cooling_Water_Flow", "tag": "XMV_52 / XMEAS_22", "direction": "increase", "dev_range": (14, 32)}],
        "cascading": [
            {"sensor": "Separator_Temperature", "tag": "XMEAS_11", "direction": "increase", "delay_min": (2, 5), "dev_range": (2, 7)},
        ],
        "severity": "high",
        "physical_mechanism": (
            "Random temperature fluctuations in cooling water entering condenser E-102 induce continuous modulation in "
            "cooling water flow valve XMV_52, creating minor thermal and pressure oscillations in Separator V-102."
        ),
        "differential_diagnosis": (
            "Distinguished from Fault 5 (step change) by lack of persistent step shift, and from Fault 15 (valve sticking) "
            "because valve XMV_52 is moving freely rather than stuck in place."
        ),
        "recommended_action": (
            "Review condenser cooling water supply header temperature and check E-102 temperature control loop."
        ),
        "reasoning": (
            "Random variations in condenser cooling water supply temperature caused continuous hunting in valve XMV_52 and "
            "separator temperature."
        ),
        "action": "Inspect cooling tower supply water temperature stability and check E-102 valve operation.",
    },
    13: {
        "id": 13,
        "name": "Reaction kinetics slow drift",
        "subsystem": "reactor_system",
        "equipment": ["R-101 Catalyst Bed"],
        "valves": ["XMV_47 (Purge valve / XMV 6)"],
        # Direction is data-derived, not copied from the TEP documentation: the
        # stored trend for XMEAS_7 is increasing in 102 of 102 Protocol A train
        # events where it is ranked (the deviation sign is mixed, 61 positive /
        # 41 negative, so the level starts below baseline and climbs). Recorded as
        # an owner-approved KB correction in docs/PROTOCOL.md; the prose below
        # still describes the older "drift downwards" reading and is left as-is
        # because it is not what the menu line or the filter reads.
        "initiating": [{"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "increase", "dev_range": (3, 10)}],
        "cascading": [
            {"sensor": "Reactor_Temperature", "tag": "XMEAS_9", "direction": "decrease", "delay_min": (3, 8), "dev_range": (2, 7)},
            {"sensor": "Component_B_Stream6", "tag": "XMEAS_24", "direction": "increase", "delay_min": (5, 12), "dev_range": (2, 6)},
        ],
        "severity": "low",
        "physical_mechanism": (
            "Slow catalyst aging, active site poisoning, or gradual fouling inside R-101 causes a slow decrease in the "
            "reaction rate constants. Over many hours, conversion decreases, causing Reactor_Pressure and reaction heat release "
            "to slowly drift downwards while unreacted reactants accumulate."
        ),
        "differential_diagnosis": (
            "Distinguished from all other faults by very slow time constant: detector CVA flags slow drift dynamics "
            "(high slow_drift_ratio > 0.6) with low instantaneous anomaly score, with no sudden valve or step triggers."
        ),
        "recommended_action": (
            "Review long-term reactor conversion trends, check catalyst age logs, and plan routine catalyst regeneration "
            "or replacement during next scheduled turnaround."
        ),
        "reasoning": (
            "Gradual catalyst deactivation caused slow downward drift in reactor reaction kinetics and operating pressure."
        ),
        "action": "Review long-term reactor conversion metrics and schedule catalyst performance evaluation.",
    },
    14: {
        "id": 14,
        "name": "Reactor cooling water valve sticking",
        "subsystem": "reactor_cooling_system",
        "equipment": ["R-101 Reactor Jacket", "Valve Actuator XMV_51 (XMV 10)"],
        "valves": ["XMV_51 (Reactor Cooling Water Flow valve / XMV 10)"],
        "initiating": [{"sensor": "Reactor_Cooling_Water_Flow", "tag": "XMV_51 / XMEAS_21", "direction": "decrease", "dev_range": (35, 60)}],
        "cascading": [
            {"sensor": "Reactor_Temperature", "tag": "XMEAS_9", "direction": "increase", "delay_min": (1, 4), "dev_range": (18, 40)},
            {"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "increase", "delay_min": (3, 7), "dev_range": (10, 26)},
        ],
        "severity": "critical",
        "physical_mechanism": (
            "Mechanical stiction or actuator diaphragm failure causes cooling water valve XMV_51 (XMV 10) to stick at a "
            "restricted position. Cooling water flow suddenly drops by 35-60% (from nominal 38.2 m3/h down to <20 m3/h). "
            "Because the reaction is highly exothermic, heat generation vastly exceeds remaining jacket heat removal. "
            "Reactor_Temperature escalates rapidly from 120.4 °C toward the 129 °C emergency trip limit, with "
            "Reactor_Pressure spiking toward 2950 kPa. Extreme risk of thermal runaway and emergency relief venting!"
        ),
        "differential_diagnosis": (
            "CRITICAL DIFFERENTIAL VS FAULT 4: In Fault 4 (inlet temp step), valve XMV_51 is fully operational and OPENS WIDE "
            "(+25-38% flow) trying to reject heat. In Fault 14, valve XMV_51 is physically STUCK, causing flow to COLLAPSE "
            "(-35-60%) while reactor temperature surges upward uncontrollably."
        ),
        "recommended_action": (
            "CRITICAL EMERGENCY: Dispatch field technician immediately to manually stroke and unstick valve XMV_51 or open the "
            "manual jacket bypass valve. If Reactor_Temperature exceeds 125 °C, initiate emergency cutback of reactant feeds "
            "Stream 1, 2, and 4 to prevent reactor trip."
        ),
        "reasoning": (
            "Mechanical valve stiction on XMV_51 choked reactor jacket cooling water flow. Heat removal collapsed first, "
            "causing rapid escalation in reactor temperature and pressure toward emergency safety envelopes."
        ),
        "action": "EMERGENCY: Dispatch technician to stroke valve XMV_51, open jacket bypass, and cut feeds if temperature > 125 °C.",
    },
    15: {
        "id": 15,
        "name": "Condenser cooling water valve sticking",
        "subsystem": "condenser_cooling_system",
        "equipment": ["E-102 Overhead Condenser", "Valve Actuator XMV_52 (XMV 11)", "V-102 Separator"],
        "valves": ["XMV_52 (Condenser Cooling Water Flow valve / XMV 11)"],
        "initiating": [{"sensor": "Condenser_Cooling_Water_Flow", "tag": "XMV_52 / XMEAS_22", "direction": "decrease", "dev_range": (30, 55)}],
        "cascading": [
            {"sensor": "Separator_Pressure", "tag": "XMEAS_13", "direction": "increase", "delay_min": (2, 5), "dev_range": (12, 30)},
            {"sensor": "Separator_Temperature", "tag": "XMEAS_11", "direction": "increase", "delay_min": (3, 7), "dev_range": (8, 20)},
        ],
        "severity": "critical",
        "physical_mechanism": (
            "Actuator mechanical stiction on valve XMV_52 (XMV 11) freezes the coolant flow to condenser E-102 at a restricted "
            "stroke, reducing flow by 30-55% (from nominal 15.0 m3/h down to <8 m3/h). With cooling water starved, condensation "
            "duty collapses. Hot vapor from R-101 passes through E-102 largely uncondensed, carrying vast gas volumes and sensible "
            "heat directly into Separator V-102. Separator_Pressure surges from 2633 kPa toward 2900 kPa, and Separator_Temperature "
            "climbs from 80.1 °C to >92 °C."
        ),
        "differential_diagnosis": (
            "CRITICAL DIFFERENTIAL VS FAULT 5: In Fault 5 (inlet water temp step), valve XMV_52 is functional and DRIVES OPEN "
            "(+25-38% flow) trying to condense vapor. In Fault 15, valve XMV_52 is physically STUCK and flow DROPS (-30-55%), "
            "confirming mechanical actuator failure rather than a utility temperature disturbance."
        ),
        "recommended_action": (
            "CRITICAL SAFETY INTERLOCK: Dispatch operator to inspect valve XMV_52 positioner and pneumatic supply. "
            "Manually open condenser cooling water bypass valve. Prepare to vent purge gas via valve XMV_47 if Separator_Pressure "
            "approaches relief valve threshold (2900 kPa)."
        ),
        "reasoning": (
            "Mechanical stiction on condenser cooling water valve XMV_52 choked E-102 cooling duty. Flow collapsed first, "
            "causing rapid accumulation of uncondensed vapor and steep pressure/temperature spikes in Separator V-102."
        ),
        "action": "CRITICAL: Stroke stuck valve XMV_52, open manual cooling bypass, and monitor Separator V-102 pressure interlocks.",
    },
    16: {
        "id": 16,
        "name": "Unknown disturbance in separation/reactor system",
        "subsystem": "separation_system",
        "equipment": ["V-102 Separator", "R-101 Reactor", "K-101 Recycle Compressor"],
        "valves": ["XMV_46 (Compressor Recycle Valve)", "XMV_47 (Purge Valve)"],
        "initiating": [{"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "increase", "dev_range": (5, 14)}],
        "cascading": [
            {"sensor": "Separator_Pressure", "tag": "XMEAS_13", "direction": "increase", "delay_min": (2, 6), "dev_range": (4, 11)},
        ],
        "severity": "medium",
        "physical_mechanism": (
            "An unmeasured process disturbance (such as subtle recycle compressor guide-vane drift or tray liquid foaming) "
            "causes coupled pressure changes across reactor R-101 and separator V-102 without a single dominant primary sensor alarm."
        ),
        "differential_diagnosis": (
            "No single initiating transmitter demonstrates >3.5 sigma departure before others; multivariate CVA detects "
            "distributed dynamics rather than an isolated localized valve failure."
        ),
        "recommended_action": (
            "Review compressor K-101 suction pressure, check separator mist eliminator differential pressure, and run "
            "historian cross-correlation analysis."
        ),
        "reasoning": (
            "Multivariate pressure perturbation across reactor and separator without single dominant isolated initiating sensor."
        ),
        "action": "Review historian trends across compressor and separator, and conduct field transmitter survey.",
    },
    17: {
        "id": 17,
        "name": "Unknown disturbance in reactor/feed system",
        "subsystem": "reactor_system",
        "equipment": ["R-101 Reactor", "Stream 6 Combined Feed Line"],
        "valves": ["XMV_45 (A and C Feed Flow valve)", "XMV_42 (D Feed Flow valve)"],
        "initiating": [{"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "increase", "dev_range": (5, 14)}],
        "cascading": [
            {"sensor": "Separator_Pressure", "tag": "XMEAS_13", "direction": "increase", "delay_min": (3, 7), "dev_range": (3, 9)},
        ],
        "severity": "medium",
        "physical_mechanism": (
            "Unmeasured feed header density or viscosity shifts propagate into R-101, causing mild pressure escalation."
        ),
        "differential_diagnosis": (
            "Distinguished from known faults (1-15) by lack of primary valve alarm or stoichiometric analyzer step."
        ),
        "recommended_action": (
            "Monitor feed header densities, check reactor agitator power draw, and audit instrumentation."
        ),
        "reasoning": (
            "Unclassified disturbance in reactor system; evidence supports process excursion but lacks isolated initiating tag."
        ),
        "action": "Audit reactor instrumentation and check upstream feed raw material density.",
    },
    18: {
        "id": 18,
        "name": "Unknown disturbance in reactor thermal loop",
        "subsystem": "reactor_system",
        "equipment": ["R-101 Reactor", "R-101 Agitator"],
        "valves": ["XMV_51 (Reactor Cooling Water Flow valve)"],
        "initiating": [{"sensor": "Reactor_Temperature", "tag": "XMEAS_9", "direction": "increase", "dev_range": (4, 12)}],
        "cascading": [
            {"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "increase", "delay_min": (2, 6), "dev_range": (3, 9)},
        ],
        "severity": "medium",
        "physical_mechanism": (
            "Subtle internal heat transfer resistance or reaction kinetic shift causes reactor temperature and pressure to rise "
            "within regulatory controller limits."
        ),
        "differential_diagnosis": (
            "No valve failure (XMV_51) detected; temperatures stay well below runaway thresholds (<124 °C)."
        ),
        "recommended_action": (
            "Verify reactor temperature sensor calibration and check reactor jacket fouling factor."
        ),
        "reasoning": (
            "Mild reactor thermal excursion without cooling valve saturation or external step trigger."
        ),
        "action": "Verify reactor thermowell calibration and monitor thermal margins.",
    },
    19: {
        "id": 19,
        "name": "Unknown disturbance in separation condenser loop",
        "subsystem": "condenser_cooling_system",
        "equipment": ["E-102 Overhead Condenser", "V-102 Separator"],
        "valves": ["XMV_52 (Condenser Cooling Water Flow valve)"],
        "initiating": [{"sensor": "Separator_Pressure", "tag": "XMEAS_13", "direction": "increase", "dev_range": (4, 12)}],
        "cascading": [
            {"sensor": "Separator_Temperature", "tag": "XMEAS_11", "direction": "increase", "delay_min": (2, 6), "dev_range": (3, 8)},
        ],
        "severity": "medium",
        "physical_mechanism": (
            "Separator vapor-liquid equilibrium fluctuation caused by ambient cooling tower drift or minor entrainment."
        ),
        "differential_diagnosis": (
            "Distinguished from Fault 15 by lack of cooling water valve stiction; condenser flow remains within nominal limits."
        ),
        "recommended_action": (
            "Inspect Separator V-102 demister pad and check condenser coolant outlet temperature."
        ),
        "reasoning": (
            "Mild separator pressure and temperature drift without mechanical actuator stiction."
        ),
        "action": "Check separator demister pad differential pressure and condenser heat balance.",
    },
    20: {
        "id": 20,
        "name": "Unknown disturbance in feed composition",
        "subsystem": "feed_system",
        "equipment": ["Stream 6 Combined Feed Line", "R-101 Reactor"],
        "valves": ["XMV_45 (A and C Feed Flow valve)"],
        "initiating": [{"sensor": "Component_B_Stream6", "tag": "XMEAS_24", "direction": "increase", "dev_range": (4, 11)}],
        "cascading": [
            {"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "increase", "delay_min": (3, 7), "dev_range": (3, 8)},
        ],
        "severity": "medium",
        "physical_mechanism": (
            "Minor trace inert variations in incoming feed lines cause subtle pressure shifts in the recycle gas loop."
        ),
        "differential_diagnosis": (
            "Small excursion magnitude (<2.5 sigma) handled by regulatory purge valve without trip risk."
        ),
        "recommended_action": (
            "Log feed stream chromatography and verify upstream blending controls."
        ),
        "reasoning": (
            "Minor feed composition excursion without severe process upset or valve saturation."
        ),
        "action": "Log gas chromatograph trends and monitor recycle loop purity.",
    },
    21: {
        "id": 21,
        "name": "Valve position stuck (stream 4 feed valve)",
        "subsystem": "feed_system",
        "equipment": ["Stream 4 Feed Line", "Valve Actuator XMV_45 (XMV 4)", "R-101 Reactor"],
        "valves": ["XMV_45 (A and C Feed Flow valve / XMV 4)"],
        "initiating": [{"sensor": "A_C_Feed_Flow", "tag": "XMV_45 / XMEAS_4", "direction": "decrease", "dev_range": (25, 48)}],
        "cascading": [
            {"sensor": "Reactor_Feed_Rate", "tag": "XMEAS_6", "direction": "decrease", "delay_min": (2, 5), "dev_range": (10, 22)},
            {"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "decrease", "delay_min": (4, 8), "dev_range": (5, 14)},
        ],
        "severity": "high",
        "physical_mechanism": (
            "Mechanical valve stem seizure or positioner failure physically jams valve XMV_45 (Stream 4 feed) at a fixed "
            "stroke. When process demands or upstream pressure fluctuations occur, the valve cannot modulate to maintain "
            "the desired flow setpoint. A_C_Feed_Flow drops and remains frozen, starving R-101 of reactants A and C."
        ),
        "differential_diagnosis": (
            "Distinguished from Fault 1 (ratio step) and Fault 7 (header pressure drop) because the physical valve stem "
            "fails to move despite controller error signal changes; current-to-pressure (I/P) transducer shows fixed stem output."
        ),
        "recommended_action": (
            "Dispatch instrument technician to check air supply pressure and stroke valve XMV_45 manually using the handwheel. "
            "Check smart positioner error codes for mechanical binding."
        ),
        "reasoning": (
            "Mechanical stiction on Stream 4 feed valve XMV_45 locked the valve stem, freezing reactant delivery and causing "
            "downstream drops in reactor feed rate and pressure."
        ),
        "action": "Inspect valve XMV_45 smart positioner, verify air supply, and lubricate valve packing/stem.",
    },
    22: {
        "id": 22,
        "name": "Valve position stuck (stream 2 D feed valve)",
        "subsystem": "feed_system",
        "equipment": ["Stream 2 D Feed Line", "Valve Actuator XMV_42 (XMV 1)", "R-101 Reactor"],
        "valves": ["XMV_42 (D Feed Flow valve / XMV 1)"],
        "initiating": [{"sensor": "D_Feed_Flow", "tag": "XMV_42 / XMEAS_2", "direction": "decrease", "dev_range": (20, 42)}],
        "cascading": [
            {"sensor": "Reactor_Feed_Rate", "tag": "XMEAS_6", "direction": "decrease", "delay_min": (2, 5), "dev_range": (8, 18)},
            {"sensor": "Reactor_Pressure", "tag": "XMEAS_7", "direction": "decrease", "delay_min": (4, 8), "dev_range": (4, 12)},
        ],
        "severity": "high",
        "physical_mechanism": (
            "Mechanical actuator binding on control valve XMV_42 (Stream 2 D feed) locks the valve in place. The flow controller "
            "loses the ability to adjust D feed delivery, causing D_Feed_Flow to drift off setpoint and disrupting the stoichiometric "
            "reactant balance in reactor R-101."
        ),
        "differential_diagnosis": (
            "Distinguished from Fault 3 (temperature step) because Stream 2 temperature is normal; the excursion is purely "
            "volumetric flow starvation caused by a mechanically jammed valve stem on XMV_42."
        ),
        "recommended_action": (
            "Inspect valve XMV_42 actuator diaphragm and pneumatic positioner. Exercise valve stem manually to clear "
            "debris or packing friction."
        ),
        "reasoning": (
            "Actuator jamming on D feed valve XMV_42 restricted reactant D delivery, causing reactor feed deficit and pressure decline."
        ),
        "action": "Check valve XMV_42 positioner, exercise stem manually, and verify Stream 2 flow controller output.",
    },
}


def knowledge_for_fault(fault_id: int) -> Dict[str, Any]:
    """Retrieve detailed engineering knowledge for a specific TEP fault."""
    if fault_id not in FAULT_KNOWLEDGE:
        raise KeyError(f"Fault {fault_id} not present in knowledge base.")
    return FAULT_KNOWLEDGE[fault_id]


def all_fault_ids() -> List[int]:
    """Return sorted list of all 22 TEP fault IDs."""
    return sorted(FAULT_KNOWLEDGE.keys())