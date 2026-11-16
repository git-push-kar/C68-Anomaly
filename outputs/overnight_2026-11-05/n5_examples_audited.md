# 30 mechanism rows - machine audit

rows: 30 (C 19, B 11)

| verdict | rows | meaning |
|---|---|---|
| compatible | 18 | the window names this measurement specifically: the checker's flag is a false positive |
| insufficient_evidence | 9 | the window holds only family-level words: the sentence must decide |
| class_mismatch | 2 | the window names a DIFFERENT instrument: a real mechanism error |
| unknown_tag | 1 | the tag does not exist in the table |

Every `claimed` window is a phrase of the sentence shown: **0 of 30** unmatched.

| # | cond | tag | canonical | window used | verdict | the sentence as printed |
|---|---|---|---|---|---|---|
| 1 | C | XMV_50 | Stripper_Steam_Valve | he onset order shows the valve position change | **compatible** | The dynamic residual attribution shows XMV_50 and XMEAS_16 as dominant contributors with high residual shares, but the onset order shows the valve position change (XMV_50) occurring before t |
| 2 | C | XMV_52 | Condenser_Cooling_Water_Flow | system and verify if the stripper steam valve | **class_mismatch** | Check the steam supply system and verify if the stripper steam valve (XMV_52) is operating correctly, as the trend in Stripper_Steam_Flow (XMEAS_19) is decreasing while temperatures rise. |
| 3 | C | XMEAS_23 | Component_A_Stream6 | and its associated A_Feed_Flow measurement | **insufficient_evidence** | Check the A_Feed_Flow valve position (XMV_44) and its associated A_Feed_Flow measurement (XMEAS_23) for possible valve sticking or control loop issues. |
| 4 | C | XMEAS_38 | Component_E_Stream11 | C feed flow measurement | **insufficient_evidence** | Check the A/C feed valve position and steam valve position for unexpected movement, and verify if the A/C feed flow measurement (XMEAS_38) is trending as expected. |
| 5 | C | XMEAS_11 | Separator_Temperature | followed by temperature | **compatible** | The onset order shows pressure measurements (XMEAS_7, XMEAS_11, XMEAS_13, XMEAS_16) rising first, followed by temperature (XMEAS_11) and flow (XMEAS_20). |
| 6 | C | XMEAS_20 | Compressor_Work | and flow | **insufficient_evidence** | The onset order shows pressure measurements (XMEAS_7, XMEAS_11, XMEAS_13, XMEAS_16) rising first, followed by temperature (XMEAS_11) and flow (XMEAS_20). |
| 7 | C | XMV_52 | Condenser_Cooling_Water_Flow | Check the reactor feed valve position | **class_mismatch** | Check the reactor feed valve position (XMV_52) and verify if the cooling water flow (XMV_52) is stuck open or if there is a slow change in cooling water inlet temperature. |
| 8 | C | XMV_52 | Condenser_Cooling_Water_Flow | and verify if the cooling water flow | **compatible** | Check the reactor feed valve position (XMV_52) and verify if the cooling water flow (XMV_52) is stuck open or if there is a slow change in cooling water inlet temperature. |
| 9 | C | XMV_19 |  | system and verify if the stripper steam valve | **unknown_tag** | Check the steam supply system and verify if the stripper steam valve (XMV_19) is operating correctly. |
| 10 | C | XMV_51 | Reactor_Cooling_Water_Flow | but the cooling water flow | **compatible** | All feed streams are deviating simultaneously with small negative trends, but the cooling water flow (XMV_51) is decreasing slightly. |
| 11 | C | XMV_50 | Stripper_Steam_Valve | easing trends in both the manipulated variable | **insufficient_evidence** | The evidence shows a slow-drift trend in multiple measurements including Stripper_Steam_Flow (XMEAS_19), Stripper_Steam_Valve (XMV_50), and Stripper_Temperature (XMEAS_18), with increasing t |
| 12 | C | XMV_51 | Reactor_Cooling_Water_Flow | le and verify the cooling water valve position | **compatible** | Check if the cooling water inlet temperature is stable and verify the cooling water valve position (XMV_51) is not stuck open. |
| 13 | C | XMV_51 | Reactor_Cooling_Water_Flow | water inlet temperature and flow control valve | **compatible** | Check the cooling water inlet temperature and flow control valve (XMV_51) for possible blockage or control issues, as the observed behavior does not match any standard fault pattern. |
| 14 | C | XMV_52 | Condenser_Cooling_Water_Flow | let conditions and flow control valve position | **compatible** | Check the condenser cooling water inlet conditions and flow control valve position (XMV_52) as these are the most affected by the disturbance. |
| 15 | C | XMV_52 | Condenser_Cooling_Water_Flow | let conditions and flow control valve position | **compatible** | Check the condenser cooling water inlet conditions and flow control valve position (XMV_52) first, as these are the most directly affected by the disturbance. |
| 16 | C | XMV_52 | Condenser_Cooling_Water_Flow | water inlet conditions and flow control valve | **compatible** | Check the condenser cooling water inlet conditions and flow control valve (XMV_52) first, as these are the most directly affected by the observed temperature change. |
| 17 | C | XMV_52 | Condenser_Cooling_Water_Flow | let conditions and flow control valve position | **compatible** | Check the condenser cooling water inlet conditions and flow control valve position (XMV_52) as these are the most affected by the disturbance. |
| 18 | C | XMV_44 | A_Feed_Flow | let conditions and flow control valve position | **compatible** | Check the condenser cooling water inlet conditions and flow control valve position (XMV_44) first, as these are the most directly affected by the anomaly. |
| 19 | C | XMV_46 | Compressor_Recycle_Valve | but the valve position | **compatible** | The dynamic residual attribution points to XMEAS_7 and XMEAS_13 as dominant contributors, but the valve position (XMV_46) shows a strong increasing trend while the measured flows/temperature |
| 20 | B | XMV_50 | Stripper_Steam_Valve | he onset order shows the valve position change | **compatible** | The dynamic residual attribution shows XMV_50 and XMEAS_16 as dominant contributors with high residual shares, but the onset order shows the valve position change (XMV_50) occurring before t |
| 21 | B | XMV_51 | Reactor_Cooling_Water_Flow | is ruled out because the flow valve position | **compatible** | FAULT_14 (Reactor cooling water valve sticking) is ruled out because the flow valve position (XMV_51) is increasing, not decreasing, and the cooling water flow measurement (XMEAS_21) shows n |
| 22 | B | XMEAS_21 | Reactor_Cooling_Water_Outlet_Temperature | and the cooling water flow measurement | **compatible** | FAULT_14 (Reactor cooling water valve sticking) is ruled out because the flow valve position (XMV_51) is increasing, not decreasing, and the cooling water flow measurement (XMEAS_21) shows n |
| 23 | B | XMV_51 | Reactor_Cooling_Water_Flow | and the dominant sensor is the valve position | **insufficient_evidence** | FAULT_01 (A/C feed ratio step change) is ruled out because the Reactor_Temperature shows no significant deviation (-0.00%) and the dominant sensor is the valve position (XMV_51), not a feed  |
| 24 | B | XMV_51 | Reactor_Cooling_Water_Flow | is ruled out because the valve position | **insufficient_evidence** | FAULT_14 (Reactor cooling water valve sticking) is ruled out because the valve position (XMV_51) is increasing, not decreasing, and the cooling water flow (XMEAS_21) shows no significant dev |
| 25 | B | XMEAS_21 | Reactor_Cooling_Water_Outlet_Temperature | and the cooling water flow | **compatible** | FAULT_14 (Reactor cooling water valve sticking) is ruled out because the valve position (XMV_51) is increasing, not decreasing, and the cooling water flow (XMEAS_21) shows no significant dev |
| 26 | B | XMV_51 | Reactor_Cooling_Water_Flow | d the valve position is a manipulated variable | **insufficient_evidence** | FAULT_14 (Reactor cooling water valve sticking) is ruled out because the flow is increasing, not decreasing, and the valve position is a manipulated variable (XMV_51) not a sensor. |
| 27 | B | XMV_51 | Reactor_Cooling_Water_Flow | d the valve position is a manipulated variable | **insufficient_evidence** | FAULT_14 (Reactor cooling water valve sticking) is ruled out because the flow is increasing, not decreasing, and the valve position is a manipulated variable (XMV_51) not a sensor. |
| 28 | B | XMV_51 | Reactor_Cooling_Water_Flow | is ruled out because the flow valve position | **compatible** | FAULT_14 (Reactor cooling water valve sticking) is ruled out because the flow valve position (XMV_51) is increasing, not decreasing, and the cooling water flow measurement (XMEAS_21) shows n |
| 29 | B | XMEAS_21 | Reactor_Cooling_Water_Outlet_Temperature | and the cooling water flow measurement | **compatible** | FAULT_14 (Reactor cooling water valve sticking) is ruled out because the flow valve position (XMV_51) is increasing, not decreasing, and the cooling water flow measurement (XMEAS_21) shows n |
| 30 | B | XMV_51 | Reactor_Cooling_Water_Flow | is ruled out because the valve position | **insufficient_evidence** | FAULT_14 (Reactor cooling water valve sticking) is ruled out because the valve position (XMV_51) is increasing, not decreasing, and the flow measurement (XMEAS_21) shows no deviation. |

## What is left for a human

* **2 class_mismatch** rows: the window names a different instrument, so these are real errors unless the sentence overrides it.
* **9 insufficient_evidence** rows: the quoted window holds only family-level words ('valve', 'flow', 'manipulated variable'), so it cannot prove a wrong-class reference either way. The sentence column now prints the text; judge from that.
* **18 compatible** rows can be dismissed without reading.
* **1 unknown_tag** rows are decidable from the table alone.

The rung-3 input is the count of confirmed real errors. An insufficient row is not an error and not a pass: keep it in a separate unsettled bucket, never in the numerator and never silently in the denominator.

