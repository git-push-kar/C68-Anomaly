# P2 - merged classifier row

Items scored by all three systems: **497**.

| system | root-cause accuracy (95 % CI) | abstains / gold abstains | precision | recall | false-abstention | gate |
|---|---|---|---|---|---|---|
| menu_rule | 0.9980 (0.9887-0.9996) | 125 / 125 | 1.0 | 1.0 | 0.0 | pass |
| hybrid | 0.9940 (0.9824-0.9979) | 128 / 125 | 0.9766 | 1.0 | 0.0081 | pass |
| adapter | 0.8330 (0.7977-0.8632) | 94 / 125 | 0.8723 | 0.656 | 0.0323 | pass |

| contrast | discordant for first | discordant for second | exact |
|---|---|---|---|
| menu_rule_vs_hybrid | 0 | 2 | p=0.5 |
| menu_rule_vs_adapter | 0 | 82 | p<1e-06 |
| hybrid_vs_adapter | 1 | 81 | p<1e-06 |

Subsystem and severity are not reported: they are knowledge-base functions of the predicted fault id, so a separate accuracy for them would restate the root-cause number rather than measure anything new.
