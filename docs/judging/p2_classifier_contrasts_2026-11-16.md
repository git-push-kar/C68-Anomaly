# P2 - merged classifier row

**Generation: pre-vocabulary-fix classifier; adapter = checkpoint-300** (0.8330 = 414/497, macro-F1 selected).
Retrained table: `vocabfix/classifier_contrasts.json` — adapter **0.7646 (0.7253–0.7998) = checkpoint-400**,
hybrid 0.9960, menu rule 1.0000, LightGBM 0.9980, MLP 0.9799; hybrid vs adapter 2 vs 117.

Items scored by all three systems: **497**.

| system | root-cause accuracy (95 % CI) | abstains / gold abstains | precision | recall | false-abstention | gate |
|---|---|---|---|---|---|---|
| menu_rule | 0.9980 (0.9887-0.9996) | 125 / 125 | 1.0 | 1.0 | 0.0 | pass |
| hybrid | 0.9940 (0.9824-0.9979) | 128 / 125 | 0.9766 | 1.0 | 0.0081 | pass |
| adapter (checkpoint-300) | 0.8330 (0.7977-0.8632) | 94 / 125 | 0.8723 | 0.656 | 0.0323 | pass |
| gbdt | 0.9940 (0.9824-0.9979) | 125 / 125 | 1.0 | 1.0 | 0.0 | pass |
| mlp | 0.9698 (0.9508-0.9816) | 122 / 125 | 0.9918 | 0.968 | 0.0027 | pass |

| contrast | discordant for first | discordant for second | exact |
|---|---|---|---|
| menu_rule_vs_hybrid | 0 | 2 | p=0.5 |
| menu_rule_vs_adapter | 0 | 82 | p<1e-06 |
| menu_rule_vs_gbdt | 1 | 3 | p=0.625 |
| menu_rule_vs_mlp | 0 | 14 | p=0.000122 |
| hybrid_vs_adapter | 1 | 81 | p<1e-06 |
| hybrid_vs_gbdt | 2 | 2 | p=1 |
| hybrid_vs_mlp | 1 | 13 | p=0.001831 |
| adapter_vs_gbdt | 81 | 1 | p<1e-06 |
| adapter_vs_mlp | 75 | 7 | p<1e-06 |
| gbdt_vs_mlp | 1 | 13 | p=0.001831 |

Subsystem and severity are not reported: they are knowledge-base functions of the predicted fault id, so a separate accuracy for them would restate the root-cause number rather than measure anything new.
