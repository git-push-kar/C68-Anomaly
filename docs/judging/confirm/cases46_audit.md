# 46 calibration cases - machine audit

46 cases: anchor 8, contradiction 15, numeric 9, tag_role 14

* checker/label disagreements: **0**
* certified by table lookup or arithmetic: **31**
* awaiting a human judgement: **15** (all contradiction shapes)

## The rows that need you

| # | shape | text | why the checker flags it |
|---|---|---|---|
| 1 | names a fault or verdict | The event is caused by FAULT_07. | names another fault as the cause |
| 2 | names a fault or verdict | Diagnosis: FAULT_10 | diagnosis line names another fault |
| 3 | names a fault or verdict | This is attributable to FAULT_02. | attribution names another fault |
| 4 | names a fault or verdict | FAULT_12 is the fault. | is-the-fault form |
| 5 | verdict claim | The verdict is no_fault. | verdict claim contradicts a classified row |
| 6 | normal-operation claim | This is a normal operation. | normal claim on an unclassified row |
| 7 | names a fault or verdict | The best fit is FAULT_09. | best-fit names another fault |
| 8 | rejection / hypothetical / comparison | This pattern contradicts FAULT_06 (A feed loss), which would show A_Feed_Flow decreasing first. | rejection plus hypothetical |
| 9 | rejection / hypothetical / comparison | FAULT_07 is ruled out by the separator pressure. | explicit rejection |
| 10 | rejection / hypothetical / comparison | Unlike FAULT_14, the stripper level is unchanged. | comparison |
| 11 | negation or abstention | This is not FAULT_02; the reactor temperature falls. | negated alternative |
| 12 | verdict claim | The verdict is classified. | verdict claim agrees |
| 13 | negation or abstention | No listed fault is asserted for this event (UNCLASSIFIED_OPEN_SET). | the abstention phrasing is not a normal claim |
| 14 | names a fault or verdict | The evidence points to FAULT_01 as the cause. | cause matches the supplied diagnosis |
| 15 | names a fault or verdict | If this were FAULT_08, the reactor cooling water flow would rise first. | hypothetical, not an assertion |

## Machine-certified rows

| # | check | label | evidence |
|---|---|---|---|
| 1 | tag_role | flag | XMEAS_17=Stripper_Underflow_Stream11; phrase '' -> names a different class |
| 2 | tag_role | flag | XMV_47=Purge_Valve; phrase '' -> names a different class |
| 3 | tag_role | flag | XMEAS_18=Stripper_Temperature; phrase '' -> names a different class |
| 4 | tag_role | flag | XMEAS_16=Stripper_Pressure; phrase '' -> names a different class |
| 5 | tag_role | flag | XMEAS_17=Stripper_Underflow_Stream11; phrase '' -> names a different class |
| 6 | tag_role | flag | XMV_52=Condenser_Cooling_Water_Flow; phrase '' -> names a different class |
| 7 | tag_role | no flag | XMEAS_1=A_Feed_Stream1; phrase '' -> names a different class |
| 8 | tag_role | no flag | XMV_50=Stripper_Steam_Valve; phrase '' -> names a different class |
| 9 | tag_role | no flag | XMEAS_9=Reactor_Temperature; phrase '' -> names a different class |
| 10 | tag_role | no flag | XMEAS_23=Component_A_Stream6; phrase '' -> names a different class |
| 11 | tag_role | no flag | XMEAS_16=Stripper_Pressure; phrase '' -> names a different class |
| 12 | tag_role | no flag | XMEAS_12=Separator_Level; phrase '' -> names a different class |
| 13 | anchor | no flag | XMEAS_1 z=18.12; text quotes 18.12 (\|delta\|=0.000, tolerance 0.01 -> exact) |
| 14 | anchor | no flag | XMEAS_16 z=31.08; text quotes 31.08 (\|delta\|=0.000, tolerance 0.01 -> exact) |
| 15 | anchor | no flag | XMEAS_11 z=-2.96; text quotes -2.96 (\|delta\|=0.000, tolerance 0.01 -> exact) |
| 16 | anchor | no flag | XMEAS_1 z=18.0; text quotes 18.0 (\|delta\|=0.000, tolerance 0.01 -> exact) |
| 17 | anchor | flag | XMEAS_1 z=18.12; text quotes 222.71 (\|delta\|=204.590, tolerance 0.01 -> contradicting) |
| 18 | anchor | flag | XMEAS_1 z=18.12; text quotes nothing |
| 19 | anchor | flag | XMEAS_1 z=18.12; text quotes 18.1 (\|delta\|=0.020, tolerance 0.01 -> contradicting) |
| 20 | anchor | flag | XMEAS_16 z=31.08; text quotes 30.98 (\|delta\|=0.100, tolerance 0.01 -> contradicting) |
| 21 | numeric | flag | LSTM autoencoder reconstruction error: stored 0.6971 vs threshold 0.6870; CVA Q-statistic: stored 10960.5 vs threshold 194.6 -> asserts 0.6971 is below 0.6870, but 0.6971 > 0.687 |
| 22 | numeric | flag | LSTM autoencoder reconstruction error: stored 0.6971 vs threshold 0.6870; CVA Q-statistic: stored 10960.5 vs threshold 194.6 -> quoted value(s) not in the store: [12345.6] |
| 23 | numeric | flag | LSTM autoencoder reconstruction error: stored 0.6971 vs threshold 0.6870; CVA Q-statistic: stored 10960.5 vs threshold 194.6 -> quoted value(s) not in the store: [0.7, 0.6] |
| 24 | numeric | no flag | LSTM autoencoder reconstruction error: stored 0.6971 vs threshold 0.6870; CVA Q-statistic: stored 10960.5 vs threshold 194.6 -> no contradiction in the quoted quantities |
| 25 | numeric | no flag | LSTM autoencoder reconstruction error: stored 0.6971 vs threshold 0.6870; CVA Q-statistic: stored 10960.5 vs threshold 194.6 -> no contradiction in the quoted quantities |
| 26 | numeric | no flag | LSTM autoencoder reconstruction error: stored 0.6971 vs threshold 0.6870; CVA Q-statistic: stored 10960.5 vs threshold 194.6 -> no contradiction in the quoted quantities |
| 27 | numeric | no flag | LSTM autoencoder reconstruction error: stored 0.6971 vs threshold 0.6870; CVA Q-statistic: stored 10960.5 vs threshold 194.6 -> no contradiction in the quoted quantities |
| 28 | numeric | flag | LSTM autoencoder reconstruction error: stored 0.6971 vs threshold 0.6870; CVA Q-statistic: stored 10960.5 vs threshold 194.6 -> stored value 10960.5 is printed as the threshold |
| 29 | numeric | no flag | LSTM autoencoder reconstruction error: stored 0.6971 vs threshold 0.6870; CVA Q-statistic: stored 10960.5 vs threshold 194.6 -> no contradiction in the quoted quantities |
| 30 | tag_role | no flag | XMV_51=Reactor_Cooling_Water_Flow; phrase '' -> names a different class |
| 31 | tag_role | flag | XMV_51=Reactor_Cooling_Water_Flow; phrase '' -> names a different class |

## Freeze candidate

`cases46_freeze_candidate.json` carries the hashes of both source files and `signed_by: null`. Signing it is the human act; nothing here may fill that in.

```
d7470445a4897708d635e706d1d5a037364ec05618ba93495cf0343c3df11d5f  scripts/recheck_cases.py
cee3ed06923a4cec3d4f74220c2f1041482a5593eb94a1ce0f9e880d0aeed320  scripts/cases46.py
```
