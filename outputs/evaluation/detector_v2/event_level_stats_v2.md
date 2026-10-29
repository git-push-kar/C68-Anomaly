# Event-level statistics at the frozen v2 operating point

`detector_v2_window_op_2026-10-30` — persistence 12 consecutive alarmed windows, min separation 20, max event 200, onset sample 160.

## Fault-free runs (TEP_FaultFree_Testing 1-500)

- events per run: **0.020**
- runs with at least one event: **10 / 500 (2.0%)**
- event-count histogram: `{"0": 490, "1": 10}`

## Faulty runs (Protocol A validation, 20 faults x 100 runs)

- events per run: **0.912**
- events: 1824 total — **0 pre-onset (0.0%)**, 1824 post-onset

| Fault | events/run | runs with post-onset event | post-onset rate | pre-onset share of events |
|---|---|---|---|---|
| 1 | 1.00 | 100/100 | 100% | 0% |
| 2 | 1.00 | 100/100 | 100% | 0% |
| 3 | 0.08 | 8/100 | 8% | 0% |
| 4 | 1.00 | 100/100 | 100% | 0% |
| 5 | 1.00 | 100/100 | 100% | 0% |
| 6 | 1.00 | 100/100 | 100% | 0% |
| 7 | 1.00 | 100/100 | 100% | 0% |
| 8 | 1.00 | 100/100 | 100% | 0% |
| 9 | 0.16 | 16/100 | 16% | 0% |
| 10 | 1.00 | 100/100 | 100% | 0% |
| 11 | 1.00 | 100/100 | 100% | 0% |
| 12 | 1.00 | 100/100 | 100% | 0% |
| 13 | 1.00 | 100/100 | 100% | 0% |
| 14 | 1.00 | 100/100 | 100% | 0% |
| 15 | 1.00 | 100/100 | 100% | 0% |
| 16 | 1.00 | 100/100 | 100% | 0% |
| 17 | 1.00 | 100/100 | 100% | 0% |
| 18 | 1.00 | 100/100 | 100% | 0% |
| 19 | 1.00 | 100/100 | 100% | 0% |
| 20 | 1.00 | 100/100 | 100% | 0% |

**Fault 15:** 100/100 runs have a post-onset event (100%), 1.00 events/run.