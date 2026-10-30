# Follow-up routing table (live dispatcher vs template)

Phase 2C inspection (a). Event fixture: `tests/fixtures/response/classified.json` (ANOM-0488, leading cause `FAULT_15`).
Routing does not depend on the event; only the answer text and the clarification triggers do.

`(none: model)` is the live path's LLM branch: the dispatcher claimed no intent, so `RCAInference.answer_followup` generates the answer.

## Per intended intent (n = 50 each)

| intended | family | template | (none: model) | clarify | guardrail-eligible | routes |
|---|---|---|---|---|---|---|
| `sensor_values` | live_template | 50 | 0 | 4 | 0 | sensor_values 50 |
| `evidence_why` | live_template | 50 | 0 | 0 | 0 | evidence_why 50 |
| `system_sensors` | live_template | 50 | 0 | 0 | 0 | system_sensors 50 |
| `sensor_scores` | live_template | 50 | 0 | 0 | 0 | sensor_scores 50 |
| `detector_severity` | live_template | 50 | 0 | 0 | 0 | detector_severity 50 |
| `detector_scores` | live_template | 50 | 0 | 0 | 0 | detector_scores 50 |
| `severity` | live_template | 50 | 0 | 0 | 0 | severity 50 |
| `overview` | live_template | 46 | 4 | 4 | 0 | overview 46, (none: model) 4 |
| `sensors_inspect` | live_template | 50 | 0 | 0 | 0 | sensors_inspect 50 |
| `action` | live_template | 49 | 1 | 0 | 0 | action 49, (none: model) 1 |
| `overview_fallback` | live_template | 50 | 0 | 5 | 0 | overview_fallback 45, overview 5 |
| `hypothesis_comparison` | benchmark_only | 16 | 34 | 1 | 0 | (none: model) 34, overview 12, overview_fallback 4 |
| `out_of_scope_refusal` | benchmark_only | 1 | 49 | 0 | 0 | (none: model) 49, overview 1 |
| `clarification_reply` | benchmark_only | 0 | 50 | 0 | 0 | (none: model) 50 |
| `valve_actuator` | live_free_form | 0 | 50 | 0 | 0 | (none: model) 50 |
| `causal_why` | live_free_form | 0 | 50 | 0 | 0 | (none: model) 50 |
| `mechanism` | live_free_form | 4 | 46 | 0 | 0 | (none: model) 46, action 4 |

## By family

| family | labels | n | template | model | model_share |
|---|---|---|---|---|---|
| live_template | 11 | 550 | 545 | 5 | 0.0091 |
| live_free_form | 3 | 150 | 4 | 146 | 0.9733 |
| benchmark_only | 3 | 150 | 17 | 133 | 0.8867 |

`live_template` is the 11 intents the dispatcher claims; `live_free_form` is the three case types the dispatcher has no rule for, so the live path always sends them to the model; `benchmark_only` is the three proposed intents, off by default.


## Benchmark configuration (proposed intents on)

| intended | benchmark primary |
|---|---|
| `sensor_values` | sensor_values 46, hypothesis_comparison 4 |
| `evidence_why` | evidence_why 50 |
| `system_sensors` | system_sensors 50 |
| `sensor_scores` | sensor_scores 50 |
| `detector_severity` | detector_severity 50 |
| `detector_scores` | detector_scores 50 |
| `severity` | severity 50 |
| `overview` | overview 46, (none) 4 |
| `sensors_inspect` | sensors_inspect 50 |
| `action` | action 49, out_of_scope_refusal 1 |
| `overview_fallback` | overview_fallback 45, overview 5 |
| `hypothesis_comparison` | hypothesis_comparison 50 |
| `out_of_scope_refusal` | (none) 41, out_of_scope_refusal 8, overview 1 |
| `clarification_reply` | clarification_reply 50 |
| `valve_actuator` | (none) 47, out_of_scope_refusal 3 |
| `causal_why` | (none) 45, out_of_scope_refusal 5 |
| `mechanism` | (none) 31, out_of_scope_refusal 15, action 4 |

## Reply-shaped statements (no clarification keyword)

n = 10; clarification triggers fired on 0. Routing: detector_scores 1, (none: model) 7, evidence_why 2

These are answers to a pending clarifying question, so the keyword router cannot see them; the dataset resolves them from dialogue state instead (see `llm/followup/cases.py`).

## Existing corpora, measured both ways

| corpus | measured | n | model | model share | routes |
|---|---|---|---|---|---|
| data/llm_1/train | followup:ask | 576 | 0 | 0.0 | overview 288, severity 144, sensors_inspect 144 |
| data/llm_1/train | grounded_qa:ask | 1008 | 0 | 0.0 | detector_scores 144, evidence_why 144, sensor_values 144, sensors_inspect 144, overview 144, severity 144, action 144 |
| data/llm_1/train | grounded_qa:full | 1008 | 0 | 0.0 | evidence_why 864, sensor_values 144 |
| data/llm_1/val | followup:ask | 32 | 0 | 0.0 | overview 16, severity 8, sensors_inspect 8 |
| data/llm_1/val | grounded_qa:ask | 56 | 0 | 0.0 | detector_scores 8, evidence_why 8, sensor_values 8, sensors_inspect 8, overview 8, severity 8, action 8 |
| data/llm_1/val | grounded_qa:full | 56 | 0 | 0.0 | evidence_why 48, sensor_values 8 |
| data/llm_1/test | followup:ask | 32 | 0 | 0.0 | overview 16, severity 8, sensors_inspect 8 |
| data/llm_1/test | grounded_qa:ask | 56 | 0 | 0.0 | detector_scores 8, evidence_why 8, sensor_values 8, sensors_inspect 8, overview 8, severity 8, action 8 |
| data/llm_1/test | grounded_qa:full | 56 | 0 | 0.0 | evidence_why 48, sensor_values 8 |
| data/llm/train | followup:ask | 576 | 0 | 0.0 | overview 288, severity 144, sensors_inspect 144 |
| data/llm/val | followup:ask | 32 | 0 | 0.0 | overview 16, severity 8, sensors_inspect 8 |
| data/llm/test | followup:ask | 32 | 0 | 0.0 | overview 16, severity 8, sensors_inspect 8 |

`...:full` is the stored `question` field routed verbatim (telemetry preamble included); `...:ask` is the text after `Operator question:`, which is what the live call site passes.

