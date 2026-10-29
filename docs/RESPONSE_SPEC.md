# RESPONSE_SPEC — TEP RCA adapter benchmark contract (v2)

**Repo:** `C:\Users\Admin\Desktop\anomaly` (branch `benchmark-upgrade`, base `4d4b3be`)
**Status:** Phase 0 contract. Nothing below describes new model behaviour; every
row cites where the behaviour already lives in this checkout.

This file replaces the earlier illustrative tag examples. All tags are written in
**this repo's convention**, not an invented one.

---

## 1. Variable tags

Canonical set: `XMEAS_1` .. `XMEAS_41` and `XMV_42` .. `XMV_52`, i.e. the keys of
`SENSOR_NAMES` in [`evidence/process_relationships.py`](../evidence/process_relationships.py).
The manipulated variables use the **global** index 42–52; the 1-based alias
1–11 (`"XMV 11"`, CSV column `xmv_11`) is an alias, never the canonical form.

`XMV_52` is a valid canonical tag. It is the canonical form of `XMV 11`
(offset `+41`), exactly as the knowledge base writes it:

| Fault | KB string (`scripts/fault_knowledge.py`) | Canonical tags |
|---|---|---|
| 1 | `XMV_45 (A and C Feed Flow valve / XMV 4)` / `XMV_45 / XMEAS_4` | `XMV_45`, `XMEAS_4` |
| 2 | `XMV_47 (Purge valve / XMV 6)` | `XMV_47`, `XMEAS_24` |
| 4, 14 | `XMV_51 (Reactor Cooling Water Flow valve / XMV 10)` / `XMV_51 / XMEAS_21` | `XMV_51`, `XMEAS_21` |
| 5, 15 | `XMV_52 (Condenser Cooling Water Flow valve / XMV 11)` / `XMV_52 / XMEAS_22` | `XMV_52`, `XMEAS_22` |
| 21 | `XMV_45 (A and C Feed Flow valve / XMV 4)` | `XMV_45`, `XMEAS_4` |

Rules:

1. Benchmark outputs, metrics and reports must print the **canonical** form
   (`XMV_52`), optionally followed by the repo's alias gloss
   (`XMV_52 (Condenser Cooling Water Flow valve / XMV 11)`).
2. Inputs may use any spelling: `XMV_52`, `XMV 11`, `XMV11`, `xmv_11`,
   `XMV_52 (…) / XMV 11`. Resolution lives in one place:
   [`scripts/tags.py`](../scripts/tags.py) — `canonical_tag()`, `iter_tags()`,
   `display_name()`, `subsystem()`, `fault_tags()`.
3. An unresolvable tag (`XMV_99`, `XMV 12`) returns `None` and must be treated as
   an error, never silently mapped.

---

## 2. Follow-up routing (what actually runs)

`main.py:394` `answer_followup` routes every follow-up. Order
([`main.py:419-467`](../main.py)):

1. `prefer_llm` (`llm.followup.prefer_llm`, default false) bypasses both the
   router **and** the guardrail → `route=llm-forced`.
2. Otherwise `RCAInference._dispatch_with_intent` first
   ([`llm/inference.py:641`](../llm/inference.py)) → `route=deterministic`.
3. Otherwise the adapter answers, then `format_followup_answer`
   ([`llm/inference.py:592`](../llm/inference.py)) repairs formatting, then
   `_guard_llm_answer` ([`llm/inference.py:456`](../llm/inference.py)) rejects
   number-free answers → `route=deterministic-guardrail`.
4. With no adapter loaded → `route=deterministic-fallback`.

`_dispatch_operator_intent` ([`llm/inference.py:445`](../llm/inference.py)) is a
thin wrapper that returns only the answer text; the naming pair is not a
separate router.

### 2.1 The real intent set — 11 intents, not 7

Intents are matched by keyword sets defined at
[`llm/inference.py:655-680`](../llm/inference.py) and returned by the branches
below. **First matching branch wins**, so this order is the priority order.

| # | Intent | Branch | Returned at | Trigger (as coded) | Format class |
|---|---|---|---|---|---|
| 1 | `sensor_values` | `if has_sensor_values:` | `inference.py:696` → `726` | sensor/ideal/normal/baseline/measured/observed **value(s)** | numeric |
| 2 | `evidence_why` | `if has_evidence_why:` | `729` → `754` | evidence, why is/was this an anomaly, alarm trigger, forces this to be an anomaly | numeric |
| 3 | `system_sensors` | `if has_system_sensors:` | `757` → `780` | which sensors are used, sensors in the system, system/all sensors | sensor-name |
| 4 | `sensor_scores` | `if has_sensor_scores:` | `783` → `801` | "sensor" **and** "score" | numeric |
| 5 | `detector_severity` | `if has_detector_scores and has_severity:` | `804` → `817` | detector score/metric/threshold **and** severity/critical/danger | numeric |
| 6 | `detector_scores` | `if has_detector_scores:` | `820` → `830` | score(s), metric(s), reconstruction error, q-stat, threshold | numeric |
| 7 | `severity` | `if has_severity:` | `833` → `846` | severity, how serious/dangerous, critical, alarm/risk level, danger | sensor-name |
| 8 | `overview` | `if has_overview:` | `849` → `864` | which/what anomaly or fault, fault number/type, what happened, root cause, cause, overview, diagnose, summary | sensor-name |
| 9 | `sensors_inspect` | `if has_sensors:` | `867` → `884` | sensor(s), inspect, deviat, which sensor | sensor-name |
| 10 | `action` | `if has_action:` | `887` → `894` | what should I do, what to do, action, recommend, fix, mitigat, next step, procedure | none |
| 11 | `overview_fallback` | `if has_generic_why:` | `898` → `913` | `why `, `why?`, explain, diagnos | sensor-name |

Format classes ([`llm/inference.py:527-537`](../llm/inference.py)):

* **numeric** — `NUMERIC_FORMAT_INTENTS = {sensor_values, evidence_why, sensor_scores,
  detector_scores, detector_severity}`. An answer with fewer than 3 markdown
  table lines is repaired by appending `_evidence_markdown_tables`
  ([`inference.py:539`](../llm/inference.py)); if evidence is empty the status is
  `unrepairable`.
* **sensor-name** — `SENSOR_NAME_INTENTS = {overview, overview_fallback,
  sensors_inspect, system_sensors, severity}`. An answer naming none of the top-3
  deviating sensors gets a `**Primary sensors:** …` line appended.
* **none** — `action` carries no table or sensor-name obligation.

Routing consequences that matter when writing test questions:

* `overview` is checked **before** `action`, and its keyword set contains `cause`;
  "what is the root cause and how do I fix it" routes to `overview`.
* `evidence_why` is checked **before** `sensor_scores`/`detector_scores`, so any
  question containing "evidence" is answered from the evidence template.
* Anything containing `threshold` is a detector-score question, even if it also
  names a sensor.

### 2.2 Proposed intents — PROPOSED, NOT IMPLEMENTED

These three are **not** in the dispatcher. They are proposed additions to close
coverage gaps found in Phase 0; nothing below is scored until the code exists.
Priorities are the insertion points in the renumbered order.

| # | Intent | Trigger (proposed) | Why it is needed |
|---|---|---|---|
| 8 | `hypothesis_comparison` | distinguish, differential, lookalike, "how is this different", "vs", versus, alternative causes | Today "How do we distinguish this from alternative causes?" is swallowed by `overview` (its keyword set contains `cause`), so a two-hypothesis comparison is answered as a single diagnosis. |
| 12 | `clarification_reply` | fewer than 3 content tokens, or "what do you mean", "can you clarify" | A too-short question currently falls through to `overview_fallback` or the LLM with no clarification path. |
| 13 | `out_of_scope_refusal` | no process keyword and no intent matched, or an explicit attempt to make the system answer off-domain | The adapter should decline rather than diagnose an unrelated question. |

Resulting order: 1 `sensor_values`, 2 `evidence_why`, 3 `system_sensors`,
4 `sensor_scores`, 5 `detector_severity`, 6 `detector_scores`, 7 `severity`,
**8 `hypothesis_comparison`**, 9 `overview`, 10 `sensors_inspect`, 11 `action`,
12 `overview_fallback`, **13 `clarification_reply`**, **14 `out_of_scope_refusal`**.

Placing `hypothesis_comparison` before `overview` is what makes it reachable;
placing the refusal last keeps it from stealing any in-domain question. These
are proposals for approval, not descriptions of current behaviour.

**Status:** the live dispatcher is unchanged. The three intents exist only as
candidates in [`llm/response/router.py`](../llm/response/router.py) behind
`ResponseRouter(enable_proposed=True)`; with the flag off (the default) the
router's candidate list stops at the 11 real intents. `clarification_reply`
needs an explicit clarification ask or a stub too short to route — a short real
question such as "What happened?" keeps its own intent.

---

## 3. Report generation

`RCAInference.generate_report` ([`llm/inference.py:327`](../llm/inference.py))
prompts with `REPORT_SYSTEM_PROMPT` (`inference.py:28`) + `SYSTEM_PROMPT`
([`llm/dataset.py:30`](../llm/dataset.py)), parses one JSON object, and falls
back to a deterministic stub with `root_cause="unable_to_parse"` when parsing
fails. Required keys: `ANSWER_KEYS` in [`llm/dataset.py:96`](../llm/dataset.py).

### 3.1 Severity is computed by code, not by the model

Severity is an authoritative field produced by
`compute_deterministic_severity` ([`evidence/process_limits.py:74`](../evidence/process_limits.py))
from detector state and sensor-limit envelopes. It is **not** a diagnostic
output of the adapter: the benchmark must not score severity as a model skill,
and any severity value quoted in a report must come from that function. The
prompt currently hands the model a fault-number → severity table
([`llm/dataset.py:42-49`](../llm/dataset.py)), which is why the old
`severity_accuracy` is uninterpretable; §7 proposes the replacement.

---

## 4. Fault scope

| Scope | Faults | Source of record |
|---|---|---|
| **Headline (real)** | 1–20 | `data/raw/faults/TEP_Faulty_Testing.csv` **only** — 960 samples/run, 0-based onset 160, 500 runs/fault |
| **Open set** | 16–20 | KB names are "Unknown disturbance in …"; accepted label is `UNCLASSIFIED_OPEN_SET` |
| **Unobservable** | 3, 9 | Flagged weak-signature; scored and reported separately, never pooled into the headline |
| **Synthetic only** | 21, 22 | `data/llm` v0 (all splits) and `data/llm_v1` test — generated, no real run exists |

Rules:

1. Headline results cover faults **1–20** only.
2. Faults 21 and 22 are **excluded from headline results**. Where they appear
   (e.g. the v1 holdout test set, which contains 15 and 21), the row, table cell
   and caption must be labelled `synthetic`.
3. `TEP_Faulty_Training.csv` is **not** used for benchmark events: it is a
   different run shape (500 samples/run, onset 20) than the Testing file, and
   pooling the two would make run length and pre-anomaly context file-dependent.
4. The label space, including the open-set and unobservable flags, lives in
   [`scripts/fault_ids.py`](../scripts/fault_ids.py).

---

## 5. Scoring identity and splits

Unique run key: **`(source_file, faultNumber, simulationRun)`**.

* Split contract: [`data/splits_v2.json`](../data/splits_v2.json), written by
  [`scripts/build_splits_v2.py`](../scripts/build_splits_v2.py).
* **Protocol A** (seen faults, run-disjoint): faults 1–20, train runs 1–300 /
  val 301–400 / test 401–500.
* **Protocol B** (leave-faults-out): fold 1 holds out {4, 11, 15}, fold 2 holds
  out {14, 15}; held-out faults appear in test only, seen faults keep runs
  401–500 unused.
* Validation: `python scripts/check_splits_v2.py` — recomputes each split's
  recorded SHA-256 over its expanded sorted key list, rejects any key shared by
  two splits, and enforces fault disjointness for the Protocol B folds.
* In-distribution (A) and fault-disjoint (B) numbers must be reported
  separately and labelled as such; the v0 set contains every fault in every
  split ([`data/llm/split_metadata.json`](../data/llm/split_metadata.json),
  `all_faults_in_train: true`).

---

## 6. Known confounds in the current benchmark (carry into every report)

1. **Severity is leaked, and is no longer a model output.** `SYSTEM_PROMPT`
   states the fault-number → severity mapping ([`llm/dataset.py:42-49`](../llm/dataset.py)),
   and every v0 training row's `severity` equals the KB severity (0/2376
   mismatches). `severity_accuracy` is a lookup reproduction, not a diagnostic
   result. Severity is computed by code (§3.1); the rubric replacement is
   proposed in §7.
2. **Teacher coverage is partial.** Only `report`-kind rows are teacher-written
   (264/2376 = 11.1% of v0 train carry `teacher_status: teacher_llm`); the
   2112 `grounded_qa` rows are deterministic physics output with
   `teacher_model: null`, even though `split_metadata.json` records a single
   global `teacher_model: qwen2.5:32b`.
3. **Root-cause matching is loose.** `evaluate.py` accepts a hit when the
   normalised reference name is a substring of the prediction **or** when a
   single token overlaps (`top_k=1` default). Quoting a raw tag such as
   `XMEAS_1` is scored as a foreign sensor in `evaluate_adapter`, because only
   `display_name` enters the allowed set. Treat `root_cause_accuracy` and
   `hallucination_rate` as upper/lower-bounded heuristics until both are fixed.

---

## 7. Proposed SYSTEM_PROMPT severity rubric replacement — PROPOSED, NOT IMPLEMENTED

The leak is four lines at [`llm/dataset.py:42-49`](../llm/dataset.py): a severity
rubric that enumerates fault numbers (`Faults 14, 15` critical; `Faults 4, 5, 6,
7, 11, 12, 21, 22` high; `Faults 1, 2, 3, 8, 9, 10, 16-20` medium; `Fault 13`
low). Its severities match the KB exactly, which is why every v0 row's severity
equals the KB value.

Proposed replacement (deletes those four lines, adds this block):

```text
SEVERITY (supplied, not inferred):
The host system computes the severity level from detector state and sensor
safety envelopes and supplies it in the evidence block as
"Assessed severity level". Restate that exact value in the "severity" field.
Do not upgrade or downgrade it, and do not infer severity from a fault number
or fault family.
```

Nothing else in `SYSTEM_PROMPT` changes: the chain-of-thought instruction, the
hedging rule and the answer schema stay as they are, so the JSON contract is
stable.

Consequences to plan for (Phase 1+, not Phase 0):

* `severity` stays a required answer key, but it becomes an echo/format check
  rather than a scored model skill; `severity_accuracy` is retired.
* The training answers must be regenerated (their severity currently derives from
  the same KB lookup) and the adapter retrained; applying the prompt edit alone
  would leave the labels and the prompt disagreeing.
* The synthetic faults 21 and 22 disappear from the prompt, removing the last
  place where they appear in headline-adjacent text.

## 8. Status of the proposed pieces (Phase 1 patch, B3)

Recorded so a later reader cannot mistake a proposal for shipped behaviour.

**Benchmark answers are schema-shaped.** A scored answer is a `Report` or a
`FollowupAnswer` (1-4 sentences, at most 3 suggested follow-ups, at most 1
clarifying question, a table id from the allowed catalog). **Tables are rendered
by code** from citation keys resolved through `StoreAccess`; the model never
writes a table cell, a detector row or a severity word. Anything outside the
schema is a validation failure (`schema`), not a scoring variant.

**Proposed intents stay benchmark-only.** `hypothesis_comparison`,
`out_of_scope_refusal` and `clarification_reply` are specified and routed behind
`ResponseRouter(enable_proposed_intents=True)`. **The live flag is off.**
`llm/inference.py` keeps its 11 intents untouched; the router mirrors the live
dispatcher on live-format questions, and a test fails if the mirror drifts.

**Trigger thresholds are placeholders.** `ambiguous_hypotheses` fires when the
top-2 confidence margin is below `0.15`. That number is a guess about adapter
behaviour: it cannot be calibrated until a trained adapter produces confidence
distributions, and it must be revisited against measured margin histograms
before any abstention number is quoted. `unclassified`, `ambiguous_question`
and `context_not_in_store` fire on deterministic conditions, not thresholds.

**Severity is code-computed, and the authority is the production rule.**
`evidence.process_limits.compute_deterministic_severity` decides the level.
`llm/response/severity.py` keeps its own rule only as a cross-check and exposes
`components['severity_agreement']` and `components['severity_gap']`. The model
still must not emit severity (`extra="forbid"`, a `severity_from_model`
failure). On all 489 stored events the cross-check agrees with the authoritative
rule on **79 (16.2%)** and is systematically lower — 218 events sit two levels
below, 205 of them `critical -> medium` — so it was not adopted; see
`outputs/evaluation/severity_agreement.md`. The renderer and the lookup answers
show the authoritative level only.

**Integer claims in prose.** A bare integer is accepted only when it equals a
stored count (`event.n_windows`, `count.top_sensors`, `count.timeline`,
`count.sensors_total`) or when it reads as a fault identifier ("Fault 15",
`FAULT_15`). Tag indices (11, 41, 52) are accepted only inside a tag
(`XMV_52`), and enumerated-list markers ("1.", "2.") are not treated as claims.
The previous rule accepted every integer 1-22 unconditionally, which let
"17 windows" pass on a two-window event.

The SYSTEM_PROMPT rubric replacement in §7 remains PROPOSED and unimplemented:
`llm/dataset.py` is unchanged in this patch.
