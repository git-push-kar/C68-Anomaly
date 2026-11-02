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

## 9. Hybrid diagnosis contract — PROPOSED, NOT IMPLEMENTED

Recorded while the hybrid is being evaluated (PROTOCOL §16). **Nothing in this section is in
the live path**: `llm/inference.py`, the System prompt in `llm/dataset.py`, the benchmark
items and every dataset hash are unchanged by it. It is written down so the contract is
reviewable before anyone builds it.

**The decision moves out of the model.** Today the report's `fault_id` and `verdict` are
generated: the model decodes a label as the first field of its JSON, and §14/§15 measure how
well it does that (macro-F1 0.9263 on validation at step 500). Under the proposal, both fields
are supplied by the caller:

1. the detector opens the event (unchanged);
2. a classifier reads the evidence features and predicts the true fault;
3. a menu rule turns that prediction into a diagnosis — `UNCLASSIFIED_OPEN_SET` when the
   prediction is not on the item's menu or the calibrated confidence is below the abstention
   threshold, `NORMAL_TRANSIENT` with the `no_fault` verdict when there is no fault;
4. the model is asked to *continue* the answer from a prefix the system has already written:

   ```
   {"fault_id": "<diagnosis>", "verdict": "<verdict>", 
   ```

   and produces only `ranked_hypotheses`, `physical_reasoning`, `narrative`;
5. everything else (mitigation, primary subsystem, table selection, severity, follow-up
   suggestions, clarifying question) is built in code from the KB and the router at render
   time, as §3 and §12.1 of PROTOCOL already require.

**The prompt contract.** The System prompt is unchanged in wording; what changes is that the
assistant turn starts with the prefix above instead of an empty string, and the instructions
must say that `fault_id` and `verdict` are given and must not be restated differently. A
continuation that emits a second copy of `fault_id`, or a ranking whose top entry is not the
supplied diagnosis, is a **validation failure**, not a scoring variant — the same treatment
§3 already gives to a model-written table cell.

**Why it is proposed.** Fault identity and prose are different problems: the classifier sees
the evidence features directly and is measured on identification alone, while the adapter is
left with the part it is good at (mechanism, narrative, operator-facing wording). The
counterweight is that the adapter can no longer be credited with the diagnosis, and a prose
writer handed a wrong label can be *coherently wrong*; that failure mode is what the
counterfactual check in `scripts/hybrid_faithfulness.py` prices, and it is why the proposal is
not a default.

**What a deployment would still need, before this is anything but a proposal:** a classifier
calibrated for the deployment population (not on a benchmark validation split), a monitored
abstention rate, a visible distinction in the rendered report between "classified" and
"open set", and an operator path for the open-set case. None of those exist today.

### 9.1 Three things §17 says this proposal would need (dated 2026-11-01, still a proposal)

Recorded after the hardening measurements in PROTOCOL §17. **Nothing here is implemented**; the
live path, the prompts, the datasets and their hashes are unchanged, and §9 above stays as it was.

**1. The diagnosis needs a second channel for "I do not recognise this".** Across the thirteen
leave-one-out faults, every confidence-shaped signal separates an unseen fault from the seen ones
about equally well — margin **0.881**, calibrated maximum **0.882**, entropy **0.883**, novelty
**0.846** mean AUROC — so the ranking is not where the problem is. (An earlier draft of this note
said the calibrated maximum scored 0.12; that was a sign error in my own AUROC call, not a
property of the classifier: an unseen class gets a *lower* maximum, so the raw comparison must be
complemented before it can be read. Corrected here, before anything was built on it.)

The problem is the **threshold**. With isotonic saturation, 774 of 834 calibrated maxima are still
at or above 0.979, so a confidence threshold abstains 4 times in 497 menu-present items while the
fold-2 classifier names FAULT_11 on 23 of the 50 items whose true fault it never saw. What a
hybrid needs is not a better confidence number but a reference: the nearest training events, or
the KB's expected sensor signature for the candidate, compared against the evidence. Novelty is
the weakest separator of the four and still 0.85, and it is the only one that is a *distance to
the training set* rather than a restatement of the classifier's own opinion.

**2. The prompt needs a grounding verdict it is allowed to use.** Forced-diagnosis generation
contradicted its prefix on 0 of 242 validation items and rationalised a deliberately wrong prefix
on 58 of 58. A prose writer that cannot disagree is not a second opinion, it is a paraphrase of
whatever it was handed. The proposal would therefore supply a system-computed field — evidence
strength and whether the event's signature matches the diagnosis — and an instruction that the
answer must *report* a mismatch rather than argue around it. That is a target-level change to the
answer schema (`grounding` or similar), and it must come with a validator that fails an answer
whose grounding verdict contradicts the system's.

**3. The tag kinds have to be enforced, not just printed.** 21 of 242 answers (8.7 %) described a
manipulated variable as a measurement or the reverse, even though every tag is printed with its
kind. The follow-up path already has the rule (`llm/followup/filters.py`); the report path has
none. Any prompt that hands the model a diagnosis should carry the retry instruction that
`llm/hybrid/generate.py::RETRY_MISLABEL` states, and the report validator should reject a prose
answer that mislabels a tag's kind.

A bounded retry experiment (§17.7) says the instruction alone is not enough: regenerating the 21
items once with the retry sentence fixed **7**, left **13** mislabelled exactly as before, and
moved **1** to a different mislabel (it had called a measurement a valve position; it came back
calling a valve position a temperature). Prompting is a 33 % fix on this failure mode, and the
retry cannot be trusted to stay put; a validator that rejects the answer, and a rewrite that
reuses the previous prose rather than regenerating it, is what the report path would need.

What §17 also showed, and what keeps this a proposal: on the twelve unobservable rows both halves
agree the event looks normal, and the dataset scores that as wrong on six of them. A "no trace"
answer needs a decision about what the right answer *is* before it needs a model improvement.

### 9.2 A prose-only training contract — PROPOSED, NOT BUILT (dated 2026-11-01)

§9 moved the diagnosis out of the model at inference time. This proposes moving it out of the
**training target** too, which is a different and larger change: §15.4 measured that the label
field is only 3 % of the supervised tokens, so the adapter spends almost all its gradient learning
prose while still being scored on a label it barely practised. **Nothing here is implemented.**
`llm/train_adapter.py`, `llm/dataset.py`, the prompts, `train_mixed.jsonl` and every hash are
untouched, and no adapter was trained for this section.

**The contract.**

1. **Supplied in the prompt, not generated:** `fault_id` and `verdict` (the classifier-plus-menu-rule
   decision), and the classifier's **top-2 with calibrated probabilities**. The model is told these
   are given and must not be restated differently.
2. **Loss masked to the prose:** `ranked_hypotheses`, `physical_reasoning`, `narrative`, plus
   follow-up answers. The supplied prefix is context, not target.
3. **Shuffled menu** in training and in evaluation, so the menu's order cannot become a feature.
4. **Tag-kind rejecting validator with one retry** on the report path
   (`llm/hybrid/tag_kind.py`), because §9.1's third item assumed a failure mode that 3B-7 could not
   find in 242 answers — see the correction at the end of this section.

**Counting, from the existing `train_mixed.jsonl` (tokenizer of the trained model; no new files):**

| quantity | value |
|---|---|
| supervised tokens today (report 573,061 + dialogue 71,067) | **644,128** |
| supervised tokens with `fault_id` and `verdict` masked | **609,697** |
| removed | **22,917 tokens, 5.35 %** (`fault_id` 13,774, `verdict` 9,143) |
| expected change in step time | **≈ 0** |
| events whose supplied label would be wrong if the cross-fitted prediction were used | **8 of 1,919 (0.42 %)** |
| events whose supplied label would be an abstention where the gold is a named fault | **254 of 1,919 (13.2 %)** |
| events where the supplied label differs from the gold at all | **262 of 1,919 (13.7 %)** |

* **Step time does not change, and saying otherwise would be false.** Masking changes which
  positions contribute to the cross-entropy, not the sequence length, so the forward and backward
  passes do the same work. The gain is in gradient quality, not speed; the only speed change would
  come from *shortening* the answer, which this contract does not do.
* **The label counts use the deduplicated 1,919 events, not the 2,880 rows.** The classifier is
  fitted per event (an event appears once in the report partition), and the 961 dialogue rows carry
  no label to supply.
* **13.2 % is the number to worry about.** Supplying the cross-fitted prediction hands the model an
  abstention on one event in eight where the truth is a named fault. A mask-and-supply contract
  trained on that would teach the model to write confident prose around "no listed fault" for
  events that do have one — the §17 lookalike failure, moved into the training set.

**Data rebuild steps, and which need teacher calls:**

| step | teacher calls |
|---|---|
| 1. add the supplied fields to each report row's prompt (the fields come from the cross-fitted classifier, which already exists) | **none** — programmatic |
| 2. mask the two fields in the answer text | **none** — programmatic |
| 3. shuffle the menu per row (seeded) | **none** — programmatic |
| 4. rebuild the manifest and hashes; re-run leak/split/open-set checks | **none** |
| 5. regenerate dialogue answers if their prompts change (they carry the event, so they may not need to) | **teacher calls**, ≈ 961 rows, ≈ 250–330 calls at the observed 3–4 rows/call — and only if step 1 changes a dialogue prompt |
| 6. train and evaluate | **none** — GPU, not teacher calls |

**Correction owed to §9.1, recorded here rather than edited into it.** Item 3 of §9.1 says the
adapter mislabelled a tag's kind in 21 of 242 answers. 3B-7 hand-read all 21 and reproduced the
40-character window the old rule read: **none is a mislabel.** Twenty are a neighbouring sensor's
name (`Stripper_Steam_Valve (XMV_50) and Stripper_Temperature (XMEAS_18)`), and the rest are a fault
name, another tag's own kind, or a downstream consequence. The new rule
(`llm/hybrid/tag_kind.py`, attribution-only, names masked) flags **0 of 242**, 0 of 58
counterfactual answers and 0 of the 21 retries. A rejecting validator stays in the contract as a
guard, but the failure mode it guards against is not one this adapter has been observed to have,
and no retry budget should be spent on it.

**What would make this proposal wrong.** If the 13.2 % abstention rate on supplied labels cannot be
reduced by supplying the *gold* label to the classifier's own errors (which is a dataset question,
not a model one), the contract trains the lookalike failure in rather than around it. Deciding that
needs an experiment this section does not run.
