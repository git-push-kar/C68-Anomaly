# Phase 2C — Follow-up dialogue dataset

**Branch:** `benchmark-upgrade` · **CPU only** · **no teacher calls**: every build
in this phase used the fake teacher. The GPU was busy with the Phase 2B teacher
job and no Ollama request was issued (`grep -c ollama` on the demo log: 0).

---

## 1. What this phase produced

| Path | What it is |
|---|---|
| [docs/FOLLOWUP_SPEC.md](../FOLLOWUP_SPEC.md) | the contract: 14 intents, TEMPLATE/LLM split, case mix, filters, leak rules, record format |
| [llm/followup/__init__.py](../../llm/followup/__init__.py) | package surface |
| [llm/followup/paraphrases.py](../../llm/followup/paraphrases.py) | 17 question labels × (40 train + 10 held-out) phrasings, false-premise templates, reply-shaped statements |
| [llm/followup/cases.py](../../llm/followup/cases.py) | case mix, per-case turn plans, clarification/refusal decisions, context rendering |
| [llm/followup/filters.py](../../llm/followup/filters.py) | answer filters (reuse + follow-up rules) |
| [llm/followup/teacher.py](../../llm/followup/teacher.py) | prompt, resume cache, retry, fake teacher, Ollama client (unused here) |
| [llm/followup/leak.py](../../llm/followup/leak.py) | leak check, split overlap, paraphrase disjointness |
| [scripts/build_followup_dataset.py](../../scripts/build_followup_dataset.py) | the builder |
| [scripts/inspect_followup_routing.py](../../scripts/inspect_followup_routing.py) | inspection (a): live routing measurement |
| [scripts/make_followup_fixture_events.py](../../scripts/make_followup_fixture_events.py) | synthetic event set in the Phase 2B schema, for CPU-only builds and tests |
| [tests/test_followup_filters.py](../../tests/test_followup_filters.py), [tests/test_followup_dataset.py](../../tests/test_followup_dataset.py) | 67 tests over every guardrail and case type |
| `outputs/evaluation/followup_routing.{md,csv,json}` | the routing artefact |

One line changed outside the new package: [`llm/response/router.py`](../../llm/response/router.py)
(`_hypotheses`), without which no clarification case could be built — see §9.

---

## 2. Inspection (a) — what the live path templates and what it sends to the model

`scripts/inspect_followup_routing.py` routes 50 realistic operator questions per
label through `RCAInference._dispatch_with_intent` — the exact call
`main.py:answer_followup` makes first — and also through `ResponseRouter` with the
proposed intents off (live) and on (benchmark).

### 2.1 Per label (n = 50 each)

| intended label | family | template | model (`None`) | clarification (flag on) | routes actually taken |
|---|---|---:|---:|---:|---|
| `sensor_values` | live intent | 50 | 0 | 4 | sensor_values 50 |
| `evidence_why` | live intent | 50 | 0 | 0 | evidence_why 50 |
| `system_sensors` | live intent | 50 | 0 | 0 | system_sensors 50 |
| `sensor_scores` | live intent | 50 | 0 | 0 | sensor_scores 50 |
| `detector_severity` | live intent | 50 | 0 | 0 | detector_severity 50 |
| `detector_scores` | live intent | 50 | 0 | 0 | detector_scores 50 |
| `severity` | live intent | 50 | 0 | 0 | severity 50 |
| `overview` | live intent | 46 | **4** | 4 | overview 46, model 4 |
| `sensors_inspect` | live intent | 50 | 0 | 0 | sensors_inspect 50 |
| `action` | live intent | 49 | **1** | 0 | action 49, model 1 |
| `overview_fallback` | live intent | 50 | 0 | 5 | overview_fallback 45, overview 5 |
| `hypothesis_comparison` | benchmark only | 16 | 34 | 1 | model 34, overview 12, overview_fallback 4 |
| `out_of_scope_refusal` | benchmark only | 1 | 49 | 0 | model 49, overview 1 |
| `clarification_reply` | benchmark only | 0 | 50 | 0 | model 50 |
| `valve_actuator` | free-form | 0 | **50** | 0 | model 50 |
| `causal_why` | free-form | 0 | **50** | 0 | model 50 |
| `mechanism` | free-form | 4 | **46** | 0 | model 46, evidence_why 4 |

### 2.2 By family

| family | labels | n | template | model | model share |
|---|---:|---:|---:|---:|---:|
| the 11 live intents | 11 | 550 | 545 | 5 | **0.0091** |
| free-form, no intent | 3 | 150 | 4 | 146 | **0.9733** |
| benchmark-only intents | 3 | 150 | 17 | 133 | 0.8867 |

### 2.3 Verdict on the brief's hypothesis

> "confirm whether only valve_actuator_status-type questions and free-form
> questions reach the LLM"

**Confirmed, with one correction and one caveat.**

* There is **no valve intent in the live dispatcher**. The 11 live intents are all
  template intents; valve/actuator questions match none of them and go to the
  model **50/50**. Same for free-form causal and mechanism questions (50/50 and
  46/50).
* The **11 live intents are template-answered, but not perfectly**: 5 of 550
  questions (0.9%) escape the keyword sets and reach the model. Four are
  `overview` phrasings that start with "Summarise…" — the gate looks for
  `summary`, so the British spelling is an unhandled synonym — and one is an
  `action` phrasing. So the boundary is *keyword presence*, not intent class:
  anything that fails to mention sensor/score/evidence/cause/action/severity
  vocabulary reaches the model, including off-domain and malformed questions.
* The other direction leaks too: 4/50 mechanism questions say "evidence" and are
  claimed by `evidence_why`. This is why the builder routes **every** candidate
  question through the dispatcher and rejects paraphrases that route as the wrong
  branch (`paraphrases_skipped` per turn), instead of trusting the label.

Consequence for the dataset: training targets the **free-form population**, which
is 97.3% model-routed; the 11 template intents are lookup-path items and appear in
training only as the 5% format sample (§4).

---

## 3. Inspection (b) — `grounded_qa` format versus what the router receives

The stored rows and the live call site do not see the same string:
`main.py:answer_followup(event_id, user_question)` passes **only the operator
ask**, while `grounded_qa.question` carries the whole telemetry preamble and ends
with `Operator question: …`.

| corpus | measure | n | rows with telemetry preamble | model-routed | where they land |
|---|---|---:|---:|---:|---|
| `data/llm_1/train` | `grounded_qa:full` (verbatim) | 1,008 | 1,008 | 0 | evidence_why 864, sensor_values 144 |
| `data/llm_1/train` | `grounded_qa:ask` (live format) | 1,008 | 1,008 | 0 | 6 intents, 144 each |
| `data/llm_1/{val,test}` | `grounded_qa:full` | 56 each | 56 | 0 | evidence_why 48, sensor_values 8 |
| `data/llm_1/{val,test}` | `grounded_qa:ask` | 56 each | 56 | 0 | 6 intents, 8 each |
| `data/llm*/train` | `followup:ask` | 576 each | 0 | 0 | overview 288, severity 144, sensors_inspect 144 |

Three findings:

1. **Routed verbatim, the preamble hijacks the route**: 85.7% of `grounded_qa`
   rows collapse into `evidence_why`, because the preamble itself says "why was
   this flagged". The stored format therefore measures the template path against a
   string the live system never passes.
2. **Routed as the operator ask, they spread evenly and still never reach the
   model** (0/1,008). The existing grounded_qa rows exercise the template path
   only; they contain no training signal for the branch this phase targets.
3. **Row count on this checkout:** the grounded_qa rows that exist are 1,120 in
   `data/llm_1` (1,008 + 56 + 56). `data/llm` (v0) holds 720/40/40 rows here and
   **no** grounded_qa rows at all, although its `split_metadata.json` records
   3,168 samples for v0 — the committed arrow files are a subset of the metadata's
   claim. The brief's "2,112 grounded_qa rows in data/llm v0" is therefore not
   reproducible from this checkout; the analysis above uses the rows that exist.

The hand-written `followup` rows are the only pre-existing operator-ask data, and
they cover three intents (`overview`, `severity`, `sensors_inspect`) — no valve, no
causal, no mechanism question. That gap is what Phase 2C fills.

---

## 4. Inspection (c) — the response layer

| Piece | What it does | Used by the dataset as |
|---|---|---|
| `FollowupAnswer` ([schemas.py](../../llm/response/schemas.py)) | intent + answer + table/params + ≤6 citations + ≤1 clarifying question + ≤3 suggestions | the record shape a turn must satisfy |
| `validators.py` | citation resolvability, number support, fault-id scope, tag grounding | shared rules; the follow-up filters extend them |
| `router.py` | keyword gates for the 11 live intents, proposed intents behind `enable_proposed`, 6 clarification triggers | the **decision** for clarifications (`clarifying_question`, `clarification_trigger`, options) |
| `lookup_answers.py` | per-intent deterministic lead + `detail` document | the answer text of template turns, verbatim |
| `store_access.py` | key-by-key access, `UnknownKeyError`, contradiction checks | the **decision** for refusals and false premises |
| `renderer.py` | tables and markdown blocks per table id | `table` / `table_params` per turn |

Two defects surfaced and one was fixed:

* **`ambiguous_hypotheses` fired on every turn.** Phase 2B's targets leave
  `confidence: None`; `_hypotheses` coerced it to `0.0`, so any two-hypothesis
  report looked like a 0.0-vs-0.0 tie and the clarification trigger fired on every
  question. Fixed: only a *recorded* confidence takes part in the margin (§9).
  Without this, the `ambiguous_question` and `clarification_reply` cases could not
  be built at all — the clarification would fire everywhere, not where intended.
* **The KB's remediation text opens with "CRITICAL SAFETY INTERLOCK: …".** On the
  lookup path that is the live answer and is left alone. But the 5% template
  sample trains on lookup text, so those labels are skipped for the sample
  (`template_excluded: ["action:severity_claim", …]`) rather than teaching a
  severity claim to the adapter.

---

## 5. Task 1 — the taxonomy, as finalised

`docs/FOLLOWUP_SPEC.md` §2 carries the full table. Summary:

* **14 intents**: the 11 live ones, plus `hypothesis_comparison`,
  `out_of_scope_refusal`, `clarification_reply` — the last three are
  benchmark-only and stay behind `ResponseRouter(enable_proposed=True)`; the live
  flag is **off** and nothing in this phase changes `main.py`.
* **TEMPLATE (11)**: `sensor_values`, `evidence_why`, `system_sensors`,
  `sensor_scores`, `detector_severity`, `detector_scores`, `severity`, `overview`,
  `sensors_inspect`, `action`, `overview_fallback`. Answered by
  `lookup_answers`, never sent to a model, **scored on the lookup path**.
* **LLM (3)**: `hypothesis_comparison`, `out_of_scope_refusal`,
  `clarification_reply`.
* **Three free-form case types have no intent** (`valve_actuator`, `causal_why`,
  `mechanism`): the live dispatcher has no rule for them, so the schema's intent
  field is `null`, no table is attached, and prose is grounded through citations.
  Adding a 15th intent would fix the shape; it is out of this phase's taxonomy and
  is listed as a known gap.
* **Training scope**: LLM-routed intents and cases only. Template intents are
  excluded from adapter training except a **5% template sample** (configurable,
  including to zero) that keeps the model's voice aligned with the lookup answers
  it sits beside. The sample uses lookup text verbatim — zero teacher cost.

---

## 6. Task 2-6 — the builder

**Dialogues, not turns, are the unit**: one event → 2-4 turns, where turn 1 is
answered after the automatic report (the report is context, not a turn).

| Case type | Route | Share | Plan | Exercises |
|---|---|---:|---|---|
| `valve_actuator` | LLM | 0.22 | valve → valve | valve/actuator status |
| `causal_why` | LLM | 0.18 | causal → mechanism | free-form causal explanation |
| `mechanism` | LLM | 0.12 | mechanism → causal | physical mechanism |
| `hypothesis_comparison` | LLM | 0.12 | comparison → valve | differential diagnosis (benchmark intent) |
| `out_of_scope` | LLM | 0.06 | out_of_scope → valve | refuse what the store does not hold |
| `false_premise` | LLM | 0.08 | false_premise → valve | wrong sensor/fault/value, contradicted by the store |
| `repeated_question` | LLM | 0.06 | valve → repeat → valve | same question twice, answer must differ |
| `ambiguous_question` | LLM | 0.06 | ambiguous → valve | router clarification trigger |
| `clarification_reply` | LLM | 0.05 | ambiguous → reply | reply to a pending clarification |
| `template_sample` | **TEMPLATE** | 0.05 | template → template | format anchor, lookup text, no teacher |

Shares are per-dialogue and normalised; `--mix` overrides them.

**Deterministic before any model call** (per turn): `intent`/`resolved_intent`,
`answerability`, `table` + `table_params`, `citations` (resolved store keys only),
`clarifying_question`/`clarification_trigger`/options, `refusal` (from
`StoreAccess`: `key_missing`, `contradicted`, `off_domain`, plus alternatives each
routed through the dispatcher first), `premise` (kind, claim, verdict, store key,
actual value), `suggested_followups` (router-verified, non-duplicate, not the
question just asked), `answer_source`. The teacher writes **one** field: `answer`,
1-4 sentences. Template turns take their text from `lookup_answer` and make no
teacher call.

**Question sources**: seeded templated paraphrases, 40 train + 10 held-out per
label (17 labels). Train dialogues draw only from the 40; val and test only from
the 10, so a test phrasing never appears in train. Slot-filled questions
(comparison, false premise, missing key) are built from the item's own store, so
no fault outside the menu can appear. The teacher-paraphrase hook is **off by
default** and the fake teacher is what tests use.

**Answer filters** (rejections, never repairs): the Phase 2B set unchanged
(`number_not_in_store`, `unknown_tag`, `sensor_not_grounded`, `absolute_wording`,
`severity_claim`, `fault_out_of_scope`), plus `empty_answer`, `answer_length`,
`answer_is_question`, `clarify_text_repeated`, `near_duplicate` (≥ 0.6 of the
answer's 4-grams already in the previous assistant turn), `confidence_claim`,
`refusal_missing`, `refusal_no_alternative`, `names_removed_fault`,
`table_reference`, `valve_flow_mislabel`, `event_id_leak`. Clarification turns
also enforce answer-first, one question at most, and never re-asking (a trigger on
a repeat/reply turn is recorded as `clarification_suppressed`).

**Cache and provenance** as Phase 2B: append-only JSONL keyed by
`sha256(model|seed|temperature|prompt)`, retry once with the rejection codes in
the prompt, and per turn `prompt_sha256`, `prompt_cache_hit`, `answer_attempts`,
`teacher_model`, `teacher_seconds`, `filter_codes`, `rejected_answer`.

**Leak check** on every context and question, aborting on a hit: event ids,
dataset item ids, split/fold names, run numbers, seed, filenames, and any fault
number outside the menu that item shows (for an open-set item, the removed fault).
The context carries `sha256(event_id)[:6]` as the incident reference — stable, and
it encodes none of the split/fault/run structure the benchmark ids carry.

**Splits** are Phase 2B's, and the build proves disjointness before the first
teacher call: no `(source_file, fault, run)` key and no event id in two splits, no
duplicate inside a split, every built key assigned to that split by
`data/splits_v2.json`, and faults 3/9 excluded from train (counted, not silently
dropped).

---

## 7. Build result (fake teacher, CPU)

The Phase 2B event set is not on disk on this checkout (`outputs/benchmark_v2/events`
is git-ignored and absent), and regenerating it is blocked by a detector-artefact
mismatch — see §11. The build was therefore verified end to end on
`outputs/benchmark_v2/followup_fixture_events`: a synthetic event set in the
Phase 2B schema (39 train / 37 val / 37 test events, faults 1-20, normal runs,
open-set candidates), so every code path runs with real store keys and real
router decisions.

```
python scripts/build_followup_dataset.py \
  --events outputs/benchmark_v2/followup_fixture_events \
  --out outputs/followup_fixture_demo/data --teacher fake --audit 100 --seed 20261030
```

| split | events | dialogues | turns | teacher turns | accepted | rejected |
|---|---:|---:|---:|---:|---:|---:|
| train | 39 | 37 | 84 | 84 | 84 | 0 |
| val | 37 | 37 | 89 | 87 | 89 | 0 |
| test | 37 | 37 | 88 | 86 | 88 | 0 |

Train skipped 2 events: faults 3 and 9. Cases per split, from the manifest:

| case type | train | val | test |
|---|---:|---:|---:|
| `valve_actuator` | 14 | 12 | 10 |
| `causal_why` | 5 | 3 | 7 |
| `mechanism` | 6 | 8 | 5 |
| `hypothesis_comparison` | 2 | 3 | 2 |
| `out_of_scope` | 1 | 1 | 2 |
| `false_premise` | 2 | 1 | 3 |
| `repeated_question` | 3 | 2 | 1 |
| `ambiguous_question` | 3 | 3 | 2 |
| `clarification_reply` | 1 | 3 | 4 |
| `template_sample` | 0 | 1 | 1 |

`case_substituted` fired 3 times (an event that cannot carry the drawn case falls
back to `valve_actuator`, recorded). Every one of the ten case types is exercised
across the run; the train column is 0 for `template_sample` only because the
fixture has 37 train dialogues and the case is drawn at 5% — at full size that is
~97 of 1,939. Split integrity: zero overlap, three pairwise comparisons, 0
duplicates. Paraphrase disjointness: 17/17 labels disjoint (40 vs 10).

**Audit sheet** (`audit_sample.csv`, 100 rows, round-robin across case types,
scoring columns blank): `dialogue_id, case_type, split, turn, question, answer,
answer_source, accepted, intent, table, citations, clarifying_question,
refusal_reason` + `correct, grounded, refusal_appropriate, not_repetitive, useful,
notes`. The judgement columns are empty on purpose — a builder-filled score would
be a quality claim nothing measured.

---

## 8. Task 7 — mix proposal and cost (proposal only, nothing fixed)

| | value |
|---|---|
| Phase 2B train report items | 1,939 |
| follow-up dialogues from the same train events | 1,939 (one per event) |
| teacher turns per dialogue (measured, fake teacher) | 2.32 |
| projected teacher calls, full train | **4,489** |
| at the measured 10.9 s/call | **13.6 h serial** |
| with one retry per 12% rejection | 15.2 h serial (5,000 calls) |
| template sample share | 5% of dialogues, 0 calls |
| LLM-routed share of built turns | ~95% (the sample is the only lookup share) |

**Recommended ratio: 1 report item : 1 dialogue** — i.e. pair every Phase 2B report
item with one dialogue drawn from the same event. That is 1 report item against
2.3 model answers per epoch; the report task stays the headline metric while the
follow-up branch gets enough volume to learn refusal/clarification behaviour.
**2:1** (one dialogue per two report items) is the conservative option if report
label accuracy regresses; **1:2** would put follow-up answers at ~4.6× the report
volume and is not recommended.

The measured LLM-routed share is a property of the *balanced probe bank*
(50 per label), not of operator traffic: 97.3% of free-form questions reach the
model, 0.9% of template-intent questions do. Turning that into a traffic share
needs production question logs, which this phase does not have; the mix above is
therefore justified by the task split (report vs follow-up), not by a traffic
estimate.

Cost at 13.6-15.2 h serial is **over 30 minutes**, so the real run is not started
here (and was forbidden in this phase). Nothing in the repo starts it by default:
`--teacher` defaults to `fake`.

---

## 9. Change log

| Change | Why |
|---|---|
| `llm/response/router.py:_hypotheses` skips hypotheses whose `confidence` is `None` | a missing confidence was coerced to `0.0`, so two-hypothesis reports always looked tied and `ambiguous_hypotheses` fired on every question. Only a recorded confidence can produce a margin. Behaviour for Phase 2B targets is unchanged (they carry no confidence, so they now simply never trigger the tie rule). |
| Template-sample labels are skipped when the lookup text carries a severity or absolute claim (`TEMPLATE_BLOCKED`) | the KB remediation for fault 15 opens with "CRITICAL SAFETY INTERLOCK: …"; training the 5% sample on it would teach a severity claim. Recorded per turn as `template_excluded`. |
| `scripts/inspect_followup_routing.py` measures all 17 labels and reports a by-family summary | the first version measured only the 14 intents, which left the three free-form case types — the ones the live path always sends to the model — unmeasured, and a pooled average over a balanced bank implies a traffic share that does not exist. |
| `scripts/build_followup_dataset.py` reads the family block for its mix proposal | same reason: report `template_bank_model_share` and `free_form_bank_model_share` separately instead of one misleading pooled number. |

---

## 10. Verification

| Check | Command | Result |
|---|---|---|
| Full suite | `python -m pytest -q` | **261 passed, 1 failed** |
| Follow-up + router tests | `python -m pytest -q tests/test_followup_filters.py tests/test_followup_dataset.py tests/test_response_router.py` | **94 passed** |
| Routing artefact | `python scripts/inspect_followup_routing.py` | 17 labels × 50; families 0.0091 / 0.9733 / 0.8867 |
| Fake-teacher build | builder command in §7 | 111 dialogues, 261 turns, 257 teacher turns, 0 rejections |
| Split overlap | in-build | zero overlap, 0 duplicates |
| Paraphrase hold-out | in-build + `check_paraphrase_disjointness()` | 17/17 disjoint |
| Leak check | in-build, every context and question | 0 violations, build aborts on a hit |

The one failure is **pre-existing and environmental**:
`tests/test_splits_v2.py::test_committed_file_matches_the_builder` compares
`render(build())` with the committed `data/splits_v2.json`, and the rendered
document embeds the absolute checkout path — the committed file says
`…/Desktop/anomaly`, this checkout is `…/Desktop/C68-Anomaly`. Only that path
differs. It is unrelated to this phase (no split file, no builder code for splits
was touched) and is left alone rather than papered over by rewriting a committed
benchmark artefact.

New tests (68), all passing:

* **filters** (18): grounded answer accepted; invented number, severity claim,
  confidence number, absolute wording, table reference, answer-as-question,
  length, repeated clarifying question, removed-fault naming (and the allowed
  denial inside a refusal), valve/flow mislabel, event-id leak, refusal that does
  not decline, refusal without an answerable alternative, near-duplicate of the
  previous turn rejected while a different answer on the same topic passes, and
  the 4-gram overlap rule's symmetry.
* **router** (1): a report whose hypotheses carry no confidence does not trigger
  `ambiguous_hypotheses` (neither all-missing nor one-missing), which is the
  regression the `_hypotheses` change in §9 fixes.
* **dataset** (49): every case type builds end to end with the fake teacher (10
  parametrised); the deterministic fields resolve in the store (10 parametrised);
  the template sample answers are the lookup text; open-set dialogues never name
  the removed fault (4 parametrised) and stay grounded; false premises are checked
  against the store (3 parametrised) and a confirming answer is rejected; the
  refusal is required and offers a routed alternative, and a bare refusal is
  rejected; a clarification is asked at most once per dialogue and never re-asked
  on a reply; a repeated question is not answered in the same words (and a
  verbatim repeat is caught by the filter); the leak check aborts on an event id,
  a split name, a run number and a fault outside the menu, while the context stays
  clean with an opaque reference; split overlap is detected and the fixture set is
  clean; the hold-out is disjoint and train/val/test draw from their own halves;
  the same seed reproduces a dialogue; the cache resume skips the teacher; the mix
  override changes the distribution; the audit sheet is stratified; the full build
  writes every split and the manifest.

---

## 11. Gate

| Gate item | Status | Evidence |
|---|---|---|
| Routing table from the live router shown | **pass** | §2, `outputs/evaluation/followup_routing.md` |
| Spec complete | **pass** | `docs/FOLLOWUP_SPEC.md` — 14 intents, TEMPLATE/LLM, 5% sample, filters, leak, record format, gaps |
| Dataset builds with the fake teacher | **pass (fixture events)** | §7 — 111 dialogues, all 10 case types |
| Leak check clean | **pass** | 0 violations; build aborts on a hit |
| Split overlap clean | **pass** | zero overlap, 0 duplicates, faults 3/9 excluded from train |
| All case types covered by tests | **pass** | 10 parametrised case types + 11 guardrails |
| Paraphrase hold-out verified | **pass** | 17/17 disjoint, 40 train vs 10 held-out |
| Counts and cost estimate reported | **pass** | §7 counts, §8 cost (4,489 calls, 13.6-15.2 h) |
| Full test suite passes | **fail (1 pre-existing)** | 261 passed, 1 environmental failure in `tests/test_splits_v2.py` (§10) |
| No real teacher called | **pass** | `--teacher fake`; no Ollama request issued |

---

## 12. Limitations and what comes next

1. **The full-size build has not been run.** The Phase 2B events are absent, and
   regenerating them is blocked: `outputs/anomaly_detector/model.pt` is
   `hidden=32/latent=16` while `configs/config_a5000.yaml` and
   `outputs/anomaly_detector/config.json` declare 128/64, so the loader fails
   before a single window is scored. Which architecture produced the frozen
   threshold in `outputs/anomaly_detector/threshold.json` has to be settled first;
   that is a Phase 2B artefact repair, not a Phase 2C change. Counts in §7 are
   therefore from the schema-faithful fixture set, and §8 is a projection.
2. **Real teacher quality is unmeasured.** The fake teacher proves the pipeline,
   the filters and the case coverage; it says nothing about qwen3:30b's answers.
   The audit sheet is ready for that, and the cost is 13.6-15.2 h serial.
3. **Free-form turns carry no intent and no table** (§5). Acceptable for this
   phase; a 15th intent would make the record uniform.
4. **Two keyword gaps stay unfixed**, because fixing them changes the live router:
   "summarise" (British spelling) misses `overview`, and `versus` collides with
   `hypothesis_comparison`. Both are recorded in the spec as known gaps.
5. **The template sample teaches lookup formatting**, including its quirks (a
   `detail` document with markdown tables). It is 5% by weight and can be set to
   zero with `--mix '{"template_sample": 0}'` if the adapter's voice drifts.

Reproduction:

```
python scripts/inspect_followup_routing.py
python scripts/make_followup_fixture_events.py
python scripts/build_followup_dataset.py --teacher fake \
  --events outputs/benchmark_v2/followup_fixture_events \
  --out outputs/followup_fixture_demo/data --seed 20261030
python -m pytest -q tests/test_followup_filters.py tests/test_followup_dataset.py
```
