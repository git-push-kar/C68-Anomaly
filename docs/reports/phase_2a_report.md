# Phase 2A report — response layer (schemas, validators, renderer, router wrapper, store access, severity)

**Repo:** `C:\Users\Admin\Desktop\anomaly` · branch `benchmark-upgrade` · CPU only; no
GPU job, no training, no adapter load. `llm/inference.py` is **unmodified**.
Phase 0 patch approved as the precondition.

Phase 2A is a *contract* layer. It does not generate anything: it decides what a
model answer is allowed to contain, renders tables only from stored values, and
scores nothing on test data.

---

## 1. Summary

1. New package [`llm/response/`](../llm/response) with 9 modules: `schemas`,
   `severity`, `allowed_set`, `store_access`, `renderer`, `router`,
   `validators`, `lookup_answers`, `__init__`.
2. **The live path is untouched.** `RCAInference._dispatch_with_intent` still
   decides every follow-up; the router mirrors its 11 keyword gates in the
   dispatcher's own priority order, and a test asserts the mirror equals the live
   result on 14 live-format questions. If the two drift, that test fails.
3. **Every table cell is a citation key**, resolved through `StoreAccess`; an
   unknown key raises. Fidelity is asserted per cell in tests, not per table.
4. **Severity is code-computed and is not a schema field.** `build_report` keeps
   parsing a payload that carries `severity`, records `severity_from_model`, and
   overwrites the value with the computed assessment.
5. **One tag set** (`allowed_set`) accepts canonical form, alias and display
   name, so `XMV_52`, `XMV 11`, `xmv_11` and the condenser flow display name are
   all legitimate — and `XMV_99` is not.
6. **Tests: 155 passed, 0 failed** (67 of them new in Phase 2A). Two real bugs
   were caught by the tests written here (a shadowed loop variable that computed
   severity for the wrong hypothesis, and a trigger check that let thread history
   fake a missing trigger).
7. **One owner decision needed:** the proposed severity rule and the production
   rule (`evidence/process_limits.compute_deterministic_severity`) **disagree on
   a real stored event** (High vs Low, §5). Both are code; only one can be
   quoted.

---

## 2. Changes (commit hashes)

| Commit | Change |
|---|---|
| `7659a80` | `llm/response/` — schemas, allowed set, store access, severity, renderer, router, validators, lookup answers |
| `12f4a69` | `scripts/make_response_fixtures.py`, `tests/fixtures/response/*.json`, `tests/test_response_renderer.py` |
| `49c1c43` | `tests/test_response_router.py` (mirror + triggers + flag) |
| `fac610c` | `tests/test_response_validators.py`, `tests/test_response_severity.py` |
| _(this commit)_ | `docs/RESPONSE_SPEC.md` §2.2 status note, this report |

No file outside `llm/response/`, `tests/`, `scripts/make_response_fixtures.py`
and `docs/` was modified.

---

## 3. Inspection findings (tasks a–d)

### a. Event store schema — what is stored per event

`outputs/events.db` (489 rows):

```sql
CREATE TABLE events (event_id TEXT PRIMARY KEY, start_time TEXT, detection_time TEXT,
  end_time TEXT, max_score REAL, mean_score REAL, severity TEXT,
  fault_label INTEGER, event_json TEXT, report_json TEXT, evidence_json TEXT, created_at TEXT);
CREATE TABLE followups (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL,
  question TEXT, answer TEXT, created_at TEXT, FOREIGN KEY (event_id) REFERENCES events(event_id));
```

Consequences for the layer: `evidence_json` is the authoritative store (sensors,
detector blocks, timeline); `report_json` is free-text model output and is
**never** a cell source; `severity` at row level is the value written by
`compute_deterministic_severity` at event time; `followups` stores question and
answer text only — no intent, no citations, so thread history can suppress a
repeat clarification but cannot reconstruct what was cited.

### b. The live dispatcher — 11 intents, priority order

`_dispatch_operator_intent` (`llm/inference.py:445`) is a thin wrapper returning
text only; `_dispatch_with_intent` (`llm/inference.py:641`) returns
`(intent, answer)`. Branch order **is** priority:

`sensor_values` (726) → `evidence_why` (754) → `system_sensors` (780) →
`sensor_scores` (801) → `detector_severity` (817) → `detector_scores` (830) →
`severity` (846) → `overview` (864) → `sensors_inspect` (884) → `action` (894) →
`overview_fallback` (913).

Reused, not rewritten: `lookup_answers.lookup_answer` calls
`_dispatch_with_intent` and keeps its output as `detail`, writing only a 1–2
sentence lead that fits the schema. `VERIFIED` — the mirror in
`llm/response/router.py` matches all 11 gates.

### c. What the live path sends the LLM for non-template questions

`RCAInference.answer_followup` (`llm/inference.py:351`): a fixed system prompt
(`inference.py:366-386`), then `_render_event_context(event, report)`, then a
scripted assistant turn, then **the last 2 history messages**, then the question
(`inference.py:413-431`). There is **no schema, no citation protocol and no
table contract** in that prompt — the answer is free-form markdown, repaired
afterwards by `format_followup_answer` (`inference.py:592`). That is precisely
why this layer exists: the contract has to be enforced after the fact.

### d. Consumers of reports and follow-ups

* `api/server.py:145` attaches `followups` to the event payload from
  `event_store.get_followups`; `api/server.py:149-152` `POST
  /api/events/{id}/followup` → `app.answer_followup`.
* `ui/app.py:220` reads `ev["report"]`; `ui/app.py:311` calls
  `app.answer_followup(selected, q)`.
* `main.py:394` `answer_followup` routes: `prefer_llm` bypass →
  `_dispatch_with_intent` (`main.py:427`) → adapter → guardrail.

Both consumers take **plain text**, so the renderer emits markdown plus a UI dict
(`title`, `meta`, `blocks`, `citations`, `type`) and needs no consumer change.

---

## 4. What each task produced

| # | Deliverable | Where | Note |
|---|---|---|---|
| 1 | `Report` / `FollowupAnswer` models, enums from `scripts/fault_ids.py`, tags from `scripts/tags.py`, 1–4 sentences, ≤3 suggestions, ≤1 clarifying question, no severity field | `schemas.py` | `ReportBody` is the model-authored part (`extra="forbid"`); `Report` adds `event_id`/`severity`/`table`/`citations`/`source` |
| 2 | Code-computed severity | `severity.py` | Rule + thresholds in §5 — **proposed, not tuned** |
| 3 | One shared allowed set (canonical, alias, display name) | `allowed_set.py` | Used by `validators` and available to the Phase 2B evaluator |
| 4 | Typed citation-key accessor, unknown key → error | `store_access.py` | 40+ key forms; `UnknownKeyError` carries a hint listing what *is* available |
| 5 | `render_report` / `render_followup` → markdown + UI dict; 7 table ids | `renderer.py` | `none, evidence, hypotheses, detectors, timeline, comparison, actions`; each column declares a format spec, so a cell is a pure function of the stored value |
| 6 | Router wrapper: candidates, answerability, 4 clarification triggers, 3 proposed intents behind a flag | `router.py` | `ambiguous_hypotheses` margin 0.15 (configurable), `unclassified`, `ambiguous_question`, `context_not_in_store` |
| 7 | Validators with stable codes | `validators.py` | 12 codes, one test each |
| 8 | Deterministic lookups reusing live templates | `lookup_answers.py` | Calls the dispatcher; does not duplicate a template |
| 9 | Tests | `tests/test_response_*.py` | 67 tests; router tests use live-format questions only |

Answerability is one of `deterministic | llm | clarify | refuse`; with the flag
off, `refuse` is unreachable, which is intended — the refusal intent is not live.

---

## 5. Severity rule — PROPOSED, needs approval

Inputs are three kinds of stored evidence plus the knowledge base. No test data
was consulted; the numbers are chosen to be monotone and interpretable.

```
score = 0.35·kb_base + 0.30·detector + 0.20·sensor + 0.15·persistence

detector   = clamp((max_ratio − 1.0) / 1.5)          max_ratio = max(LSTM score/threshold, CVA score/threshold)
sensor     = clamp((max|z| − 2.0) / 4.0)             over the event's top anomalous sensors
persistence= 0.6·clamp(n_windows/20) + 0.4·sustain   sustain = mean_score / max_score
kb_base    = KB severity of the diagnosed fault      low 0.25 · medium 0.50 · high 0.75 · critical 1.00
             (open set → medium; normal transient → low)

cuts:  score ≥ 0.78 critical · ≥ 0.55 high · ≥ 0.30 medium · else low
```

Three modifiers, each with a stated reason:

1. **KB floor.** The level is raised to the KB level of the diagnosed fault
   family — the family's physics is a calibrated prior. A *short or decaying*
   event (`n_windows < 5` or `sustain < 0.70`) lowers that floor by one step, so
   a two-window blip cannot inherit `critical`.
2. **Escalation by one step** when both detectors are alarmed **and**
   `persistence ≥ 0.6` **and** `sensor ≥ 0.5` — i.e. corroboration, not a single
   large ratio.
3. **Normal transients cap at medium**; nothing fault-justified was diagnosed.

Observed behaviour on the three fixtures: classified → High (0.43), unclassified
→ Medium, normal excursion → Low.

**The disagreement you need to rule on.** `compute_deterministic_severity`
(`evidence/process_limits.py:74`, the production rule) wrote `low` for the
`ANOM-0488` event; the proposed rule says `high`, because that rule has no
evidence envelope in the event for the cooling loop but reads the fault family's
`critical` KB base and the event's 2 windows. Both are code, both are defensible,
and the follow-up now discloses the divergence in one clause ("The event store
carries LOW from the older rule"). Three ways forward:

* (a) adopt the proposed rule and stop reading `evidence.severity` anywhere —
  one number, and the KB floor disappears from the stored event;
* (b) keep `compute_deterministic_severity` authoritative and demote this module
  to a cross-check that scores agreement, not severity;
* (c) merge them: use `compute_deterministic_severity` as the `kb_base`
  component when present, and keep the detector/persistence terms as the
  response layer's adjustment.

I recommend (c) only if you want the KB floor kept; otherwise (b), because two
authoritative severity rules in one product is a defect, not a feature.

---

## 6. Commands and outputs

```
> python -m pytest -q
155 passed in 5.17s

> python -m pytest tests/test_response_router.py tests/test_response_renderer.py \
    tests/test_response_validators.py tests/test_response_severity.py -q
67 passed in 3.52s

> python scripts/make_response_fixtures.py
classified ANOM-0488 sensors 5
unclassified ANOM-0481 sensors 5
normal_excursion ANOM-0489 sensors 5
```

Re-running the generator reproduces the fixtures byte for byte.

Rendered report and follow-up are in
[phase2a_examples.md](../../outputs/evaluation/phase2a_examples.md); stderr-free
reproduction:

```python
import json; from pathlib import Path
from llm.response import StoreAccess, build_report, render_report
ev = json.loads(Path("tests/fixtures/response/classified.json").read_text())
report, result = build_report(ev["report"], ev)      # result.ok == True
print(render_report(report, StoreAccess(ev)).markdown)
```

| Detector | Score | Threshold | Margin | Status |
|---|---|---|---|---|
| LSTM reconstruction error | 0.6352 | 0.69 | 0.92x | below threshold |
| CVA Q-statistic | 248.2991 | 239.75 | 1.04x | ALARMED |
| Overall event score | 0.7577 | — | — | ALARMED |

Every cell above is `format_cell(store.resolve(key), spec)`; the test recomputes
each one and compares, so a fabricated cell fails the suite rather than the
review.

---

## 7. Tests

| File | Tests | What it locks |
|---|---|---|
| `tests/test_response_renderer.py` | 15 | golden render of 3 fixtures; per-cell fidelity; synthetic label; follow-up renders through the same tables; lookup answers stay valid |
| `tests/test_response_router.py` | 26 | mirror equals `_dispatch_with_intent` on 14 live questions; priority (`evidence_why` > fallback, `system_sensors` ≠ `sensor_values`); 4 triggers; margin configurability; flag off/on; history suppression |
| `tests/test_response_validators.py` | 17 | one failure case per rule (12) + positive controls: valid answer, rounded number accepted, fixture report valid, severity matches root cause |
| `tests/test_response_severity.py` | 9 | rule version marked proposed; monotone in margin; KB floor; step-down for short events; normal cap; open-set base; store keys; missing detector blocks; lookup uses the computed level |
| **Phase 2A total** | **67** | **all pass** |
| Whole suite | 155 | all pass |

Two bugs the tests caught, both fixed in this phase:

* `validators.build_report` passed `fault=` but the hypothesis loop shadowed it,
  so severity was computed for the **last hypothesis** while the report said the
  first (FAULT_14 vs FAULT_15).
* `validate_followup` asked the router with history, so an already-asked
  question made the trigger look absent (`clarify_not_triggered` fired
  spuriously). Trigger presence is now a property of the event; history only
  adds `clarify_reasked`.

---

## 8. Gate table

| Gate item | Status | Evidence |
|---|---|---|
| Tests pass | **PASS** | 155 passed, 0 failed |
| Renderer table fidelity equals the store on fixtures | **PASS** | `test_every_table_cell_traces_to_the_store` (all 3 fixtures, every cell), `test_detector_table_matches_store_numbers` |
| Validator rejects each listed failure | **PASS** | 12 rules × 1 dedicated failing test; `test_valid_answer_passes` as the control |
| One rendered report and one rendered follow-up shown | **PASS** | §6 and `outputs/evaluation/phase2a_examples.md` |
| `llm/inference.py` behaviour unchanged | **PASS** | file not in any Phase 2A commit; `git log -- llm/inference.py` still ends at `be05f1f`, which is an ancestor of `master` |
| Router uses the live dispatcher, not a rewrite | **PASS** | `lookup_answers` calls `_dispatch_with_intent`; mirror test |
| Router tests use live-format questions | **PASS** | 14 questions, no dataset prompt in the file |
| Severity not a model field | **PASS** | `ReportBody` has no `severity`; `extra="forbid"`; `severity_from_model` code |
| Severity rule proposed, not tuned | **PASS** | rule version `severity-rule-v1-proposed`; no test/validation data read |
| Validator number rule is tight | **FAIL (by design)** | see deviation 3 — integers 1–22 are always accepted |

---

## 9. Deviations from the brief

1. **`comparison` table exists but no intent is routed to it yet.** The
   `hypothesis_comparison` intent is behind the flag, so the table is only
   reachable if a caller names it; `table_allowed_for` permits it for that intent
   only. Building it now avoided a schema change later.
2. **`out_of_scope_refusal` has an empty keyword set.** Refusal is defined by
   *absence* (no intent matched and no in-domain token), not by a phrase list,
   which is what the spec asked for; it also means off-domain questions phrased
   with a process word are answered rather than refused.
3. **The number check has a deliberate hole.** `LABEL_INTEGERS` accepts
   `1..22` (fault numbers) and `{11, 41, 52}` (tag indices) unconditionally, so
   "17 windows" passes even if the event had 2. Tightening it needs a decision:
   numbers that are fault labels are not measurements, but a wrong count is also
   not a fabrication. Suggested fix: accept a small integer only when it equals a
   stored count (`event.n_windows`, `count.top_sensors`, `count.timeline`) or
   appears as a fault id (`FAULT_15`, "Fault 15"). Not implemented, because it
   changes what "correct" means and that is your call.
4. **`severity.stored` and `severity.computed` are both exposed.** I chose
   disclosure over silent override in the lookup answer; §5 asks which rule wins.
5. **No API/UI wiring.** The brief stopped at the layer; `api/server.py` and
   `ui/app.py` still receive plain text and were not modified.

---

## 10. Open questions (owner)

1. **Which severity rule is authoritative** — §5 (a), (b) or (c)? This decides
   whether `severity_accuracy` is retired outright or replaced by an agreement
   metric.
2. **Number rule** — tighten as in deviation 3, or keep label integers free?
3. **Trigger thresholds** — `ambiguous_hypotheses` at margin 0.15 and
   `ambiguous_question` at a 15% relative margin are guesses. Phase 1's per-fault
   confidence spread (once the adapter exists) is the right place to set them;
   until then they are placeholders.
4. **Proposed intents** — approve inserting `hypothesis_comparison` before
   `overview` (it changes live behaviour for questions containing "cause" only if
   the flag is on) or keep all three benchmark-only?
5. **Answer length** — the schema allows 1–4 sentences; the live prompts ask for
   sections and bullets. The benchmark answers will be much shorter than live
   answers, so a model trained on live-style data will fail length unless
   retrained on schema-shaped answers. Worth deciding before data regeneration.

---

## 11. Cost

Phase 2A itself was **CPU-only and is complete**: 67 tests in 3.5 s, whole suite
5.2 s, no GPU second. Remaining Phase 2A-adjacent work, if you want it:

| Work | Estimate |
|---|---|
| Evaluator rewrite on top of `allowed_set` + `store_access` (Phase 2B) | 1 day CPU |
| Re-generating training answers in the schema shape (needed for open questions 4–5) | depends on the teacher: ~0.5 s/item × ~2.4k rows ≈ 20 min local, plus regeneration of the 2,112 grounded rows |
| Adapter retraining, 3 configs × 3 seeds | as costed in the Phase 0 addendum: 8,877 s/seed measured, 7 configs |
