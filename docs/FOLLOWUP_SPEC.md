# FOLLOWUP_SPEC — follow-up dialogue contract (Phase 2C)

**Repo:** branch `benchmark-upgrade`.
**Status:** implemented in this phase (`llm/followup/`,
`scripts/build_followup_dataset.py`). The live path is untouched: the three
benchmark-only intents stay behind `ResponseRouter(enable_proposed=True)` and the
live flag remains **off**.

This document is the contract the dataset is built to. Every number quoted here
was measured on this checkout; the commands that produce them are in
`docs/reports/phase_2c_report.md`.

---

## 1. What this dataset is for

The report dataset (Phase 2B) teaches the first turn: read evidence, choose a
label from a menu, justify it. A real operator then asks questions, and the live
system answers some of them from templates and the rest from the adapter:

* `main.py:answer_followup` calls `RCAInference._dispatch_with_intent` first and
  returns the template answer if a keyword hits;
* only when nothing hits does it call `RCAInference.answer_followup` and generate.

The follow-up dataset therefore exists to train **exactly the branch the live path
delegates**: the questions that reach the model, and the behaviours around them
(refuse what the store cannot support, ask at most one clarifying question, do not
repeat itself on a repeated question).

---

## 2. Intent taxonomy — 14 intents

Eleven are live. Three are benchmark-only (`llm/response/schemas.py:Intent`,
routed only with `enable_proposed=True`).

`TEMPLATE` means the answer is produced by `llm/response/lookup_answers.py` from
store keys and **never sent to a model**. `LLM` means the answer text must be
generated.

| # | Intent | Origin | Class | Live behaviour | Benchmark behaviour |
|---|---|---|---|---|---|
| 1 | `sensor_values` | live | **TEMPLATE** | template | template |
| 2 | `evidence_why` | live | **TEMPLATE** | template | template |
| 3 | `system_sensors` | live | **TEMPLATE** | template | template |
| 4 | `sensor_scores` | live | **TEMPLATE** | template | template |
| 5 | `detector_severity` | live | **TEMPLATE** | template | template |
| 6 | `detector_scores` | live | **TEMPLATE** | template | template |
| 7 | `severity` | live | **TEMPLATE** | template | template |
| 8 | `overview` | live | **TEMPLATE** | template | template |
| 9 | `sensors_inspect` | live | **TEMPLATE** | template | template |
| 10 | `action` | live | **TEMPLATE** | template | template |
| 11 | `overview_fallback` | live | **TEMPLATE** | template | template |
| 12 | `hypothesis_comparison` | benchmark-only | **LLM** | no branch → **model** (or `overview` when the question says "cause") | `comparison` intent, table `comparison`/`hypotheses` |
| 13 | `out_of_scope_refusal` | benchmark-only | **LLM** | no branch → **model** | refuse; decision from `StoreAccess`, wording from the model |
| 14 | `clarification_reply` | benchmark-only | **LLM** | no branch → **model** | reply to a pending clarifying question, resolved from dialogue state |

### 2.1 Three LLM-routed cases with *no* intent

| case | Live behaviour | Why there is no intent |
|---|---|---|
| valve / actuator status | **model** | no keyword set matches; the dispatcher returns `None` |
| free-form causal question ("how does a stuck valve raise the reactor temperature?") | **model** | same |
| mechanism question ("by what mechanism does stiction oscillate?") | **model** | same |

These are recorded with `intent: null`. The `FollowupAnswer` schema requires an
intent, so a free-form turn carries no table (`none`) and its citations list the
keys its prose may quote. Forcing them into `overview_fallback` would misreport
which branch answered: the live path never reaches that template for them.

### 2.2 The routing measurement behind the table

`scripts/inspect_followup_routing.py` routes 50 realistic operator questions per
label — the 14 intents **and** the 3 free-form case types that have no intent
(17 × 50 = 850) — through the live dispatcher and the router.
Artefact: `outputs/evaluation/followup_routing.md` / `.csv`.

| intended | template | model (`None`) | clarifications (flag on) |
|---|---|---|---|
| 7 numeric/sensor/severity intents | 50/50 each | 0 | 0-4 |
| `overview` | 46/50 | 4 | 4 |
| `action` | 49/50 | 1 | 0 |
| `overview_fallback` | 50/50 (5 via `overview`) | 0 | 5 |
| `hypothesis_comparison` | 16/50 (as `overview`) | 34 | 1 |
| `out_of_scope_refusal` | 1/50 | 49 | 0 |
| `clarification_reply` | 0/50 | 50 | 0 |
| `valve_actuator` (no intent) | 0/50 | **50** | 0 |
| `causal_why` (no intent) | 0/50 | **50** | 0 |
| `mechanism` (no intent) | 4/50 (as `evidence_why`) | **46** | 0 |

| family | n | template | model | model share |
|---|---|---|---|---|
| the 11 live intents | 550 | 545 | 5 | 0.0091 |
| free-form, no intent | 150 | 4 | 146 | 0.9733 |
| benchmark-only intents | 150 | 17 | 133 | 0.8867 |

So the brief's hypothesis holds with one correction: **the 11 live intents are
template-answered when their keywords are present, and everything else reaches the
model** — in practice 99.1% of template-intent questions never reach the model,
and 97.3% of free-form questions always do. It is not only valve questions: a
question that never says "sensor/score/evidence/cause…" also reaches the model,
and the 5 escapes are `overview` × 4 ("Summarise…") and `action` × 1. Nor is the
line clean in the other direction: 4 mechanism questions contain a keyword the
`evidence_why` gate claims, which is why the builder routes every candidate
question through the dispatcher before trusting it (§6.1).

The three found collisions that matter for the taxonomy:

* "Summarise the event." does not match `overview` (the gate is `summary`), so a
  summary request reaches the model. British spelling is an unhandled synonym.
* Under the flag, "Report the observed values **versus** baseline values." routes
  to `hypothesis_comparison`: the proposed comparison keyword `versus` is too
  loose for ordinary comparisons.
* `out_of_scope_refusal` fires only when the question contains **no** plant
  vocabulary. "What is the compressor bearing vibration?" is in-domain in
  vocabulary but absent from the store, so the router lets it through to the
  model. This is why the refusal *decision* for the dataset comes from
  `StoreAccess` (§6.2), not from the keyword intent.

---

## 3. What the adapter is trained on

**The adapter is trained only on LLM-routed intents and cases.** The 11 template
intents are scored on the lookup path (deterministic, exact) and are excluded from
training, with one controlled exception:

* a **template sample, default 5%** of dialogues (`mix["template_sample"]`),
  answered with the *lookup text itself* — no teacher call. It exists so the
  model keeps seeing the answer format the lookup path produces and does not
  drift into a different voice. It is configurable, including to zero:
  `--mix '{"template_sample": 0}'`.

Rationale: training a model to reproduce text that code produces adds nothing
measurable and costs teacher calls, but removing *all* template-shaped answers
from training lets the model's style diverge from the answers the live path shows
beside it in the same conversation.

Consequence for scoring: a template intent is judged on the lookup path (exact
match to the deterministic answer, or a citation/table check), never on generated
prose.

---

## 4. Case types and mixture

A **dialogue** = 2-4 turns for one event. Turn 1 is answered after the automatic
report; the report itself is context, not a turn.

| Case type | Class | Default share | Turn plan | What it exercises |
|---|---|---|---|---|
| `valve_actuator` | LLM | 0.22 | valve → valve | valve/actuator status, the classic LLM-only question |
| `causal_why` | LLM | 0.18 | causal → mechanism | free-form causal explanation |
| `mechanism` | LLM | 0.12 | mechanism → causal | physical mechanism |
| `hypothesis_comparison` | LLM | 0.12 | comparison → valve | benchmark-only intent, differential diagnosis |
| `out_of_scope` | LLM | 0.06 | out_of_scope → valve | refuse what the store does not hold |
| `false_premise` | LLM | 0.08 | false_premise → valve | wrong sensor / fault / value, contradicted by the store |
| `repeated_question` | LLM | 0.06 | valve → repeat → valve | same question twice, answer must differ |
| `ambiguous_question` | LLM | 0.06 | ambiguous → valve | router clarification trigger |
| `clarification_reply` | LLM | 0.05 | ambiguous → reply | reply to a pending clarification |
| `template_sample` | **TEMPLATE** | 0.05 | template → template | format anchor, lookup text, no teacher |

Shares are per dialogue and normalised (`DEFAULT_MIX` in
`llm/followup/cases.py`); override with `--mix file.json` or inline JSON.

Case conditioning: `template_sample` is only used on classified, non-open-set
events (an open-set report raises a clarification on the first turn, which would
replace the lookup answer); `hypothesis_comparison` needs a KB lookalike;
`false_premise` needs a diagnosis to contradict. When a drawn case does not fit the
event, the builder substitutes `valve_actuator` and records `case_substituted`.

---

## 5. Deterministic part versus teacher part

For **every** turn the code decides, before any model is called:

| Field | Source |
|---|---|
| `intent` / `resolved_intent` | live dispatcher (template turns) or the case type (LLM turns) |
| `answerability` | `deterministic` / `llm` / `clarify` / `refuse` |
| `table`, `table_params` | `TABLE_FOR_INTENT` + store (`tags`, `hypotheses`) |
| `citations` | resolved store keys only (`StoreAccess.resolve`); an unresolvable key is never written |
| `clarifying_question`, `clarification_trigger`, `clarification_options` | `ResponseRouter` triggers |
| `refusal` | `StoreAccess` check (§6.2): `key_missing`, `contradicted`, `off_domain`, plus the alternatives |
| `premise` | the checked premise: kind, claim, verdict, store key, actual value |
| `suggested_followups` | router-verified, non-duplicate, not the question just asked |
| `answer_source` | `lookup` or `teacher` |

The teacher writes **one** string: `answer`, 1-4 sentences. It may not emit an
intent, a table, a citation, a clarifying question, a severity level or a
confidence number; those are either given to it as fixed facts or rejected.

Template turns take the answer verbatim from `lookup_answer(...)` — the
dispatcher's own formatting is kept in `detail`, and the graded lead is the
1-2 sentence controlled text from `lookup_answers._lead_for`.

---

## 6. Questions

### 6.1 Sources

* **Templated paraphrases per intent**, seeded. `llm/followup/paraphrases.py` has
  10 base phrasings per label × 4 surface prefixes = **40 train** questions, plus
  **10 held-out** hand-written phrasings per label.
* **Held-out half reserved for val/test**: train dialogues draw only from the 40;
  val and test dialogues draw only from the 10. `check_paraphrase_disjointness()`
  proves the two halves share no normalised question across all 17 labels, and the
  build aborts if they do.
* **Slot-filled questions**: comparison, false-premise and missing-key questions
  are filled from the item's own hypotheses/store, so no fault outside the menu can
  appear in a question.
* **Teacher-paraphrase hook: OFF by default.** `build_followup_dataset.py` never
  asks a model for a paraphrase; the seam would be a new case type, and the fake
  teacher is what the tests use.

Each turn records its paraphrase id (`label#index` in the half) and
`paraphrases_skipped`, the number of paraphrases rejected because they did not
route as intended. A bank question is never trusted: the builder routes every
candidate through the live dispatcher before using it.

---

## 7. Teacher prompt and answer filters

### 7.1 Prompt

`llm/followup/teacher.py:build_prompt` renders: the stored context (opaque
incident reference, detector readings, variables that moved, leading diagnosis),
the fixed decisions, the conversation so far, the question, and the rules. It never
contains an event id, split name, run number or filename. A retry appends only the
rejection codes.

### 7.2 Filters (rejections, never repairs)

Reused from Phase 2B (`llm/benchmark/filters.py`): `number_not_in_store`,
`unknown_tag`, `sensor_not_grounded`, `absolute_wording`, `severity_claim`,
`fault_out_of_scope` — same constants, same number helper.

Added for follow-ups (`llm/followup/filters.py`):

| Code | Rule |
|---|---|
| `empty_answer` | no text |
| `answer_length` | outside 1-4 sentences |
| `answer_is_question` | the answer asks instead of answering |
| `clarify_text_repeated` | the answer restates the clarifying question |
| `near_duplicate` | ≥ 0.6 of the answer's 4-grams already in the previous assistant turn |
| `confidence_claim` | a numeric confidence ("87% confident") |
| `refusal_missing` | a refusal that does not decline |
| `refusal_no_alternative` | a refusal that offers nothing the store can answer |
| `names_removed_fault` | an open-set answer that establishes the removed family (the operator's own words are not a leak) |
| `table_reference` | "as shown in the table below" |
| `valve_flow_mislabel` | a manipulated variable described as a measurement (`XMV_52 temperature`) |
| `event_id_leak` | a benchmark/dataset id in the answer |

Refusal format: the answer must contain a decline marker **and** at least one of
the alternatives the decision supplied, each of which was routed through the live
dispatcher before being offered (so "offer an alternative that exists in the
store" is checked, not asserted).

Clarification format: at most one clarifying question per turn (it is a field, and
the schema rejects two `?`), the answer must come first (non-empty, no `?`, not the
clarifying question's text), and a repeat or a reply turn may never re-ask — a
trigger that fires there is recorded as `clarification_suppressed`.

### 7.3 Retry and cache

Retry **once** on rejection, with the codes in the prompt. Provenance per turn:
`prompt_sha256`, `prompt_cache_hit`, `answer_attempts`, `teacher_model`,
`teacher_seconds`, `filter_codes`, `rejected_answer`. The resume cache is an
append-only JSONL keyed by `sha256(model|seed|temperature|prompt)`, so a changed
rule, context or model is a different key and never serves stale text.

---

## 8. Leak checks and splits

Applied to **every** follow-up context and question, aborting the build on a hit:

* event ids (`EVT-`, `NORM-`, `ANOM-`, and the literal event id), dataset item ids
  (`::`, `item_id`), split and fold names, run numbers, `seed`, CSV filenames;
* **fault numbers outside the menu that item is shown** — for an open-set item
  that includes the fault whose entry was removed;
* the opaque incident label (`INC-xxxxxx`) is allowed in the context and refused in
  a question.

The context carries no event id: it uses `sha256(event_id)[:6]`, which is stable
and carries none of the split/fault/run structure the benchmark ids encode.

Splits are Phase 2B's: train dialogues from train events, val from val, test from
test. The build proves, before any teacher call:

* no `(source_file, fault, run)` key appears in two splits, and no event id either;
* no duplicate key inside a split;
* every built key is a key the split file assigns to that component (fault-free
  runs excluded, since the faulty split file does not cover them);
* faults 3 and 9 are excluded from train (counted, not silently dropped).

---

## 9. Record format

```jsonc
{
  "dialogue_id": "FD-train-00042",
  "event_key": "TEP_Faulty_Testing.csv|15|19",   // provenance only, never in the context
  "event_sha256": "...",
  "split": "train", "variant": "menu" | "removed",
  "case_type": "false_premise", "requested_case_type": "...", "case_substituted": null,
  "open_set": false, "normal": false, "fault": 15,
  "context": "...",                              // leak-checked, opaque incident id
  "turns": [
    {
      "kind": "false_premise", "question": "...", "paraphrase": "...",
      "intent": "out_of_scope_refusal", "resolved_intent": null,
      "answerability": "refuse", "table": "none", "citations": ["detector.lstm.score", "..."],
      "clarifying_question": null, "clarification_trigger": null,
      "clarification_suppressed": null,
      "refusal": {"required": true, "reason": "contradicted",
                  "premise": {...}, "alternatives": [{"question": "...", "intent": "detector_scores"}]},
      "premise": {"kind": "fault", "claim": "FAULT_07", "verdict": "contradicted", "store_key": "...", "actual": "FAULT_15"},
      "suggestions": ["..."],
      "answer_source": "teacher", "answer": "...", "accepted": true, "filter_codes": [],
      "prompt_sha256": "...", "answer_attempts": 1, "teacher_model": "fake-followup-v1"
    }
  ]
}
```

`outputs/.../manifest.json` records: counts per split and case type, acceptance and
rejection codes, split integrity, paraphrase disjointness, leak-check status, file
hashes, teacher metadata, the cost estimate, the full-run projection and the mix
proposal.

---

## 10. Audit sheet

`audit_sample.csv`: 100 rows (or the population, if smaller) round-robin across
case types, one row per turn, with the scoring columns left **blank**:

`dialogue_id, case_type, split, turn, question, answer, answer_source, accepted,
intent, table, citations, clarifying_question, refusal_reason` then
`correct, grounded, refusal_appropriate, not_repetitive, useful, notes`.

The four judgement columns map to the filters: `grounded` (numbers/tags),
`refusal_appropriate` (declines and offers a stored alternative),
`not_repetitive` (not a near-duplicate of the previous turn), `useful` (would an
operator act on it). A builder-filled score would be a quality claim nothing
measured, so they stay empty.

---

## 11. Known gaps (carried into the report)

1. **Free-form turns have no intent**, so no table. The schema ties tables to an
   intent, and the live dispatcher has no intent for these questions. The dataset
   records `intent: null` and grounds the prose through citations instead. A
   15th intent would fix the shape, but it is not in this phase's taxonomy.
2. **The refusal intent is store-driven, not keyword-driven.** The proposed
   `out_of_scope_refusal` trigger fires on 8/50 of the probe questions, because it
   treats plant vocabulary as "in domain". The dataset therefore records the
   refusal *decision* from `StoreAccess` and labels the turn with the refusal
   intent; the keyword router would let most of them reach the model.
3. **`clarification_reply` is state-driven.** Reply-shaped statements ("the
   detector scores view") contain no clarification keyword: 7/10 reach the model
   under the flag. The dataset resolves them from the pending clarifying question.
4. **`ambiguous_hypotheses` needed a fix to be usable.** Phase 2B leaves
   `confidence: None`; the router defaulted it to 0.0, so every two-hypothesis
   report looked tied and asked a clarification on every turn. The trigger now
   requires a measured confidence (see the report's change log).
5. **Two keyword gaps stay unfixed** because fixing them changes the live router:
   "summarise" (British spelling) misses `overview`, and `versus` collides with
   `hypothesis_comparison`.
