# Phase 2B final patch 2 — report

**Repo:** `C:\Users\Admin\Desktop\anomaly`, branch `benchmark-upgrade`.
**Scope of this pass:** provenance first, then the AFFECTED count. **No teacher
call was made** (0 of the 120-call budget). The count came back at **493 items**
against the 120 ceiling, so the phase stops at step 2 as instructed: the KB edit
was not applied, nothing was re-called, and no dataset file was rebuilt.

## 1. Summary

The fault-13 direction edit is technically small (one word in one KB entry) and
textually enormous (the line is rendered into 5,731 of 5,870 prompts), so the
owner's rule was to price the *prose* blast radius before touching it. Measured
on the 1,925 accepted train answers, the union of "answers that mention the
affected variable" and "answers either form of the rule change would reject" is
**493 items — 4.1× the 120 ceiling** — and the stop condition fired.

The composition explains why. The edited claim is about **XMEAS_7 /
Reactor_Pressure**, and reactor pressure is a legitimate cascaded variable in
eight other faults' stories and in the open-set family's list of ruled-out
candidates. Only **68** of the 493 name `FAULT_13` at all, and only **4** of
those name it while diagnosing a different fault. The other 425 mention the
variable without naming the fault, so the content-based definition — correct as
a safety net — cannot tell "this answer depends on fault 13's menu line" from
"this answer happens to discuss reactor pressure".

Two findings from the same pass matter for the decision:

1. **The rule reads less prose than it appears to.** `_direction_conflict`
   searches for the literal canonical tag (`xmeas_7`). Of the 212 accepted
   answers that state a direction near a mention of this variable, **80 use only
   the display name** (`Reactor_Pressure`, "reactor pressure") and **34 use only
   the tag**; 98 use both. Any answer that says "reactor pressure is falling"
   and never writes the tag is invisible to the rule — so the 5 "contradicting"
   answers are a lower bound on prose that disagrees with the new sign, not a
   count of it.
2. **The 5 contradicted answers are not all what they seem.** One of them reads
   `XMEAS_7 (…) with a decreasing trend (z=-8.87, deviation -2.47%)` — the
   prose is anchored to a number printed in the evidence. The KB flip is derived
   from the *trend label* (increasing in 102/102 train events), while the
   stored **deviation sign is mixed 61/41**. A one-word rule cannot tell "the
   level is below baseline" from "the trend is falling", so flipping the sign
   would reject at least one answer that quotes the evidence faithfully.

The provenance work (step 1) is in, with the requested separation: `menu.py`
and `fault_knowledge.py` are **not** in `FILTER_CODE_FILES`. Every new record
and the manifest now carry `menu_revision` (`r1`), `menu_sha256` and `kb_sha256`,
and reuse requires the filter hash *and* the revision, unless the record carries
an explicit `menu_revision_unaffected` marker.

## 2. Changes

| commit | what |
|---|---|
| `510300b` | `llm/benchmark/provenance.py` (revision label, menu hash, KB hash, `reuse_decision`); `select_pending()` factored out of `run_teacher`; provenance wired into both build paths and the manifest; `tests/test_benchmark_provenance.py` (16 tests) |
| `e0c0d8d` | `scripts/count_fault13_affected.py`; `outputs/evaluation/fault13_affected.json`; manifest teacher-provenance key list extended |

`FILTER_CODE_FILES` is unchanged, so no cached answer was invalidated by this
pass.

## 3. Commands and outputs

### 3.1 Provenance (`510300b`)

```
$ python -m pytest tests/test_benchmark_provenance.py -q
................                                                         [100%]
16 passed in 3.54s
```

The rule as implemented:

```python
def reuse_decision(record, provenance):
    if record["filter_sha256"] != provenance["filter_sha256"]:
        return False, "filter_changed"
    if (record.get("menu_revision") or "r1") == current_revision:
        return True, "revision_match"
    if current_revision in record.get("menu_revision_unaffected", []):
        return True, "revision_unaffected"
    return False, "menu_revision_changed"
```

Tests cover: matching revision; legacy records read as `r1`; filter change
winning over an unaffected marker; revision bump invalidating an unmarked
record; the marker surviving *its* revision only (`r2` marker does not exempt
`r3`); the dict form with a reason; and the ordering `run_teacher` uses
(`select_pending` returns both the pending items and the reason counts).

Current values on this checkout:

```
menu_revision  r1
menu_sha256    1d63c8bd1bab…
kb_sha256      f01236ffe01b…
filter_sha256  a014a9a5a369…
```

### 3.2 The reuse rule exposes a pre-existing cache split

Running the new rule against the cache as it stands:

```
$ python - <<'PY'  # load train.jsonl + teacher_cache.jsonl, select_pending
train items: 1925   cache entries: 1971
would be pending: 1770 | reasons: {'filter_changed': 1770,
                                   'reuse_revision_match': 155}
PY
```

The cache holds **four** distinct filter hashes:

| filter_sha256 | records | written by |
|---|---|---|
| `c3888e7d3509` | 1,939 | the full teacher run |
| `5069f696f82a` | 200 | the scaffolding re-call |
| `a014a9a5a369` | 169 | the recall pass under the current filters |
| `df0919d21204` | 45 | the fault-5 re-run |

This is **not** a data defect: every published answer was attached through
`replay_cache`, which re-judges the raw text under the current filters, so the
1,925 rows are valid today. It is a *provenance* gap: only 155 rows carry the
current hash, so under the strict rule a future `--teacher` run would re-call
1,770 answers (≈5.3 h at 10.8 s) that the replay has already cleared.

The clean fix is not to weaken the rule but to let re-judgement count as
evidence: when `replay_cache` accepts a raw answer under the current filter
hash, stamp it (`validated_filter_sha256`) so the answer is reusable without a
call. Proposed, not implemented — it changes the reuse rule the owner just
specified, so it needs a decision.

### 3.3 AFFECTED count (`e0c0d8d`)

```
$ python scripts/count_fault13_affected.py
train_items        1925
accepted_items     1925
fault13_accepted   119
groups:
  mentions        493  {classified: 350, unclassified: 143}
  contradicting     5  {classified: 5}
  gerund_rejected  21  {classified: 21}
affected_total     493  {classified: 350, unclassified: 143}
overlap:
  contradicting_not_in_mentions  []
  gerund_not_in_mentions         []
```

By fault (`mentions`, i.e. the whole union):

| fault | items | | fault | items |
|---|---|---|---|---|
| 13 | 113 | | 8 | 42 |
| 7 | 93 | | 20 | 45 |
| 6 | 82 | | 18 | 37 |
| 5 | 41 | | 12 | 36 |
| 10 | 2 | | 19 | 2 |

By variant: `menu` 434, `removed` 59. The two rule-derived groups are subsets of
the mention group (both overlap lists empty), as they must be.

What the mention group actually contains:

| mention pattern | answers |
|---|---|
| `XMEAS_7` and `Reactor_Pressure` | 230 |
| `Reactor_Pressure` only | 103 |
| `XMEAS_7` only | 92 |
| also names `FAULT_13` | 68 (of which 4 diagnose another fault: 10, 19, 20) |

Inside fault 13's own 119 accepted answers: 113 mention the variable, 6 mention
nothing, and the rest split across the pattern combinations above.

### 3.4 Which answers the rules can actually see

```
accepted answers with a direction word within 60 chars of a mention: 212
  tag literal only: 34   display name only: 80   both: 98
fault-13 subset: 60 (tag-only 4, display-name only 29)
```

Scenarios, recomputed on the current 1,925 rows:

| scenario | newly rejected |
|---|---|
| gerunds added, KB unchanged | **21** (all fault-13, prose says the tag rises) |
| KB flipped, gerunds unchanged | **0** (`decrease` is not a substring of `decreasing`) |
| both | **5** (114 stay valid) |

The five contradicted windows, for judgement:

| item | window | note |
|---|---|---|
| `EVT-train-f13-r100` | `) at onset, with a trend decreasing first, consistent with` | asserts the old KB claim |
| `EVT-train-f13-r030` | `) with a decreasing trend (z=-8.87, deviation -2.47%)` | anchored to a printed negative z |
| `EVT-train-f13-r098` | `) with a decreasing trend starting at +0.0 min, followed by` | asserts the old claim |
| `EVT-train-f13-r070` | `) with a decreasing trend starting at +0.0 min, followed by` | asserts the old claim |
| `EVT-train-f13-r038` | `), and separator (xmeas_13) systems, alongside a decreasing` | window ends before the noun; ambiguous |

### 3.5 Cost of each scope, if one is approved

| scope | items | expected calls (×1.12) | model time | wall incl. retries |
|---|---|---|---|---|
| rule-affected only (21 + 5) | 26 | 29 | 4.7 min | ~5 min |
| answers naming `FAULT_13` | 68 | 76 | 12.2 min | ~14 min |
| all fault-13 answers that mention the variable | 113 | 127 | 20.3 min | ~23 min — **over the 120-call cap** |
| all fault-13 accepted answers | 119 | 133 | 21.4 min | ~24 min — over the cap |
| the measured union | 493 | 552 | 88.7 min | ~99 min — far over the cap |

At the measured 10.8 s/item and a 12% first-pass rejection rate.

## 4. Tests

| suite | result |
|---|---|
| `tests/test_benchmark_provenance.py` | **16 passed** |
| `tests/test_benchmark_dataset.py`, `tests/test_benchmark_teacher_policy.py` | **73 passed** |
| full suite, app environment | **374 passed** |
| full suite, `venv_train` | **374 passed** |

## 5. Gate table

| gate | result |
|---|---|
| AFFECTED-set count reported before any call | **PASS** — 493, from `scripts/count_fault13_affected.py`, JSON written |
| union > 120 → stop and report | **PASS** — stopped; **0 teacher calls** |
| provenance fields (`menu_revision`, `menu_sha256`, `kb_sha256`) in every new record and the manifest | **PASS** — `510300b`; `e0c0d8d` extends the manifest key list |
| `menu.py` / `fault_knowledge.py` **not** added to `FILTER_CODE_FILES` | **PASS** — file unchanged |
| reuse-rule test: KB edit + bumped revision marks affected stale, leaves unaffected reusable | **PASS** — both directions tested |
| change applied in one commit with tests | **NOT APPLIED** — stop gate fired before step 3 |
| re-call results by fault, retention table | **NOT RUN** |
| old menu line absent from all prompts (script) | **NOT RUN** — dataset not rebuilt |
| fault-13 contradictions 0 or listed | **NOT MET** — 5 listed above; the flip is what would clear them |
| final class counts, leak/split/open-set checks | **NOT RUN** — no rebuild |
| full suite, app environment | **PASS** — 374 |
| full suite, `venv_train` | **PASS** — 374 |

## 6. Deviations

1. **Stopped after step 2**, per the instruction's own ceiling. No KB edit, no
   re-call, no rebuild, no audit-sheet regeneration.
2. **Provenance was implemented before counting**, as ordered, so it is
   committed and tested even though the change it guards was not applied.
3. **`select_pending()` extracted** from `run_teacher` so the reuse rule is
   testable without a teacher; behaviour is identical (a record with a missing
   `filter_sha256` is still pending).
4. **The strict reuse rule marks 1,770 rows stale** on this checkout. That is the
   rule working as specified against a cache written under four different filter
   hashes; it is reported rather than worked around, with the replay-validation
   stamp proposed in §3.2.

## 7. Open questions

1. **Which scope?** The measured union (493) is dominated by prose that mentions
   reactor pressure as *another* fault's cascade. The honest candidates are 26
   (what the rules can see), 68 (what names the fault) or 113/119 (every
   fault-13 answer, since its prompt's claim changed). All except the last two
   fit the 120-call cap; the last two need the cap raised slightly.
2. **Should `_direction_conflict` also read display names?** 80 of 212
   direction-bearing answers never write the tag, including 29 of fault 13's 60.
   Leaves the current rule checking half the prose it claims to check.
3. **Is "decrease" in those five answers a claim about the trend or the
   deviation?** The KB flip is justified on the trend label (102/102), but the
   deviation sign is mixed 61/41, and `EVT-train-f13-r030` quotes a negative z.
   If the KB line means "trend", the flip is right and those answers are stale;
   if a reader takes it to mean "below baseline", they are not wrong.
4. **Replay-validation stamp** (§3.2) — needed before any future teacher run,
   otherwise the 1,770 already-cleared answers are re-called at ~5.3 h.

## 8. Cost of what remains

If a scope is approved, the remaining work is: one commit (KB sign, gerund
vocabulary, `MENU_REVISION = "r2"`, tests both ways), the re-call (5–99 min
depending on scope), a full rebuild (`--replay-cache --exclude-rejected`, ~6 min
per the previous pass), the prompt-line check, the direction re-run and the two
test suites — under 30 minutes of wall time for any scope up to 119 items, and
about 110 minutes for the full 493.
