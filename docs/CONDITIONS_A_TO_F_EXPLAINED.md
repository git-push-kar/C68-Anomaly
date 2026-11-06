# What A–F are, and what each one is for

Status: **explanatory context**, companion to [HANDOVER_BENCHMARK_PLAN_V2.md](HANDOVER_BENCHMARK_PLAN_V2.md).
It adds no measurement and changes no rule. Where it disagrees with a dated `docs/PROTOCOL.md`
section, the protocol section wins. Numbers marked *(verified here, 2026-11-05)* were read off
artefacts on this checkout; numbers marked *(inherited)* come from the dated record and were not
re-derived.

This document exists because the letters get used three different ways in the record — as plans,
as cells, and as labels in a cost table — and a reader arriving on the GPU device needs the
distinction before touching anything.

---

## 1. The invariant, stated once

In every condition below, **the diagnosis, verdict, severity and confidence are code outputs.**
They are computed before any writer runs, handed to the writer as fixed input, and never revised by
it.

The conditions therefore do not differ in what the system concludes. They differ only in **who
writes the sentences around the conclusion**. A preference score between two conditions is a
statement about prose; it is never a statement about diagnostic accuracy, and no result from this
grid may be reported as one.

Two corollaries that constrain the whole design:

- The classifier and detector surfaces are measured separately and never enter this grid.
- A condition that improves prose while changing a verdict is not a condition, it is a defect.

---

## 2. Vocabulary, because the record is inconsistent

| term | meaning here |
|---|---|
| **item** | one of the 520 events scored. Every item appears in every condition. |
| **condition / cell** | a whole configuration of writer assignment, applied to every item. |
| **seam** | a split of one report into two authorship regions — numeric sentences vs mechanism sentences — with a rule for which side writes which. C and M are seams. |
| **arm** | a configuration on a *different* surface (detector, dialogue). Avoided for report conditions, because "arm" already means a monitor setting. |
| **gate** | a check that can **veto** a condition outright. Not a score. |
| **rung** | a tie-break step in the declared selection rule, applied in order. |
| **instrument** | the code that counts defects. It is itself measured (46 cases, precision/recall), because a defect count from an uncalibrated instrument is not quotable. |

**Disambiguation, because it costs a day if missed:** "A–F" names **seven** labels, and they do not
all belong to the report surface.

- **Report surface arms (six):** A, C, M, B, D, E.
- **Dialogue surface arms (three):** T (template), U (adapter as-is), F (retrained adapter).

So "the six cells" in the plan means A/B/C/M/D/E scored on 520 items. **F is not one of them.** F
competes with T and U on 97 stored dialogue turns. A and F are not rivals: A never touches the
model on the report path, F never touches the report.

---

## 3. The grid

The report grid is a 2×2 over who writes which half of a report, plus two toggles:

|  | **numbers by code** | **numbers by adapter** |
|---|---|---|
| **mechanism by code** | **A** — fallback v3.1 | **M** — A's grammar, adapter's numbers |
| **mechanism by adapter** | **C** — hybrid seam | **B** — full adapter |

| label | one line | trains? | status on this checkout |
|---|---|---|---|
| **A** | deterministic assembler; templates fill every number from the store | no | **live** |
| **B** | adapter authors the whole report | no | **stored** — 520 × 3 orders |
| **C** | code writes numeric sentences, adapter writes mechanism | no | **no code**; seam over stored B |
| **M** | adapter's numeric values, code's mechanism grammar | no | **no code**; seam over stored B |
| **D** | B delivered through check → retry → fallback | no | validator exists; **retry needs weights that are absent** |
| **E** | adapter retrained on report prose, 3 seeds | **yes** | script only |
| **F** | adapter retrained on follow-ups | **yes** | spec only; 961 rows, 180 steps |

---

## 4. Each condition, and the question it actually answers

### A — fallback v3.1 authors every report

**What it is.** A deterministic assembler. Fixed sentence templates keyed on the supplied
diagnosis, with store values slotted in. No model, no inference, no reasoning step.

**The question it answers.** *Can a template do this well enough that a language model is not
needed on the report path at all?*

**Why it is in the grid.** It is the null condition, and unusually for a null it is already ahead:
0 flagged rows of 40 on the pilot and anchor exact 40/40 *(40-row pilot, on this checkout)*, and
at 520 items 19 numeric rows (3.65%), 0 grounding rows, 329/329 anchor exact *(verified here)*. Every other condition must
beat it to exist, and the default rung gives it any tie.

**What it cannot show.** It cannot produce a mechanism sentence it was not given. Its prose is
flatter exactly where the library is thin — on the menu-removed open-set stratum the adapter is
preferred 8 to 1 *(40-row pilot, 10-item stratum)*. That is a quality ceiling, not a correctness
risk.

**Failure mode.** None that is measurable here. A template that fills `"{tag} (z={z})"` from the
store cannot emit a wrong inequality direction, because nothing in it composes a comparison. A
condition that loses to A on numeric defects has to explain why it was worth a GPU.

### B — the adapter authors the whole report

**What it is.** The fine-tuned 3B-17 adapter generates the entire report, numbers included, from
the same prompt A's templates are keyed on.

**The question it answers.** *Does the model's fluency buy anything an operator would notice, once
the numbers are its responsibility?*

**Why it is in the grid.** It is the measured failure the whole study is built around, and it is
already paid for: 1,560 stored generations, three prompt orders, **0 GPU left to spend** *(verified
here)*. At 520 items it produces 62 numeric-defect rows (11.92%), 33 grounding rows (6.35%), and
anchors the top sensor exactly on 170 of 329 items (51.67%) against A's 329/329 *(verified here)*.

**What it cannot show.** Its defect profile is uniformly mechanical — numbers, tag roles,
grounding. It is not evidence about reasoning quality, and a preference win for B would not mean
B's causal claims are better.

**Failure mode.** It is the reason C, M and E exist. Its 11.92% numeric rate is 8.3 points above A,
far outside the 1.0-point gate margin, so B cannot pass rung 1 as it stands. B is retained as the
upper bound of adapter authorship, not as a shipping candidate.

### C — code writes the numbers, adapter writes the mechanism

**What it is.** Every numeric sentence in the report is written by code, from the store. The
adapter's section is **number-free by construction** — it is not handed numbers to paraphrase. The
report becomes two regions with a defined seam.

**The question it answers.** *If the adapter is relieved of numeric obligation entirely, does the
defect class disappear while the fluency advantage survives?*

**Why it is in the grid.** It is the only design matched to the measured failure. All of B's defect
instances are numeric, tag-role or grounding; none is a reasoning error. C removes the first and
third outright and narrows the second, while preserving the stratum where the adapter is genuinely
better — the menu-removed rows where it is preferred 8 to 1.

**What it cannot show.** It cannot show that adapter mechanism prose is *valid*. That is rung 3's
job, on a separate instrument, and C can win the gate and still lose the mechanism count.

**Failure mode.** Assembly reads as two authors; an operator notices the voice change. And the
adapter can still misattribute a tag role inside a mechanism sentence, which is why the seam must
be certified by the instrument — not by a reader — as containing no numeric sentence from the
adapter.

**How it is built here.** As a text-level seam over stored B: split B at sentence level, keep B's
mechanism sentences, replace every numeric sentence with A's. No generation pass, ~1 h CPU. This is
lawful only because B's 1,560 generations are on disk; without them C would need a retrain first,
because the adapter weights are not in this checkout.

### M — the mirror seam: adapter's numbers, code's mechanism

**What it is.** The inverse of C. A's mechanism sentences, with the adapter's numeric values
substituted in.

**The question it answers.** *Is it the "code writes numbers" rule that helps, or merely having a
seam?*

**Why it is in the grid.** This is the reason the design is a 2×2 rather than a list. A grid with
A, B and C alone cannot distinguish two stories that both fit the data:

- code-authored numbers are what removes the defects (C beats M, and M is no better than B);
- any split into two authorship regions helps, e.g. by shortening each writer's remit
  (M ≈ C).

M is the control that makes that distinction estimable, and it costs about the same hour as C. If M
matches C, then a paper claiming "numeric authorship is the fix" is wrong, and the honest claim is
about seam structure. That is a cheaper mistake to make now than after a confirmatory pass.

**What it cannot show.** M is deliberately the *worse* assignment on correctness grounds — the
adapter's numbers are the defect source. A poor M result is expected and is not a finding; it is
the reference point that gives C's result its meaning.

**Failure mode.** M is the condition most likely to be quietly dropped when time is short, because
it looks like a duplicate of C. Dropping it converts a 2×2 into a one-way comparison and forfeits
the interaction term. If it is dropped, say so and cap the claim at "C beats B and A", never
"because numbers were code-written".

### D — check, retry, fallback, on top of B

**What it is.** A delivery wrapper, not a writer. The four instrument checks run on the generated
text; one corrective retry on failure; anything still flagged ships as fallback text.

**The question it answers.** *How much of B's damage does validation alone repair?*

**Why it is in the grid.** It isolates the value of the wrapper from the value of the writer. It is
what a team would plausibly ship if told only "add validation", so the study has to show what that
buys.

**What it cannot show.** Validation retries text generation; it does not recompute a threshold. At
B's 52.5% first-pass flag rate *(40-item pilot)*, D must rescue more than half of all
rows merely to reach the 25% fallback ceiling. A D that meets the ceiling by falling back on most
rows is A with extra steps, and must be reported that way.

**Status caveat, and it matters.** With no adapter weights on this checkout, D's retry cannot run.
D degenerates to "B with fallback substituted on invalid rows" and **must be labelled exactly
that** in every table. Labelling it "D" without the qualifier would attribute a retry effect that
never happened.

### E — retrain the adapter on report prose

**What it is.** The adapter authors reports after being retrained on a corpus whose numeric
sentences are correct by construction (fallback's numeric sections) plus mechanism narrative. Three
seeds.

**The question it answers.** *Can training close the numeric gap, or is numeric fidelity simply not
what a 2B model does reliably?*

**Why it is in the grid.** It is the only condition that tests the training hypothesis rather than
working around it. Without E, a negative result is unfalsifiable: a reviewer can always say the
adapter was simply undertrained.

**What it cannot show.** It cannot show that *this* adapter generalises — three seeds bound the
variance, they do not establish a population. And it must beat a deterministic writer that scores
0 defects on 40 rows, which is a demanding bar for a condition whose training data is largely that
writer's own text.

**Cost.** 3.0–5.2 h GPU for three seeds, sequential. The device peaked at 16.2 of 24 GiB, so **two
training jobs cannot run concurrently.** Needs a dated §30, because it is outside the §29 order.

### F — retrain the adapter for follow-ups only

**What it is.** Reports come from A. The adapter serves the dialogue path: mechanism, comparison,
refusal and clarification prose. Trained on 961 dialogue rows, 180 steps, hyperparameters
unchanged.

**The question it answers.** *Does the adapter earn its keep on the surface where it has no numeric
obligation at all?*

**Why it is in the grid.** It is the cheapest spend that touches a surface the deterministic writer
genuinely cannot serve. A lookup table handles the canned intents; it degrades the free-form ones —
valve and causal questions — into refusals. F is where the measured strength of the adapter
(mechanism and comparison prose, no anchor obligation) actually lands.

**What it cannot show.** It cannot be cited as a report result, and a good F does not license C. The
dialogue surface is context for a report; it is not a second copy of the report gate.

**Cost.** 38–65 min at the two recorded step times. Note that no per-item generation rate is
recorded anywhere in this repository, so the *inference* half of every estimate is a projection
from output length, not a measurement. Whichever session runs F should record its own wall clock.

---

## 5. What the letters are not

Three readings that the record invites and that are wrong:

1. **"A–F is a ranking."** It is not. The output is a tuple —
   `report = … · dialogue = … · detector = … at FAR … · classifier = … · cost = … · reason = …` —
   and the reason names the rung that decided. A ranking with no reason line cannot be audited.
2. **"The six cells are six competitors."** C, M and D are *derived* from the same stored B text.
   Treating them as independent samples inflates what the preference differences mean. They are
   paired by construction, which is why every contrast is paired and why the selection-bias
   correction assumes correlated conditions, not independent ones.
3. **"F competes with E for the same budget."** It does, in a scheduling sense, and that is a
   decision, not a finding. In a design sense they touch different surfaces and answer different
   questions; neither substitutes for the other.

---

## 6. How the grid is adjudicated

Three instruments, never averaged into a single score:

| tier | instrument | may it overrule? |
|---|---|---|
| gate | numeric-defect non-inferiority vs A, paired | **yes — veto** |
| primary | blind paired human preference | decides among gate-passers |
| corroboration | judge, validated on the human sample (κ reported) | no |
| objective | mechanism validity, order invariance, sufficiency, abstention | no — explanatory |

The six rungs, declared before any result is visible: gate → preference → mechanism validity →
order invariance → cost → A. The gate margin `m` is recommended at 1.0 percentage point (~5 items
of 520), far below the 4.9-point MDE, so it can only veto a large visible gap and never adjudicate
noise.

Why the declared rule matters: with four report cells and paired correlated conditions at n = 520,
selecting the best-looking cell carries an optimistic bias of about 0.54 points and a 52%
probability of picking a non-best cell. With twelve conditions it is 1.24 points and 89%. Those
inflations are the same order as the gaps between the near-tied cells, which is why the rule is
fixed in advance and the confirmatory pass is a single paired shot.

---

## 7. What the grid already shows, and what it does not

Measured on 520 items, adapter (kb) against A *(verified here)*:

| reading | B | A |
|---|---|---|
| numeric-defect rows | 62 / 520 = 11.92% | 19 / 520 = 3.65% |
| grounding rows | 33 = 6.35% | 0 |
| tag_role rows | 10.96% *(one reader)* / 76.15% *(another)* | 0 |
| contradiction rows | 0 *(calibrated)* / 462 = 88.85% *(uncalibrated)* | 0 |
| anchor exact, top \|z\| ≥ 1 | 51.67% | 100% |

Read carefully, that table says: **A is ahead on every automated reading, and two of the five
readings are not yet trustworthy.** The tag_role disagreement is 7× on identical text, and the
contradiction check fires on 88.85% of rows only because the gate is wired to the uncalibrated
duplicate. Until P0 lands, those two columns are not quotable and neither check may gate anything.

What the grid does not yet show is the part it was built for: **where C, M and E land.** The
ordering is already visible and paper-level work will not change who wins. It changes whether the
margin is measured, intervaled and defensible — and the near-tied cells are where the study has
something real to decide.

---

## 8. The honest one-paragraph summary

The letters describe the same diagnosis rendered by seven different authorship arrangements, none of
which may move the diagnosis. A is the deterministic assembler and the default. B is the
fine-tuned adapter writing everything, already measured as mechanically defective and already
paid for. C removes numeric obligation from the model; M is its mirror, and exists so that a win
for C can be attributed to numeric authorship rather than to the mere presence of a seam. D wraps B
in validation and measures what validation alone repairs. E asks whether training closes the gap.
F moves the model to the surface where it has no numbers to get wrong. C, M and D are cheap
derivations from text already on disk; E and F are the only conditions that spend GPU, and they
spend it sequentially, at 3.0–5.2 h and 38–65 min respectively.
