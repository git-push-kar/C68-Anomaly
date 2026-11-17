# DRAFT — unsigned. Nothing here takes effect until the owner signs it.

# Decision: the merged classifier row's scope, 2026-11-16

> **Generation note.** The numbers below are pre-vocabulary-fix (adapter 0.8330 = 414/497). The retrained table
> is `vocabfix/classifier_contrasts.json`: adapter **0.7646 (0.7253–0.7998)**, hybrid 0.9960, menu rule 1.0000.

## The situation

The plan asks for root cause, subsystem and severity as one merged row. On this checkout the three
compared systems emit **one output: the fault id**. Subsystem and severity are then looked up from
that id in the plant knowledge base, so they are derived, not predicted — a system that gets the fault
right gets both for free.

Measured on the 497 items all three scored (`docs/judging/p2_classifier_contrasts.md`):

| system | root-cause accuracy | abstention precision / recall | false abstention |
|---|---|---|---|
| menu rule | 0.9980 | 1.0000 / 1.0000 | 0.0000 |
| hybrid | 0.9940 | 0.9766 / 1.0000 | 0.0081 |
| adapter (pre-vocabulary-fix) | 0.8330 | 0.8723 / 0.6560 | 0.0323 |

Paired exact McNemar: menu rule vs hybrid 0/2 (p = 0.5); menu rule vs adapter 0/82; hybrid vs adapter
1/81 (both p < 1e-06).

## Option 1 — recommended: publish the row with two absences

> **Accepted.** The classifier row reports **root cause** as measured, and **subsystem = not measured**
> and **severity = not measured**. In this benchmark both are knowledge-base functions of the predicted
> fault id and are emitted independently by none of the three systems; a separate accuracy for either
> would restate the root-cause accuracy, which §12 prohibits as a composite score.

**Why this is the right option.** The row stays readable and true: three systems compared on one output
they all actually produce, with the abstention behaviour reported next to it. Nothing is lost from the
comparison — the deciding evidence is accuracy plus abstention recall (the adapter names a fault on 31
of the 125 "not on the menu" items), and neither subsystem nor severity would sharpen that.

## Option 2 — commission a system that emits them

Build heads that predict subsystem and severity from evidence rather than looking them up, then score
them with the same pairing. This is new modelling work, it would most likely reproduce the lookup's
accuracy, and it delays the P2 row for a claim the comparison does not need.

## Signature

Option chosen: ______ (1 = two absences / 2 = commission)

Signed: ______________________   Date: ____________   Channel: ____________
