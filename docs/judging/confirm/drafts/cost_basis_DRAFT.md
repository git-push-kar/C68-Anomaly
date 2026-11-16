# DRAFT — unsigned. Nothing here takes effect until the owner signs it.

# Decision: what rung 5 compares, 2026-11-16

## Why this is needed now

Rung 3 can only adjudicate two eligible cells. After the mechanism sheet is extended, C and B both
qualify — but if their intervals overlap, the signed amendment sends the selection to rung 4, and rung
4 has data for B alone (order invariance exists only for the adapter's three orders, and A, C and D are
assembled rather than generated). That leaves **rung 5, cost**, as the only measured rung able to
decide — and if rung 5 has no input, the ladder falls to rung 6 and the receipt stays provisional,
which keeps D33 closed.

No cost basis has been signed. The handover's table asks for "$/1,000", and that figure depends on
hardware and on which price you assume, which is exactly the kind of choice that has to be declared
before it decides something.

## Option 1 — recommended: compare measured seconds, report no dollar column

> **Accepted.** Rung 5 compares cells on **measured wall-clock seconds per 520-row pass**, taken from
> the stored generation timings already on disk. The table's cost column is headed "latency (s / 520
> rows)" and no $/1,000 figure is published, because no price basis has been declared. Any published
> dollar figure would be a separate decision with its own dated note.

**Why this is the right option.** The seconds are already measured on this machine and cannot be
tuned after the fact: A performs no model call, and the adapter spends ≈ 0.325 s per item. A latency
comparison is a real, checkable difference and it decides rung 5 without inventing a price. It does
mean the paper says "slower" rather than "more expensive", which is the weaker but true statement.

## Option 2 — declare a dollar basis

Choose a GPU hourly price and an assumed utilisation, convert the measured seconds, and publish
$/1,000 with both assumptions in the caption. Stronger claim, but the number then depends on your
assumption and a reviewer will test it.

## Option 3 — leave rung 5 unset

Then rung 6 decides and the receipt is provisional, which keeps the confirmatory pass closed and
D33 unrun. This is a valid choice only if you intend to run the preference study first, since that
would let rung 2 decide instead.

## Signature

Option chosen: ______ (1 = seconds / 2 = dollars / 3 = leave unset)

If option 2: GPU $/hour = ______, utilisation assumption = ____________

Signed: ______________________   Date: ____________   Channel: ____________
