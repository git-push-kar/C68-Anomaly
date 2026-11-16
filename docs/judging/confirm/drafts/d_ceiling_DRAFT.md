# DRAFT — unsigned. Nothing here takes effect until the owner signs it.

# Decision: D's fallback ceiling, 2026-11-16

## The situation

D is the validator writer: it writes, checks its own output against the evidence store, retries, and
hands the case to plain A when its retries are exhausted. The plan declared that D may hand off on at
most **25 %** of rows, above which it is not a distinct condition.

Measured on the 520-row fit set: **142 rows handed off = 27.3 %**, twelve rows over the line
(`outputs/overnight_2026-11-05/n4_cells.json:seam_certification.D_fallback_rows`).

Because the hand-off rows are scored by A, D's own writer is judged only on the 73 % it wrote. That is
why D shows the lowest numeric-defect count in the table (7) — the count is real, but it belongs to a
smaller denominator than every other cell's.

## Option 1 — recommended: accept, and report D as "A with extra steps"

> **Accepted.** D fell back on 142 of 520 rows (27.3 %) against the declared 25 % ceiling. D is
> therefore reported as **"A with extra steps"**, not as a distinct writer: its numeric count of 7 is
> published with the 27.3 % fallback rate beside it, and no claim is made that D reduces defects on
> the rows it wrote itself. No cell is re-cut and no threshold is moved.

**Why this is the right option.** By 2026-11-16 the number 142 was already visible. Re-defining what
counts as a fallback at this point would be a rule change made after a result was known, which §12
forbids — and a reader would be right to ask what else moved once the answer was in view. Twelve rows
over a declared line do not touch the paper's primary contrast (B vs A: 60 vs 17 discordant, p ≈ 0),
so the cost of accepting is one sentence and the credibility gained is the whole point of having
declared a ceiling.

## Option 2 — re-cut D and re-run the cell

Change the fallback definition (for example: count only hard validation failures, or allow more
retries) and re-run the N4 stage for D. This would put D inside its ceiling with a defensible
definition, but the definition would be chosen knowing what it has to accommodate. If you want this,
it should be dated and stated as a post-hoc re-cut, with the original 27.3 % still reported beside it.

## Signature

Option chosen: ______ (1 = accept / 2 = re-cut)

Signed: ______________________   Date: ____________   Channel: ____________
Transcribed by: agent (this file is a draft; the signed text will be committed separately)
