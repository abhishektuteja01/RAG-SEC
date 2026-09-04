# Reading the RETR-7/RETR-8 numbers — caveats before you quote anything

## The Arm 1 / Arm 2 comparison is CONFOUNDED. Do not quote a delta.

Arm 1, post-RETR-7, corrected labels: **recall@10 0.337, recall@50 0.473**.
Arm 1 as published in `SESSION.md`:    recall@10 0.329, recall@50 0.466.

That is **not** a +0.008 improvement from RETR-7. **Two things changed at once:**

1. the corpus (RETR-7/RETR-8 changed 47.5% of chunk texts), and
2. the labels — the published Arm 1/Arm 2 rows are the last two still on the OLD
   pre-`RETR-35` matcher, which `SESSION.md` says understates by roughly 0.025.

Labels are recomputed from chunk text at eval time, so this run is automatically on
corrected labels. Expected Arm 1 on corrected labels *alone*, with no RETR-7, would be
roughly 0.329 + 0.025 = ~0.354. Observed is 0.337. Taken naively that reads as RETR-7
*hurting* by ~0.017 — but the 0.025 is a rough project-wide estimate, never measured for
Arm 1 specifically, so that subtraction is not sound either.

**Honest statement: Arm 1 on the new corpus with corrected labels is 0.337 / 0.473, and
there is no directly comparable prior number.** This is exactly the failure mode
`SESSION.md` §4 names — "reusing a number without checking which job produced it."

## How to get a clean attribution, if you want it

Restore `~/rag-sec-backups/chunks_pre_retr7_20260904.dump` into a *scratch* database and
run Arm 1 and Arm 2 against it. That yields old-corpus + corrected-labels, which is the
missing cell. With it you get a proper 2x2 and can attribute the movement to the corpus
rather than to the relabeling. Not run tonight: it needs a second database and it would
have contended with the arms (concurrent CPU passes measured 8x slower, `SESSION.md` §4).

Cost: one 652 MB restore, then two CPU eval passes. No GPU, no API spend.

## What RETR-7 predicted

`RETR-33` measured `no_gold_chunk` = 0% twice, and the heading is a median 0.91% of a
chunk's tokens, so the predicted retrieval effect was ~0 in either direction. A small
movement of either sign is consistent with that prediction. **The fix shipped as a
correctness fix, not a recall play, and that framing should not change retroactively
because a number moved 0.008.**

## Still not restored

- Arm 4-B / 4-C: deliberately left on pre-RETR-7 headings, so they are no longer
  text-comparable with variant A.
- `RETR-33` failure triage: needs the Arm 3 scores.
- All `COST-*` compression and answer-accuracy numbers: need LLM spend, untouched.
