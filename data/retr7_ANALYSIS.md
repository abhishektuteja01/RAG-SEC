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

## Arm 2, same caveat

Arm 2, post-RETR-7, corrected labels: **recall@10 0.514, recall@50 0.708**.
Arm 2 as published:                    recall@10 0.495, recall@50 0.687.

Same confound, same refusal to quote a delta. But the two arms together are mildly
informative. Applying `SESSION.md`'s rough +0.025 old-label correction as a *prediction*:

| arm | published (old labels) | naive corrected prediction | observed (new corpus) |
|---|---|---|---|
| Arm 1 | 0.329 | ~0.354 | **0.337** (-0.017) |
| Arm 2 | 0.495 | ~0.520 | **0.514** (-0.006) |

Both land slightly *below* the label-correction-only prediction, Arm 2 well inside its own
+/- 0.014 confidence interval. That is consistent with RETR-7/RETR-8 being **neutral to
very slightly negative** for first-stage retrieval — which is what `RETR-33` predicted
(`no_gold_chunk` = 0% twice; the heading is ~0.91% of a chunk's tokens).

It is *not* proof: the 0.025 is a project-wide estimate, never measured per arm, so this
table is a sanity check, not an attribution. The scratch-database run below is what would
settle it.

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

## Where the Arm 3 jobs are (morning orientation)

Login nodes are round-robin and `tmux` is per-host. As of 03:50:

- `rr-dev`  -> tmux on **explorer-01**
- `rr-test` -> whichever node the pipeline's ssh landed on; check both
- your original `reembed` session -> **explorer-02**

```bash
ssh tuteja.a@explorer-01.explorer.northeastern.edu 'tmux ls'
ssh tuteja.a@explorer-02.explorer.northeastern.edu 'tmux ls'
squeue -u tuteja.a          # same from either node
```

Job 9950592 (Arm 3 dev) went in **PENDING (Priority)** — waiting for a free V100, which is
the queue risk flagged before launch. Cluster-side logs are `~/rr-dev.log` and `~/rr-test.log`
on shared home, readable from either login node.
