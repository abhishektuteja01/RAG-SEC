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

## Wall-clock margin on the test job — check this first if Arm 3 test is missing

Both rerank jobs were submitted with `--time=08:00:00`.

| job | started | 8h wall expires | projected finish | margin |
|---|---|---|---|---|
| 9950592 Arm 3 dev  | ~03:50 | ~11:50 | ~09:22 | comfortable |
| 9950641 Arm 3 test | ~04:00 | ~12:00 | **~11:16** | **~45 min** |

Measured rates as of 06:53: dev 4.0 q/min (640/1235), test 3.6 q/min (600/1546). Test is the
larger split (1,546 vs 1,235 questions) *and* the slower rate, so it has little headroom. A
15% slowdown puts it into the wall.

**If it hits the wall, nothing is lost — but the pipeline will not resubmit for you.**
`rerank_hpc.py` checkpoints per question id and skips what is already in the output file, so
resuming is re-running the identical command:

```bash
ssh tuteja.a@login.explorer.northeastern.edu
tmux new -s rr-test2
srun --partition=gpu --gres=gpu:v100-sxm2:1 --cpus-per-task=4 --mem=48G --time=08:00:00 --pty /bin/bash
module load python/3.13.5 && source ~/rerank-env/bin/activate
cd ~ && python -u rerank_hpc.py retr7_rr_test_payload.json retr7_rr_test_scores.jsonl
```

Then, on this machine:

```bash
scp tuteja.a@xfer.discovery.neu.edu:~/retr7_rr_test_scores.jsonl data/
uv run scripts/pipeline/05_arm3_rerank.py score --scores data/retr7_rr_test_scores.jsonl \
    --split test --out data/retr7_arm3_test_results.json
```

Do **not** pass `--baseline` — the scores file already carries `unfiltered_raw`, and the
cached `day6_arm4_A` baseline is pre-RETR-7 and would mix two corpora. `rerank_score.py` now
detects this and skips the merge automatically.

The pipeline writes `## Arm 3 TEST -- DID NOT COMPLETE` into the results file in this case,
so its absence will be explicit rather than silent.

## Arm 3 dev — the clean before/after, and the answer to "did RETR-7 move retrieval?"

Unlike Arm 1/Arm 2, this comparison is **not confounded**: all four Arm 3 cells were already
re-scored under `RETR-35`'s corrected labels, and this run uses the same labeller. Only the
corpus changed.

| cell (dev) | published, pre-RETR-7 | post-RETR-7 | delta |
|---|---|---|---|
| Arm 3 = unfiltered_raw, recall@10 | 0.634 | **0.629** | -0.005 |
| Arm 3 recall@50 | 0.713 | **0.708** | -0.005 |
| filter+strip, recall@10 | 0.765 | **0.760** | -0.005 |
| filter+strip, recall@50 | 0.806 | **0.802** | -0.004 |

**RETR-7/RETR-8 changed retrieval by -0.005, well inside the +/-0.013 stderr. No measurable
effect.** That is exactly what was predicted before building it: `RETR-33` measured
`no_gold_chunk` = 0% twice, and the heading is a median 0.91% of a chunk's tokens. The fix
shipped as a correctness fix with a stated expectation of ~0, and the expectation held.

This also retro-explains Arm 1/Arm 2: their apparent -0.017 / -0.006 against the naive
label-corrected prediction was the +0.025 estimate being imprecise, not RETR-7 doing harm.

### The superadditivity result replicates on the new corpus

| | pre-RETR-7 | post-RETR-7 |
|---|---|---|
| filter alone | +0.040 | **+0.040** |
| strip alone | +0.021 | **+0.017** |
| both together | +0.148 | **+0.131** |
| additive prediction | 0.061 | 0.057 |
| ratio | 2.4x | **2.3x** |

The interaction is reproduced on an independently re-embedded corpus. `RETR-6`'s explanation
survives: without the filter the company name genuinely discriminates among 799 filings, so
stripping it destroys signal; with the filter every candidate is already the right company,
so the name only rewards boilerplate.

Note `unfiltered_stripped` moves recall@50 by exactly +0.000 while moving recall@10 by
+0.017 — stripping reorders the top of the list without changing what is in the pool at all,
which is what a rerank-only change should do.

### The stale-baseline guard fired

`rerank_score.py` printed: *"scores file already has unfiltered_raw for all 1235 questions
-- NOT merging the cached baseline day6_arm4_A_rerank_scores.jsonl"*. Without the fix made
before this run, the pre-RETR-7 cached cell would have been merged into a post-RETR-7 2x2
and the whole ablation would have been silently mis-attributed.
