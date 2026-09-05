# Where the project is, and what to do next

Living document. Overwrite stale lines; don't append to them. Numbers and reasoning live
in `DECISIONS.md` — this file only says where things stand and what to pick up.

Last updated: 2026-09-04, with a paid Day 9 run in flight — see §2's first block before
doing anything. Day 8 closed out, including the `RETR-7`/`RETR-8` re-index
(`RETR-39`) and the last cheap cost item (`COST-34`/`COST-35`/`COST-36`). **Day 10's
observability was then pulled forward ahead of Day 9's run** (`OBS-1`..`OBS-10`) so that
run produces its own traces instead of needing a second paid pass.

**"Day N" is a unit of planned work in `spec.md`, not a calendar date** — Day 8 spanned
several days. The `COST-`/`RETR-` IDs follow the project label, not the calendar, so a
`day8_*` data filename says nothing about when it was written. Scripts carry no day prefixes:
`scripts/` is organised by job and every file is named for what it does. `INVENTORY.md` maps
every file to what it actually does.

---

## 1. Where things stand

**The headline, confirmed out of sample.** Two changes, both resolved from question text
alone, neither needing re-embedding: filter candidates to the company the question names,
and strip the company/filing framing from the query *before reranking only*.

| test split (untouched until the end) | before | after |
|---|---|---|
| recall@10 | 0.607 | **0.747** |
| nDCG@10 | 0.468 | 0.617 |

Re-measured 2026-09-04 on the post-`RETR-7` corpus (`RETR-39`); the pre-re-index pair was
0.604 -> 0.752, every cell moving -0.002 to -0.005, inside stderr. The gain has survived two
corpus-level changes: +0.145 pre-label-correction, +0.148 after it, **+0.140 after the
re-index**. MRR is deliberately absent — not comparable across labelings (`RETR-35`).

Superadditive, and it replicated on the re-embedded corpus: dev filter alone +0.040, strip
alone +0.017, together **+0.131**; test +0.037 / +0.025 / **+0.140**. Both 2.3x the additive
prediction. The gain is *larger* on test than dev, which is the anti-overfitting evidence,
since dev informed the resolver's design. **Quote the test number, 0.747.**

**Trustworthy right now**, recall@10 / recall@50, all variant-clean and on `RETR-35`'s
corrected labels unless marked:

| arm (dev unless stated) | recall@10 | recall@50 |
|---|---|---|
| Arm 1, dense only | 0.337 | 0.473 |
| Arm 2, + BM25/RRF | 0.514 | 0.708 |
| Arm 3, + reranker (= Arm4-A) | 0.629 | 0.708 |
| Arm 3 + filter + strip | **0.760** | 0.802 |
| Arm 4-B / 4-C (NOT re-indexed) | 0.237 / 0.236 | 0.268 / 0.270 |
| **Arm 3 + filter + strip, TEST** | **0.747** | 0.785 |

**All rows are post-`RETR-7` and on corrected labels except Arm 4-B/C**, deliberately not
re-embedded and no longer text-comparable with variant A. Arm 1/Arm 2 movement against the old
published 0.329/0.495 is **confounded** — corpus and labels changed at once, so quote the
level, never a delta. `data/retr7_ANALYSIS.md` records how to get the missing cell from the
pre-re-index `pg_dump`.

**Compression was measured and never shipped.** `agent.py` sends uncompressed evidence —
`rag_sec/compress.py` and the slice scripts exist only in the offline experiments. slices@1500
would cut ~$13.00/pass to ~$5.27 (2.5x, not the 4.1x once claimed) and cost roughly 9 points of
answer accuracy. The failure mode is safe: the model refuses rather than fabricates.

**Cost work is closed (`COST-36`).** dev+test uncompressed on batch at `medium` is **$14.69**
against a ~$30 envelope, so cost is no longer a design constraint. Compression's entire saving
across both splits is ~$9 for 9-11 accuracy points — **do not slice at any budget**, including
the 3000 that looks like the compromise. Drop cost arguments from decisions that also have
accuracy arguments. Batch is now justified by rate limits, not price.

---

## 2. Open questions, in the order worth doing them

### Day 9 IS RUNNING RIGHT NOW — read this before touching anything

**A live paid run is in progress** (started 2026-09-04 evening, ~65/198 questions at last check,
0 errors, ETA ~3h). **First action in any new session: check whether it is still alive**, because
everything below forks on the answer:

```bash
pgrep -f '[a]gent_run.py' && echo ALIVE || echo STOPPED
wc -l < data/day9_arm6_dev_results.jsonl        # rows done
grep -c ERROR data/day9_run.log                 # should be 0
tail -3 data/day9_run.log
```

- **If ALIVE:** do not edit `src/rag_sec/**` or `scripts/eval/agent_run.py`. The run is resumable,
  so editing mid-flight means half the results file came from a different build — the provenance
  `spec.md:136` requires. Docs, analysis scripts and new files are fine.
- **If STOPPED before 200 rows:** re-run the identical command below. It keeps successes, retries
  errors, drops duplicates, and wipes each question's stale checkpoint. Do NOT delete the results
  file — that was only correct once, when `AGENT-16` invalidated every row in it.

```bash
caffeinate -is uv run scripts/eval/agent_run.py -n 200 --concurrency 1 2>&1 | tee -a data/day9_run.log
```

**Environment requirements, all learned the hard way:**
- **`--concurrency 1`, always.** Concurrent MPS model construction segfaults the machine
  (`AGENT-10`/`AGENT-17`). The default is now 1; do not raise it.
- **Mains power.** A 3-4h MPS-saturating run flattens this battery, and `caffeinate` prevents
  sleep, not battery death.
- **Docker must keep running** — Postgres is the `rag-sec-postgres-1` container on `localhost:5432`,
  and every question retrieves from it and checkpoints into it. Quitting Docker kills the run.
- **Nothing else CPU- or memory-heavy.** 16 GiB total against a Docker VM plus two transformer
  models; swap hit ~10 GB during this run. Two concurrent CPU passes measured 8x slower, not 2x.

**When the run finishes, in order (all free, no GPU, no API):**
1. `uv run scripts/eval/agent_analyze.py` — the authoritative trajectory / cost / accuracy numbers.
2. `uv run scripts/eval/worst_failures.py` — **must be regenerated**; the earlier output was deleted
   as stale, having been built on pre-`AGENT-16` rows (`spec.md` 2.2 rule 5, `OBS-12`).
3. Capture Langfuse dashboard and trace screenshots (`OBS-11`) — **Hobby retention is 30 days**, and
   Day 14's README needs them, so do this during Day 10, not at the end.
4. Commit `data/day9_arm6_dev_results.jsonl` and `data/day9_run.log`. They are deliberately
   uncommitted while in flight; everything else is committed as of `d066358`.
5. Then Day 10's real half: diagnose the ten worst failures from their traces.

**Where each number goes:**
- Trajectory, cost, latency, judge accuracy, paired answer accuracy -> `DECISIONS.md`, a new Day 9
  row, and the results table at the top of that file if an arm-level figure changes.
- `$/question` comes from `agent_analyze.py`, never from the Langfuse dashboard — per-trace cost is
  not expressible there (`OBS-11`).
- Per-arm cost/latency comparisons come from the two separate traces, never summed (`OBS-9`).
- **Stage latencies from this run are NOT publishable.** See the contamination note below.
- Nothing from a partial file is a result. Quote nothing until the run completes.

**Owed once it finishes, and each one is a correction, not new work:**
- **`AGENT-13` is probably wrong and must be revised.** It says `COST-30`'s 15.1% insufficiency rate
  "does not transfer", written on n=5. At n=60 the first-iteration rate is **10.0%** — so it roughly
  does transfer and the pilot's 16-60% was noise. Recompute at n=200 and rewrite the row.
- **`OBS-10`'s cost split and loop multiple need restating at n=200.** The multiple fell 3.6x (n=3)
  -> 1.80x (n=60) as single-iteration questions dominated; the row already says not to quote it.
- **A clean stage-latency re-measurement.** `embed_s` p50 read **12.89s** at n=60 against 0.33-0.57s
  in the quiet two-question pilot, with model construction already excluded (`model_init_s` = 0.00).
  A 20-40x slowdown on embedding one short query is paging, not compute. So `spec.md:121`'s p50/p95
  and Day 13's gated p95 must come from a re-measurement on a quiet machine — free, since retrieval
  latency needs no LLM calls — not from this run.

**Early signal at n=60, explicitly not a result:** looping adds essentially no retrieval. Iteration 1
alone scored recall 0.692 / nDCG@10 0.600; all iterations unioned scored 0.708 / **0.600**; the static
arm scored 0.708 / 0.604. Mean iterations 1.25, cap hit 5%. Answer accuracy favoured the loop
73.1% vs 69.2% on 2-0 discordant pairs, which at n=52 is not significant. This is the direction
`COST-30` predicted, and if it holds at n=200 it is the honest headline for Day 9.

**One trace gotcha before you open a trace.** There are more root observations than questions x 2:
deterministic trace-id seeding (`OBS-8`) makes a re-run question *merge* into its existing trace
rather than fork one — which is what makes resume idempotent. So the ~21 questions that also ran in
the aborted attempt carry two `answer-question` roots in one trace, one from pre-`AGENT-16` code.
**Disambiguate by timestamp**, or you will diagnose the buggy attempt.

---

### How this run came to be restarted

**A first attempt was stopped 18 questions in and archived** (`data/archive/day9_static_unsorted*`):
the static baseline was reading the published rankings in first-stage order, so it was Arm 2 +
filter rather than Arm 3 + filter + strip (`AGENT-16`). Nine further fixes came out of the
pre-restart audit (`AGENT-17`), three of them run-killers. Everything below is post-fix.

**The battery hypothesis was wrong, and this is worth remembering.** The aborted attempt's
rerank drift (22.9s -> 66.8s mean, 2.9x) was blamed on battery power. It reproduced almost
identically on mains, at the same inflection point. The cause is **memory pressure**: swap grew
from 1.7 GB to ~10 GB during the run on a 16 GiB machine, with the Docker VM holding Postgres and
two transformer models in MPS unified memory. Neither `ps` RSS nor thermal warnings show it —
`ps` cannot see MPS buffers or the VM's footprint, and no thermal warning was ever recorded.
**Check `sysctl vm.swapusage` before blaming heat or power.** The drift also partly reverses when
a competing process is killed, so CPU contention compounds it.

**Tracing is live and the run needs no extra flag.** Spans go to Langfuse Cloud whenever
`LANGFUSE_*` is in `.env` and no-op silently when it is not, so a keyless machine or CI runs
unchanged. Two traces per question (loop arm, static arm) grouped by `session_id` = question
id; every row carries `trace_id`, `trace_id_static` and `session_id` as the join. **Hobby
retention is 30 days, so capture the Day 10/14 screenshots within a month of the run.**

**What the run produces**, one JSON row per question — built as a one-off that also feeds
Days 10/11/13, because re-running costs money: trajectory + per-iteration queries/verdicts,
`usage` per LLM call (tokens, cost, latency, `cached_input_tokens`), `stage_latency`
(embed/search/rerank), top-10 + 50 pre-rerank candidates + full reranked list per iteration,
`gold_chunk_ids`, both gold answer fields, `trace_id`/`trace_id_static`/`session_id`, and
**`static_baseline`** — the paired Arm 3+filter+strip answer over the *published* `RETR-39`
ranking (`data/retr7_rr_dev_scores.jsonl:filtered_stripped`), so retrieval is not re-run and the
comparison is paired by construction. **That file stores candidates in first-stage order with
rerank scores merely attached, so the loader must re-sort** — not doing so was `AGENT-16`, and
`scripts/checks/static_ranking_order.py` now fails the run rather than trusting it.

**Bugs fixed before and during this run are in `DECISIONS.md`, not here**: `AGENT-9`..`AGENT-12`
(answer-format, MPS thread-safety, file-level resume, errored-rows-counted-as-done), `AGENT-16`
(the static-baseline ordering bug that forced the restart) and `AGENT-17` (nine more from the
pre-restart audit, three of them run-killers). **The two operational rules that follow from them:**
run serial — concurrent MPS model construction segfaults the machine, and the reranker is the bulk
of wall clock so concurrency buys nothing — and run on **mains power**. The concurrency at which
the original SIGSEGV was reproduced is unestablished (`SESSION` said 4, the code comment said 2, no
log survives); `AGENT-10` records it as ">1" and it should not be quoted.

**Measured on the pilot, and it moves the numbers:** `COST-30`'s 15.1% insufficiency rate
**does not transfer** — it was measured on one-shot `COST-13` responses, not a loop's partial
evidence. Observed **11-60%** depending on sample (`AGENT-13`; the 16% low end does not
reproduce), and questions terminating on the **cap** rather than a verdict. `AGENT-8` is
**confirmed** — prefix caching fires — but **quote neither the cache share nor the loop's cost
multiple from a pilot**: both swing hard with the iteration mix (multiple 3.6x at n=3 vs 1.96x at
n=9; cache 12.4-26.1%), because a single-iteration question caches nothing and costs little
(`OBS-10`, `AGENT-14`). Take both from the finished run.

**Still owed after the run:** regenerate the worst-failures files — `scripts/eval/worst_failures.py`
exists and its earlier output was **deleted as stale**, having been built on the pre-`AGENT-16` rows.
Then Day 10's real half: diagnose the ten worst failures from their traces, and capture dashboard
screenshots while the traces are still in retention. **The documentation debt is now cleared** —
`AGENT-9`..`AGENT-17`, `COST-38` (flex, deliberately unestablished), `COST-39` (CI gate) and
`OBS-1`..`OBS-12` are all written.

**Budget:** $40 total, ~$6 spent (including ~$0.75 on the aborted attempt and ~$0.5 on pilots).
Day 9 projects **$7-13** — the spread is the cap-hit rate, not uncertainty about rates. Reserve
after Days 10-14 (~$19): ~$2-8.

### Decided: Day 9's premise is false, and we report that

**Every question in dev and test maps to exactly one filing** (1235 and 1546, checked
2026-09-04). `spec.md:383` asks for Arm 6 measured "specifically on the multi-document
questions" — that subset is **empty**, so the loop's best case is untestable on this
benchmark. This sharpens rather than softens `COST-30`, which deferred iteration's published
strength (multi-hop composition *across* documents) to Day 9. The only `multi_doc` flag in the
repo (`scripts/analysis/pack_variants.py:287`) measures whether *retrieved chunks* span
filings — `RETR-3`'s contamination finding, a property of the result, not of the question.

**Decided** (`AGENT-15`): run the trajectory half — calls, tokens, dollars, wall clock,
sufficiency-judge accuracy, all runnable and all single-document — **and report the negative**.
Written up in `data/day9_multidoc_negative_ANALYSIS.md`, which re-derived the property
independently and strengthened it: train is single-filing too (8,958, zero), so widening the pool
cannot rescue it, and `page_number` is single-valued as well — one *page* per question, not merely
one filing. Also quantified there: **50.4% of dev questions name two or more distinct years and
still need exactly one filing**, because 10-Ks reprint prior-year comparatives — the same
reprinting `RETR-3` blames for retrieval failures. Building a multi-hop set was rejected as scope
`spec.md` never budgeted.

### Free — no GPU, no API spend

- **Day 6's actual question, deferred not dropped.** For questions needing arithmetic across
  >=2 table cells, does any of the A/B/C table-indexing strategies surface all the needed
  evidence — or is it a retrieval-vs-computation gap? Data dependency satisfied (`GOLD-1`);
  the analysis has never been started.
- **A trap, not a lead.** Restricting `COST-27`'s figure guard to the top chunk alone scored
  +1.3 (6 gained / 2 lost, **p=0.29**) and was the best of six swept variants. A hypothesis
  with a test-split price on it. Do not quote it; do not ship it on the dev number.

### Then back to `spec.md`

**Days 10-14.** 10: **instrumentation half is done** (`OBS-*`) — what remains is the half
`spec.md` calls "the skill", diagnosing the ten worst failures from their traces, plus a dashboard
screenshot pass. 11: CI
quality gate, citation grounding, unanswerable set, first interview drill. 12-13: AWS, MCP
server, latency pass with a stated p95. 14: README, writeup, final drill.

### Decided against

- **Arm 5 (late interaction / multi-vector).** Dropped without building: 378GB of storage
  against Arm 1's 389MB, plus MaxSim needs an index pgvector doesn't have.
- **Merging `judge` into `answer`** (`COST-30`). Its cost leg is now moot (`COST-36`), but it
  stands on the other two: CRAG's loop scores recall@5 0.658 against hybrid+rerank's 0.816 on
  T²-RAGBench itself, and judges discard answers that were often already right (models answer
  correctly 35-62% of the time under insufficient context).
- **Loop on/off ablation.** Arm 3 already *is* the loop-off arm.
- **`slices50`.** Lost at all five budgets on the clean pool.
- **`RETR-2`, widening `TOP_K`** (`RETR-33`). Recovered 4.3 points on the Day 6 ordering;
  recovers 1.2 on the shipped one, because the whole reranker bucket is now 1.5 points.
- **`thinking_level=low`** (`COST-34`). Real saving, wrong trade — see §3.

---

## 3. Things worth saying in an interview

Each with its caveat attached — saying the caveat first is what makes the claim land.

**1. I diagnosed before I optimized, and it moved the target.**
Sorted all 1,235 dev questions into single failure buckets: 73.5% fine, 6.9% the reranker's
fault, 19.6% the gold never entered the candidate pool, 0% chunking. The reranker was already
at 89% of its own ceiling — 7.6 points of headroom — so I stopped working on it. Then 100
hand-read failures showed 74% of retrieved chunks came from the wrong document, because 10-Ks
reprint themselves near-verbatim and neither retriever weights the year token.
*Caveat:* the 0% chunking bucket relies on a lenient whole-page matcher, so it rules out the
catastrophic version of that hypothesis, not the subtle one.

**2. The fix was superadditive, and I can say why.**
Filter alone +0.040, strip alone +0.017, together +0.131 (dev, post-re-index). Without the
filter the company name genuinely discriminates among 799 filings, so stripping it destroys
real signal; with the filter every candidate already matches the company, so the name only
rewards corporate boilerplate. I ran the strip-alone cell specifically so the interaction
would be attributable — with three cells the conclusion would have been "stripping helps,"
which is false.
*Caveat:* dev informed the resolver's design. The generalization rests on the untouched test
split, which is why the headline is test's **0.747** and not dev's 0.760.

**3. I found a bug that had been silently corrupting my results, and the guard isn't the
obvious one.**
Retrieval queries omitted a `variant` predicate, so a dropped experiment's rows were eligible
candidates and got scored against the gold labels as if they were different chunks — 9.2% of
filtered candidate slots. Found because the database reported 106,027 chunks against a
documented 99,654. The interesting part: the contamination ran *against* my hypothesis — it
suppressed the measured gain rather than inflating it, so every number rose after the fix. The
guard is a **per-variant** count, because the total was the only thing that diverged, plus an
AST scan that fails any `FROM chunks` read whose WHERE clause omits `variant`, run pre-connect
so it fires with the database down.
*Caveat:* it cost a 4.4-hour GPU re-run, and it blocks a contaminated pass from *starting* —
it can't vet a result already written to disk.

**4. I measured a cost optimization and then argued against shipping it.**
`gemini-3.7-flash` bills reasoning as output, and thinking was 86% of billed output — a
~$3.9/pass floor that compression cannot touch, because it stays near-constant as the prompt
shrinks. `thinking_level=low` cuts that floor 35%. I kept `medium`: the saving is $1.1/pass in
absolute terms, all 4 discordant pairs went the other way, and the whole benchmark costs
$14.69. Knowing the number is what let me decline it.
*Caveat:* `p=0.13` means no *detectable* harm at n=278, not that low is safe — and `COST-27`'s
n=30 pilot sign-flipped at n=300.

**If pushed on cost:** the compression work is a genuine accuracy-for-cost trade whose headline
had to be retracted once retrieval improved. Being able to say "my earlier conclusion didn't
survive better retrieval, and here's the measurement that killed it" is worth more than the
original claim.

---

## 4. Housekeeping

- **The recurring failure mode: reusing a number without checking which job produced it.**
  `RETR-24` (B/C chunks scored as A), `RETR-30`'s mis-anchored 4.3 GPU-h estimate (that was
  `RETR-22`'s chunk-level figure), and the near-miss filename clobber are the same mistake — a
  value correct in one context, silently wrong in the next. Check provenance before quoting a
  number, including your own.
- **Still owed a row: the 9.4x loop cost multiplier — provenance could NOT be established.**
  Searched 2026-09-03: nothing on disk produces it, and git only ever shows it in this bullet.
  The nearest candidates are coincidences — `$16.97 -> $3.10/pass` is 5.5x, and `COST-24`
  marked every one of those figures suspect. It most likely came from the deleted Day 8 session
  file. **No row was added, deliberately:** a row is a provenance claim, and inventing one is
  worse than the number having no home. Don't quote it; re-derive it or drop it.
  **This is the standing argument against deleting this file.**
- **A documented GA feature can still be unusable.** `service_tier="flex"` 503'd ~9 of 10
  requests and `COST-13` actually ran at `standard` — recovered from latency (p50 2.74s vs
  flex's 1-15 min target) and billing ($2.22 vs the $2.23 dashboard), since the stored rows
  carry no `tier` field. **Batch, not flex, is the 50% route**, and the two do not stack.
- **Batch caps enqueued tokens per account, not per job** (`COST-35`). Two concurrent 1.63M
  submissions 429'd the second at `batches.create`; sequential succeeded, nothing billed for
  the failure. `--max-enqueued` splits only *within* one invocation. Run arms sequentially and
  call a level comparison same-day, not simultaneous. **Still unchecked:** the key's tier in AI
  Studio, which `COST-29` asked for and would distinguish the 3M cap from a concurrent-job
  limit of 1.
- **This machine is the binding constraint on any GPU run, and it is memory, not heat.** 16 GiB
  total against a Docker VM holding Postgres plus two transformer models in MPS unified memory.
  Swap reached ~10 GB during the Day 9 run and `embed_s` inflated from ~0.4s to ~13s — paging, not
  compute. **Diagnose with `sysctl vm.swapusage`, not `ps` RSS** (which cannot see MPS buffers or
  the VM) and not `pmset -g therm` (no thermal warning has ever been recorded on this machine).
  Any latency this project publishes needs the machine state recorded next to it.
- **Don't run two CPU passes concurrently.** Measured 2026-09-01: contended ~0.5 it/s vs
  ~3.9 it/s alone — **8x, not 2x**. Each compression pass took ~5 min alone against a
  projected 40. Run them sequentially.
- **`RUNBOOK.md` holds the HPC procedure as commands** — hosts, module/venv setup, and the
  round-robin login-node gotcha (`tmux` is per-host, so a session can look missing).
  **Interactive `srun` was chosen over `sbatch`** for live visibility; the two `.sbatch` files
  stay as the unattended fallback. Stagger submissions past model load — both pull the same
  reranker into one HF cache, and two cold downloads to one path risks corruption.
- **`CLAUDE.md` is gitignored on purpose** (personal working rules). Keep the standing context
  there short and number-free — an unversioned file is a bad last home for a number.
- **Guards that exist and should stay green:** `scripts/checks/candidate_sql.py` (fails any
  `FROM chunks` read whose WHERE omits `variant`), and the per-variant chunk count.
