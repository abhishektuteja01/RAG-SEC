# Where the project is, and what to do next

Living document. Overwrite stale lines; don't append to them. Numbers and reasoning live
in `DECISIONS.md` — this file only says where things stand and what to pick up. **§5 is the
Day 9-14 checklist**; it refers to §2's numbered command lists rather than repeating them.

Last updated: 2026-09-05, **Day 9's run is COMPLETE** — 200/200 questions, 0 errors, 339.6
min. Everything on the old run-list is done except the two things that need a human
(Langfuse screenshots, AWS). Day 8 closed out, including the `RETR-7`/`RETR-8` re-index
(`RETR-39`) and the last cheap cost item (`COST-34`/`COST-35`/`COST-36`). Day 10's
observability was pulled forward ahead of the run (`OBS-1`..`OBS-12`), so it produced its
own traces. **Read §2's first block before quoting any Day 9 number** — three of them are
not what the pilot predicted, and two published numbers turned out to be non-comparable.

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
| Arm 6 loop, iteration 1 only (n=200 subset) | 0.703 | — |
| Arm 3 + filter + strip, same 200 (paired) | 0.736 | — |

**The last two rows are the Day 9 paired subset (n=200), not the full dev split** — they are
comparable to each other and to nothing else in this table, and the loop is *behind* because
iteration 1 searches a `plan`-rewritten query (`AGENT-19`). Arm 6's case rests on answer
accuracy (69.2% vs 61.6%, p=0.00098), not on retrieval.

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

### Day 9 is DONE — read this before quoting any of its numbers

**The run finished 2026-09-05: 200/200 questions, 0 errors, 339.6 min, serial, on mains.**
`data/day9_arm6_dev_results.jsonl` and `data/day9_run.log` are committed. Full numbers are in
`DECISIONS.md` (`AGENT-19`..`AGENT-24`, `DEPLOY-5`/`DEPLOY-6`, `CI-2`); this section says only
what changed and what to be careful with.

**The headline reversed, and that IS the interview story.** The n=60 early signal said the loop
was a negative result and `AGENT-15` had already decided to report one. At n=200 the loop wins
answer accuracy **69.2% vs 61.6%**, discordant **14-1**, McNemar exact **p=0.00098**. But its
retrieval is *worse*: iteration-1 recall@10 **0.703** against the static arm's **0.736**,
because iteration 1 searches with a query `plan` rewrote rather than the raw question that
filter+strip was tuned for. **So the win is real and it is not a retrieval win** — the answer
stage sees the deduped union of every iteration (28-30 chunks vs 10) and refuses less often.
Quote the accuracy pair and the p-value; never quote a retrieval gain (`AGENT-19`).

**Two numbers this run printed are NOT comparable and must not be quoted** (`AGENT-22`):
`agent_analyze.py` scores the union row at `k=len(got)` — up to 40 chunks — against the static
row's recall@10, so its `0.756 vs 0.736` compares a 40-slot budget to a 10-slot one. And the
union's nDCG@10 is *identically* iteration 1's by construction, because the union is built in
iteration order, so the 0.593/0.593 match is arithmetic, not a finding.

**`AGENT-13` and `OBS-10` are revised, as they were owed.** First-iteration insufficiency is
**13.0%** at n=200, so `COST-30`'s 15.1% roughly DOES transfer and the pilot's 11-60% was
noise. Cost split is answer 64% / judge 19% / plan 17% at a **1.77x** loop multiple (not 3.6x),
and prefix cache is **4.0%** (not 26.1%) — all three moved because 174 of 200 questions ran a
single iteration, and a single-iteration question neither replays history nor caches.

**Stage latency is now measurable, and the old diagnosis was wrong** (`AGENT-20`/`AGENT-24`).
The drift was never battery, heat, or contention with the Docker VM: it is MPS's caching
allocator never returning freed blocks on a 16 GiB unified-memory machine. `ps` RSS cannot see
it (0.1 GB against 7 GB of swap growth) and swap fell 9.8 -> 2.8 GB the instant the process
exited. **One line — `torch.mps.empty_cache()` per retrieval, now in `retrieve.py` — removes
it.** Laptop steady state: embed p50 **0.075s**, search **0.241s**, rerank **32.60s**, total
**32.97s** / p95 **36.53s**, swap flat. The run's own p50s (embed 12.98s, rerank 59.59s) were
saturated-regime numbers and are superseded.

**69.2% is a floor, not an estimate** (`AGENT-21`/`AGENT-23`). Four questions whose gold is
yes/no are scored wrong in both arms because `answer_eval` requires a number on the ANSWER
line, though the model answered correctly; replaying with `yes->1.0`/`no->0.0` gives arm6
71.5% / static 63.4%, moving the gap by 0.5. **Not fixed deliberately** — redefining a metric
after seeing its results is a human decision. And at least one 'reasoning failure' is a wrong
gold: `finqa_dev_147`'s own filing table gives 78.93%, which is what the model said, against a
gold of 34.96 that reads the 2009 column.

**What still needs a human, and nothing else does:**

1. **Langfuse dashboard + trace screenshots** (`OBS-11`). Hobby retention is 30 days from
   2026-09-05. Disambiguate by timestamp: ~21 questions carry two `answer-question` roots in
   one trace because deterministic trace-id seeding merges a re-run into its existing trace,
   and the older root is pre-`AGENT-16` code.
2. **AWS**, the binding ~10h block — but see `DEPLOY-6` before sizing anything.
3. **Decide the yes/no scorer question** (`AGENT-21`). Free to replay either way.
4. **Decide the container's reranker route** (`DEPLOY-6`). This one gates Day 12/13.
5. **Review `retrieve.py` and `api.py`** — both were changed during this session.

**Two things found by finally building and running what Day 10-12 had only written:**

- **`DEPLOY-1`'s image had never been built and could not have built** (`DEPLOY-5`): five
  defects, three fatal to the build, two that would have shipped silently — `stage_latency`
  was always `{}` (so Day 13's p95 gate would have gated an empty dict), and `python:slim`
  ships no wordlist, so `RETR-11`'s guard is **off in the container** and the deployed
  resolver is not the benchmarked one. **That last one is still open.**
- **The CPU container is far too slow to serve interactively** (`DEPLOY-6`): 212.4s for one
  question, 207.8s of it the cross-encoder, at 755% CPU. A 2-vCPU free-tier host implies
  ~800s/question. Correctness is fine — top citation 0.996, right answer. **Do not size the
  EC2 host until the reranker route is chosen.**

**Budget:** $40 total. Day 9 came in at **$4.10** for 200 questions ($0.0205/q), plus ~$0.60
for the unanswerable pass and ~$0.01 for one container smoke test. Roughly $29-30 remains.

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

**Days 10-14 — §5 holds the item-by-item state; this is only what moved.** Three halves are now
done ahead of their day and all three were written during the Day 9 run, so **none has touched real
data or a real build**: Day 10's instrumentation (`OBS-*`), Day 11's gate scripts and workflow
(`CI-1`, tested only on a synthetic fixture), and Day 12's serving layer and `Dockerfile`
(`DEPLOY-1`, never built). What is genuinely untouched: diagnosing the ten worst failures from
their traces, citation grounding, the unanswerable set, both drills, **AWS (~10h, the binding
block)**, the MCP server, the p95 pass — which `DEPLOY-1` moves into the container — and Day 14's
README and writeup.

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
- **This machine WAS the binding constraint on any GPU run, and one line fixed it**
  (`AGENT-24`). The cause was never heat, battery, or contention with the Docker VM: MPS's
  caching allocator does not return freed blocks, which is free on a discrete GPU and ruinous
  on 16 GiB of unified memory. Proof it was intra-process: a retrieval-only pass running
  ALONE reproduced the whole curve, and swap fell 9.8 -> 2.8 GB the moment that process
  exited. `retrieve()` now calls `torch.mps.empty_cache()` per question and stage latency is
  flat over 25+ questions. **Still diagnose with `sysctl vm.swapusage`, never `ps` RSS** —
  RSS reported 0.1 GB against 7 GB of swap growth, because it cannot see MPS buffers — and
  never `pmset -g therm` (no thermal warning has ever been recorded here). **Numbers measured
  before 2026-09-05 in a long pass are saturated-regime and are not the system's latency.**
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


---

## 5. Day 9-14 checklist

Where each day stands. The old "run-list #N" numbering is retired — that list is finished.
**Everything unchecked below is either a human task or a decision, except where marked.**

### Day 9 — Arm 6 agentic loop — COMPLETE
- [x] Paid run — 200/200, 0 errors, 339.6 min, $4.10
- [x] Trajectory/cost/accuracy numbers (`AGENT-19`, `AGENT-22`)
- [x] Worst-failures files regenerated, both arms
- [x] Results jsonl + log committed
- [x] `AGENT-13` revised, `OBS-10` restated at n=200
- [x] Multi-doc negative written up (`AGENT-15`)

### Day 10 — Observability
- [x] Instrumentation half (`OBS-1`..`OBS-12`)
- [ ] **Dashboard + trace screenshots — YOU. Retention expires ~2026-10-05.** Disambiguate the
      ~21 double-rooted traces by timestamp; the older root is pre-`AGENT-16` code
- [x] Ten worst failures diagnosed — `data/day9_failure_diagnosis_ANALYSIS.md`, a **machine
      draft**; one claim verified by hand against the filing (`AGENT-23`), the rest are not
- [ ] Read that draft and keep or cut its unverified claims

### Day 11 — Gate and drill
- [x] Gate scripts + workflow (`CI-1`)
- [x] Real fixture built (400q), gate green on real data (`CI-2`)
- [x] Gate proven to fire — degraded fixture exits 1, clean exits 0
- [x] Refusal correctness measured — 47/48, and the one miss is our label (`EVAL-2`)
- [ ] Fix or drop `unans_034`, which is answerable (`EVAL-2`)
- [x] Unanswerable set written and validated (`EVAL-1`)
- [ ] Citation-grounding check — not started
- [ ] Interview drill #1 — not started
- [ ] Answer-accuracy leg, on-merge not per-push (`COST-39`) — now unblocked, baseline exists

### Day 12 — AWS
- [x] Serving layer, `Dockerfile`, `.dockerignore` (`DEPLOY-1`)
- [x] Image built, boots offline as non-root, `/health` `/ready` `/ask` all good (`DEPLOY-5`).
      Five defects fixed; **9.06GB image, 2.17-2.75GiB resident**
- [ ] **Fix the container wordlist** (`DEPLOY-5`) — `python:slim` has none, so `RETR-11`'s guard
      is off and the deployed resolver is NOT the benchmarked one. Install `wamerican` or bake a
      list and set `COMPANY_WORDLIST`. **Do this before deploying anything**
- [ ] **Decide the reranker route** (`DEPLOY-6`) — 207.8s per question on CPU. Cut `CANDIDATE_K`
      for serving / ONNX-quantise / bigger instance / go async. **Gates instance sizing**
- [ ] **AWS account** — user-owned (card, MFA). Pick the **Free** plan at signup, not Paid
- [ ] Verify the $100 landed: Billing -> **Credits**, a real row with amount + expiry. Empty page = fall back
- [ ] Zero-spend Budget **before any workload** — the "Free plan shuts down instead of billing" claim is
      unconfirmed by AWS
- [ ] Four remaining $20 tasks, then re-check Credits that they paid out. Delete the RDS instance after
- [ ] Report whether the second $100 arrived — it sets instance size and how long the URL stays up
      (`DEPLOY-4`). All five gate everything below
- [ ] IAM, ECR, S3 — no local build needed, and the image now exists to push
- [ ] Both containers on one EC2 host via docker-compose, arm64/Graviton (`DEPLOY-4`). Postgres from
      `Dockerfile.postgres` — **not** RDS; `pg_search` rules it out (`DEPLOY-2`). **Size from
      `DEPLOY-6`'s decision, not from the 2.17GiB alone**
- [ ] Corpus load — needs a `pg_dump` of the 2.1 GB live DB. **Unblocked now**
- [x] Bedrock dropped, generation stays Gemini (`DEPLOY-3`, closes `AGENT-18`)

### Day 13 — Ship and latency
- [ ] Deploy finished, live URL, single EC2 host, up through Day 14 (`DEPLOY-4`) — depends on
      Day 12 and on the credit check; falls back to deploy/screenshot/teardown if credits fall short
- [ ] MCP server — not started
- [ ] p95 measured **in the container** (`DEPLOY-1`) — the field it reads is fixed and populated
      (`DEPLOY-5`), but the number it currently gives is 212s (`DEPLOY-6`)
- [ ] Gate enforces p95 — **blocked on `DEPLOY-6`**, not on the gate

### Day 14 — Ship
- [ ] README benchmark table + trace screenshots — depends on the screenshots above
- [ ] The written post
- [ ] Final drill

**What actually remains is smaller than it was, and differently shaped.** Days 9 and 11 are
done. Day 10 needs only screenshots and a read-through. Day 12's *code* half is proven rather
than drafted — but building it surfaced two blockers that did not exist as known work
yesterday (`DEPLOY-5`'s wordlist, `DEPLOY-6`'s 207.8s rerank), and `DEPLOY-6` is a design
decision, not a task. **AWS is still the binding block, and it is now second in line behind
deciding how the reranker gets served.**
