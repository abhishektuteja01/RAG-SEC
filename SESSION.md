# Where the project is, and what to do next

Living document. Overwrite stale lines; don't append to them. Numbers and reasoning live
in `DECISIONS.md` — this file only says where things stand and what to pick up. **§5 is the
Day 9-14 checklist**; it refers to §2's numbered command lists rather than repeating them.

Last updated: 2026-09-06 (very late). **Days 9, 10, 11 are COMPLETE and Day 12 is ~85% done —
the service is DEPLOYED AND ANSWERING on real AWS hardware.** `POST /ask` returns a correct,
cited answer from a self-hosted Postgres on Graviton3 in **157.7s, of which rerank is 156.9s
(99.5%)** — `DEPLOY-18`. Both reranker latency routes were already exhausted (`DEPLOY-11`, whose
stated reason was wrong — see `DEPLOY-11-CORRECTED`; and `DEPLOY-14`), and 157.7s measured on the
target confirms the conclusion rather than rescuing it. **The real remaining bottleneck is not
AWS: the repo has NO GIT REMOTE and has never been pushed.** Day 8 closed out, including the
`RETR-7`/`RETR-8` re-index (`RETR-39`) and the last cheap cost item
(`COST-34`/`COST-35`/`COST-36`). Day 10's observability was pulled forward ahead of the run
(`OBS-1`..`OBS-12`), so it produced its own traces. **Read §2's first block before quoting any
Day 9 number** — three of them are not what the pilot predicted, and two published numbers
turned out to be non-comparable.

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

**The company filter was silently OFF for 60% of the loop's own iterations, and it is now
fixed** (`AGENT-25`). Found by opening one trace, not from any metric. `retrieve()` resolved
the company from whatever query it was handed, and after iteration 1 that is the planner's
rewrite — which drops the company name and invents a `filing_stem:` syntax nothing parses. So
`resolve()` returned nothing, the unfiltered branch ran across all 799 filings, and
`finqa_dev_168`'s iterations 2-3 searched Host Hotels, Hologic and Philip Morris while looking
for Global Payments. **The trace said so outright** — `search-candidates` carries
`"filtered": false` — so the instrumentation was reporting it before anyone read it. Fixed via
a `resolve_from` argument defaulting to the old behaviour, so only the agent path changes;
unresolved fell 59.6% -> 14.9%. **Consequence for every Day 9 number: they all predate this
fix, so 69.2% is a FLOOR measured on a partly-broken loop.** Do not re-run for it now (~$4).

**The dashboard needs the run's own time window or it lies** (`OBS-13`). Last-1-day reads 185
questions, last-7-days reads 225, the truth is 200 — the short window truncates the run's
start, the long one sweeps in the aborted pre-`AGENT-16` attempt. On
`2026-09-04 20:10 -> 2026-09-05 02:00` it matches the results file exactly on six counts,
which makes it a cross-validation of two independent code paths. **And its stage-latency tile
is still pre-`AGENT-24` saturated data** (embed 13.13s against a true 0.075s) — regenerate it
before Day 13 gates a p95 off it, which its own caption says it will.

### AWS is LIVE — the running system, in one block

**Deployed 2026-09-06 and answering.** Everything here exists right now; nothing below is a plan.

| what | value |
|---|---|
| account | Paid plan (the **Free plan cannot offer an 8 GiB Arm instance** — `DEPLOY-17`) |
| credits | **$160** = $100 signup + 3 × $20 (EC2, Budgets, RDS). Lambda + Bedrock skipped — no project use |
| region | **us-east-2 only.** Anything in another region is invisible from the console you are looking at |
| instance | `i-0ebb938d759088661`, **`c7g.2xlarge`** (8 vCPU, 16 GiB, Graviton3, non-burstable), ~$0.29/hr |
| address | Elastic IP **`3.149.223.184`** — stable across stop/start, which is why it exists |
| ssh | `ssh -i ~/.ssh/ragsec-key.pem ec2-user@3.149.223.184` |
| IAM | user `ragsec-deploy` (local CLI, profile `ragsec`); instance role `ragsec-ec2-ecr` so **no keys live on the box** |
| image | ECR `294417174821.dkr.ecr.us-east-2.amazonaws.com/ragsec:d12.1`, built **on the host** (arm64 native, no 4GB home upload) |
| Postgres | self-hosted from `Dockerfile.postgres`, `pg_search` 0.25.1 + `vector` 0.8.6, bound to `127.0.0.1` |
| corpus | restored from a 670MB `pg_dump`: **A=99654 / B=4708 / C=1665**, all four indexes |
| deploy dir | `~/deploy` on the host — `compose.yml` + `.env` (0600), NOT in the repo |
| endpoints | `/health` and `/ready` green; `/ready` re-runs `store.preflight`, so it independently re-verifies the corpus |
| **not yet done** | **port 8000 is closed to the internet** — reachable only from inside the box |

**Stop the instance when idle.** Containers restart automatically and warmup is 75s.

**What still needs a human — and AWS is no longer the top of the list:**

1. **PUBLISH THE REPO. There is no git remote and nothing has ever been pushed.** ~30 min, and
   it is the only thing genuinely standing between this work and a public artifact. It was
   never AWS.
2. **Decide port 8000's exposure.** `/ask` calls Gemini per request with no auth, so a public
   URL is a spend vector once it is in a README. Suggested: keep the security group on your own
   IP and widen to `0.0.0.0/0` only while demoing — a ten-second edit either way.
3. **The p95 target: restate it as non-interactive.** `DEPLOY-18` measures 157.7s on the target
   with rerank at 99.5% and embed+search at 0.74s combined, so no first-stage work can move it.
   Reaching interactive needs ~50-75x; K=10 buys 5x for 0.224 recall. One route remains genuinely
   open and it is now cheap — **ORT int8 vs torch on Graviton3, ~1h, on the host that exists**
   (`DEPLOY-11-CORRECTED`; Graviton3 has SVE and i8mm, unlike the M3, and unlike t4g's Graviton2).
4. **`DEPLOY-12`'s sizing basis is now re-derivable from real hardware** — supersede the
   container's 207.8s with `DEPLOY-18`'s 156.9s on 8 known cores.

### The two latency routes, both now measured and both closed

**Route 1, ONNX/int8: dead on macOS, but `DEPLOY-11`'s stated reason was WRONG and the
correction matters** (`DEPLOY-11-CORRECTED`). `DEPLOY-11` compared `latency-fp32` (torch
CrossEncoder) against `latency-int8` (an ONNX Runtime session) and read the gap as precision.
Those are **different backends**. Verified on this machine: torch reports
`BLAS_INFO=accelerate`, so its matmuls reach Apple's undocumented **AMX** coprocessor; ORT's
MLAS never does and gets plain NEON. `scripts/archive/ort_fp32_latency.py` measured the missing
cell — three legs, **one backend per process**, same 3 questions, same 50 pairs:

| leg | p50 | per question |
|---|---|---|
| torch-fp32 (AMX) | **58.5s** | 61.7 / 58.5 / 55.5 — flat |
| ort-fp32 (NEON) | 415.7s | 278.9 / 415.7 / 422.6 |
| ort-int8 (NEON) | 276.3s | 106.2 / 276.3 / 349.0 |

**int8 is ~1.5x FASTER than fp32 inside ORT** (0.38 / 0.66 / 0.83 per question — it wins all
three). The 3.1x was backend, not precision. **Why the fast kernel is missing:** ORT gates its
i8mm QGEMM kernels behind `#if defined(__linux__)`; this M3 reports `FEAT_I8MM: 1` and ORT
detects it, then declines the kernel because the OS is not Linux (and the SVE fallback is
useless here — no M-series has SVE). **So the Mac measurement says NOTHING about Graviton**,
which is Linux + i8mm, and where torch conversely loses Accelerate. The container's 207.8s is
the *weak* configuration. **Caveats that must ship with these numbers:** n=3, and the ORT legs
drift upward within the run (fp32 1.5x, int8 3.3x) while torch is flat at 1.11x — the
directions are robust, the magnitudes are not. **Do not quote 7.11x; quote 4.5-7.6x.**

**Route 2, cutting `CANDIDATE_K`: measured on both splits, and it cannot do the job**
(`DEPLOY-14`, via `scripts/archive/candidate_k_curve.py` — free, replayed from disk, no GPU).

| K | test recall@10 | vs 50 | dev | est. container rerank |
|---|---|---|---|---|
| 50 | 0.747 | — | 0.760 | 207.8s |
| 40 | 0.736 | -0.012 | 0.741 | ~166s |
| 30 | 0.711 | -0.037 | 0.708 | ~125s |
| 25 | 0.669 | -0.078 | 0.682 | ~104s |
| 20 | 0.627 | -0.120 | 0.654 | ~83s |
| 10 | 0.523 | -0.224 | 0.553 | ~42s |

Rerank cost is linear in K, so an interactive number needs a **10-20x cut**, and **K=10 costs
0.224 test recall — more than the entire filter+strip headline gain of +0.140.** Measured recall
tracks the *pool ceiling* at every K (K=30: 0.711 vs 0.730; K=20: 0.627 vs 0.637), so cutting K
destroys the **candidate pool**, not the reranker's ordering — the gold is spread through the
first stage's tail. That is `RETR-33` restated: **first-stage candidate generation is the
constraint.** **Adopted: K=30 as a 40%-of-the-work trim, explicitly NOT as the p95 answer.**
Rejected: any K below 25 at any latency benefit.

**Kept only so the ONNX artifacts are not re-created by accident:** `models/onnx/` is ~2.9GB and
gitignored; `onnx_rerank_export.py` recreates it in ~30s. The extra is `uv sync --extra onnx` —
`onnxruntime` + `onnx` + `onnxscript`, deliberately **not** `optimum[onnxruntime]`, which cannot
install against our pinned `transformers==5.8.1` (`DEPLOY-8`). Do not relax the pin to "fix" it.
`onnx_rerank_parity.py` still has never completed a parity run, and no longer needs to.

**Settled earlier, all recorded in `DECISIONS.md`** — note `DEPLOY-6-RESOLVED` (reranker route =
ONNX/int8, `CANDIDATE_K` unchanged) has since been **reversed twice** and is dead; see the two
routes above. Still standing: the yes/no scorer reports
both, headline unchanged (`AGENT-27`); `unans_034` is retired and the set scores 47/47
(`EVAL-3`); the container wordlist is baked from the benchmark's own list rather than installed
(`DEPLOY-7`); the four changed files were reviewed and five defects fixed (`AGENT-26`); and the
failure-diagnosis draft is verified claim by claim (`AGENT-28`).

**Two things found by finally building and running what Day 10-12 had only written:**

- **`DEPLOY-1`'s image had never been built and could not have built** (`DEPLOY-5`): five
  defects, three fatal to the build, two that would have shipped silently — `stage_latency`
  was always `{}` (so Day 13's p95 gate would have gated an empty dict), and `python:slim`
  ships no wordlist, so `RETR-11`'s guard is **off in the container** and the deployed
  resolver is not the benchmarked one. **That last one is still open.**
- **The CPU container is far too slow to serve interactively** (`DEPLOY-6`): 212.4s for one
  question, 207.8s of it the cross-encoder, at 755% CPU. A 2-vCPU free-tier host implies
  ~800s/question. Correctness is fine — top citation 0.996, right answer. **Its 207.8s is now
  explained** (`DEPLOY-11-CORRECTED`): that is torch on NEON, having lost Apple's AMX. Sizing no
  longer waits on a reranker route — both are closed — but read `DEPLOY-12` before reusing the
  number.

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

**Still owed after the run:** regenerate the worst-failures files — `scripts/archive/worst_failures.py`
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
repo (`scripts/archive/pack_variants.py:287`) measures whether *retrieved chunks* span
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

**5. I caught my own published conclusion being backwards, and the fix was measuring one
missing cell.**
I had written up "ONNX int8 is 3.1x slower than fp32, route abandoned." Reading it back, the
comparison changed two variables: the fp32 leg was the torch CrossEncoder, the int8 leg was ONNX
Runtime. Torch on this Mac links against Accelerate and reaches Apple's AMX coprocessor; ORT's
MLAS gets plain NEON. Thirty minutes of measurement — three legs, one backend per process — and
int8 turned out **1.5x faster** than fp32 within ORT, with the 3.1x being a 4.5-7.6x backend gap
wearing a precision costume. The abandonment was still right for macOS; the stated reason was
not, and the reason is what determines whether it transfers to Graviton (Linux + i8mm — where
ORT's fast kernel is compiled in and torch loses AMX).
*Caveat:* n=3, and the ORT legs drift within a run on a fanless laptop, so I quote the range and
the machine state, never the point estimate.

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
- **Latency drift has now had THREE different mechanisms and nobody guessed any of them.**
  Battery/thermals were wrong; it was MPS memory (`AGENT-24`). On AWS the drift was CPU
  credits: a `t4g` burstable in `Standard` mode ran a single question past 1029s while
  `CPUCreditBalance` fell 34.4 -> 0.04 (`DEPLOY-17`). **Diagnose with the metric for the
  machine you are on** — `sysctl vm.swapusage` on the laptop, `CPUCreditBalance` on a
  burstable instance — and never publish a latency without the machine state beside it. The
  deployment host is deliberately **non-burstable** so this class cannot recur.
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

### Day 10 — Observability — COMPLETE except one tile
- [x] Instrumentation half (`OBS-1`..`OBS-12`)
- [x] Dashboard + trace screenshots captured — `images/langfuse_dashboard_{1,2,3}_*.png` and
      `images/langfuse_trace_loop_graph.png`, on the `OBS-13` window so counts match the file
- [ ] Regenerate the stage-latency tile from a post-`AGENT-24` run (`OBS-13`) — its caption
      says Day 13 gates on it, and it currently shows saturated-regime numbers. **The only
      Day 10 item left**, and it needs a serial pass on a quiet machine
- [x] Ten worst failures diagnosed — `data/day9_failure_diagnosis_ANALYSIS.md`, a **machine
      draft**; one claim verified by hand against the filing (`AGENT-23`), the rest are not
- [x] Draft verified claim by claim (`AGENT-28`) — 38 claims, 34 confirmed, 1 refuted, 3
      corrected in place, none cut; corrections are marked inline in the file

### Day 11 — Gate and drill — gate COMPLETE
- [x] Gate scripts + workflow (`CI-1`)
- [x] Real fixture built (400q), gate green on real data (`CI-2`)
- [x] Gate proven to fire — degraded fixture exits 1, clean exits 0
- [x] Refusal correctness measured — 47/48, and the one miss is our label (`EVAL-2`)
- [x] `unans_034` retired; set is 47 questions and scores **47/47** (`EVAL-3`). The results
      file keeps the measured row — the questions file now defines membership
- [x] Unanswerable set written and validated (`EVAL-1`)
- [ ] Citation-grounding check — not started
- [ ] Interview drill #1 — not started
- [ ] Answer-accuracy leg, on-merge not per-push (`COST-39`) — now unblocked, baseline exists

### Day 12 — AWS
- [x] Serving layer, `Dockerfile`, `.dockerignore` (`DEPLOY-1`)
- [x] Image built, boots offline as non-root, `/health` `/ready` `/ask` all good (`DEPLOY-5`).
      Five defects fixed; **9.06GB image, 2.17-2.75GiB resident**
- [x] **Container wordlist fixed** (`DEPLOY-7`) — baked, not installed: `wamerican` disagrees
      with the benchmark's list on 27 of 250 aliases, so it would have shipped a different
      resolver. Guard now runs at build AND startup. **Full 9.06GB rebuild not yet run**
- [x] **Reranker route decided** (`DEPLOY-6-RESOLVED`) — ONNX + int8, `CANDIDATE_K` unchanged
- [x] **ONNX/int8 abandoned for macOS** (`DEPLOY-11`), **but its reason was wrong**
      (`DEPLOY-11-CORRECTED`) — the 3.1x was torch-AMX vs ORT-NEON, and int8 is ~1.5x *faster*
      than fp32 within ORT. **Graviton is Linux + i8mm, so it is untested there, not ruled out**
- [x] **`CANDIDATE_K` cut measured on BOTH splits and it FAILS as a latency route** (`DEPLOY-14`).
      K=30 adopted as a 40% trim (test recall 0.711); no K reaches an interactive p95
- [x] **Sizing basis re-derived on real hardware** (`DEPLOY-17`/`DEPLOY-18`) — API is **3.95 GiB
      resident**, so `t4g.small`'s 2 GiB was never viable and the free tier is ruled out by RAM
      before latency. `t4g` burstable also throttles (credits 34.4 -> 0.04 in 35 min), making
      latency a function of uptime. Host is `c7g.2xlarge`, non-burstable
- [x] **AWS account** — created, **upgraded to Paid** (Free plan has no 8 GiB Arm type)
- [x] Credits verified: **$160**. EC2 needed launch **and terminate** to pay out; Lambda and
      Bedrock deliberately skipped (no project use, $40 left on the table)
- [x] Zero-spend Budget created (it was also one of the $20 tasks)
- [x] RDS throwaway created and **deleted**, no final snapshot retained
- [x] IAM — `ragsec-deploy` + instance role `ragsec-ec2-ecr`. **S3 skipped**: the `pg_dump` went
      over `scp` in 79s, so a whole service was avoided rather than added
- [x] ECR — repo `ragsec`, image `d12.1`. **Built ON the host**, not pushed from the laptop:
      4.4GB of weights come down over AWS's backbone instead of up a home connection
- [x] Both containers on one host via docker-compose, arm64/Graviton. Postgres from
      `Dockerfile.postgres`, **not** RDS (`DEPLOY-2`)
- [x] Corpus loaded — 670MB dump, sha verified both ends, A=99654/B=4708/C=1665, 4 indexes
- [x] `/health` + `/ready` green; wordlist guard verified ON in a running container, so
      **`DEPLOY-7` is now proven rather than merely built**
- [ ] **Open port 8000** — the last step to a live URL, and a decision (see §2, item 2)
- [ ] **ORT int8 vs torch on Graviton3, ~1h.** Runs as-is now that Postgres is on the host;
      the one route `DEPLOY-11-CORRECTED` leaves genuinely open
- [x] Bedrock dropped, generation stays Gemini (`DEPLOY-3`, closes `AGENT-18`)

### Day 13 — Ship and latency
- [x] Deploy done on a single EC2 host; **only the firewall stands between it and a live URL**
- [ ] MCP server — not started. **Cuttable (2-3h, nothing depends on it)**
- [x] p95 basis measured **in the container on the target** — `stage_latency` populates correctly
      (`DEPLOY-5`'s whitelist fix), and one warm question gives 157.7s total / 156.9s rerank
- [ ] Gate enforces p95 — **unblocked**: pick a threshold off `DEPLOY-18` and state the host next
      to it. No longer blocked on a reranker route, because both are closed

### Day 14 — Ship
- [ ] README benchmark table + trace screenshots — depends on the screenshots above
- [ ] The written post
- [ ] Final drill

**What remains is publishing, not building.** Days 9-11 are done bar one dashboard tile, and
Day 12 is done bar the firewall. **The bottleneck all along was never AWS: this repo has no git
remote and has never been pushed.** That is ~30 minutes.

**Roughly 15-20 hours left**, in the order worth doing them:

1. **Publish the repo** — 30 min. Nothing else is gated on it and everything benefits.
2. **ORT int8 vs torch on Graviton3** — ~1h, only needs the host that already exists, and it
   closes the last open measurement in the writeup.
3. **Open port 8000** — 10 min plus the exposure decision.
4. **Regenerate the stale dashboard tile** (`OBS-13`) — 30 min; it currently shows
   saturated-regime numbers its own caption says Day 13 gates on.
5. **The deliberately-broken commit proving the gate fires** — 20 min. `spec.md §7` names it
   explicitly; the gate exists and is proven on a fixture, but not in the history.
6. Citation grounding (1-2h) · p95 gate threshold (1h) · README + table + screenshots (2h, and
   it still publishes **0.752** where the current number is **0.747**) · the written post (2-3h)
   · two drills (2-3h).
7. **Script reorg** — 65 scripts, 9 near-duplicate copies of one ranking loader (the `AGENT-16`
   bug lived in one of them). Full job 22-30h; the **publishable subset is 4-5h and is all
   documentation** — a `scripts/README.md` giving the run order beats renaming files, which
   would break CI, the Dockerfile and the `.sbatch` fallbacks.

**Budget is not a constraint:** ~$160 of AWS credit, valid 12 months from signup, and ~$29 of
the $40 API budget. The live host costs ~$0.29/hr and should be **stopped when idle**.

**The p95 decision is made in all but writing: restate it as non-interactive.** `DEPLOY-18`
measures 157.7s on the deployment target with rerank at 99.5%, so this is a property of the
workload, not of a bad host. Saying that with four measured routes behind it is stronger than a
number fudged into range. Optional scope cuts if you want to publish sooner: the **MCP server**
(2-3h) and the **second drill**.

**If you run anything on this laptop, read `DEPLOY-12` first:** fanless M3 Air, and ORT rerank
latency drifted 1.5-3.3x within *three* questions while torch stayed flat. Record per-question
times and machine state, never just a p50/p95.

**Two standing cautions for whoever picks this up.** Every Day 9 figure predates `AGENT-25`,
so quote 69.2% as a floor and never as the loop's ceiling. And the four numbers that this
session proved non-comparable — the union recall@40 (`AGENT-22`), the pre-`AGENT-24` stage
latencies, the pilot-era `AGENT-13`/`OBS-10` figures, and any dashboard count on a default
time window (`OBS-13`) — are all still sitting in the log above their corrections, because a
row is a record and not a claim. Read the correction, not the row.
