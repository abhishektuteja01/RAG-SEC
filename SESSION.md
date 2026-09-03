# Where the project is, and what to do next

Living document. Overwrite stale lines; don't append to them. Numbers and reasoning live
in `DECISIONS.md` — this file only says where things stand and what to pick up.

Last updated: 2026-09-02, end of Day 8.

---

## 1. Where things stand

**Day 8 was supposed to be the LangGraph agentic loop.** The loop got built and works
(3 dev questions, all correct, none looped). But a triage of all 1,235 dev questions found
retrieval — not the reader, not the loop — was the binding constraint, so most of Day 8
became unplanned retrieval work. That was the right call; it means Day 8 absorbed work
`spec.md` never scheduled.

**The headline, confirmed out of sample.** Two changes, both resolved from question text
alone, neither needing re-embedding: filter candidates to the company the question names,
and strip the company/filing framing from the query *before reranking only*.

| test split (untouched until the end) | before | after |
|---|---|---|
| recall@10 | 0.581 | **0.726** |
| nDCG@10 | 0.461 | 0.618 |
| MRR | 0.490 | 0.658 |

They are superadditive: filter alone +0.042, strip alone +0.015, together **+0.145** —
2.5x the additive prediction. The gain is *larger* on test than on dev (+0.126), which is
the anti-overfitting evidence, since dev informed the resolver's design. **Quote the test
number, 0.726.**

**Compression is a real trade, not free money.** slices@1500 cuts ~$13.00/pass to ~$5.27
(2.5x, not the 4.1x once claimed) and costs roughly 9 points of answer accuracy. The
failure mode is safe — the model refuses rather than fabricates.

**Trustworthy right now** (all variant-clean): Arm 1 0.329/0.466, Arm 2 0.495/0.687,
Arm 3 0.609/0.685. With filter+strip, Arm 3 dev 0.736/0.776.

---

## 2. Open questions, in the order worth doing them

### Free — no GPU, no API spend

**(a) Compressed prompts lose their labels and their order.** Found 2026-09-02, not yet
acted on. `compress.pack_by_score` emits slices in *score* order with no `[stem chunk N]`
prefix. The uncompressed control keeps chunk order and all 10 provenance labels. So the
two arms of `COST-23` differ in three ways — how much text, what order, and whether the
model can see which filing and year each block came from — when the write-up treats it as
one.

Why this probably matters: `RETR-3` showed 74% of retrieved chunks come from the wrong
document, half of them the right company's wrong year. Worked case `convfinqa_1222` is
stratum B (gold survived) and its compressed prompt holds *two near-identical UNP income
statements from different years, both unlabeled*. The answer is present and unusable.

This is a fourth candidate explanation for the stratum-B channel, alongside the three in
`COST-26`, and it's the cheapest. Fix: group packed slices by source chunk and print each
group under its heading in document order — about 5% of the token budget. `compress.compress`
already does exactly this and is currently unused; it is the template.

**(b) `COST-26`'s three slice-ranking fixes.** The gold table slice sits in chunk rank 0-2
but scores near zero and lands at slice rank 22-46, beaten by its own caption and by
wrong-year copies; only 12-19 slices fit in the budget. Three candidates: require a
3+ digit non-year figure in some packed slice, bind a caption to its table's rows, cap the
tokens any one chunk can take. Slice scores are already on disk.

**Measure (a) and (b) against 69.3% figure survival, not 84.9% matcher survival**
(`COST-25`). Only pay for a re-run of `COST-13` if the offline number moves.

**(c) Wire the query strip into `retrieve.py`.** The module `agent.py` calls applies the
company filter but never calls `strip_entity_framing` — so the shipping path implements
the filter-only cell (+0.042) rather than the confirmed +0.145. The strip currently exists
only in the offline payload builder. Deferred deliberately so today's commit stayed
mechanical; it is a small standalone change and should get its own `DECISIONS.md` row.

### Cheap

- **`COST-12` — merge `judge` into `answer`.** One send instead of two, ~-22% cost, and it
  keeps a sufficiency signal rather than deleting one. Independent of everything else. ~1h.
- **Cap the thinking budget.** `gemini-3.7-flash` bills ~662 internal reasoning tokens per
  call against ~110 visible, and that cost is near-constant across arms — a floor of about
  $3.9/pass that compression cannot touch. For a task whose answer is one number, capping
  it attacks the floor directly. Parameter name and 3.7-flash support not yet verified.
- **`RETR-2` — widen `TOP_K` 10 → 20.** Recovers 4.3 of the reranker's 6.9-point bucket;
  gold sits at median rank 18 when it misses. Only affordable once a token budget ships,
  because the budget then caps tokens regardless of `k`.
- **`RETR-9` — label noise.** ~15% of the 19.6% candidate-miss bucket looks mislabeled.
  Measure before optimizing against it. Cases: `convfinqa_2530`, `convfinqa_1124`,
  `finqa_dev_840`. The reranker bucket is *not* noisy — 19/20 are real failures.

### Expensive, and it invalidates everything upstream

- **`RETR-7`/`RETR-8` — the heading bug.** 46.7% of chunks carry a heading set *after* some
  of their own body text: a title atom below `TARGET_CHUNK_TOKENS` overwrites
  `current_heading` instead of flushing. Observed: a balance sheet headed `# CREDIT RISK`.
  `RETR-8` batches in stripping page furniture (`# F-58`, `# /s/ KPMG LLP`). **This is the
  only open item that changes chunk text**, so it invalidates every embedding, the BM25
  index and Arms 1-4. Do it last, in one re-index cycle, as has been the plan.

### Then back to `spec.md`

- **Day 6's actual question, deferred not dropped.** For questions needing arithmetic
  across ≥2 table cells, does any of the A/B/C table-indexing strategies surface all the
  needed evidence — or is it a retrieval-vs-computation gap? The data dependency is
  satisfied (`GOLD-1`); the analysis has never been started.
- **Days 9-14.** 9: Arm 6 vs the best static arm on multi-document questions. 10:
  observability, then diagnose the ten worst failures from traces. 11: CI quality gate,
  citation grounding, unanswerable set, first interview drill. 12-13: AWS, MCP server,
  latency pass with a stated p95. 14: README, writeup, final drill.

### Decided against

- **Arm 5 (late interaction / multi-vector).** Dropped without building: 378GB of storage
  against Arm 1's 389MB, plus MaxSim needs an index pgvector doesn't have. Reported rather
  than built, per `spec.md`'s own framing.
- **Loop on/off ablation.** ~$15, and Arm 3 already *is* the loop-off arm. Published work
  says iterative retrieval doesn't pay on single-document financial table QA.
- **`slices50`.** Lost at all five budgets on the clean pool. Dead on replicated evidence.

---

## 3. Three things worth saying in an interview

Each with its caveat attached — saying the caveat first is what makes the claim land.

**1. I diagnosed before I optimized, and it moved the target.**
Sorted all 1,235 dev questions into single failure buckets: 73.5% fine, 6.9% the
reranker's fault, 19.6% the gold never entered the candidate pool, 0% chunking. The
reranker was already at 89% of its own ceiling — 7.6 points of headroom — so I stopped
working on it. Then 100 hand-read failures showed 74% of retrieved chunks came from the
wrong document, because 10-Ks reprint themselves near-verbatim and neither retriever
weights the year token.
*Caveat:* the 0% chunking bucket relies on a lenient whole-page matcher, so it rules out
the catastrophic version of that hypothesis, not the subtle one.

**2. The fix was superadditive, and I can say why.**
Filter alone +0.042, strip alone +0.015, together +0.145. Without the filter the company
name genuinely discriminates among 799 filings, so stripping it destroys real signal; with
the filter every candidate already matches the company, so the name only rewards corporate
boilerplate. I ran the strip-alone cell specifically so the interaction would be
attributable — with three cells the conclusion would have been "stripping helps," which is
false.
*Caveat:* dev informed the resolver's design. The generalization rests on the untouched
test split, which is why the headline is 0.726 and not 0.736.

**3. I found a bug that had been silently corrupting my results, and the guard isn't the
obvious one.**
Retrieval queries omitted a `variant` predicate, so a dropped experiment's rows were
eligible candidates and got scored against the gold labels as if they were different
chunks — 9.2% of filtered candidate slots. Found because the database reported 106,027
chunks against a documented 99,654. The interesting part: the contamination ran *against*
my hypothesis — it suppressed the measured gain rather than inflating it, so every number
rose after the fix and my published dense/hybrid rows had been understating retrieval. The
guard is a **per-variant** count, because the total was the only thing that diverged, plus
an AST scan that fails any `FROM chunks` read whose WHERE clause omits `variant`, run
pre-connect so it fires with the database down.
*Caveat:* it cost a 4.4-hour GPU re-run, and it blocks a contaminated pass from *starting*
— it can't vet a result already written to disk.

**If pushed on cost:** the compression work is a genuine accuracy-for-cost trade and its
own headline had to be retracted once retrieval improved. Being able to say "my earlier
conclusion didn't survive better retrieval, and here's the measurement that killed it" is
worth more than the original claim.

---

## 4. Housekeeping

- **~1.5 GB of superseded payloads are still on disk** and gitignored pending deletion.
  They are all scored and rebuildable; nothing reads them:
  ```
  rm -f data/day8_slice_payload_t150.json data/day8_slice_payload_t150_filtered_stripped.json \
        data/day8_retr16v2_dev_payload.json data/day8_retr18_test_payload.json \
        data/day5_rerank_payload.json data/day6_arm4_B_rerank_payload.json \
        data/day6_arm4_C_rerank_payload.json data/day8_slice_scores_t150.jsonl \
        data/day8_retr16_scores.jsonl data/day8_cost13_smoke.jsonl
  ```
- **No finalize script writes a results file.** `day8_finalize_retr16.py`,
  `day8_finalize_compression.py` and `day8_cost13_score.py` all print and exit, so
  `RETR-29`/`RETR-31`/`COST-18`/`COST-23` exist only as prose in `DECISIONS.md`. Every
  earlier arm has a `data/*_dev_results.json`. Adding `--out` and re-running off the score
  files already on disk is free.
- **Retrieval SQL is duplicated across 8 files.** `rag_sec.company` has one home;
  dense/BM25/RRF does not. That duplication is what let `RETR-24` hide in seven places at
  once. The AST guard now catches that specific failure, but the debt stands.
- **Two `DECISIONS.md` corrections owed**, both verified against the data on
  2026-09-02: `COST-21` says the scorer accepts `{identity, x100, /100}` — it accepts seven
  factors, including the "(in thousands)"/"(in millions)" conventions (5 of 278 verdicts,
  all genuine, ~0.7pt net). And it says 125+34=159 questions were excluded from the strata;
  the code excludes 127, keeping the 32 that have one parseable gold field.
- **`CLAUDE.md` is gitignored on purpose** (personal working rules). Keep the standing
  context there short — `DECISIONS.md` is the source of truth for numbers.
