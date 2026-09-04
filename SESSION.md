# Where the project is, and what to do next

Living document. Overwrite stale lines; don't append to them. Numbers and reasoning live
in `DECISIONS.md` — this file only says where things stand and what to pick up.

Last updated: 2026-09-04. Day 8 plus two follow-on sessions: `COST-27`/`COST-28`, `RETR-33`/`RETR-34`/`RETR-35`, a `COST-25` provenance correction, then a housekeeping pass (`RETR-36`, `INFRA-10`, two `COST-21` corrections) and the Layer-3
diagnosis (`RETR-37`/`RETR-38`).

**"Day N" is a unit of planned work in `spec.md`, not a calendar date** — Day 8 spanned
several days. The `COST-`/`RETR-` IDs follow the project label, not the calendar, so a
`day8_*` data filename says nothing about when it was written. Scripts no longer carry day
prefixes at all: `scripts/` is organised by job and every file is named for what it does.
`INVENTORY.md` maps every file to what it actually does.

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
| recall@10 | 0.607 | **0.747** |
| nDCG@10 | 0.468 | 0.617 |

Re-measured 2026-09-04 on the post-`RETR-7` corpus (`RETR-39`). The pre-re-index pair was
0.604 -> 0.752; every cell moved -0.002 to -0.005, inside stderr, so the headline is
unchanged in substance.

Both cells re-scored under `RETR-35`'s corrected labels. The pre-correction pair was
0.581 -> 0.726; the *gain* is unchanged (+0.148 against +0.145), only the levels moved.
MRR is deliberately absent: it is not comparable across labelings (`RETR-35`).

They are superadditive: filter alone +0.040, strip alone +0.021, together **+0.148** —
2.4x the additive prediction, with all four cells now scored under `RETR-35`'s corrected
labels. The gain is *larger* on test than on dev, which is the anti-overfitting
evidence, since dev informed the resolver's design. **Quote the test number, 0.752.**

**Compression is a real trade, not free money.** slices@1500 cuts ~$13.00/pass to ~$5.27
(2.5x, not the 4.1x once claimed) and costs roughly 9 points of answer accuracy. The
failure mode is safe — the model refuses rather than fabricates.

**Trustworthy right now**, recall@10 / recall@50, all variant-clean and all under
`RETR-35`'s corrected labels unless marked:

| arm (dev unless stated) | recall@10 | recall@50 |
|---|---|---|
| Arm 1, dense only | 0.337 | 0.473 |
| Arm 2, + BM25/RRF | 0.514 | 0.708 |
| Arm 3, + reranker (= Arm4-A) | 0.629 | 0.708 |
| Arm 3 + filter + strip | **0.760** | 0.802 |
| Arm 4-B / 4-C (NOT re-indexed) | 0.237 / 0.236 | 0.268 / 0.270 |
| **Arm 3 + filter + strip, TEST** | **0.747** | 0.785 |

**All rows above are post-`RETR-7` and on corrected labels except Arm 4-B/C**, which were
deliberately not re-embedded and stay on pre-`RETR-7` headings — they are no longer
text-comparable with variant A. Arm 1 and Arm 2 were finally re-run (`RETR-39`), clearing the
standing blocker, but **their movement against the old published 0.329/0.495 is confounded**:
corpus and labels changed at once, so quote the level, never a delta. `data/retr7_ANALYSIS.md`
records how to get the missing cell from the pre-re-index `pg_dump`.

---

## 2. Open questions, in the order worth doing them

### Free — no GPU, no API spend

Everything previously listed here is done: the prompt-format fix (`COST-28`), `COST-26`'s
three slice-ranking fixes (`COST-27`, all three lose, nothing shipped), the survival-pool
provenance question (`COST-25` corrected — quote 67.0%), the failure-triage refresh
(`RETR-33`), the label audit (`RETR-34`), the matcher rewrite (`RETR-35`) and the Layer-3
diagnosis (`RETR-37`; two candidate fixes built and reverted, `RETR-38`). Layer 3's own
remaining bugs are logged there as known and parked — not on this list. What is left:

- **Re-scoring: done except Arm 1 and Arm 2.** All four Arm 3 ablation cells and Arm 4 A/B/C
  are now on corrected labels (`rescore_labels.py`, `RETR-35`). **Arm 1 and Arm 2 are blocked**
  — their results files persist only `top_5_retrieved`, so recall@10/@50 is unrecoverable and
  they need a retrieval re-run against Postgres. No GPU, but it is a re-run, not a re-grade.
- **A trap, not a lead.** Restricting `COST-27`'s figure guard to the top chunk alone scored
  +1.3 (6 gained / 2 lost, **p=0.29**) and was the best of six swept variants. A hypothesis
  with a test-split price on it. Do not quote it; do not ship it on the dev number.

### Cheap

- **`thinking_level` medium -> low: wired, unmeasured.** `answer_ab_run.py --thinking` ships
  (`COST-31`). The API default for `gemini-3.7-flash` is **medium**, and the script passed no
  thinking config, so every cost number to date was billed at medium. `thinking` is part of the
  checkpoint key, which makes the 278 stored rows the `medium` arm for free — only the `low`
  arm needs paying for (~$2.02 standard, ~$1.01 batch). Google publishes no per-level token
  counts, so the saving can only be measured, not estimated. Caveat before spending: the medium
  rows are from 2026-09-02, so reusing them puts any API drift entirely in one arm — either
  re-run both interleaved (~$4) or re-run ~60 medium rows as a drift check.
- **`COST-12` is closed, decided against (`COST-30`).** Do not merge `judge` into `answer`.
  The measured insufficiency rate is 15.1% against a ~15-25% break-even, so it was a coin flip
  rather than the claimed 22%; and CRAG's loop loses to hybrid+rerank on T²-RAGBench itself.
  `COST-13` still has no row of its own — its design and result live in `COST-20`/`COST-23`.
- **`RETR-2` — widen `TOP_K`: DEAD (`RETR-33`).** Recovered 4.3 points on the Day 6 ordering;
  recovers 1.2 on the shipped one, because the whole reranker bucket is now 1.5 points.

### Expensive, and it invalidates everything upstream

- **`RETR-7`/`RETR-8` — DONE, and the re-index is applied (`RETR-39`).** Both fixes are live:
  47,312 of 99,654 chunks re-embedded, boundaries and chunk count unchanged, gold labels
  unmoved, every arm re-measured. Effect on retrieval: **-0.002 to -0.005 everywhere, inside
  stderr** — the ~0 that `RETR-33` predicted. The flags
  (`RAG_SEC_MULTI_HEADING`, `RAG_SEC_STRIP_TITLE_FURNITURE`) now default **on**, matching the
  stored corpus — left off, `atom_replay.py` matched only 52.52% and `rag_sec.compress` would
  have replayed a different document than the one indexed. Set them to `0` only to reproduce
  the pre-re-index corpus. Results in `data/retr7_results_0904.md`, caveats in
  `data/retr7_ANALYSIS.md`, procedure in `RUNBOOK.md`.

### Then back to `spec.md`

- **Day 6's actual question, deferred not dropped.** For questions needing arithmetic
  across ≥2 table cells, does any of the A/B/C table-indexing strategies surface all the
  needed evidence — or is it a retrieval-vs-computation gap? The data dependency is
  satisfied (`GOLD-1`); the analysis has never been started.
- **Days 9-14.** 9: Arm 6 vs the best static arm on multi-document questions — and
  `COST-30` sharpens the hypothesis: iteration's published wins are multi-hop *composition
  across* documents, while on single-document table QA the corrective loop **loses** to
  hybrid+rerank (CRAG recall@5 0.658 vs 0.816, arXiv:2604.01733). So Day 9 is not "does the
  loop work" but "does it earn its cost on the multi-document minority, given it demonstrably
  does not on the majority". Run it on the untouched split. 10:
  observability, then diagnose the ten worst failures from traces. 11: CI quality gate,
  citation grounding, unanswerable set, first interview drill. 12-13: AWS, MCP server,
  latency pass with a stated p95. 14: README, writeup, final drill.

### Decided against

- **Arm 5 (late interaction / multi-vector).** Dropped without building: 378GB of storage
  against Arm 1's 389MB, plus MaxSim needs an index pgvector doesn't have. Reported rather
  than built, per `spec.md`'s own framing.
- **Loop on/off ablation.** ~$15, and Arm 3 already *is* the loop-off arm. Published work
  says iterative retrieval doesn't pay on single-document financial table QA — that assertion
  now has a number and a citation on our own benchmark (`COST-30`).
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
test split, which is why the headline is test's **0.752** and not dev's 0.765 (both under
`RETR-35`'s corrected labels; the pre-correction pair was 0.726 / 0.736).

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

- **Done: results files, the SQL dedupe, and both `COST-21` corrections.** `--out` shipped
  and the four missing `data/*_results.json` written (`INFRA-10`); dense/BM25/RRF now has one
  home in `rag_sec.candidates` with `variant` required, byte-identical SQL, guarded by a new
  `scripts/checks/candidate_sql.py` (`RETR-36`); `COST-21`'s row now says seven scale factors
  and 127 exclusions, both re-verified against dev on 2026-09-03. `rerank_score.py` already
  had `--out` — only the dev results file was missing.
- **Still owed a row: the 9.4x loop cost multiplier — and provenance could NOT be
  established.** Searched 2026-09-03: no artifact on disk produces it, nothing in `data/`
  records it, and git only ever shows it in this bullet (the smoke test prints per-node token
  counts but persists nothing). The nearest candidates are coincidences, not sources —
  `$16.97 -> $3.10/pass` is 5.5x and `COST-24` marked every one of those figures suspect. It
  most likely came from the deleted Day 8 session file. **No row was added, deliberately:** a
  row is a provenance claim, and inventing one is worse than the number having no home. Don't
  quote it; re-derive it from a real run or drop it.
- **`CLAUDE.md` is gitignored on purpose** (personal working rules). Keep the standing
  context there short and number-free — `DECISIONS.md` is the source of truth for numbers,
  and an unversioned file is a bad last home for one.
- **The recurring failure mode: reusing a number without checking which job produced it.**
  `RETR-24` (B/C chunks scored as A), `RETR-30`'s mis-anchored 4.3 GPU-h estimate (that was
  `RETR-22`'s chunk-level figure), and the near-miss filename clobber are all the same
  mistake — a value correct in one context, silently wrong in the next. Check provenance
  before quoting a number, including your own.
- **A documented GA feature can still be unusable.** `service_tier="flex"` 503'd ~9 of 10
  requests and `COST-13` actually ran at `standard` — the stored rows carry no `tier` field, so
  this was recovered from latency (p50 2.74s vs flex's 1-15 min target) and from billing ($2.22
  standard vs the $2.23 dashboard). **Batch, not flex, is the 50% route**, and the two do not
  stack. `COST-29`. Check the key's tier in AI Studio before designing a batch runner.
- **Don't run two CPU passes concurrently.** Measured 2026-09-01: contended ~0.5 it/s vs
  ~3.9 it/s alone — **8x, not 2x**. Each compression pass took ~5 min alone against a
  projected 40. Run them sequentially.
- **Interactive `srun` was chosen over `sbatch`** for live visibility; the two `.sbatch`
  files (`scripts/retrieval/rerank_hpc.sbatch`, `scripts/compression/slice_rerank_hpc.sbatch`) stay as
  the unattended fallback. Stagger submissions past model load — both pull the same reranker
  into one HF cache, and two cold downloads to one path risks corruption.
- **Unverified, from the deleted Day 8 session file:** Batch tier may need only one
  submission for `COST-13` (1.6M tokens vs a ~3M cap), not chunking. Re-check the cap before
  relying on it.
