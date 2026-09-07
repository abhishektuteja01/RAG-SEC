# Inventory

Every folder and every file in this project: what it is, where it came from, what uses it,
and whether you still need it. Written 2026-09-02, last checked 2026-09-04, by reading the
files, not by running them. Scripts sections rewritten 2026-09-06 for the `pipeline/` reorg:
20 scripts became 7 numbered phases plus 2 cluster legs, 5 dead scripts were deleted, and 28
one-off measurements moved to `archive/`. The `data/` sections below were not revisited.

Words used throughout:

- **chunk** — a filing cut into a ~900-word piece. The unit everything is stored and searched as.
- **embedding** — a list of 1024 numbers standing for a chunk's meaning, so similar text sits close together.
- **reranker** — a slower, more careful model that re-orders a shortlist after the fast search picks it.
- **the cluster** — the university GPU machines. Jobs there are booked and queued, not instant.
- **variant** — which table layout a chunk was built with. `A` is the real corpus; `B` and `C` were an experiment that lost.

Status labels:

- **Keep** — still used, or cost real money or GPU time to make.
- **Rebuildable** — a script here can recreate it cheaply; safe to delete.
- **Replaced** — a newer file does the same job better.
- **Delete** — nothing uses it and nothing would miss it.

---

## Root

Twenty things sit at the top level: six documents, six setup files, two environment files,
one leftover script, and four folders.

### Documents

**`spec.md`** — 28 KB, in git.
The original plan, written before any code. Covers why the project exists, how a number is
allowed to count as real, and a day-by-day schedule for fourteen days.
**Status: Keep.** It's the plan you're still working against.
Note: it's about a week behind reality. Day 8 is one line in it but became a whole phase of
retrieval work, one planned experiment was dropped, and days 9-14 haven't started.

**`DECISIONS.md`** — 92 KB, in git.
The log of every choice made and what it measured. Starts with the results tables, then rows
tagged by phase. If a number appears anywhere else in this project, this file is the one that's right.
**Status: Keep.** The most current file here and the main thing you'd show someone.

**`CLAUDE.md`** — 3 KB, **not in git** (deliberately).
Two parts: nine working rules, and a running snapshot of where things stand.
**Status: Keep.**

**`SESSION.md`** — 11 KB, in git.
Where things stand and what to do next, with open work sorted by what it costs to try.
**Status: Keep.** The only file that ranks what's next.
Note: its first section repeats the results tables from `DECISIONS.md`. The housekeeping list in
section 4 is genuinely unique — that part is worth protecting.

**`README.md`** — 3.3 KB, in git.
The front door: what this is, how to set it up, and the commands to rebuild the corpus in order.
**Status: Keep, but stale in three ways.** It has no results table (the plan says that's the first
thing "done" requires), the chart it shows was made before several corrections and shows numbers
you've since withdrawn, and its list of documents leaves out `SESSION.md`.

**`RUNBOOK.md`** — in git.
The step-by-step for running a GPU job on Northeastern's Explorer cluster: transfer
through the `xfer` host, allocate with `srun` inside tmux, run, copy back, verify. Includes the
full `RETR-7`/`RETR-8` re-index procedure with its backup step.
**Status: Keep.** It is the only place the cluster hostnames and module/venv setup are written
as commands rather than buried in a decision row — and `ARM3-2`'s copy is stale (Discovery, not Explorer).

**`INVENTORY.md`** — this file.
What every folder and file is, where it came from, and whether it's still needed.
**Status: Keep.** Written by reading the files, so it goes stale as they change — the file
counts and sizes are a snapshot, the judgements last longer.

### Setup and packaging

**`pyproject.toml`** — the dependency list.
**Status: Keep.**
Still open: there is no test or linting setup at all.

**`uv.lock`** — 489 KB, in git. The exact resolved versions, 129 packages. This is what actually makes the setup reproducible. **Status: Keep.**

**`.python-version`** — says 3.12, matches what's installed. **Status: Keep.**

### Infrastructure

**`docker-compose.yml`** — starts one Postgres container holding the entire corpus:
chunk rows, embeddings, and the keyword index. Settings match what the code expects.
**Status: Keep.** Three gaps: no health check (things can start before the database is ready),
no memory tuning, and **no backup anywhere** — that one volume holds roughly 13.5 hours of GPU work.

**`Dockerfile.postgres`** — builds that container, adding the keyword-search extension at an exact version.
**Status: Keep.** The best-pinned file in the project.

**`Dockerfile`** — builds the *application* container: the search pipeline plus a web server,
CPU-only, with both model weight files copied in rather than downloaded on startup.
**Status: Keep, unbuilt.** Written during the Day 9 run, so it has never been built — the build
needs a dependency re-lock, and re-locking mid-run would have changed the environment underneath it.

**`.dockerignore`** — keeps the data folder, the virtual environment and the notes out of that image.
**Keep.** Without it the build would copy roughly the whole project, including the corpus.

**`.github/workflows/ci.yml`** — the automatic checks that run on every push. Two stages: the
self-contained guards, then the search-quality gate. **Keep, never run.** Written before the
fixture it scores existed.

**`scripts/checks/ci_fixture_build.py`** — makes the small committed file the gate scores:
400 dev questions, each with its correct chunks and the ranking the published run produced.
Run by hand, rarely. **Keep, not yet run.**

**`scripts/checks/retrieval_gate.py`** — scores that file and fails the build if quality dropped
more than two points. Borrows the scoring functions from `eval.py` rather than copying them.
**Keep.** Tested against a made-up file: passes clean, fails a planted regression.

**`scripts/checks/unanswerable_validate.py`** — proves the trick questions really have no answer
in the corpus, for the 30 of 48 where that can be checked by machine. **Keep.** Catches the
non-obvious case: a year with no filing of its own can still be answerable, because annual
reports reprint several earlier years.

**`scripts/archive/unanswerable_run.py`** — asks all 48 and reports how often the system correctly
says it does not know, broken down by kind of question. **Keep, not yet run.** Costs about $0.60.

### Environment and settings

**`.env`** — **not in git**, correctly. Holds real credentials.
**Status: Keep**, but it contains a `GROQ_API_KEY` that nothing uses anymore — you moved off that
service. Dead credential, worth removing.

**`.env.example`** — the template others copy. **Status: Keep.**

**`.gitignore`** — 2.7 KB, in git.
Unusually careful: it explains, per file, *why* each exclusion is safe. **Status: Keep.**

**`.claude/settings.json`** — three permission entries, no secrets. **Status: Keep.**

**`.claude/settings.local.json`** — your personal permissions, no secrets. **Status: Keep.**

### Other folders

**`images/results_chart.png`** — 25 KB, the chart on the README.
**Status: Replaced in content.** Made before three separate corrections, so the front page of the
project shows numbers you've withdrawn. **No script makes it** — it's hand-made, which is exactly
why it went stale.

**`.venv/`** — 1.2 GB, not in git. Rebuilt any time with `uv sync`. **Status: Rebuildable.**

**`.git/`** — 442 MB. Large for a project whose text is under a megabyte. Something bulky was
probably committed early, before the ignore rules matured. Worth a look someday, not urgent.

### What could be tidied at the root

- Cut the snapshot in `CLAUDE.md` down to a few pointing lines so numbers live in one place only.
- Keep `SESSION.md` about the future and `DECISIONS.md` about the past — that's a clean split, but section 1 currently blurs it.
- Fix the README chart and add a results table.
- The clearest doc order for someone new: README to set up, `spec.md` for what was promised, `DECISIONS.md` for what happened, `SESSION.md` for what's next, `CLAUDE.md` for how to work here.

---

## `src/rag_sec/` — the shared library

Seventeen files, about 3,280 lines. This is the code that the scripts all borrow from. Nothing here
runs on its own; everything here gets imported.

**`__init__.py`** — empty, marks the folder as importable. **Keep.**

**`config.py`** — 23 lines. Names the two models used, and picks the accelerator, in one place, so the piece that counts words
and the piece that does the search never disagree. **Keep** — small but load-bearing.

**`dataset.py`** — downloads the question set and splits it into train/dev/test. The split is
grouped by document so the same filing can't appear on both sides. **Keep** — every question comes through here.

**`edgar.py`** — downloads filings from the SEC, politely rate-limited. Matches on the fiscal
year end rather than the filing date, which fixed 26 wrong years out of 100.
**Keep, but parked** — the corpus is complete at 799 filings, so this only runs again if it grows.

**`parsing.py`** — turns filing HTML into a clean list of paragraphs and tables. Repairs tables
that filing software splits across cells (rejoining `$` and `6,635`).
**Keep** — one of its functions is still used live during compression.
Note: it leans on internal pieces of an outside library pinned to one version; upgrading breaks it.

**`chunking.py`** — Packs the parsed pieces into ~900-word chunks without ever cutting
a table row in half. **Keep** — this defines the corpus.
Note: the **heading bug is now fixed** (`RETR-7`/`RETR-8`), behind two environment flags that now default **on**, matching the re-indexed corpus. 45.9% of atoms used to be labelled with a *different* section's heading.
The fix deliberately does not move a single chunk boundary, so gold labels and chunk numbering
survive it; only chunk *text* changes, on 47.5% of chunks. Flip the flags on as part of a
re-index — until then the file behaves exactly as it always did, verified byte-for-byte on all
99,654 chunks.

**`summarize.py`** — asks a model to summarize a table, used only for the losing table experiment.
**Keep as history.** It's the only file here that costs money to run, and it should not run again.

**`store.py`** — the database connection and table definitions, plus a check that the corpus still
has the expected number of rows. **Keep** — every database touch goes through it.

**`preflight.py`** — reads the project's own source code and refuses to start if any database query
forgets to say which variant it wants. This is the guard against the bug described under `RETR-24`,
where the wrong table layout's chunks were being scored as if they were the real ones.
**Keep.** It runs before connecting, so it works even with the database down.
Its limit: it stops a bad run from starting, it cannot check a result already written to disk.

**`company.py`** — 338 lines. Works out which company a question is about, from the question text
alone, and strips the company name back out before searching. Together these are the **biggest
measured win in the project.** **Keep.**

**`candidates.py`** — the one home for first-stage search: the dense query, the BM25 query, the
fusion of the two, and the text lookup for the result. Used to be eleven copies scattered across
five files, which is how the variant bug needed the same fix seven times. **Keep.** Which table
layout to search is a required argument here, never a default — that is the whole point.

**`retrieve.py`** — the full search pipeline as one callable function: company filter plus the
query strip, the latter applied to the reranker only. This is the best measured setup. **Keep.**

**`api.py`** — the web layer: two health checks and one `/ask` endpoint that runs the best
measured setup and returns the answer with the chunks it cited. Reuses the answer prompt from
`agent.py`, so what gets served is the same thing that was measured. **Keep, unrun.**

**`agent.py`** — the multi-step loop: plan, search, judge, answer.
**Keep, parked.** Verified working on 3 questions, then deferred on cost evidence. One prompt
inside it is still used by the current cost work.

**`eval.py`** — 341 lines. Decides which chunks count as correct answers, and computes the scores.
**Every published retrieval number in this project comes from this file. Keep.**
Note: its thresholds were tuned on a 200-row hand sample, so the numbers are only as good as that
tuning — roughly 15% of one failure category is still thought to be mislabelled.

**`answer_eval.py`** — checks whether a generated answer's number matches the expected one,
allowing for units like thousands and millions. **Keep.**

**`compress.py`** — cuts evidence down to fit a token budget, which is the main cost lever.
**Keep** — the next planned fixes land here.
Note: one function, `compress()`, is unused but deliberately kept as the template for that fix.

**`tracing.py`** — 452 lines. The Langfuse wrapper: one context manager per observation kind
(`span`, `generation`, `retriever`, `embedding`) plus `question_trace` and `flush_tracing`.
**The only file allowed to import `langfuse`** — everything else goes through here, so the
backend is swappable in one place. **Keep.** Two properties it is built around: it no-ops
entirely when `LANGFUSE_*` is unset, so keyless machines and CI import it fine, and it never
raises into its caller — the first failure anywhere latches tracing off for the process rather
than retrying ~8,000 times or killing a 4.4-hour paid run. `scripts/checks/tracing_offline.py`
is what holds it to both.

### What could be tidied in `src/`

- Most of what looks messy here is deliberate and documented. Don't "clean" it without reading the reasons.
- `eval.py` does two unrelated jobs — labelling and scoring — and would be easier to test if split.
  Not worth touching a dozen scripts' imports for no behaviour change; do it on the next re-index
  cycle, when those scripts are being edited anyway.

---

## `scripts/` — the layout

Three folders, named by role rather than by day. The old `dayN_` prefixes are gone and so are
the nine topic folders (`corpus/`, `index/`, `retrieval/`, `compression/`, `eval/`,
`observability/`, `analysis/`) — a reader had to know which arm a script belonged to before they
could find it. The day tags live in `DECISIONS.md`, where they belong.

| Folder | What lives there |
|---|---|
| `pipeline/` | seven numbered phases, in the order a corpus is actually built and measured |
| `pipeline/hpc/` | the two cluster legs, plus their sbatch job files |
| `checks/` | guards and regression tests, not investigations |
| `archive/` | one-off measurements whose findings are already in `DECISIONS.md`, and superseded scripts kept as the record of how a number was made |

Every `*_hpc.*` file is deliberately self-contained: it runs on the cluster where the
`rag_sec` package isn't installed, so it must never import from it (`ARM3-2`, `INFRA-6`).

---

## `scripts/pipeline/` — the seven phases

Twenty scripts collapsed into seven, one per stage, each a subcommand-per-leg CLI. Numbered so
the order is the file listing. Every phase docstring carries PRODUCES / READS / the
`DECISIONS.md` rows it backs / when it actually ran / its traps — so the provenance the old
per-folder split carried in this file now lives next to the code.

**`01_corpus.py`** — the one way to build the corpus: fetch all 799 filings from EDGAR, parse
them, chunk them. **Keep, active.** Replaced `day2_ingest.py`, `day4_ingest_next200.py` and
`day2_chunk.py` in `INFRA-8`, and now `corpus/build_corpus.py`; all are in git history.
No sampling and no seed — 799 *is* the whole pool, so there was never anything to sample, and the
seeds were what allowed `DATA-6`'s accidental duplicate run. Each stage skips its own output, so one
run reaches 799 and re-running is a no-op. `--rechunk` (~4 min) and `--reparse` (~30 min) rebuild
after a `chunking.py`/`parsing.py` change without re-downloading 2.9 GB; `--limit N` smoke-tests.
Note: `--rechunk` rewrites all 799 chunk files to add the `standalone` field they predate
(`ARM4-2`) — content-identical, but it touches every file.

**`02_gold_labels.py`** — the ground truth every metric is scored against. Two independent legs.
Absorbed `eval/resolve_gold_evidence.py` and `archive/arm4_identify_gold_tables.py`.

- `evidence` turns pointers like `table_6` into the actual rows and sentences, so scoring never
  has to touch the huge original dataset files. **Keep — and this is the most fragile thing in
  the project.** Two problems, both confirmed: the file it reads,
  `data/day7_gold_inds_matched_full.json`, has **no producer anywhere in this project** (the
  matching that made it was done by hand and never saved), and it reads a `data/raw/` folder that
  **does not exist on this machine**. `INFRA-9`'s two guards now make it refuse rather than
  quietly overwrite good ground truth with an almost-empty file. Everything you can currently
  measure depends on those two files and neither can be rebuilt from this repository.
- `tables` works out which raw table answers which question, so the B and C layouts only had to
  be built for the ~498 tables that mattered. **Keep** — it holds the **last surviving copy of
  the old labelling method** (superseded by `GOLD-1`); deleting it removes that approach from the
  project entirely. Runnable today: no DB, no GPU, no money. Its output
  `data/day6_gold_tables.json` is still read by `eval.py` and by phase 06.

**`03_index.py`** — makes the corpus searchable. Four legs, absorbing `index/embed_local.py`,
`index_bm25.py`, `embed_prepare.py`, `embed_load.py` and `reembed_changed.py`. **Both routes are
kept on purpose:** local is how a fresh machine gets a corpus, the cluster split job is how this
project's real passes were actually run.

- `local` — embeds on this laptop and loads into the database, then builds HNSW. ~1.4 chunks/s on
  this M3's MPS (`INFRA-13`): fine for a smoke test, ~9.3 h for a full pass.
- `bm25` — builds the keyword search index. **Nothing from Arm 2 onward works without it.**
  Note: the count it prints includes the B and C rows, so it reports 106,027 rather than the real
  99,654. Harmless here, but that mismatch is what eventually exposed the contamination bug.
- `new-filings --prepare` / `--load` — the corpus-growth split job. `--load` skips a whole filing
  rather than loading half of one if any chunk is missing its embedding.
- `changed-chunks --prepare` / `--load` — the `RETR-7`/`RETR-8` re-embed of the 47,312 chunks
  whose text moved. `--load` refuses rather than half-applies if any changed chunk has no
  embedding or any filing's chunk count moved, and drops/rebuilds HNSW + BM25 around the bulk
  update. See `RUNBOOK.md` for the full cycle.

**`04_arms_first_stage.py`** — Arms 1 and 2, one positional argument apart, which is why they
belong in one file (`spec.md` 2.2: change one thing per arm). Absorbed `retrieval/arm1_dense.py`
and `arm2_hybrid.py`, which differed by 12 lines. **Keep, active.** `--company-filter` is what
produced Arm 1's 0.329 -> 0.384.

**`05_arm3_rerank.py`** — Arm 3, the headline arm. Absorbed `retrieval/rerank_prepare.py`,
`rerank_score.py` and the abandoned `archive/arm3_*` trio. Because the reranker can't run on a
laptop in reasonable time, this is a split job: `prepare` (needs the database) → the cluster leg
→ `score` (no GPU). **Keep, active — this produced the numbers you quote.**

- `prepare` builds all four cells in one GPU booking so the filter/strip improvement can be
  attributed properly.
- `local` is the single-process CPU/MPS alternative, same model and same cell definitions,
  writing the identical scores format. It exists so the headline arm is reproducible **without
  cluster access** — the one real gap the old `retrieval/` folder had. ~32.6 s/question/cell on
  an M3, so use `--n`; resumable per question id, and it writes to
  `data/local_rr_{split}_scores.jsonl`, deliberately not the published `retr7_rr_*` files.
- `score` computes the headline comparison and **writes a results file**
  (`data/retr7_arm3_{dev,test}_results.json`), which `rerank_score.py` never did. It requires all
  four cells present and counts what it skipped, so a missing cell cannot shrink the denominator
  invisibly.
- Note: the internal default filenames still carry Day-8 drift (`day8_retr16v2_*`) while the live
  artifacts are `retr7_rr_{dev,test}_scores.jsonl`. Pass `--scores` explicitly. Left alone so the
  merge is behaviour-preserving against the scripts it replaced.

**`06_arm4_tables.py`** — the table-layout experiment that lost, A vs B vs C. Absorbed
`corpus/build_table_variants.py` and `archive/arm4_rerank_{prepare,score}.py`. **Keep.**
`variants` **writes the live index and can cost money** (Strategy C summarises tables with
claude-haiku-4-5, `ARM4-6`); all 498 summaries are already cached on disk, so a normal run spends
nothing, and a cache miss now **refuses** rather than silently re-spending. `score` is the odd leg
out: it computes a current number, not a historical one — its variant-A output is the live
baseline and the `unfiltered_raw` control every Day 8 gain is measured against, and it is still
the only thing that can score B and C.

**`07_arm6_loop.py`** — the Day 9 Arm 6 agentic loop plus its paired static baseline. Absorbed
`eval/agent_run.py` and `agent_analyze.py`.

- `run` — **NOT FREE.** Live paid Gemini calls, ~$8 for n=300, now behind an explicit
  `--allow-paid-run` gate; `-n` is the other brake. Resume is at the **file level**: completed ids
  in `--out` are skipped and errored rows are not counted as done, so a kill costs nothing but
  deleting that file costs money again. Must stay at `--concurrency 1` — concurrent model
  construction on MPS segfaults the machine (`AGENT-10`/`AGENT-17`).
- `analyze` — trajectory, judge accuracy, retrieval and answer scores, p50/p95 per stage.
  **Free, read-only**, safe on a partial results file mid-run. **The authoritative source for
  $/question**: it prices each usage record from the model ids in the row, so the number is not
  typed in anywhere.
- Its static baseline is where `AGENT-16` was found — the rankings it slices are not in
  descending score order as stored, and `checks/static_ranking_order.py` now guards that.

### `scripts/pipeline/hpc/` — the two cluster legs

**`embed_hpc.py`** — BGE-M3 embedding on a cluster GPU node. **Unchanged** from the day-5
corpus-growth run: payload `{filing_stem, chunk_index, text}` in, `{filing_stem, chunk_index,
embedding}` out. Resumable if the job is killed. **Keep.**

**`rerank_hpc.py`** — the current reranker on the cluster. Ran 4.4 h for dev and 7.2 h for test.
Checkpoints per question id and skips what is already in the output file. **Keep, active.**
**Its basename is part of the contract:** it is scp'd to the cluster home directory and invoked
by bare name, so renaming it breaks a submitted job rather than this repo. Same for argv —
exactly two positionals.

**`rerank_hpc.sbatch`** — the cluster job file for the test run. Moved here from
`retrieval/` to sit beside the `.py` it submits. **Keep.**

Both import nothing from `rag_sec`; do not add an import.

---

## `scripts/checks/` — guards and regression tests

Twelve files. Not investigations — each has an ongoing obligation, which is exactly what filing
them as "Day 8 one-offs" used to hide. `mps_leak_probe.py` left for `archive/` in the reorg: it
reproduces a behaviour rather than asserting one, so nothing should gate on it.

Five are invoked by path from CI (`.github/workflows/ci.yml`) — `variant_predicates.py`,
`candidate_sql.py`, `static_ranking_order.py`, `tracing_offline.py`, `retrieval_gate.py` — and
`container_wordlist.py` is invoked by the `Dockerfile` at both build and runtime. **Those six
filenames are load-bearing outside this repo's Python; moving one breaks a pipeline, not an
import.** Four more are described only in passing below and are worth a pass of their own:
`ci_fixture_build.py` (distils the gitignored scores + chunks into the one committed ~1 MB
fixture the retrieval gate replays, `CI-1`), `retrieval_gate.py` (the gate itself, `CI-2`),
`container_wordlist.py`, and `unanswerable_validate.py`.

**`variant_predicates.py`** — checks the project's own code for the query bug.
**Keep**, but it's a safety check, not an investigation. The same check now runs automatically.
Its remaining use is running with the database down or before committing — and right now
**nothing actually invokes it**, so that guard is only half wired up.

**`candidate_sql.py`** — pins the exact database queries `candidates.py` sends, fully filled in,
so a "harmless tidy-up" of the shared search code can't silently change what any published number
was measured on. **Keep — this is a regression test**, and it needs no database.

**`company_resolver.py`** — 27 hand-written cases checking the company matcher doesn't
confuse "apple prices" or "visa applications" for tickers. **Keep — this is a real regression test**,
not a one-off. It belongs in a `tests/` folder.

**`atom_replay.py`** — proves the corpus can be rebuilt exactly as stored, all 99,654
chunks byte for byte. **Keep — this is an integrity check, not a one-off.** It must be re-run after
any change to chunking or parsing.

**`heading_fix.py`** — the acceptance check for `RETR-7`/`RETR-8`. Asserts four things: with the
flags off the packer is byte-identical to the pre-fix one, with them on every chunk boundary is
unchanged, heading misattribution goes 45.9% -> 0.0%, and (with `--labels 300`) no dev gold label
moves. **Keep — this is what makes the fix safe to ship dark**, and it must pass before the
re-index cycle flips the flags on. Read-only, no GPU, no API calls.

**`tracing_offline.py`** — 462 lines, **nine checks** over `src/rag_sec/tracing.py`.
**No keys, no network, no GPU, no cost** — the enabled cases point the SDK at a port that is
never listening, so the failure paths are exercised for real rather than mocked. Each case runs
in its own subprocess, because the module latches its enabled/disabled decision once per
process. **Keep — this is what makes `OBS-4`'s "tracing can never kill the run" a checked claim
instead of a comment**, and it exits non-zero on the first failure so it can go in CI.

**`static_ranking_order.py`** — 81 lines. Fails if the static rankings `agent_run.py` slices
`[:TOP_K]` from are not in descending rerank-score order **at the point of use**. Read-only, no
GPU, no API. **Keep** — this is `AGENT-16`'s guard: `rerank_hpc.py` attaches the score to the
first-stage RRF order and promises nothing about ordering, the published scorer sorts on load
and `agent_run.py` did not, and 0 of 1235 dev cells were in order. Fourth instance of the
project's recurring bug class, which is why it is mechanical now.

**`agent_loop_smoke_test.py`** — **the dangerous one.** It looks like a test, but every run makes live
paid model calls (about $0.04) and writes rows into your database. There is **no dry-run and no
confirmation prompt.** **Keep, but give it a guard.** It lives here rather than with the compression
work because it tests the deferred multi-step loop, not compression; its docstring now leads with
the cost warning.

---

## `scripts/archive/` — one-offs and superseded scripts

Thirty files, in two groups. Every one carries a docstring header saying what it did, which
`DECISIONS.md` IDs and `data/` files came from it, what replaced it, and whether it is safe to
run today. **Archived does not mean dead data** — several of these produced files that live code
still reads. **Delete nothing here:** each is either the only way to regenerate a published
number, or the record of how one was made.

Nothing in this folder is needed to reproduce an arm. That is the line: if a script is on the
path from EDGAR to a headline number it is a `pipeline/` phase; if it answered one question whose
answer is now a `DECISIONS.md` row, it is here.

### The one-off measurements (moved here, not retired)

Each was written to answer one question, printed a table, and the answer went into
`DECISIONS.md`. None of them save their output, so deleting one means rewriting it to get that
number back.

**`failure_triage.py`** — the most consequential file here. Sorted every failing question into a
cause and found that the reranker is only 6.9% of the problem while 19.6% of the time the right
answer never even reached the shortlist. **This reframed the whole project onto search.** Re-run
on the shipped ordering (`RETR-33`): 87.0 / 1.5 / 11.5 / 0. The old contaminated ordering is still
reachable via `--scores` and reproduces the earlier numbers exactly.

**`company_filter_ab.py`** — the test that proved the company filter works (+8.1 points). Needs
the database and embeds 1,235 questions locally — minutes, not hours. It has a `--split test`
option; **don't use it casually**, the test split is meant to stay untouched.

**`chunk_length_vs_recall.py`** — asks whether search quality drops on longer chunks. It does:
0.50 short, 0.41 medium, 0.26 long. Warning: it reads a results file that has been rewritten
twice since, so re-running gives different numbers than the ones recorded.

**`chunk_interpretability.py`** — how much evidence is unreadable on its own even before
compression. Answer: 7% genuinely at risk — the standing reason *not* to re-cut the corpus.

**`gold_atom_floor.py`** / **`slice_floor.py`** — the smallest useful evidence piece averages 684
words, 76% of its chunk, which killed one compression approach; thinner slices make a small
budget workable. Two sizes of the same measurement, and they should be one script with an option.
Note: `slice_floor.py` prints the other's result as a **typed-in string** rather than computing
it, so it will silently go stale.

**`stratum_b_channel.py`** — checked whether the earlier compression conclusion held up. It
didn't: the loose way of judging "did the evidence survive" over-reports, and unevenly, which
**doubled the real gap between the two approaches.**

**`pack_variants.py`** — built and measured six ways of assembling the compressed prompt, all
offline. None of the three proposed fixes to evidence selection works (`COST-27`); the one change
that did ship — labelling each block with its filing and year, keeping a table's rows together —
costs a point of evidence survival and buys a prompt the reader can attribute (`COST-28`). It
imports the loaders from `stratum_b_channel.py` by literal filename rather than copying them, so
**the two must stay in the same directory**, and it **asserts per question that its rebuilt prompt
still reproduces the one whose numbers are published.**

**`rescore_labels.py`** / **`label_matcher_ab.py`** / **`label_audit_prepare.py`** /
**`label_audit_score.py`** — the `RETR-9`/`RETR-34`/`RETR-35` label work: the A/B between the old
shingle matcher and `gold_inds` positional matching, the blind audit that sized the label error
rate, and the rescore that moved every arm onto coverage-based labels. The audit is closed and
its rates, mechanism and cross-check are all written up in `DECISIONS.md`.

**`slice_prepare.py`** / **`slice_rerank_hpc.py`** / **`slice_rerank_hpc.sbatch`** /
**`slice_budget_sweep.py`** — the compression measurement chain: cut candidates into ~150-word
slices, score 555,337 of them on the cluster (~1 h 52 m of GPU), sweep budgets across four
approaches. The sbatch carries a real operational rule in its comments: submit it **second**,
after the other GPU job has loaded its model, because both pull the same model into the same
cache and two cold downloads at once risks corrupting it. `slice_budget_sweep.py` **prints the
wrong number by default** — the stricter, honest survival measure lives only in
`stratum_b_channel.py` and was never folded back in; the file admits this in its own comments.
Note: `slice_prepare.py`'s default input file isn't on disk.
**Compression was measured and never shipped** (`COST-36`): `agent.py` sends uncompressed.

**`answer_ab_prepare.py`** / **`answer_ab_run.py`** / **`answer_batch_run.py`** /
**`answer_ab_score.py`** — the paid answer A/B. **The only scripts in the project that spend real
money.** `prepare` and `score` are free and offline on purpose, so prompts can be eyeballed
before any spending and responses re-scored afterwards. `answer_ab_run.py` sends synchronously at
`standard` (~$2.02/run; **not** `flex`, which 503'd ~9 of 10 requests, `COST-29`);
`answer_batch_run.py` is the same payload and row shape via the Batch API at 50% (~$1.01), joining
results back by an explicit `id|arm|thinking` key rather than by position. Both share the same
resume file, so neither re-pays for the other's rows — **while `data/day8_cost13_responses.jsonl`
exists a re-run is free; delete or move it and it costs money again.** `--limit` is a wiring
check, **not a sample**: the payload is stratum-sorted, so any limit under 89 is 100%
`A_gold_lost` (`COST-33`).

**`latency_measure.py`** — the per-stage latency pass behind the figures `spec.md` publishes.
Shares its sampling seed with phase 07 and `mps_leak_probe.py`, so all three see the same
questions. Run on mains power: battery throttling moves the very numbers it measures.

**`mps_leak_probe.py`** — moved out of `checks/`, because it is a **diagnostic, not a guard**: it
reproduces the MPS memory growth behind `AGENT-24`'s cache drain rather than asserting anything,
so nothing should gate on it.

**`candidate_k_curve.py`** — the `TOP_K`/candidate-width sweep. That question is closed
(`RETR-33`); first-stage candidate *generation* is the part still open.

**`onnx_rerank_export.py`** / **`onnx_rerank_parity.py`** / **`ort_fp32_latency.py`** — the
`DEPLOY-6`/`DEPLOY-11` reranker route: export the cross-encoder to ONNX, quantise to int8, and
**prove it does not reorder the top-10** before claiming `RETR-39`'s figures on the deployed box.
`onnx_rerank_parity.py` re-execs **itself** as a subprocess per phase so the OS reclaims each
backend at exit — three backends resident at once on 16 GiB is what killed `DEPLOY-8`.
`ort_fp32_latency.py` imports its helpers from `onnx_rerank_parity.py` as a **sibling module**,
so the two must stay in the same directory.

**`unanswerable_run.py`** — the abstention pass; `checks/unanswerable_validate.py` is the guard
half and stays in `checks/`.

**`worst_failures.py`** — writes the 20 worst failures per arm as markdown (`spec.md` 2.2 rule 5)
to `data/day9_worst_failures_{arm6,static}.md`. **Free, read-only**, safe on a partial file. Its
ordering is a stated choice, not a measurement: failures are ranked by how close the gold
evidence got to the answer model — reasoning, then rerank, then retrieval — because every wrong
number is equally wrong. Note: **regenerate it from the finished run**; the two pre-`AGENT-16`
files were deleted rather than kept, since a stale worst-failures list reads as current.

**`build_langfuse_dashboard.py`** — builds the Day 10 dashboard ("RAG-SEC -- Arm 6 loop vs
static") and its **12 widgets** by pushing them to Langfuse. **This is why the dashboard is
reproducible rather than a screenshot of some clicks.** Needs the Langfuse keys and network; no
GPU, no model calls, so it costs nothing to re-run. **Idempotent by name** — widgets are looked
up by name and PATCHed, never duplicated, and nothing is removed without `--prune`/`--delete`.
Note: it writes through the `unstable/` API, pinned to snapshot 4.16.0, a surface that **drops
unknown body keys silently instead of erroring** — so the script re-reads what it wrote and
compares field by field, because a half-built dashboard that renders is worse than a failed run.
It backs off on 429: a full build is ~30 calls against a 30/min limit. Archived because the
dashboard is built, not because it is disposable.

### The superseded script

**`arm4_rerank_hpc.py`** — Arm 4's cluster leg, stage 2 of the split job. Deliberately **not**
moved into `pipeline/hpc/`: it differs from the Arm 3 twin by **three lines of actual code** and
exists only as the record of three finished GPU passes. Its output
`data/day6_arm4_{A,B,C}_rerank_scores.jsonl` is all three real passes and all three are still on
disk — **the A file is read by live code today**, as phase 05's `unfiltered_raw` baseline and as
`failure_triage.py`'s input. Safe on a GPU box, but do not let it overwrite that A file.

The rest of the old archive is gone rather than kept. `arm3_singleprocess.py`,
`arm3_rerank_{prepare,hpc,score}.py`, `arm4_singleprocess.py`,
`arm4_rerank_{prepare,score}.py` and `arm4_identify_gold_tables.py` were all superseded by a
`pipeline/` phase — phase 05's `local` leg is now the clearest single-file statement of what Arm 3
is, and it is the *current* one rather than an abandoned Day-5 route. Their outputs are either
gone or still on disk and still read, and the scripts themselves are in git history.

### What could be tidied in `scripts/archive/`

- Delete nothing. Everything here is the evidence behind a published number.
- Fold the honest survival measure into `slice_budget_sweep.py` so the wrong number stops being
  what it prints by default.
- Several files hardcode the same constants — slice size 150, budget 1500, and which ranking to
  read. The slice size is the one that bites: slices travel as *positions*, so a mismatch quietly
  resolves to completely different text. Some files guard against this; **`stratum_b_channel.py`
  does not.**
- The "load the ranking from disk" duplication that used to run to ten copies is fixed:
  `rag_sec.eval.load_ranking` is the single loader and it **sorts on load**, which is `AGENT-16`'s
  fix rather than a tidy-up.
- Two scoring scripts record **different GPU model names** as a fixed string. Both are preserved
  verbatim because the published results files carry them; see phase 06's
  `RERANK_DEVICE_UNVERIFIED` note.

---

## `data/` — 4.9 GB, 96 entries

The biggest source of confusion, and the biggest cleanup opportunity. Roughly **1 GB is dead weight.**

### The bulk folders

**`filings/`** — 2.9 GB, 799 files. Raw filing HTML downloaded from the SEC, named
`TICKER_YEAR_CIK.htm`. Nothing reads it at runtime; it's the audit trail behind everything else.
Re-downloading is free but slow, and the SEC can change what it serves — so treat it as
effectively irreplaceable. **Keep, but it belongs in an archive folder, not next to live files.**

**`parsed/`** — 389 MB, 799 files. Each filing broken into paragraphs and tables.
**Keep — actively read** during compression.

**`chunks/`** — 377 MB, 799 files. The ~900-word chunks. **Keep — actively read.**
Scoring reads these directly, and every chunk number in every results file is meaningless without them.

### Logs from building the corpus

| File | Verdict |
|---|---|
| `day2_ingest_log.jsonl` (23 KB) | **Keep** — small, and it's the evidence behind a recorded decision |
| `day2_ingest_log_docling.jsonl` (16 KB) | **Keep** — the only surviving proof of the parser comparison; the code that made it is gone |
| `day2_chunk_log.jsonl` (12 KB) | **Decide** — nothing reads it and, since `INFRA-8`, nothing writes it; tracked in git, and the rule that used to contradict that is gone |
| `day4_ingest_log.jsonl` (128 KB) | **Keep** — the record of the corpus growing to 799 |
| `day4_ingest_stdout.log`, `day4b_...`, `day5_...` (25 KB) | **Delete** — three copies of the same library warning, no results |
| `day5_prepare_payload_stdout.log` (77 KB) | **Delete** — a progress bar; its one useful line is already in the script |
| `day5_arm3_stdout.log` (1.5 KB) | **Keep** — the timing evidence for moving reranking to the cluster |
| `day2_findings.md` (2.4 KB) | **Keep**, but it's writing, not data — it belongs in a docs folder. It also points at a folder that no longer exists |

### Results from Arms 1-3

All are small, tracked in git, and rebuildable in minutes. **Keep all of them.**

`day3_arm1_dev_results.json` and its failures list (Arm 1: 0.329 / 0.466), the same pair with
`_companyfilter` (0.384 / 0.607), `day4_arm2_...` (Arm 2: 0.495 / 0.687) and its filtered pair
(0.520 / 0.776), and `day5_arm3_...` (Arm 3: 0.609 / 0.685).

**`rerank_scores.jsonl`** (1.6 MB) — **Keep.** This one is not rebuildable cheaply — it took a GPU
pass — yet `.gitignore` classes it as disposable. That rule is wrong for this file.

**`rerank_scores_603filings.jsonl`** (1.1 MB) — **Delete.** From when the corpus was 603 filings
and the question set was 837. Nothing reads it and it can't be compared to anything current.

### The table-layout experiment

| File | Verdict |
|---|---|
| `day6_gold_tables.json` (208 KB) | **Keep — actively read** by scoring |
| `day6_table_summaries.json` (228 KB) | **Keep — this cost money** (498 model calls) |
| `day6_arm4_{A,B,C}_dev_results.json` + failures (~3 MB) | **Keep** — the published result: A beat both |
| `day6_arm4_{A,B,C}_rerank_scores.jsonl` (5.8 MB) | **Keep — three GPU passes.** The A file is still read by two current scripts |
| `day6_arm4_A_rerank_payload.json` (**230 MB**) | **Delete** — cluster input, rebuilt in minutes. B and C were already deleted; this one was missed |
| `day6_arm4_{A,B,C}_cpu_dev_*` (6 files, 21 KB) | **Delete** — 5-question warm-ups. Three of them report a score of 0.9 and are genuinely dangerous if mistaken for real results |

### Ground truth and labelling

**`day7_gold_inds_matched_full.json`** (4.8 MB) — **Keep. The highest-risk file here.**
Nothing in this project creates it. Everything you can measure depends on it.

**`day7_gold_evidence_resolved.json`** (2.4 MB) — **Keep.** Rebuildable only from the file above
plus raw dataset files that aren't present.

**`day6_gold_labeling_audit.txt`** (1.5 MB) — **Delete**, replaced by the v2 file.

**`day6_gold_labeling_audit_v2.txt`** (1.2 MB) — **Keep as archive.** A hand audit, so it can't be regenerated, but nothing reads it.

**`*.bak_old_labeler`** (12 files, 5.9 MB) — **Keep as archive.** These are every arm's numbers
under the *old* labelling method — the "before" half of a comparison `DECISIONS.md` explicitly
wants kept. They double the length of every listing though; a subfolder would fix that.

### Day 8 search results

**`day8_retr16v2_dev_scores.jsonl`** (8.6 MB) — **Keep.** 4.4 GPU-hours. The most-read scores file
in the project and the source of your current best numbers.

**`day8_retr18_test_scores.jsonl`** (14 MB) — **Keep.** 7.2 GPU-hours, on the untouched test split.
This is the evidence that your improvement isn't just fitted to the data you tuned on.

**`day8_retr16v2_dev_results.json`** (1.7 KB) and **`retr35_test_scores.json`** (1.7 KB) —
**Keep.** The scored tables behind the headline, dev and test. Tiny, and they are what stops the
best number in the project from living only in prose (`INFRA-10`).

**`day8_retr16_rerank_payload.json`** (**113 MB**) — **Delete.** The input to a run you've formally withdrawn.

**`day8_failure_cases.json`** (15 MB) — **Delete.** Free to regenerate; the conclusions are already written down.

**`day8_failure_slices/`** (5 files, 556 KB) — **Delete.** A finished reading queue.

### Day 8 compression

**`day8_slice_scores_t150_filtered_stripped.jsonl`** (28 MB) — **Keep.** Nearly two GPU-hours, and
it's about to be re-read for the next round of fixes.

**`day8_survival_flags.json`** (296 KB) — **Keep**, but flagged: these were built with the loose
measure that has since been shown to over-report. Anything built on them inherits that.

**`day8_cost13_payload.json`** (6.5 MB) — **Keep, upgraded from "delete or archive" by
`COST-34`.** It is no longer just a record: it is the fixed prompt set that both the medium and
low arms share, and that shared-ness is the whole reason their delta is clean. It is also
*not* cheaply rebuildable now — regenerating it post-`RETR-7` needs the slice-rerank GPU job,
and rebuilding it would silently change the prompts underneath both stored arms. Deleting it
makes `COST-34` unreproducible.

**`day8_cost13_responses.jsonl`** (144 KB) — **Keep. You paid for this.** It's also what makes
re-running free instead of paid. Note it holds *two* reasoning levels — 278 medium rows from
2026-09-02 plus `COST-33`'s 10-row low pilot — so it must be read with
`answer_ab_score.py --thinking`, which is why that flag exists (`COST-34`).

**`cost31_thinking_medium.jsonl`** and **`cost31_thinking_low.jsonl`** (~170 KB each) —
**Keep. You paid for these, ~$2.09 of batch.** `COST-34`'s two arms, 278 rows each, 0 failed,
both run 2026-09-04 on the payload above. The medium file is *not* redundant with the older
278 medium rows: re-running it same-day is what retired the drift confound, and comparing the
two is the only record of how small that drift was.

**`cost34_thinking_{medium,low}_results.json`** (17 KB each) — **Keep.** Per-question verdicts
at each level; the printed tables cannot be recovered from the marginals alone.

**`day8_slice_budget_dev_results.json`** (1.9 KB) and **`day8_cost13_dev_results.json`** (17 KB) —
**Keep.** The survival sweep and the paired answer result, as files rather than as terminal output
(`INFRA-10`). The second one carries the per-question verdicts, which the printed table cannot be
recovered from. Both regenerate for free from the score files above.

### Day 9 agentic loop

**`day9_arm6_dev_results.jsonl`** — **Keep. You are paying for this**, one row per question with
both arms, their trajectories, usage and per-stage latencies. Growing while the run is in
flight; `agent_run.py` resumes off it, so **deleting it re-spends the money.** Everything Days
9-13 publish about the loop is read from this file by `agent_analyze.py` and `worst_failures.py`.

**`day9_run.log`** — **Keep while the run is live**, then decide: it is the run's stdout, and its
timing evidence is also in the JSONL. Nothing reads it.

**`day9_multidoc_negative_ANALYSIS.md`** — 12 KB. **Keep.** The `AGENT-15` write-up of Day 9's
negative: `spec.md`'s multi-document subset is empty, so the loop's published best case cannot be
tested on this benchmark, and this is where the check, the 10-K comparatives trap, and what the
run still establishes are written down. Writing rather than data, like `day2_findings.md`.

**`archive/`** — 572 KB, **5 files.** The `.bak` rows the Day 9 restarts left behind, kept as
provenance for `AGENT-9` (19 rows whose answer prompt asked for the wrong format, so both arms
scored 0.0%) and `AGENT-16` (the 21 rows plus log from before the static-ordering bug stopped the
run), and the 5-question CPU pilot behind `AGENT-13`.
**Keep as archive.** Their trajectories and `usage` are valid — only the answers are unscoreable
— and several `DECISIONS.md` rows cite them by path.

### Leftover cluster inputs and other

**`day5_embed_payload.json`** (**103 MB**) — **Delete.** Rebuilt in minutes.

**`embed_results.jsonl`** (**582 MB**) — **Decide.** Already loaded into the database. It covers
only **196 of 799 filings**, so it is *not* a usable backup. If the database is backed up, delete
it. If it isn't, the right fix is to back up the database, not to keep this.

**`company_lexicon.json`** (10 KB) — **Keep — actively read** on the search path.
Careful: it's committed to git *and* rebuilds itself only when missing. A stale committed copy will
silently win over a corpus change. If the corpus grows, delete this file to force a rebuild.

**`unanswerable_questions.jsonl`** (48 questions) — trick questions with no answer in the corpus,
used to test whether the system says "I don't know" instead of inventing a number. Five kinds;
30 of them are checkable by machine, the other 18 rest on what an annual report contains and say
so in the file. **Keep — hand-written, not regeneratable.**

**`ci_retrieval_fixture.jsonl` / `ci_retrieval_baseline.json`** — the small committed slice the
automatic quality check scores, plus the numbers it compares against. **Keep once built** — they
are what lets the check run without a database or a GPU. Neither exists yet.

---

## Safe to delete

| What | Size |
|---|---|
| `data/day6_arm4_A_rerank_payload.json` | 230 MB |
| `data/day8_retr16_rerank_payload.json` | 113 MB |
| `data/day5_embed_payload.json` | 103 MB |
| `data/day8_failure_cases.json` | 15 MB |
| `data/day8_cost13_payload.json` (or archive) | 6.5 MB |
| `data/day6_gold_labeling_audit.txt` | 1.5 MB |
| `data/rerank_scores_603filings.jsonl` | 1.1 MB |
| `data/day8_failure_slices/` | 556 KB |
| `data/day5_prepare_payload_stdout.log` | 77 KB |
| `data/day6_arm4_*_cpu_dev_*` (6 files) | 21 KB |
| Three ingest `stdout.log` files | 25 KB |
| All `__pycache__/` folders | ~280 KB |
| **Running total** | **~471 MB** |
| `data/embed_results.jsonl` — only if the database is backed up | 582 MB |
| **With that** | **~1.05 GB** |

**Looks deletable, is not:** anything ending `_rerank_scores.jsonl` or `_slice_scores_*.jsonl`
(GPU hours), `day7_gold_inds_matched_full.json` (nothing can recreate it),
`day8_cost13_responses.jsonl`, `day6_table_summaries.json` and `day9_arm6_dev_results.jsonl`
(paid for — the last one is also what makes a re-run free), and the `.bak_old_labeler` and
`data/archive/*.bak` files (deliberately kept).

**Genuinely deletable and already deleted:** `data/day9_worst_failures_{arm6,static}.md`, both
generated from pre-`AGENT-16` data. They regenerate for free once the run finishes — the reason
to delete rather than keep was that a stale failure list reads as a current one.

---

## The three things worth fixing first

1. **The ground truth can't be rebuilt.** `data/day7_gold_inds_matched_full.json` has no producer,
   and the script that reads it points at a folder that doesn't exist — so an accidental re-run
   would overwrite good data with almost nothing. Either save the matching as a real script, or
   write down plainly that this file is frozen and back it up.

2. ~~**Your best number isn't saved anywhere.**~~ **Done.** `rerank_score.py` used to print and
   exit; `scripts/pipeline/05_arm3_rerank.py score` takes `--out` and writes the scored table, which is where
   `data/retr7_arm3_{dev,test}_results.json` — the `RETR-39` headline, dev and test — come from.

3. ~~**The naming is the reason this folder feels overwhelming.**~~ **Done.** Every script used to
   be called `dayN_something`, which recorded *when* it was written, not *what it does* — and it was
   already misleading, since `arm4_rerank_score.py` carried a Day 6 prefix while computing a current
   baseline. `scripts/` was first reorganised by job, and is now organised by *role* — the seven
   numbered `pipeline/` phases in the order they run, `checks/` for guards, `archive/` for
   one-offs — because organising by job still required knowing which arm a script belonged to
   before you could find it. The day tags stay in `DECISIONS.md`, where they belong.
