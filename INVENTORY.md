# Inventory

Every folder and every file in this project: what it is, where it came from, what uses it,
and whether you still need it. Written 2026-09-02, last checked 2026-09-03, by reading the
files, not by running them.

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

Fifteen files, about 2,300 lines. This is the code that the scripts all borrow from. Nothing here
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

### What could be tidied in `src/`

- Most of what looks messy here is deliberate and documented. Don't "clean" it without reading the reasons.
- `eval.py` does two unrelated jobs — labelling and scoring — and would be easier to test if split.
  Not worth touching a dozen scripts' imports for no behaviour change; do it on the next re-index
  cycle, when those scripts are being edited anyway.

---

## `scripts/` — the layout

Eight folders, named by what the scripts do rather than the day they were written. The old
`dayN_` prefixes are gone; the day tags live in `DECISIONS.md`, where they belong.

| Folder | What lives there |
|---|---|
| `corpus/` | building the corpus from EDGAR |
| `index/` | making the corpus searchable |
| `retrieval/` | the live search arms and the current rerank pipeline |
| `compression/` | evidence compression and the paid answer A/B |
| `eval/` | ground-truth resolution |
| `checks/` | guards and regression tests, not investigations |
| `analysis/` | one-off investigations behind published numbers |
| `archive/` | superseded scripts kept as the record of how a number was made |

Every `*_hpc.*` file is deliberately self-contained: it runs on the cluster where the
`rag_sec` package isn't installed, so it must never import from it (`ARM3-2`, `INFRA-6`).

---

## `scripts/corpus/` — building the corpus

**`build_corpus.py`** — the one way to build the corpus: fetch all 799 filings from EDGAR,
parse them, chunk them. **Keep, active.** Replaced `day2_ingest.py`, `day4_ingest_next200.py` and
`day2_chunk.py` in `INFRA-8`; all three are in git history if ever needed.
No sampling and no seed — 799 *is* the whole pool, so there was never anything to sample, and the
seeds were what allowed `DATA-6`'s accidental duplicate run. Each stage skips its own output, so one
run reaches 799 and re-running is a no-op. `--rechunk` (~4 min) and `--reparse` (~30 min) rebuild
after a `chunking.py`/`parsing.py` change without re-downloading 2.9 GB; `--limit N` smoke-tests.
Note: `--rechunk` rewrites all 799 chunk files to add the `standalone` field they predate
(`ARM4-2`) — content-identical, but it touches every file.

**`build_table_variants.py`** — built the alternative table layouts B and C for the experiment
that lost. **Keep as history.** It's also the only thing that ever created those B and C rows —
the same rows that later caused the contamination bug.
Note: its description still says "Groq", which you stopped using.

---

## `scripts/index/` — making the corpus searchable

**`embed_local.py`** — computes embeddings on this laptop and loads them into the database,
then builds the search index. **Keep** — the simple local path.

**`index_bm25.py`** — 24 lines. Builds the keyword search index. **Keep** — nothing from
Arm 2 onward works without it.
Note: the count it prints includes the B and C rows, so it reports 106,027 rather than the real
99,654. Harmless here, but that mismatch is what eventually exposed the contamination bug.

The next three are one job split across two machines, because embedding the whole corpus on a
laptop isn't practical:

**`embed_prepare.py`** — works out what still needs embedding and writes it to one
file to carry to the cluster. **Keep.**

**`embed_hpc.py`** — runs **on the cluster**. Deliberately imports nothing from this project,
so nothing on the cluster can break it. Resumable if the job is killed. **Keep.**

**`embed_load.py`** — loads the results back into the database. **Keep.**
Good behaviour: if any chunk is missing its embedding it skips the whole filing rather than
loading half of one.

---

## `scripts/retrieval/` — the live search arms

Each "arm" is one search setup being compared:

- **Arm 1** — meaning-based search only
- **Arm 2** — Arm 1 plus keyword search, results merged
- **Arm 3** — Arm 2 plus the reranker
- **Arm 4** — Arm 3 tried against three table layouts, A, B and C (now in `archive/`)

Because the reranker can't run on a laptop, a reranked arm is split into three files:
**prepare** (here, needs the database) → **rerank** (on the cluster, no database) →
**score** (here, no GPU). That pattern used to repeat three times; the Day 5 and Day 6 copies
are now in `archive/` and only this one is current.

**`arm1_dense.py`** — Arm 1 end to end. **Keep, active.** The `--company-filter` option is what produced the improvement from 0.329 to 0.384.

**`arm2_hybrid.py`** — Arm 2. **Keep, active.** Differs from Arm 1 by only 12 lines — they're one option apart.

**`rerank_prepare.py`** — the current, best version. Builds four combinations
in one GPU booking so the improvement can be attributed properly. **Keep, active** — this produced
the numbers you quote.
Note: it's the only script here that must be run from the project root.

**`rerank_hpc.py`** — the current reranker, on the cluster. **Keep, active.**
Ran 4.4 hours for dev and 7.2 hours for test.

**`rerank_score.py`** — computes the headline comparison. **Keep, active.**
**One real gap: it prints to the screen and saves nothing.** Every other scoring script writes a
results file. The per-question detail behind your best number exists nowhere on disk. About ten
lines to fix, and the highest-value fix in this folder.

**`rerank_hpc.sbatch`** — the cluster job description for the test run. **Keep.**

### What could be tidied in `scripts/retrieval/`

- The three prepare / rerank / score families are about 70% identical. Combining them would save
  roughly 600 of 1,900 lines. The argument against doing it now: the older files are the record of
  numbers already published, and the cluster files are valuable precisely because they're frozen.
  That is now settled the safe way: these three are the current ones and the Day 5 / Day 6 copies
  sit in `archive/`.
- One small statistics helper is copy-pasted into **seven** files. It belongs in `eval.py`. No risk.
- Two scoring scripts record **different GPU model names** as a fixed string, which will quietly
  be wrong if either is ever run elsewhere.
- Every query here now correctly says which variant it wants. That was fixed after the
  contamination bug and is now enforced automatically.

---

## `scripts/compression/` — evidence compression and the paid answer A/B

Seven files, all about measuring whether compressing the evidence saves money without losing
accuracy. **This is the only folder in the project that spends real money.**

**`slice_prepare.py`** — cuts candidate chunks into ~150-word slices and packages them
for the cluster. **Keep.**
Note: **its default input file isn't on disk** — you'd have to rebuild that first.

**`slice_rerank_hpc.py`** — scores 555,337 slices on the cluster. **Costs GPU time:** about
1 hour 52 minutes on one machine. **Keep.**

**`slice_rerank_hpc.sbatch`** — the cluster job description for the above. **Keep.**
It carries a real operational rule in its comments: submit this **second**, after the other GPU job
has loaded its model, because both pull the same model into the same cache and two cold downloads
at once risks corrupting it.

**`slice_budget_sweep.py`** — 277 lines. Sweeps budgets across four approaches and reports
how much evidence survives. **Keep — but the number it prints is the wrong one.**
The stricter, honest measure lives only in `analysis/stratum_b_channel.py` and was never folded back
in. The file admits this in its own comments. Adding it as a second column is the obvious fix.

**`answer_ab_prepare.py`** — builds both versions of each prompt and picks which questions to
test. Free and offline on purpose, so prompts can be eyeballed before any spending. **Keep.**

**`answer_ab_run.py`** — **one of the two scripts that deliberately spend money.**
Sends prompts to Gemini one at a time, synchronously. About **$2.02** for a full run at
`standard` (corrected from $1.53, which priced thinking tokens wrong — `COST-23`/`COST-31`).
It prints a price estimate first and has a dry-run mode. Sends at `standard`, **not** `flex`:
flex 503'd ~9 of 10 requests (`COST-29`).
**Keep — and know this:** it resumes by reading `data/day8_cost13_responses.jsonl`. While that
file exists, re-running is **free**. **Delete or move that file and a re-run costs money again.**
`--limit` is a wiring check, **not a sample** — the payload is stratum-sorted, so any limit
under 89 is 100% `A_gold_lost` (`COST-33`).

**`answer_batch_run.py`** — **the other money-spending script, and the cheap one.** Same
payload in, same row shape out, via the Batch API at **50% of standard** — a full run is
~**$1.01** against `answer_ab_run.py`'s ~$2.02. Asynchronous: it splits the work into jobs
under Tier 1's 3M enqueued-token cap, uploads JSONL, submits, polls, then joins results back
by an explicit `id|arm|thinking` key rather than by position. Shares the same resume file, so
the two runners are interchangeable and neither re-pays for the other's rows. Proven on a real
10-request job (`COST-32`). **Keep.** Use this one for anything large; use `answer_ab_run.py`
when you want results in seconds rather than minutes.

**`answer_ab_score.py`** — scores the responses already paid for. Free, safe to re-run.
Prices `flex` and `batch` rows at 50% and everything else at full. **Keep.**

### What could be tidied in `scripts/compression/`

- Delete nothing. Every file either generates a published number or launches one that does.
- Fold the honest survival measure into `slice_budget_sweep.py` so the wrong number stops
  being what it prints by default.
- Four files hardcode the same constants — slice size 150, budget 1500, and which ranking to read.
  The slice size is the one that bites: slices travel as *positions*, so a mismatch quietly resolves
  to completely different text. Two files guard against this; **`analysis/stratum_b_channel.py` does not.**

---

## `scripts/eval/` — one file

**`resolve_gold_evidence.py`** — turns pointers like `table_6` into the actual rows and
sentences, so scoring never has to touch the huge original dataset files.
**Status: Keep — and this is the most fragile thing in the project.**

Two problems, both confirmed:
1. **The file it reads, `data/day7_gold_inds_matched_full.json`, has no producer anywhere in this
   project.** The matching that made it was done by hand and never saved as a script.
2. It reads from a `data/raw/` folder that **does not exist on this machine**. Missing files only
   produce a warning, so re-running it today would quietly overwrite good ground truth with an
   almost-empty file.

Everything you can currently measure depends on these two files. Neither can be rebuilt from
this repository.

---

## `scripts/checks/` — guards and regression tests

Not investigations. Each of these has an ongoing obligation, which is exactly what filing them
as "Day 8 one-offs" used to hide.

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

**`agent_loop_smoke_test.py`** — **the dangerous one.** It looks like a test, but every run makes live
paid model calls (about $0.04) and writes rows into your database. There is **no dry-run and no
confirmation prompt.** **Keep, but give it a guard.** It lives here rather than with the compression
work because it tests the deferred multi-step loop, not compression; its docstring now leads with
the cost warning.

---

## `scripts/analysis/` — one-off investigations

Seven scripts. These aren't part of the pipeline. Each was written to answer one
question, printed a table, and the answer was written into `DECISIONS.md`. **Every one of them is
the only way to regenerate a number you've published**, and none of them save their output — so
deleting one means rewriting it to get that number back.

**`chunk_length_vs_recall.py`** — asks whether search quality drops on longer chunks. It does:
0.50 on short, 0.41 on medium, 0.26 on long. **Keep.**
Warning: it reads a results file that has been rewritten twice since. Re-running gives different
numbers than the ones recorded.

**`company_filter_ab.py`** — the test that proved the company filter works (+8.1 points).
**Keep.** Needs the database and embeds 1,235 questions locally — minutes, not hours.
It has a `--split test` option; **don't use it casually**, the test split is meant to stay untouched.

**`gold_atom_floor.py`** — measured that the smallest useful evidence piece averages 684 words,
76% of its chunk, which killed one compression approach. **Keep.**

**`chunk_interpretability.py`** — asks how much evidence is unreadable on its own even
before compression. Answer: 7% genuinely at risk. **Keep** — that number is the standing reason
*not* to re-cut the corpus.

**`failure_triage.py`** — the most consequential file here. Sorted every failing
question into a cause and found that the reranker is only 6.9% of the problem while 19.6% of the
time the right answer never even reached the shortlist. **This reframed the whole project onto search.**
**Keep.** Re-run on the shipped ordering (`RETR-33`): 87.0 / 1.5 / 11.5 / 0. The old contaminated
ordering is still reachable via `--scores` and reproduces the numbers above exactly.

**`stratum_b_channel.py`** — 240 lines, the newest file here. Checked whether the earlier
compression conclusion held up. It didn't: the loose way of judging "did the evidence survive"
over-reports, and does so unevenly, which **doubled the real gap between the two approaches.**
**Keep, active** — this is the next thing to run.

**`pack_variants.py`** — built and measured six ways of assembling the compressed prompt, all
offline. Answered the three open questions in one run: none of the three proposed fixes to how
evidence is picked actually works (`COST-27`), and the one change that did ship — labelling each
block with its filing and year, and keeping a table's rows together — costs a point of evidence
survival and buys a prompt the reader can actually attribute (`COST-28`). It imports the loaders
from `stratum_b_channel.py` rather than copying them, and **asserts per question that its rebuilt
prompt still reproduces the one whose numbers are published**. **Keep, active.**

**`slice_floor.py`** — showed that cutting evidence into thinner slices makes a small budget
workable. **Keep.**
Note: it prints the previous script's result as a **typed-in string** rather than computing it, so
it will silently go stale.

### What could be tidied in `scripts/analysis/`

- Delete nothing. Everything here is the evidence behind a published number.
- Two floor-measuring scripts (`gold_atom_floor.py`, `slice_floor.py`) are the same measurement at
  two sizes and should be one script with an option.
- The same "load the ranking from disk" code is written out **three times** across this folder and
  `scripts/compression/`. Three copies quietly disagreeing is exactly the failure that already cost a day.

---

## `scripts/archive/` — superseded, kept as provenance

Nine files. Every one carries a docstring header saying what it did, which `DECISIONS.md` IDs and
`data/` files came from it, what replaced it, and whether it is safe to run today. **Archived
scripts do not mean dead data** — several of these produced files that live code still reads.

**`arm3_singleprocess.py`** — Arm 3 all in one process, on the laptop.
**Status: History.** Its output files don't exist because the laptop route was abandoned as too
slow. Keep it as the clearest single-file statement of what Arm 3 actually is.

**`arm3_rerank_prepare.py`** / **`arm3_rerank_hpc.py`** / **`arm3_rerank_score.py`** —
Arm 3's three stages. **Status: Replaced** by the Day 8 versions (now `retrieval/rerank_*`), but
keep as the record of how the published Arm 3 numbers were made. `arm3_rerank_hpc.py`'s output
`data/rerank_scores.jsonl` is a real GPU pass and is still on disk.

**`arm4_singleprocess.py`** — Arm 4 all in one process. **History.** The B and C rows it created are
the 6,373 chunks that later contaminated Arms 1-3.

**`arm4_identify_gold_tables.py`** — works out which raw table answers which question, so the B
and C layouts only had to be built for the ~498 tables that mattered.
**Keep as history** — and note it holds the **last surviving copy of the old labelling method**
(superseded by `GOLD-1`). Deleting it removes that approach from the project entirely. Its output
`data/day6_gold_tables.json` is still read by `eval.py` and `corpus/build_table_variants.py`.

**`arm4_rerank_prepare.py`** / **`arm4_rerank_hpc.py`** / **`arm4_rerank_score.py`** —
Arm 4's three stages. **Keep** — the A results file is still the live baseline that two current
scripts read, and the A payload was reused to build the first slice payload (`COST-14`).
Note: `arm4_rerank_hpc.py` differs from its Arm 3 twin by **three lines of actual code.** Two
110-line files for that.
**`arm4_rerank_score.py` is the odd one out: it computes a current number, not a historical one.**
Its variant-A output is the live baseline (0.609/0.687) and the `unfiltered_raw` control every
Day 8 gain is measured against. It is also still the only script that can score variants B and C.

---

## `data/` — 4.8 GB, 73 entries

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

**`day8_cost13_payload.json`** (6.5 MB) — **Delete or archive.** Rebuildable, but it's the only
record of exactly what was sent to the model.

**`day8_cost13_responses.jsonl`** (144 KB) — **Keep. You paid for this.** It's also what makes
re-running free instead of paid.

**`day8_slice_budget_dev_results.json`** (1.9 KB) and **`day8_cost13_dev_results.json`** (17 KB) —
**Keep.** The survival sweep and the paired answer result, as files rather than as terminal output
(`INFRA-10`). The second one carries the per-question verdicts, which the printed table cannot be
recovered from. Both regenerate for free from the score files above.

### Leftover cluster inputs and other

**`day5_embed_payload.json`** (**103 MB**) — **Delete.** Rebuilt in minutes.

**`embed_results.jsonl`** (**582 MB**) — **Decide.** Already loaded into the database. It covers
only **196 of 799 filings**, so it is *not* a usable backup. If the database is backed up, delete
it. If it isn't, the right fix is to back up the database, not to keep this.

**`company_lexicon.json`** (10 KB) — **Keep — actively read** on the search path.
Careful: it's committed to git *and* rebuilds itself only when missing. A stale committed copy will
silently win over a corpus change. If the corpus grows, delete this file to force a rebuild.

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
`day8_cost13_responses.jsonl` and `day6_table_summaries.json` (paid for),
and the `.bak_old_labeler` files (deliberately kept).

---

## The three things worth fixing first

1. **The ground truth can't be rebuilt.** `data/day7_gold_inds_matched_full.json` has no producer,
   and the script that reads it points at a folder that doesn't exist — so an accidental re-run
   would overwrite good data with almost nothing. Either save the matching as a real script, or
   write down plainly that this file is frozen and back it up.

2. **Your best number isn't saved anywhere.** `rerank_score.py` prints and exits. About ten
   lines would fix that.

3. ~~**The naming is the reason this folder feels overwhelming.**~~ **Done.** Every script used to
   be called `dayN_something`, which recorded *when* it was written, not *what it does* — and it was
   already misleading, since `arm4_rerank_score.py` carried a Day 6 prefix while computing a current
   baseline. `scripts/` is now organised by job (`corpus`, `index`, `retrieval`,
   `compression`, `eval`, `checks`, `analysis`, `archive`) with every file named for what it does; the
   day tags stay in `DECISIONS.md`, where they belong.
