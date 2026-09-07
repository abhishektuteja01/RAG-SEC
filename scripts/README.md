# `scripts/` — how the project was actually run

Three folders, three jobs. **`pipeline/`** is the reproduction path: seven numbered phases,
in order, plus the two scripts that run on a GPU cluster. **`checks/`** is twelve guards —
each one locks a property that a real bug broke; five fail CI and one fails the container
build. **`archive/`** is thirty one-off measurements whose findings are already recorded in
[`DECISIONS.md`](../DECISIONS.md), plus the producer for the README's chart; nothing in the
pipeline imports them.

For the design reasoning behind any choice below, read the phase file's own docstring — it
carries the full "produces / reads / traps" record. This file is only the order and the map.

---

## Read this before any date below

**"Day N" is a unit of planned work, not a calendar date.** A `dayN_` filename says nothing
about when the file was written. The project ran **2026-08-24 → 2026-09-06**:

| Planned | Real calendar |
|---|---|
| Days 1–7 | Aug 24 – Aug 30 |
| Day 8 (alone) | Aug 31 – Sep 4 — six days |
| Days 9–14 | Sep 4 – Sep 6, and **Day 10 was executed before Day 9** |

There is no calendar "Day 7": every `day6_*` and `day7_*` artifact was written by Aug 30.
(`data/day6_gold_tables.json` has mtime 2026-08-29 and the two `day7_*` files 2026-08-30,
though all three landed in the same 08-30 commit.) Where a filename's `dayN_` prefix and its
real date disagree, both are given. Phase numbers are also not arm numbers: phase 04 is Arms
1–2, 05 is Arm 3, 06 is Arm 4, 07 is Arm 6. There is no Arm 5 code — it was measured on paper
and dropped before building (`ARM5-1`).

**Phase 02 cannot run end to end, and that is the most important fact here.** The `evidence`
leg's input `data/day7_gold_inds_matched_full.json` has **no producer in this repo** and the
raw datasets it would need (`data/raw`, `data/FinQA`, `data/ConvFinQA`, `data/TAT-DQA`) are
all absent (`INFRA-15`, `GOLD-7`). Every metric in the project is downstream of that one
4.8 MB file. It and its output are tracked in git, and that is the only backup — so:

- A **fresh clone can reproduce every number**, because the labels are committed.
- A fresh clone **cannot rebuild the labels**. Don't write a new producer; re-derived labels
  are not the labels the published numbers were measured against (`RETR-35`).

---

## Run order

Commands are copy-pasteable and were each confirmed against the script's own `--help`.

| # | Phase | Produces | Command | Really ran | DECISIONS |
|---|---|---|---|---|---|
| 01 | Corpus: EDGAR → parse → chunk | `data/filings/`, `data/parsed/`, `data/chunks/`, `data/ingest_log.jsonl` | `uv run scripts/pipeline/01_corpus.py` | 08-24→26 first pass; 08-27→28 grown to 799; 09-04 `--rechunk` under `RETR-7`/`8` | `DATA-6`, `INFRA-8`, `ARM4-2`, `RETR-7`, `RETR-8` |
| 02 | Gold labels — **half-runnable** | `data/day6_gold_tables.json` (runs) · `data/day7_gold_evidence_resolved.json` (**cannot run**) | `uv run scripts/pipeline/02_gold_labels.py tables`<br>`uv run scripts/pipeline/02_gold_labels.py evidence` | tables 08-29 00:44; frozen input 08-30 17:31; evidence 08-30 18:53 | `INFRA-15`, `GOLD-1`, `GOLD-7`, `INFRA-9`, `ARM4-3`, `DATA-7`–`DATA-9` |
| 03 | Index: embed into pgvector + build BM25 | Postgres `chunks`, HNSW index, `pg_search` BM25 index | `uv run scripts/pipeline/03_index.py local`<br>`uv run scripts/pipeline/03_index.py bm25` | on/before 08-27 first local embed (**exact date not established**; bounded by Arm 1 on 08-27); 08-27→28 corpus growth; 09-04 re-index | `ARM3-2`, `INFRA-4`, `INFRA-6`, `INFRA-12`, `INFRA-13`, `ARM2-1`, `ARM4-3`, `RETR-7`, `RETR-8` |
| 04 | Arms 1 & 2: dense, then hybrid BM25/RRF | `data/day3_arm1_dev_*`, `data/day4_arm2_dev_*` | `uv run scripts/pipeline/04_arms_first_stage.py arm1`<br>`uv run scripts/pipeline/04_arms_first_stage.py arm2` | 08-27 Arm 1; 08-28→29 Arm 2; 09-01 `--company-filter` sidecars; **09-04 both re-run and overwritten** after the re-index | `ARM1-2`, `RETR-5`, `INFRA-4`, `ARM2-1`, `RETR-35`, `RETR-36`, `RETR-7`, `RETR-8` |
| 05 | **Arm 3 — the headline arm.** Rerank + company filter + query strip | `data/retr7_rr_{dev,test}_scores.jsonl`, `data/retr7_arm3_{dev,test}_results.json` | see the three legs below | 08-28 first HPC pass; 08-30 rescored; **09-04 the published `retr7_*` dev+test passes** | `ARM3-1`, `ARM3-2`, `RETR-5`, `RETR-6`, `RETR-16`, `RETR-18`, `RETR-24`, `RETR-30`, `RETR-39`, `AGENT-16`, `AGENT-24` |
| 06 | Arm 4: A/B/C table layouts — **a dead end**, kept because `score --variant A` is a live control | `data/day6_arm4_{A,B,C}_dev_results.json`, `data/day6_table_summaries.json` | `uv run scripts/pipeline/06_arm4_tables.py score --variant A --overwrite` | 08-29 gold tables + C summaries; **08-30 the whole A/B/C run** (results 19:09–19:12). Not 09-04 — the re-index touched variant A only | `ARM4-2`…`ARM4-10`, `GOLD-5`, `INFRA-17`, `RETR-22`, `RETR-29`, `RETR-31` |
| 07 | Arm 6: LangGraph loop vs. the static shipped arm | `data/day9_arm6_dev_results.jsonl` | `uv run scripts/pipeline/07_arm6_loop.py analyze` (free)<br>`uv run scripts/pipeline/07_arm6_loop.py run --allow-paid-run -n 200` (**paid**) | started 2026-09-04 20:10, finished 2026-09-05 01:55 — 200/200, 0 errors, 339.6 min, serial, on mains power | `AGENT-1`, `AGENT-4`, `AGENT-5`, `AGENT-8`, `AGENT-10`, `AGENT-15`…`AGENT-17`, `AGENT-19`, `AGENT-21`, `AGENT-22`, `AGENT-24`, `AGENT-25`, `COST-21`, `COST-30`, `COST-36`, `OBS-10`, `OBS-13` |

### Phase 03 — the cluster route

The GPU stage needs no database, which is why it is split out (`ARM3-2`, `INFRA-6`). The
mechanics — hosts, `module load`, tmux, transfers — are specific to one university cluster and
are kept out of the repo; the commands below are the part that transfers.

```bash
# new filings, not yet in Postgres
uv run scripts/pipeline/03_index.py new-filings --prepare data/embed_payload.json
#   on the GPU node, from ~ :  python -u embed_hpc.py embed_payload.json embed_results.jsonl
uv run scripts/pipeline/03_index.py new-filings --load data/embed_results.jsonl

# chunks already in Postgres whose TEXT changed (RETR-7/RETR-8) — the only leg that can
uv run scripts/pipeline/03_index.py changed-chunks --prepare data/retr7_embed_payload.json
uv run scripts/pipeline/03_index.py changed-chunks --load data/retr7_embed_results.jsonl \
    --baseline-chunks data/chunks_pre_retr7
```

`changed-chunks` diffs `data/chunks/` against a **backup** of the pre-fix chunks, never
against the database, and refuses if any filing's chunk count moved. `local` and
`new-filings` are INSERT-only and skip any filing already present — so a re-chunk is
invisible to them.

### Phase 05 — the three legs

```bash
# 1. laptop: candidate pools -> one self-contained payload (all four 2x2 cells, both splits)
uv run scripts/pipeline/05_arm3_rerank.py prepare --splits dev,test --out data/rr_payload.json

# 2a. GPU node, from ~ (the basename is part of the contract; the sbatch calls it bare)
#     python -u rerank_hpc.py rr_payload.json retr7_rr_dev_scores.jsonl
#     or:  sbatch scripts/pipeline/hpc/rerank_hpc.sbatch

# 2b. OR no cluster at all — same model, same pool, identical scores format
uv run scripts/pipeline/05_arm3_rerank.py local --n 20 --split dev \
    --out data/local_rr_dev_scores.jsonl

# 3. laptop: replay a scores file into the published 2x2. Reads only — no GPU, no DB, no spend
uv run scripts/pipeline/05_arm3_rerank.py score --scores data/retr7_rr_dev_scores.jsonl \
    --split dev --out data/retr7_arm3_dev_results.json
```

**Always pass `--scores` and `--out` explicitly.** `prepare --out` and `score --scores` still
default to stale `day8_retr16v2_*` names, and `rerank_hpc.sbatch` hardcodes `day8_retr18_test_*`.
The names were left alone so the merged phase is behaviour-preserving; the live artifacts are
`data/retr7_rr_{dev,test}_scores.jsonl`.

**A scores file is not stored in rank order.** Cells are written in first-stage RRF order with
the rerank score merely attached. Reading one raw scored recall@10 **0.552 against 0.739**
(`AGENT-16`). Read it only through `rag_sec.eval.load_ranking`, which sorts on load.

---

## Start here: replaying the headline number

**The cheapest path is not below.** The rerank scores are committed, so scoring the published
Arm 3 result needs no GPU, no Postgres and no money -- only the corpus, for gold labels:

```bash
uv run scripts/pipeline/01_corpus.py    # ~1 h, resumable, skips existing
uv run scripts/pipeline/05_arm3_rerank.py score \
    --scores data/retr7_rr_test_scores.jsonl --split test    # ~90 s
```

That prints the `RETR-39` table: `filtered_stripped` recall@10 0.747 on 1545/1546 test
questions. Everything below is only needed to rebuild the index itself, which Arms 1-2
query live.

## Rebuilding the index: Arms 1-2 with no cluster

Do the setup in the [main README](../README.md) first (`.env`, `docker compose up -d`,
`uv sync`). Then:

```bash
uv run scripts/pipeline/01_corpus.py                          # ~1 h, resumable, skips existing
uv run scripts/pipeline/03_index.py local                     # ~1.4 chunks/s on M3 MPS -> ~9.3 h
uv run scripts/pipeline/03_index.py bm25                      # minutes, no GPU
uv run scripts/pipeline/04_arms_first_stage.py arm1           # dense baseline
uv run scripts/pipeline/04_arms_first_stage.py arm2           # + BM25/RRF
uv run scripts/pipeline/05_arm3_rerank.py local --n 20 --split dev \
    --out data/local_rr_dev_scores.jsonl                      # smoke test, ~45 min
uv run scripts/pipeline/05_arm3_rerank.py score --scores data/local_rr_dev_scores.jsonl \
    --split dev --out data/local_arm3_dev_results.json
```

Phase 02 is skipped on purpose: the labels it would build are already in git and frozen
(`INFRA-15`).

**The `local` leg is not the cluster route at a discount.** Rerank is **~32.6 s per question
per cell** on an M3 (`AGENT-24`) and **~156.9 s** on the Graviton3 deploy host. The full 2×2
over the 1,235-question dev split is therefore ≈**45 hours on the M3** and ≈**9 days on the
deploy host**. It is resumable per question id, and it has never produced a published number —
every figure in `DECISIONS.md` came from the GPU legs. Use `--n` unless you mean it.

---

## What costs money or GPU time

| Command | Spend | Gate |
|---|---|---|
| `07_arm6_loop.py run` | **~$4.10 per 200 questions** (~$0.0205/question), live Gemini | Refuses to start without `--allow-paid-run`. There is no cheaper dry run — re-run `analyze` on the file on disk instead |
| `06_arm4_tables.py variants` | 498 `claude-haiku-4-5` calls (one per gold table). **All 498 are cached, so a normal run spends $0** | An uncached summary is a hard error unless `--allow-paid-summaries`. Don't pass it — a lost cache would silently re-spend |
| `hpc/rerank_hpc.py` (test 2×2) | GPU-hours: 309,160 pairs, ~7.3 GPU-h estimated | SLURM allocation (`RETR-18`) |
| `hpc/embed_hpc.py` (re-index) | GPU-hours: 47,312 chunks in ~72 min at ~11 chunks/s on a V100 | SLURM allocation (`INFRA-12`, `INFRA-13`) |
| `05_arm3_rerank.py local` | No money — your machine's hours instead (see above) | none |
| `checks/agent_loop_smoke_test.py` | **~$0.04 per run**, live Gemini, despite the name | none, and no dry-run flag |

Everything else — every `score`, every `analyze`, all of `checks/` bar that one — reads only.

Two standing operating rules for phase 07, both learned the hard way: **run serial**
(`--concurrency` above 1 is refused; concurrent MPS model construction segfaults with no
traceback and zero rows written — `AGENT-10`, `AGENT-17`) and **run on mains power** (battery
throttling moves the per-stage p50/p95 latencies this project publishes — `AGENT-20`,
`AGENT-24`).

---

## `checks/` — the guards

Each of these exists because the thing it checks actually broke. They are runnable scripts,
not pytest, to match the rest of `scripts/`.

| Guard | Locks | Wired to |
|---|---|---|
| `variant_predicates.py` | every read of `chunks` constrains `variant` (`RETR-24` needed the same fix seven times) | **CI**, and `get_conn()` preflight |
| `candidate_sql.py` | the SQL and parameter order `rag_sec.candidates` emits, transcribed from the seven pre-`RETR-36` copies | **CI** |
| `static_ranking_order.py` | static rankings are score-descending at point of use (`AGENT-16`: 0/1235 cells were) | **CI** |
| `tracing_offline.py` | `rag_sec.tracing` no-ops with no `LANGFUSE_*` keys — runs keyless on purpose | **CI** |
| `retrieval_gate.py` | replays `data/ci_retrieval_fixture.jsonl` with `eval.py`'s own metric functions; fails if recall@10 or nDCG@10 falls more than 2.0 points below `data/ci_retrieval_baseline.json` (0.7854 / 0.6593 on 400 dev questions), or if the fixture's question count no longer matches the baseline's n=400 | **CI** |
| `container_wordlist.py` | `RETR-11`'s English-word guard is on and reading the right wordlist — silent failure otherwise (`DEPLOY-5`) | **Dockerfile, at build *and* at startup** |
| `ci_fixture_build.py` | builds the committed fixture the gate scores. Run rarely, by hand | — |
| `heading_fix.py` | `RETR-7`/`RETR-8` acceptance: reversible with flags off, chunk boundaries preserved, bug actually fixed. Local-only — it reads gitignored `data/chunks/` | — |
| `company_resolver.py` | `rag_sec.company.resolve` edge cases + precision/recall on train (calibration) and dev (confirmation) | — |
| `atom_replay.py` | the packer replayed from `data/parsed/` reproduces `data/chunks/` byte-for-byte, and chunks have enough atoms for compression to have any purchase | — |
| `unanswerable_validate.py` | the unanswerable set really is unanswerable, for the 30 of 47 questions where that is machine-checkable | — |
| `agent_loop_smoke_test.py` | the agentic loop end to end. **Paid** — see the cost table | — |

The CI workflow is [`.github/workflows/ci.yml`](../.github/workflows/ci.yml): a `guards` job
(seconds; no corpus, no database, no network) then a `gate` job. Deliberately absent, with reasons: the
answer-accuracy leg ($23.62/push at standard rates — `COST-39`) and the p95 latency leg
(must be measured in the serving container, not replayed — `DEPLOY-1`).

---

## `archive/` — claim → code

One-off measurements. Each one's finding is already in `DECISIONS.md`; this table is only so
a reader can walk back from a number to the script that produced it.

| Script | Backs |
|---|---|
| `answer_ab_prepare.py` | `COST-13`, `COST-20`, `COST-23`, `COST-27` |
| `answer_ab_run.py` | `COST-13`, `COST-29`, `OBS-10` |
| `answer_ab_score.py` | `COST-13`, `COST-20`, `COST-31` |
| `answer_batch_run.py` | `COST-29` |
| `arm4_rerank_hpc.py` | `ARM3-2`, `ARM3-3`, `INFRA-6` — the record of Arm 4's three GPU passes; three lines off the Arm 3 twin, so it was not promoted to `pipeline/hpc/` |
| `build_langfuse_dashboard.py` | `AGENT-8`, `OBS-*` |
| `candidate_k_curve.py` | `INFRA-10`, `AGENT-16`, `DEPLOY-11`, `DEPLOY-13`, `DEPLOY-14` |
| `chunk_interpretability.py` | `COST-4` |
| `chunk_length_vs_recall.py` | `ARM1-3` |
| `company_filter_ab.py` | `DATA-4`, `RETR-18` |
| `failure_triage.py` | `RETR-16`, `RETR-24` |
| `gold_atom_floor.py` | `COST-3`, `COST-6` |
| `label_audit_prepare.py` | `GOLD-1`, `RETR-9`, `RETR-31`, `RETR-33`, `COST-20` |
| `label_audit_score.py` | `RETR-9` |
| `label_matcher_ab.py` | `RETR-34`, `RETR-35` |
| `latency_measure.py` | `RETR-24`, `AGENT-16` |
| `mps_leak_probe.py` | `AGENT-17` |
| `onnx_rerank_export.py` | `DEPLOY-6` |
| `onnx_rerank_parity.py` | `DEPLOY-6`, `DEPLOY-8`, `RETR-35`, `RETR-39`, `AGENT-16` |
| `ort_fp32_latency.py` | `DEPLOY-8`, `DEPLOY-11` |
| `pack_variants.py` | `COST-25`, `COST-26`, `RETR-24` |
| `rescore_labels.py` | `RETR-35` |
| `results_chart.py` | `RETR-39` — no finding of its own; renders that table's recall@10 column as `images/results_chart.png`. Needs `uv run --with matplotlib`; matplotlib is not a project dependency |
| `slice_budget_sweep.py` | `COST-6`, `COST-7`, `COST-11`, `COST-25`, `RETR-29` |
| `slice_floor.py` | `COST-7` |
| `slice_prepare.py` | `COST-5`, `COST-7`, `COST-8`, `COST-17`, `RETR-16`, `RETR-29` |
| `slice_rerank_hpc.py`, `slice_rerank_hpc.sbatch` | `COST-7`, `COST-11` |
| `stratum_b_channel.py` | `COST-23`, `COST-25`, `COST-26` |
| `unanswerable_run.py` | `COST-34`, `AGENT-10`, `AGENT-17` |
| `worst_failures.py` | `COST-21` — the standing rule that each arm keeps its 20 worst failures in a file |
