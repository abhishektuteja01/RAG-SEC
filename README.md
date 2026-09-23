# RAG-SEC

Retrieval-augmented QA over 799 SEC 10-K filings. It measures, one technique at a time, what
dense embeddings, hybrid BM25 fusion and cross-encoder reranking each actually buy.

## The result

Held-out **test** split, cumulative — dense + BM25/RRF + `bge-reranker-v2-m3`, coverage-based
gold labels, n=1545:

| Cell | recall@10 | recall@50 |
|---|---|---|
| reranked baseline | 0.607 ± 0.012 | — |
| + company filter | 0.644 ± 0.011 | — |
| + query strip | 0.632 ± 0.012 | — |
| **+ both** (`RETR-39`) | **0.747 ± 0.010** | 0.785 |
| **+ year-proximity RRF nudge** (`RETR-40`) | **0.771 ± 0.010** | 0.825 |
| **+ dense leg embeds the stripped query** (`RETR-51`) | **0.814 ± 0.009** | 0.885 |
| **+ chunk-year candidate list at read depth 200** (`RETR-52`, **serving**) | **0.831 ± 0.008** | 0.915 |

`/ask` now answers at the **0.831 configuration** — the last two rows went from measured to
serving in `DEPLOY-25`, once they were timed on the deploy host and turned out to be free:
paired over 110 test questions in the deployed container, all three flags on cost retrieval a
mean **-0.011s, 95% CI [-0.051, +0.031]**. The recall figure itself is the offline GPU rerank
pass's, not a re-score through the HTTP path; the two share one candidate generator
(`candidates.first_stage`, `RETR-53`) and the flags were verified identical in the running
container, which is why the number transfers. They are query-side and first-stage only — the reranker still scores 50
candidates, so they cost nothing per answer. Every paired confidence interval excludes zero,
McNemar p<0.0001, and no subgroup regresses (`RETR-51`/`RETR-52`).

Individual questions can still regress even though no subgroup does, and `DEPLOY-25` documents
a measured case: a gold chunk whose own text never restates the year the question asks for is
excluded from `RETR-52`'s year list and can fall out of the candidate pool.

Company filter + query strip **together are worth 2.3x their separate gains.** Once every candidate is already the
right company, the company name left in the query only rewards whichever chunk repeats the
most boilerplate. The interaction is the finding, not either piece.

The two newest gains are **reachability, not ranking**: recall@50 moves +0.090 against
recall@10's +0.059, so most of the win is gold reaching the candidate pool at all, and the
reranker converts only part of it.

![Retrieval quality by pipeline stage](images/results_chart.png)

The chart covers Arms 1-3 and stops at the published ablation, so it does not yet show the
last two rows above. Regenerate it with
`uv run --with matplotlib scripts/archive/results_chart.py` (~3-4 min). Every number is
recomputed from `data/` at generation time and cross-checked against `DECISIONS.md`'s
baseline table; a mismatch aborts rather than shipping a stale chart.

## Reproduce it in 90 seconds

The rerank scores are committed, so both the published ablation and the newest arms replay
with no GPU, no Postgres, no API key and no money:

```bash
# the 2x2 ablation -- filter+strip, 0.747
uv run scripts/pipeline/05_arm3_rerank.py score \
    --scores data/retr7_rr_test_scores.jsonl --split test

# the two first-stage additions -- 0.771 -> 0.814 -> 0.831, with paired CIs and subgroups
uv run scripts/pipeline/05_arm3_rerank.py score --table arms \
    --scores data/arms_scores.jsonl --split test
```

One prerequisite: gold labels resolve against `data/chunks/`, so a fresh clone first runs
`uv run scripts/pipeline/01_corpus.py` — ~1 h, resumable, no GPU.

## How it works

Each arm adds exactly one technique to the one above, on the same corpus, split and labels, so
every delta is attributable. recall@10 is dev, from `RETR-39`:

| Arm | adds | recall@10 | verdict |
|---|---|---|---|
| 1 | dense BGE-M3 vectors | 0.337 | the baseline everything else is measured against |
| 2 | + BM25/RRF fusion | 0.514 | kept — the largest single gain |
| 3 | + `bge-reranker-v2-m3` | 0.629 | **shipped**; 0.791 with company filter + query strip + a year-proximity RRF nudge (`RETR-40`), and 0.847 measured with the two first-stage additions (`RETR-51`/`RETR-52`), serving defaults since `DEPLOY-25` |
| 4 | table layouts B and C | — | lost — whole-table A wins on every metric (`ARM4-10`) |
| 5 | multi-vector late interaction | — | never built: ~378 GB of vectors (`ARM5-1`) |
| 6 | agentic LangGraph loop | — | closed as a negative (`AGENT-34`): its answer-accuracy win does not survive a fair baseline (`AGENT-30`), and its retrieval deficit is not significant (`AGENT-33`) |

`/explain-arm 3` in Claude Code walks through any one of them.

## What's in the box

```
INGEST                                  QUERY
EDGAR filings (.htm)                    question
      │  edgar.py                             │  company.py   (company filter + query strip)
      ▼                                       ▼
parsed sections (.json)                 retrieve.py   (dense + BM25/RRF + bge-reranker-v2-m3)
      │  parsing.py  (sec-parser)             │
      ▼                                       ├──►  eval.py        recall@k, nDCG@k, MRR
chunks (.json)                                │
      │  chunking.py                          └──►  compress.py    (DSLR slice packing)
      ▼                                                  │
embeddings + BM25 index                                  ▼
      │  store.py                              answer_eval.py, agent.py  (Arm 6's loop)
      ▼
Postgres (pgvector + pg_search)  ◄── read by retrieve.py
```

Every module is under `src/rag_sec/`. `api.py` serves Arm 3 as `POST /ask` and is **deployed**:
both containers on one `g4dn.xlarge` (Tesla T4) EC2 host, Postgres self-hosted from
`Dockerfile.postgres` because RDS cannot load `pg_search` (`DEPLOY-2`). It answers correctly, and
the GPU cutover cut rerank from the CPU host's 157.7 s to **fp16 rerank mean 3.27 s**, with exact
top-5 parity on every test question (`DEPLOY-21`, `DEPLOY-22`).

That is the rerank stage, not the round trip. Measured end to end on the fixed ten questions
every latency decision here uses, `POST /ask` is **p50 6.7 s** — about 3.4 s of retrieval and
3 s of Gemini generation (`DEPLOY-25`). So it is usable but **not yet inside the 2-5 s
interactive target**, and the remaining headroom is generation-side, not retrieval-side. Arm 6's
traces and dashboards are in `images/langfuse_*.png`.

Eval set: [T²-RAGBench](https://huggingface.co/datasets/G4KMU/t2-ragbench) (FinQA + ConvFinQA,
799 unique filings).

## Build it yourself

The corpus (~4 GB) isn't committed; it rebuilds from EDGAR. Setup needs
[`uv`](https://docs.astral.sh/uv/) and Docker:

```bash
cp .env.example .env        # POSTGRES_PASSWORD, EDGAR_CONTACT_EMAIL
docker compose up -d        # Postgres 18 + pgvector + pg_search (BM25)
uv sync
```

[`scripts/README.md`](scripts/README.md) is the run order: every phase, what it produces, what
it costs, and what a fresh clone can and cannot rebuild.

## Docs

| File | Answers |
|---|---|
| [`DECISIONS.md`](DECISIONS.md) | why every choice was made — the source of truth for any number |
| [`scripts/README.md`](scripts/README.md) | run order, real dates, what each command costs |
| [`INVENTORY.md`](INVENTORY.md) | file-by-file map, and what's still missing |
| [`CLAUDE.md`](CLAUDE.md) | orientation: what an arm is, the glossary, the traps |

`/walkthrough` in Claude Code gives a guided tour of the repo.
