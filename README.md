# RAG-SEC

Retrieval-augmented QA over 799 SEC 10-K filings. It measures, one technique at a time, what
dense embeddings, hybrid BM25 fusion and cross-encoder reranking each actually buy.

## The result

Held-out **test** split, best arm — dense + BM25/RRF + `bge-reranker-v2-m3`, coverage-based
gold labels, n=1545 (`DECISIONS.md` `RETR-39`):

| Cell | recall@10 |
|---|---|
| reranked baseline | 0.607 ± 0.012 |
| + company filter | 0.644 ± 0.011 |
| + query strip | 0.632 ± 0.012 |
| **+ both** | **0.747 ± 0.010** |

**Together they are worth 2.3x their separate gains.** Once every candidate is already the
right company, the company name left in the query only rewards whichever chunk repeats the
most boilerplate. The interaction is the finding, not either piece.

![Retrieval quality by pipeline stage](images/results_chart.png)

Regenerate it with `uv run --with matplotlib scripts/archive/results_chart.py` (~3-4 min).
Every number is recomputed from `data/` at generation time and cross-checked against
`DECISIONS.md`'s baseline table; a mismatch aborts rather than shipping a stale chart.

## Reproduce it in 90 seconds

The rerank scores are committed, so the headline replays with no GPU, no Postgres, no API key
and no money:

```bash
uv run scripts/pipeline/05_arm3_rerank.py score \
    --scores data/retr7_rr_test_scores.jsonl --split test
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
| 3 | + `bge-reranker-v2-m3` | 0.629 | **shipped**; 0.760 with company filter + query strip |
| 4 | table layouts B and C | — | lost — whole-table A wins on every metric (`ARM4-10`) |
| 5 | multi-vector late interaction | — | never built: ~378 GB of vectors (`ARM5-1`) |
| 6 | agentic LangGraph loop | — | wins on answer accuracy, loses on retrieval (`AGENT-19`) |

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
both containers on one Graviton3 `c7g.2xlarge` EC2 host, Postgres self-hosted from
`Dockerfile.postgres` because RDS cannot load `pg_search` (`DEPLOY-2`). It answers correctly and
is not interactive — 157.7 s per question, 99.5% of it the reranker (`DEPLOY-18`). Arm 6's
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
