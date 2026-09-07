# RAG-SEC

Retrieval-augmented QA over SEC 10-K filings. A learning project measuring, arm
by arm, how much each retrieval technique (dense embeddings, hybrid BM25 fusion,
cross-encoder reranking) actually moves accuracy — not assumed, measured.

## Result

Best arm, on the **held-out test split** — dense + BM25/RRF + `bge-reranker-v2-m3`,
scored under the coverage-based gold labels (`DECISIONS.md` RETR-35):

| Cell | recall@10 |
|---|---|
| reranked baseline | 0.607 |
| + company filter | 0.644 |
| + query strip | 0.632 |
| **+ both** | **0.747** |

The two together are worth 2.3x their sum: once every candidate is the right company, the
company name in the query only rewards whichever chunk repeats the most boilerplate.

Every arm is scored on the same 799-filing corpus, same split, same labels. Full
methodology and every design decision (why this chunk size, why this fusion method, what
broke and how) is logged in [`DECISIONS.md`](DECISIONS.md).

![Retrieval quality by pipeline stage](images/results_chart.png)

⚠️ The chart above predates RETR-35's label correction and its numbers are **not**
comparable to the table — recall is uniformly higher under the corrected labels. It has no
producer script, so it cannot be regenerated; the table is the number to quote.

## Replaying the result without a GPU

The rerank scores behind the table are committed, so the headline replays in ~90 s with no
GPU, no Postgres, no API key and no money:

```bash
uv run scripts/pipeline/05_arm3_rerank.py score \
    --scores data/retr7_rr_test_scores.jsonl --split test
```

It needs the corpus on disk first (`eval.py` reads `data/chunks/` to resolve gold labels), so
a fresh clone runs `01_corpus.py` once — ~1 h, no GPU. That is the short path: it skips the
~9.3 h embed and Postgres entirely. Only Arms 1-2 need those, because they query the index
live. See [`scripts/README.md`](scripts/README.md).

## Pipeline

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
      │  store.py                              answer_eval.py, agent.py  (LangGraph, parked)
      ▼
Postgres (pgvector + pg_search)  ◄── read by retrieve.py

Every module above is under src/rag_sec/.
```

Eval set: [T²-RAGBench](https://huggingface.co/datasets/G4KMU/t2-ragbench)
(FinQA + ConvFinQA subsets, 799 unique filings).

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Docker.

```bash
cp .env.example .env        # fill in POSTGRES_PASSWORD and EDGAR_CONTACT_EMAIL
docker compose up -d        # Postgres 18 + pgvector + pg_search (BM25)
uv sync
```

## Regenerating the data

The corpus (filings, parsed sections, chunks, embeddings, rerank scores) isn't
committed — it's ~4GB and fully reproducible from EDGAR. Rebuild it in order:

Each numbered phase in `scripts/pipeline/` is one stage, in order.
[`scripts/README.md`](scripts/README.md) is the full run order — every phase, its real run
dates, what it costs, and what a fresh clone can and cannot rebuild.

```bash
uv run scripts/pipeline/01_corpus.py           # fetch all 799 filings from EDGAR, parse, chunk (~1h)
uv run scripts/pipeline/03_index.py local      # embed + load into Postgres
uv run scripts/pipeline/03_index.py bm25       # build BM25 index (pg_search)
```

`01_corpus.py` skips whatever is already on disk, so it resumes after an interruption
and re-running it is a no-op. After changing `chunking.py` or `parsing.py`, rebuild without
re-downloading:

```bash
uv run scripts/pipeline/01_corpus.py --rechunk   # re-chunk from data/parsed/ (~4 min)
uv run scripts/pipeline/01_corpus.py --reparse   # re-parse from data/filings/ (~30 min)
```

Reranking (Arm 3) is the expensive stage. Phase 05 is a split job for that
(`05_arm3_rerank.py prepare` → run `scripts/pipeline/hpc/rerank_hpc.py` on a GPU box →
`05_arm3_rerank.py score`); adapt the GPU step to whatever cluster or cloud GPU you have.
Every published rerank number came from that route (`DECISIONS.md` ARM3-2). It also runs
locally without a GPU — `05_arm3_rerank.py local` — at roughly 32.6s per question per cell
on an M3, so a full split is tens of hours. See [`scripts/README.md`](scripts/README.md).
Corpus-growth embeddings follow the same pattern in phase 03
(`03_index.py new-filings --prepare` → `scripts/pipeline/hpc/embed_hpc.py` →
`03_index.py new-filings --load`).

## Running the eval

```bash
uv run scripts/pipeline/04_arms_first_stage.py arm1        # dense only
uv run scripts/pipeline/04_arms_first_stage.py arm2        # + BM25 / RRF fusion
uv run scripts/pipeline/05_arm3_rerank.py local --n 20     # + reranker, single machine (slow on CPU; use --n)
```

The recorded Arm 3 baseline was produced by phase 05's split job instead
(`05_arm3_rerank.py prepare` → `scripts/pipeline/hpc/rerank_hpc.py` on a GPU box →
`05_arm3_rerank.py score`) — see `DECISIONS.md` ARM3-2 for why a full CPU run isn't
viable. Arm 4's table-layout variants are phase 06 and the agent loop is phase 07.

## Docs

- [`DECISIONS.md`](DECISIONS.md) — every design choice, why kept or dropped
- [`scripts/README.md`](scripts/README.md) — the pipeline in run order, with dates and costs
- [`INVENTORY.md`](INVENTORY.md) — file-by-file map: what each file is and whether it's still needed
- [`RUNBOOK.md`](RUNBOOK.md) — how to run a GPU job on the cluster, start to finish
- [`CLAUDE.md`](CLAUDE.md) — orientation if you're new: what an arm is, the traps, the glossary

Open the repo in Claude Code and `/walkthrough` gives you a guided tour; `/explain-arm 3`
walks through any single arm.

Four files are local and gitignored, so text elsewhere in the repo cites them without
linking: `spec.md` (scope, sequence, eval methodology), `SESSION.md` (what's next),
`CLAUDE.local.md` (the owner's working rules), `DECISIONS.local.md` (pruned housekeeping rows).
