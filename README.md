# RAG-SEC

Retrieval-augmented QA over SEC 10-K filings. A learning project measuring, arm
by arm, how much each retrieval technique (dense embeddings, hybrid BM25 fusion,
cross-encoder reranking) actually moves accuracy — not assumed, measured.

![Retrieval quality by pipeline stage](images/results_chart.png)

Every arm is scored on the same 799-filing corpus, same dev split, same gold
labels. Full methodology and every design decision (why this chunk size, why
this fusion method, what broke and how) is logged in [`DECISIONS.md`](DECISIONS.md).

## Pipeline

```
EDGAR filings (.htm)
        │  src/rag_sec/edgar.py
        ▼
  parsed sections (.json)
        │  src/rag_sec/parsing.py   (sec-parser)
        ▼
     chunks (.json)
        │  src/rag_sec/chunking.py
        ▼
  embeddings + BM25 index          reranked top-k
        │  src/rag_sec/store.py         │  bge-reranker-v2-m3
        ▼                               ▼
        Postgres (pgvector + pg_search)
        │
        ▼
     src/rag_sec/eval.py  →  recall@k, nDCG@k, MRR
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

```bash
uv run scripts/ingest/day2_ingest.py          # fetch filings from EDGAR → data/filings/
uv run scripts/ingest/day4_ingest_next200.py  # fetch remaining filings for full 799 coverage
uv run scripts/ingest/day2_chunk.py           # parse + chunk → data/chunks/
uv run scripts/index/day3_index_chunks.py     # embed + load into Postgres
uv run scripts/index/day4_index_bm25.py       # build BM25 index (pg_search)
```

Reranking (Arm 3) needs a CUDA GPU — on a CPU it doesn't finish in reasonable
time (see `DECISIONS.md` ARM3-2). The split-job scripts in `scripts/arms/`
(`day5_prepare_rerank_payload.py` → run `day5_hpc_rerank.py` on a GPU box →
`day5_finalize_arm3.py`) exist for that; adapt the GPU step to whatever cluster or
cloud GPU you have. Corpus-growth embeddings follow the same split-job pattern in
`scripts/index/` (`day5_prepare_embed_payload.py` → `day5_hpc_embed.py` →
`day5_load_embeddings.py`).

## Running the eval

```bash
uv run scripts/arms/day3_run_arm1.py   # dense only
uv run scripts/arms/day4_run_arm2.py   # + BM25 / RRF fusion
uv run scripts/arms/day5_run_arm3.py   # + reranker (single-machine reference; slow on CPU, use -n for a quick check)
```

The recorded Arm 3 baseline was produced by the split-job pipeline instead
(`scripts/arms/day5_prepare_rerank_payload.py` → `day5_hpc_rerank.py` on a GPU box →
`day5_finalize_arm3.py`) — see `DECISIONS.md` ARM3-2 for why a full CPU run isn't
viable.

## Docs

- [`spec.md`](spec.md) — scope, sequence, eval methodology (source of truth)
- [`DECISIONS.md`](DECISIONS.md) — every design choice, why kept or dropped
- `CLAUDE.md` (local, gitignored) — working rules + current project state
