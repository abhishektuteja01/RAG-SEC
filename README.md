# RAG-SEC

Question answering over 799 SEC 10-K filings. A question goes through hybrid retrieval and
reranking, Gemini answers from the top chunks, and the API returns those chunks as sources.

Retrieval, in order:

1. **Company filter** -- if the question names a company, search only that company's filings.
2. **Dense search** (`BAAI/bge-m3` embeddings, pgvector) and **BM25** (pg_search), 200 rows
   each. The dense leg embeds the question with the company name and filing wording removed.
3. **RRF fusion** of the two lists, plus a small bonus for filings from the year the question
   names, plus a third list of chunks whose text names that year. The top 50 survive.
4. **Rerank** the 50 with `BAAI/bge-reranker-v2-m3`, against the question with the company
   name removed. The top 10 go to the answer model (`gemini-3.7-flash`).

On the held-out test split (1,545 questions), **recall@10 is 0.831** and recall@50 is 0.915.

This repo is the slim, production version. The full research history (every retrieval
technique tried, measured one change at a time, including the ones that lost) is frozen
at git tag [`v1-research`](https://github.com/abhishektuteja01/RAG-SEC/tree/v1-research).

## Layout

```
src/rag_sec/   retrieve.py (the pipeline), candidates.py (dense + BM25 + RRF),
               company.py, fiscal_year.py, store.py (Postgres), answer.py (Gemini),
               api.py + static/index.html (HTTP API and chat page), eval.py (gold labels + metrics)
scripts/       evaluate.py, setup_db.sh, rebuild/ (optional, slow corpus rebuild)
deploy/        Docker images, GPU-host compose file, GUIDE.md
data/          the few files serving and evaluation need (alias table, wordlist, gold labels,
               stored rerank scores)
```

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync
cp .env.example .env                  # set POSTGRES_PASSWORD and GOOGLE_API_KEY
docker compose up -d                  # Postgres with pgvector + pg_search
scripts/setup_db.sh                   # download and restore the prebuilt database
uv run --env-file .env uvicorn rag_sec.api:app --workers 1
```

Open http://localhost:8000 for the chat page, or call the API:

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "What was Visa Inc.'\''s net revenue in 2015?"}'
```

The first start loads both models (~20 s). Keep `--workers 1`: more workers load more copies
of the models, and concurrent model loading crashes on Apple MPS. To deploy on a GPU host,
see [deploy/GUIDE.md](deploy/GUIDE.md).

## Evaluate

```bash
# Re-score the stored rerank output: no GPU, no database, no API key (~10 s)
uv run scripts/evaluate.py --replay            # test recall@10 0.831 (n=1545)
uv run scripts/evaluate.py --replay --split dev   # dev 0.847

# Run the real retrieval for each question (needs the database and the models)
uv run --env-file .env scripts/evaluate.py --live --split dev --limit 50
```

The first run downloads the question set (T2-RAGBench) from Hugging Face. Gold labels come
from `data/gold_chunk_ids.json`, or are computed from `data/chunks/` if you rebuilt the corpus.

## Rebuild from scratch (optional)

Only needed without the database dump. Slow: about 1 hour to fetch and chunk the filings,
and 9-10 hours to embed them on an Apple M3.

```bash
uv run --env-file .env --extra rebuild scripts/rebuild/corpus.py     # needs EDGAR_CONTACT_EMAIL
uv run --env-file .env --extra rebuild scripts/rebuild/index.py local
uv run --env-file .env --extra rebuild scripts/rebuild/index.py bm25
uv run scripts/rebuild/gold_cache.py check     # the gold labels still match the chunks
```
