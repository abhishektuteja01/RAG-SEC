# RAG-SEC

## What this is

Question answering over 799 SEC 10-K filings (the yearly reports US public companies must
file). You ask a question, like "What was Visa Inc.'s net revenue in 2015?". The system finds
the pieces of the filings that hold the answer, and Gemini answers from them, citing each
piece as a source.

## How it works

The filings are cut into about 100,000 pieces of roughly 900 words, called chunks. For each
question:

1. **Company filter and query cleanup.** If the question names a company, only that
   company's filings are searched. The company name and phrases like "as reported in the
   10-K" are removed from the question, since every candidate is from that company anyway.
2. **Two searches, merged.** Meaning search (bge-m3) turns text into a list of numbers, so
   chunks with similar meaning sit close together. Keyword search (BM25) scores chunks by
   the question's words, favouring rare words. RRF (reciprocal rank fusion) merges the two
   ranked lists by rank position. Chunks from the year the question names get a small boost.
   The top 50 go on.
3. **Rerank.** A reranker (bge-reranker-v2-m3) is a slower, more careful model. It reads
   the question and each of the 50 chunks together, and keeps the best 10.
4. **Answer.** Gemini (`gemini-3.7-flash`) answers from those 10 chunks and names its sources.

The code for steps 1-3 is `src/rag_sec/retrieve.py`.

## A bit of history

This was built in 14 days as a learning project. Each version (an "arm") added one
retrieval technique to the previous one, and every arm was scored on the same fixed set of
questions, so each gain can be traced to one change.

The score is **recall@10**: the share of the right chunks that make it into the top 10.

| Arm | Adds | Result |
|---|---|---|
| 1 | meaning search | 0.337 (dev) |
| 2 | + keyword search, merged with RRF | 0.514 (dev) |
| 3 | + reranker | 0.629 (dev); **shipped** |
| 4 | different ways of storing tables | lost to the plain layout |
| 5 | multi-vector search | never built |
| 6 | an agent that searches in a loop | no better; published as a negative result |

"Dev" is the question set used while building. The shipped system is Arm 3 plus the company
filter, the query cleanup and the two year signals. On the held-out **test** set (1,545
questions, never used for tuning) it scores **recall@10 0.831**.

The full research history, every decision and measurement, is frozen at the
[`v1-research`](https://github.com/abhishektuteja01/RAG-SEC/tree/v1-research) tag.

## Run it on your laptop

Timings below were measured on an Apple Silicon Mac.

1. Install [Docker](https://docs.docker.com/get-docker/) and
   [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repo:
   ```bash
   git clone https://github.com/abhishektuteja01/RAG-SEC.git && cd RAG-SEC
   ```
3. Make your config file and add a Gemini key:
   ```bash
   cp .env.example .env
   ```
   Get a free key from [Google AI Studio](https://aistudio.google.com/apikey) and put it in
   `GOOGLE_API_KEY`. The free tier allows only a small number of `gemini-3.7-flash` requests
   per day; the exact limit is on Google's
   [rate-limit page](https://ai.google.dev/gemini-api/docs/rate-limits).
4. Start Postgres:
   ```bash
   docker compose up -d
   ```
5. Load the database (a 652 MB download; the restore takes about 2 minutes):
   ```bash
   scripts/setup_db.sh
   ```
6. Start the server:
   ```bash
   uv run --env-file .env uvicorn rag_sec.api:app --workers 1
   ```
   The first start downloads the two models (about 6.4 GB). After that it takes about
   45 s to load them.
7. Open http://localhost:8000. When http://localhost:8000/ready says `ready`, ask away.
   Each question takes about 27-30 s, almost all of it reranking. A plain CPU is slower.

Good to know:

- **Port 5432 already taken?** Set `POSTGRES_PORT=5433` (or any free port) in `.env`.
  Both Docker and the server follow it.
- **Docker names the database volume after the folder** you cloned into (folder `RAG-SEC`
  gives the volume `rag-sec_pgdata`). Two clones in folders with the same name share one
  database. Use a different folder name, or set `COMPOSE_PROJECT_NAME` in `.env`.
- **Already have the dump file?** Skip the download:
  `DUMP_FILE=/path/to/rag_sec_chunks.dump scripts/setup_db.sh`
- Keep `--workers 1`. More workers load more copies of the models, and loading two at once
  crashes on Apple Silicon.

## Check the score

```bash
uv run scripts/evaluate.py --replay
```

This re-scores the stored reranker output and should print test recall@10 0.831 (n=1545).
It takes 1-2 minutes and needs no GPU, database or API key. The first run downloads the
question set from Hugging Face.

To run the real search for each question instead (needs the database and the models, but no
Gemini key), about 35 s per question on a Mac:

```bash
uv run --env-file .env scripts/evaluate.py --live --limit 10
```

## Deploy on a GPU server

[deploy/GUIDE.md](deploy/GUIDE.md) runs both containers on one NVIDIA GPU machine. On an
AWS `g4dn.xlarge` (T4 GPU), reranking takes about 3 s per question.

## Rebuild from scratch (optional)

Only needed if you don't use the database dump. `scripts/rebuild/` fetches and chunks the
filings (about 1 hour), then embeds them (many hours; 9-10 on an Apple M3).

## Project layout

```
src/rag_sec/
  retrieve.py        the search pipeline: filter, two searches, rerank
  candidates.py      the two searches (SQL) and RRF merging
  company.py         finds the company a question names; strips it from the query
  fiscal_year.py     finds years in text; the year boost
  store.py           Postgres connection, table schema, corpus check
  answer.py          the Gemini prompt and call; reads the ANSWER line
  api.py             HTTP API: /ask, /ready, /health, and the chat page
  static/index.html  the chat page
  config.py          model names and pinned versions, device choice
  eval.py            gold labels (which chunks are right) and metrics
  dataset.py         loads the T²-RAGBench questions
  edgar.py           downloads filings from SEC EDGAR (rebuild only)
  parsing.py         10-K HTML to text and tables (rebuild only)
  chunking.py        text and tables to chunks (rebuild only)
scripts/
  setup_db.sh        downloads and restores the database
  evaluate.py        scores recall@10 (--replay or --live)
  rebuild/           corpus.py, index.py, gold_cache.py: the slow rebuild
deploy/
  GUIDE.md           GPU server guide
  Dockerfile         API image (CPU or GPU)
  Dockerfile.postgres  Postgres with pgvector and pg_search
  docker-compose.gpu.yml  both services on a GPU host
  container_wordlist.py   startup check for the company matcher's wordlist
data/
  company_lexicon.json    company names and tickers
  wordlist_web2.txt       English words, so "visa" is not read as Visa Inc.
  gold_chunk_ids.json     cached gold labels, so scoring works without the corpus
  arms_scores.jsonl       stored reranker scores for --replay
  day6_*.json, day7_*.json  inputs to the gold labels
docker-compose.yml   Postgres for your laptop
.env.example         config template
pyproject.toml, uv.lock  Python dependencies
.github/workflows/ci.yml  CI: replays the score, fails unless it is 0.831
CLAUDE.md            notes for AI coding assistants
```

## License and data

The 10-K filings come from SEC EDGAR and are public domain. The questions come from
[T²-RAGBench](https://huggingface.co/datasets/G4KMU/t2-ragbench), used under
CC-BY-4.0, with thanks to its authors.
