# RAG-SEC -- orientation

QA over 799 SEC 10-K filings: company filter -> dense (bge-m3, pgvector) + BM25 (pg_search)
-> RRF (+ year signals) -> bge-reranker-v2-m3 -> Gemini `gemini-3.7-flash`. Test recall@10 0.831.

- `src/rag_sec/retrieve.py` is the pipeline; it has no switches, it IS the serving setup.
  `api.py` adds no retrieval of its own. Candidate SQL lives once, in `candidates.py`.
- `scripts/evaluate.py --replay` must print test recall@10 0.831 (n=1545); CI asserts it.
- Gold labels (`eval.py` label functions, `dataset.load_t2_ragbench`, three `data/day*` files)
  are fingerprinted into `data/gold_chunk_ids.json`. Changing them means
  `scripts/rebuild/gold_cache.py build` and proving the labels did not move.
- `store.preflight` fails any connection unless the table holds exactly 99,654 variant-'A'
  chunks, all embedded. Keep the `variant = %s` condition in chunk SQL: the served rankings
  were verified with it, and dropping it can change the query plan.
- `chunking.py` reproduces all 99,654 stored chunks exactly; the gold labels are chunk
  indices, so a change that moves a boundary invalidates every score.
- Rerank score files store candidates in first-stage order: read them through
  `eval.load_ranking`, which sorts. Open any on-disk artifact before trusting its shape.
- Keep one API worker: concurrent model loading segfaults on Apple MPS.
- Env comes from `uv run --env-file .env`. Model and dataset revisions are pinned by commit.
- Research history (all arms, decisions, measurements): git tag `v1-research`.
