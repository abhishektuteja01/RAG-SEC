"""First-stage candidate generation: the dense (pgvector) query, the BM25 (pg_search) query,
RRF over them, and the text fetch for the fused pool.
"""

from rag_sec.fiscal_year import extract_years, year_distance_bonus

CANDIDATE_K = 50  # pool size: how many candidates reach the reranker
# How many rows each retriever reads before fusion cuts back to CANDIDATE_K. Reading deeper
# raises the chance the right chunk is in the pool; the rerank cost stays the same.
READ_DEPTH = 200
RRF_K = 60  # the standard constant from the original RRF paper (Cormack et al. 2009)
# Every row in the table is variant 'A' (whole tables kept in one chunk). The column stays
# because the database dump has it; every read constrains it.
VARIANT = "A"
YEAR_BIAS_ALPHA = 0.01  # untuned; tune on dev before changing

# One string, used by both queries, so dense and BM25 see the same restricted pool.
_TICKER_FILTER = "AND split_part(filing_stem, '_', 1) = ANY(%s)"

Pair = tuple[str, int]  # (filing_stem, chunk_index)


def dense(conn, embedding, k: int, tickers: list[str] | None = None) -> list[Pair]:
    """Top-k by cosine distance on the HNSW index."""
    extra = _TICKER_FILTER if tickers else ""
    args = (VARIANT, tickers, embedding, k) if tickers else (VARIANT, embedding, k)
    rows = conn.execute(
        f"SELECT filing_stem, chunk_index FROM chunks WHERE variant = %s {extra} ORDER BY embedding <=> %s LIMIT %s",
        args,
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def bm25(conn, query_text: str, k: int, tickers: list[str] | None = None) -> list[Pair]:
    """Top-k by BM25 over `text`, through pg_search's own index."""
    extra = _TICKER_FILTER if tickers else ""
    args = (query_text, VARIANT, tickers, k) if tickers else (query_text, VARIANT, k)
    rows = conn.execute(
        f"""SELECT filing_stem, chunk_index, paradedb.score(id) AS s
           FROM chunks
           WHERE id @@@ paradedb.match('text', %s) AND variant = %s {extra}
           ORDER BY s DESC LIMIT %s""",
        args,
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def rrf_fuse(ranked_lists: list[list[Pair]], query_years: list[int]) -> list[Pair]:
    """Reciprocal Rank Fusion, plus a small bonus for filings near a year the question names.

    RRF: score(d) = sum over lists of 1/(RRF_K + rank of d in that list). It uses ranks, not
    raw scores, because cosine similarity and BM25 scores live on unrelated scales.

    The year bonus is soft, never a filter: a hard year filter would delete the right answer
    whenever the year extraction misses. No year in the question means no bonus.
    """
    scores: dict[Pair, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    for pair in scores:
        scores[pair] += year_distance_bonus(pair[0], query_years, YEAR_BIAS_ALPHA)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


def first_stage(
    conn, query_emb, query_text: str, *, tickers: list[str], query_years: list[int]
) -> tuple[list[Pair], dict[Pair, str]]:
    """Both retrievers, fusion, the cut to CANDIDATE_K, and the text for what survived.
    Returns (pool, texts).

    `tickers` empty means an unfiltered search. `query_years` is extracted by the caller.

    Fusion runs twice. The first pass only supplies an order. From it, a third list is made:
    the candidates whose chunk text names a year the question names, in that same order.
    Fusing all three decides which CANDIDATE_K survive. The third list has no weight or
    threshold of its own, and it can only reorder what the two retrievers already found.
    """
    lists = [
        dense(conn, query_emb, READ_DEPTH, tickers),
        bm25(conn, query_text, READ_DEPTH, tickers),
    ]
    base = rrf_fuse(lists, query_years)
    texts = chunk_texts(conn, base)
    lists.append(year_text_list(base, texts, query_years))
    pool = rrf_fuse(lists, query_years)[:CANDIDATE_K]
    return pool, texts


def chunk_texts(conn, pairs: list[Pair]) -> dict[Pair, str]:
    """Text for exactly these pairs, in one query, keyed in the caller's order."""
    if not pairs:
        return {}
    stems = [p[0] for p in pairs]
    idxs = [int(p[1]) for p in pairs]
    rows = conn.execute(
        "SELECT filing_stem, chunk_index, text FROM chunks WHERE variant = %s "
        "AND (filing_stem, chunk_index) IN (SELECT s, i FROM unnest(%s::text[], %s::int[]) AS t(s, i))",
        (VARIANT, stems, idxs),
    ).fetchall()
    lookup = {(r[0], r[1]): r[2] for r in rows}
    return {p: lookup[p] for p in pairs if p in lookup}


def year_text_list(pairs: list[Pair], texts: dict[Pair, str], query_years: list[int]) -> list[Pair]:
    """The candidates whose chunk TEXT mentions a year the question names, in input order.

    The year bonus in `rrf_fuse` reads the year from the filing name, so it cannot tell the
    2018 column from the 2019 column inside one 10-K. This list works at the chunk level.
    """
    if not query_years:
        return []
    wanted = set(query_years)
    return [p for p in pairs if p in texts and wanted & set(extract_years(texts[p]))]
