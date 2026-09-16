"""One home for first-stage candidate generation: the dense (pgvector) query, the BM25
(pg_search) query, RRF over them, and the text fetch for a fused pool. Every arm, ablation
and diagnostic goes through these -- `retrieve.py` for the shipping path, the `scripts/`
arms for the offline evals (DECISIONS.md ARM1-*/ARM2-*/RETR-31).

`variant` is a REQUIRED argument on every read, deliberately not defaulted. These four
queries used to exist as seven near-identical copies, which is how `RETR-24` -- a missing
`variant` predicate -- ended up needing the same fix in seven places. A default would put
that footgun back one level down: a caller would inherit `'A'` without saying so, exactly
the "correct in one context, silently wrong in the next" failure mode. `LIVE_VARIANT` is
here to be passed explicitly, not assumed.

Behaviour-preserving refactor: the SQL is byte-identical to the copies it replaces once the
`variant` parameter is bound (`scripts/checks/candidate_sql.py` asserts that).
"""

from rag_sec.fiscal_year import year_distance_bonus

RRF_K = 60  # standard constant from Cormack et al. 2009's original RRF paper
LIVE_VARIANT = "A"  # the only variant in the live index (ARM4-3)
YEAR_BIAS_ALPHA = 0.01  # untuned placeholder -- tune on dev before promoting to a default

# One string, used by both queries: the company filter is the same predicate in each, and
# RETR-5's whole point is that dense and BM25 see the *same* restricted pool.
_TICKER_FILTER = "AND split_part(filing_stem, '_', 1) = ANY(%s)"

Pair = tuple[str, int]


def dense(conn, embedding, k: int, variant: str, tickers: list[str] | None = None) -> list[Pair]:
    """Top-k by cosine distance on the HNSW index. Returns (filing_stem, chunk_index)."""
    extra = _TICKER_FILTER if tickers else ""
    args = (variant, tickers, embedding, k) if tickers else (variant, embedding, k)
    rows = conn.execute(
        f"SELECT filing_stem, chunk_index FROM chunks WHERE variant = %s {extra} ORDER BY embedding <=> %s LIMIT %s",
        args,
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def bm25(conn, query_text: str, k: int, variant: str, tickers: list[str] | None = None) -> list[Pair]:
    """Top-k by BM25 over `text` via pg_search's own index -- a real BM25, not a rewritten
    `ts_rank` (INFRA-4). Returns (filing_stem, chunk_index)."""
    extra = _TICKER_FILTER if tickers else ""
    args = (query_text, variant, tickers, k) if tickers else (query_text, variant, k)
    rows = conn.execute(
        f"""SELECT filing_stem, chunk_index, paradedb.score(id) AS s
           FROM chunks
           WHERE id @@@ paradedb.match('text', %s) AND variant = %s {extra}
           ORDER BY s DESC LIMIT %s""",
        args,
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def rrf_fuse_scores(ranked_lists: list[list[Pair]], k: int = RRF_K) -> dict[Pair, float]:
    """Reciprocal Rank Fusion scores: score(d) = sum_over_lists 1/(k + rank_in_list(d)).

    Rank-based, not score-based -- dense cosine similarity and BM25 scores live on
    unrelated scales, so summing raw scores would let whichever one happens to have
    bigger numbers dominate. RRF only needs each list's ordering.
    """
    scores: dict[Pair, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def rrf_fuse(ranked_lists: list[list[Pair]], k: int = RRF_K) -> list[Pair]:
    """Plain RRF order. Every offline arm and the shipping default call this one unchanged
    (RETR-36) -- do not add signals here; add a sibling fuse function instead, the way
    `rrf_fuse_year_biased` does, so past arms stay reproducible from this same code."""
    scores = rrf_fuse_scores(ranked_lists, k)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


def rrf_fuse_year_biased(
    ranked_lists: list[list[Pair]],
    query_years: list[int],
    alpha: float = YEAR_BIAS_ALPHA,
    k: int = RRF_K,
) -> list[Pair]:
    """RRF fusion plus an additive year-proximity nudge -- soft, not a filter. `RETR-5`
    rejected a hard year gate: the extraction signal is only 75-82% accurate, and a hard
    filter at that accuracy deletes the right answer whenever it misses. `query_years` empty
    (no year extracted from the question) makes every bonus 0, identical to plain `rrf_fuse`.
    `alpha` is untuned -- see `YEAR_BIAS_ALPHA`."""
    scores = rrf_fuse_scores(ranked_lists, k)
    for pair in scores:
        scores[pair] += year_distance_bonus(pair[0], query_years, alpha)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


def chunk_texts(conn, pairs: list[Pair], variant: str) -> dict[Pair, str]:
    """Text for a fused pool, in the caller's order. One query per *filing*, not per chunk:
    a pool of 50 candidates typically spans far fewer filings than chunks."""
    if not pairs:
        return {}
    stems = list({p[0] for p in pairs})
    rows = conn.execute(
        "SELECT filing_stem, chunk_index, text FROM chunks WHERE variant = %s AND filing_stem = ANY(%s)",
        (variant, stems),
    ).fetchall()
    lookup = {(r[0], r[1]): r[2] for r in rows}
    return {p: lookup[p] for p in pairs if p in lookup}
