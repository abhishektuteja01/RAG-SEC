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

from rag_sec.fiscal_year import extract_years, year_distance_bonus

CANDIDATE_K = 50  # POOL size: how many candidates reach the reranker. Re-exported by retrieve.py, which every caller imports it from
# READ depth: how many rows each first-stage leg asks the index for, before fusion cuts
# back to CANDIDATE_K. The two were one constant doing both jobs until P10 needed them
# apart -- depth pays (union recall 0.8426 -> 0.9503 at 200) while the pool, and so the
# rerank bill, stays fixed. READ_DEPTH_MAX is the ceiling a caller may ask for, and is
# what store.HNSW_EF_SEARCH is derived from: ef_search must cover the DEEPEST read, not
# the pool. Deriving it from CANDIDATE_K would put RETR-50 straight back at depth 200.
READ_DEPTH = 50  # default: unchanged from when this was CANDIDATE_K's second job
# What `retrieve()` SERVES at, since DEPLOY-25 flipped RETR-51/RETR-52 on. Deliberately a
# second constant rather than a new value for READ_DEPTH: the offline cells in
# `05_arm3_rerank.py` fall back to READ_DEPTH for every arm that predates a read-depth knob,
# so moving it would silently re-measure Arms 1-3 at depth 200 and quietly invalidate every
# number they published. Two constants because they answer two different questions -- what
# the shipped path reads, and what the historical arms read.
SERVING_READ_DEPTH = 200
READ_DEPTH_MAX = 200
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


def first_stage(
    conn,
    query_emb,
    query_text: str,
    *,
    tickers: list[str] | None,
    variant: str,
    read_depth: int = READ_DEPTH,
    pool_k: int = CANDIDATE_K,
    query_years: list[int] | None = None,
    year_bias: bool = True,
    year_text_fusion: bool = False,
    reserve: int = 0,
) -> tuple[list[Pair], dict[Pair, str], int]:
    """The whole first stage: two retrievers, fusion, the cut to `pool_k`, and the text for
    what survived. Returns (pool, texts, n_lists_fused).

    `n_lists_fused` is returned rather than re-derived by the caller for tracing: it depends
    on `reserve`, `tickers` and `year_text_fusion` together, and a caller recomputing it is a
    second copy of this function's branching that can silently disagree with it.

    ONE HOME, for the reason `RETR-36` gave for the SQL a level below. `retrieve()` and
    `05_arm3_rerank.py prepare` both build a candidate pool, and they had already drifted:
    `prepare` was still fusing with plain `rrf_fuse` months after `RETR-40` made the
    year-proximity nudge the shipped default, so the cluster leg and the local leg were
    scoring DIFFERENT POOLS while the module docstring said a scorer could not tell them
    apart. That is `INFRA-22`'s shape exactly -- a consumer assuming a producer's behaviour
    that nothing guaranteed -- and the only fix that holds is for there to be one producer.
    Add a knob here, and every leg gets it; add one at a call site, and the legs diverge.

    `reserve` keeps that many slots for unfiltered results (see `retrieve()`); 0 disables it.
    `query_years` is required for both year features and is the CALLER's extraction, not one
    made here: the agent path extracts from the original question, not the rewritten one.
    """
    fuse = (lambda ls: rrf_fuse_year_biased(ls, query_years or [])) if year_bias else rrf_fuse
    if tickers:
        n = read_depth - reserve
        lists = [
            dense(conn, query_emb, n, variant, tickers),
            bm25(conn, query_text, n, variant, tickers),
        ]
        if reserve:
            lists += [
                dense(conn, query_emb, reserve, variant),
                bm25(conn, query_text, reserve, variant),
            ]
    else:
        lists = [
            dense(conn, query_emb, read_depth, variant),
            bm25(conn, query_text, read_depth, variant),
        ]
    if year_text_fusion:
        # P10. The first fusion only supplies an ORDER for the third list, which
        # `year_text_list` inherits rather than inventing, so the feature carries no weight
        # or threshold of its own. Text for the whole union is fetched here, not for the
        # final pool -- the third list has to exist before the cut, since its entire job is
        # to change WHICH candidates survive it. No extra rerank pairs: `pool_k` is unchanged.
        # And the third list is a SUBSET of the union, so it can only reorder what the two
        # retrievers already found, never introduce a chunk neither returned, and it leaves
        # RRF's insertion-order tie-break (pinned by scripts/checks/candidate_sql.py) alone.
        base = fuse(lists)
        # ...exact, not chunk_texts: the per-filing fetch is cheaper only for a small pool
        # spanning few filings. On the union, and on the unfiltered path especially, it pulls
        # tens of MB of whole filings to use a few hundred chunks -- 39 MB mean / 100 MB worst
        # at depth 200, and 575ms against 32ms.
        texts = chunk_texts_exact(conn, base, variant)
        lists = lists + [year_text_list(base, texts, query_years or [])]
        pool = fuse(lists)[:pool_k]
    else:
        pool = fuse(lists)[:pool_k]
        texts = chunk_texts(conn, pool, variant)
    return pool, texts, len(lists)


def chunk_texts_exact(conn, pairs: list[Pair], variant: str) -> dict[Pair, str]:
    """Text for exactly these (filing_stem, chunk_index) pairs, in the caller's order.

    The per-filing fetch `chunk_texts` uses is cheaper only while a pool spans far fewer
    FILINGS than chunks, which is what a 50-candidate company-filtered pool does. P10 inverts
    both halves of that: it needs text for the whole pre-cut union, and the union is deepest
    exactly on the questions where company resolution abstains and the search is unfiltered.
    Measured there, BM25 alone at depth 200 spans a mean 68 filings, so `chunk_texts` would
    pull ~10,400 rows / 39 MB (worst seen 100 MB) to use 200 of them. This pulls the 200.

    Composite-key lookup against the `UNIQUE (filing_stem, chunk_index, variant)` index, one
    query, no per-chunk round trip. `variant` is required here for the same reason it is on
    every other read in this module (RETR-24).
    """
    if not pairs:
        return {}
    stems = [p[0] for p in pairs]
    idxs = [int(p[1]) for p in pairs]
    rows = conn.execute(
        "SELECT filing_stem, chunk_index, text FROM chunks WHERE variant = %s "
        "AND (filing_stem, chunk_index) IN (SELECT s, i FROM unnest(%s::text[], %s::int[]) AS t(s, i))",
        (variant, stems, idxs),
    ).fetchall()
    lookup = {(r[0], r[1]): r[2] for r in rows}
    return {p: lookup[p] for p in pairs if p in lookup}


def year_text_list(pairs: list[Pair], texts: dict[Pair, str], query_years: list[int]) -> list[Pair]:
    """P10's third RRF list: the candidates whose chunk TEXT mentions a year the question
    names, in the order they were handed in.

    Distinct from `year_distance_bonus`, which reads the year out of `filing_stem` and is
    therefore FILING-level -- it cannot tell the 2018 column from the 2019 column inside one
    10-K, and "right filing, wrong chunk" is the larger half of the first-stage gap. This is
    chunk-level and separates them.

    Ordering is inherited from `pairs`, not computed: pass the base RRF order and this list is
    that order restricted to year-matching chunks. Tuning-free by construction -- it adds no
    weight, no threshold and no parameter, and its only influence is through RRF's own
    1/(k+rank). The measured arm's within-list ordering was never recorded, so this is a
    RE-DERIVATION of it, not a reproduction; anything quoted from it must be re-measured here.

    `extract_years` on the chunk, not a bare `\b2019\b`: it carries the guard that skips
    `$2,019` and `2019 million`, so "mentions a year" means the same thing on the chunk side
    as on the question side. Empty `query_years` returns [], which makes the whole feature a
    no-op the same way a missed extraction already does for the year nudge.
    """
    if not query_years:
        return []
    wanted = set(query_years)
    return [p for p in pairs if p in texts and wanted & set(extract_years(texts[p]))]


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
