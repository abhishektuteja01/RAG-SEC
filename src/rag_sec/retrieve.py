"""The shipping retrieval path: dense + BM25/RRF + cross-encoder rerank, query in and
ranked chunks out. Same pipeline scripts/archive/arm3_singleprocess.py evaluates offline
(DECISIONS.md ARM2-1/ARM3-1/RETR-31); the private helpers are shared, not re-implemented.
"""

from functools import lru_cache

from rag_sec.company import resolve as resolve_companies
from rag_sec.company import strip_entity_framing
from rag_sec.config import EMBED_MODEL_NAME, RERANK_MODEL_NAME, pick_device
from rag_sec.store import get_conn

TOP_K = 10
CANDIDATE_K = 50
RRF_K = 60


@lru_cache(maxsize=1)
def _embed_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    device = pick_device()
    return CrossEncoder(RERANK_MODEL_NAME, device=device)


def _retrieve_dense(conn, embedding, k: int, tickers: list[str] | None = None) -> list[tuple[str, int]]:
    where = "AND split_part(filing_stem, '_', 1) = ANY(%s)" if tickers else ""
    args = (tickers, embedding, k) if tickers else (embedding, k)
    rows = conn.execute(
        f"SELECT filing_stem, chunk_index FROM chunks WHERE variant = 'A' {where} ORDER BY embedding <=> %s LIMIT %s",
        args,
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _retrieve_bm25(conn, query_text: str, k: int, tickers: list[str] | None = None) -> list[tuple[str, int]]:
    extra = "AND split_part(filing_stem, '_', 1) = ANY(%s)" if tickers else ""
    args = (query_text, tickers, k) if tickers else (query_text, k)
    rows = conn.execute(
        f"""SELECT filing_stem, chunk_index, paradedb.score(id) AS s
           FROM chunks
           WHERE id @@@ paradedb.match('text', %s) AND variant = 'A' {extra}
           ORDER BY s DESC LIMIT %s""",
        args,
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _rrf_fuse(ranked_lists: list[list[tuple[str, int]]], k: int = RRF_K) -> list[tuple[str, int]]:
    scores: dict[tuple[str, int], float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


def _fetch_texts(conn, pairs: list[tuple[str, int]]) -> dict[tuple[str, int], str]:
    if not pairs:
        return {}
    stems = list({p[0] for p in pairs})
    rows = conn.execute(
        "SELECT filing_stem, chunk_index, text FROM chunks WHERE variant = 'A' AND filing_stem = ANY(%s)",
        (stems,),
    ).fetchall()
    lookup = {(r[0], r[1]): r[2] for r in rows}
    return {p: lookup[p] for p in pairs if p in lookup}


def retrieve(
    query: str,
    k: int = TOP_K,
    company_filter: bool = True,
    strip_query: bool = True,
    reserve: int = 0,
) -> list[dict]:
    """Dense+BM25/RRF candidates, reranked by the cross-encoder, top-k returned as dicts
    (LLM-readable as a LangGraph tool result, and scoreable as eval input).

    Defaults are RETR-31's `filtered_stripped` cell, the winning one: recall@10 0.581 ->
    0.726 on the held-out test split. Both flags stay parameters so the earlier arms remain
    reproducible from this same code path.

    `company_filter` scopes candidate generation to whichever company the question names,
    resolved from the question text alone (`rag_sec.company`). Filter alone is +0.042.

    `strip_query` reranks against the question with company/filing framing removed, while
    candidate generation still sees the full question -- exactly the split
    scripts/retrieval/rerank_prepare.py measured. Alone it is +0.015, but
    with the filter it is +0.145: once every candidate is the right company, the company
    name only rewards whichever chunk repeats the most boilerplate (RETR-6).

    `reserve` keeps that many candidate slots for unfiltered results, as insurance against
    the one failure mode that can delete gold: a question naming an acquired business or a
    counterparty instead of the filer ("the FIS Gaming Business" in a Global Payments
    filing). Measured at 0.0% of dev and 0.20% of train, so it defaults off.
    """
    embed_model = _embed_model()
    cross_encoder = _cross_encoder()
    query_emb = embed_model.encode(query, normalize_embeddings=True)
    tickers = resolve_companies(query) if company_filter else []
    rerank_query = strip_entity_framing(query) if strip_query else query

    with get_conn() as conn:
        if tickers:
            n = CANDIDATE_K - reserve
            lists = [
                _retrieve_dense(conn, query_emb, n, tickers),
                _retrieve_bm25(conn, query, n, tickers),
            ]
            if reserve:
                lists += [
                    _retrieve_dense(conn, query_emb, reserve),
                    _retrieve_bm25(conn, query, reserve),
                ]
        else:
            lists = [
                _retrieve_dense(conn, query_emb, CANDIDATE_K),
                _retrieve_bm25(conn, query, CANDIDATE_K),
            ]
        fused = _rrf_fuse(lists)[:CANDIDATE_K]
        texts = _fetch_texts(conn, fused)

    candidates = [c for c in fused if c in texts]
    pairs = [(rerank_query, texts[c]) for c in candidates]
    scores = cross_encoder.predict(pairs, batch_size=32) if pairs else []
    order = sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)[:k]

    return [
        {
            "filing_stem": candidates[j][0],
            "chunk_index": candidates[j][1],
            "text": texts[candidates[j]],
            "score": float(scores[j]),
        }
        for j in order
    ]
