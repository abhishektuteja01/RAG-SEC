"""The shipping retrieval path: dense + BM25/RRF + cross-encoder rerank, query in and
ranked chunks out. Same pipeline scripts/archive/arm3_singleprocess.py evaluates offline
(DECISIONS.md ARM2-1/ARM3-1/RETR-31); candidate generation itself lives in
`rag_sec.candidates`, shared with every offline arm rather than re-implemented (RETR-36).
"""

from functools import lru_cache

from rag_sec.candidates import LIVE_VARIANT, bm25, chunk_texts, dense, rrf_fuse
from rag_sec.company import resolve as resolve_companies
from rag_sec.company import strip_entity_framing
from rag_sec.config import EMBED_MODEL_NAME, RERANK_MODEL_NAME, pick_device
from rag_sec.store import get_conn

TOP_K = 10
CANDIDATE_K = 50


@lru_cache(maxsize=1)
def _embed_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    device = pick_device()
    return CrossEncoder(RERANK_MODEL_NAME, device=device)


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
                dense(conn, query_emb, n, LIVE_VARIANT, tickers),
                bm25(conn, query, n, LIVE_VARIANT, tickers),
            ]
            if reserve:
                lists += [
                    dense(conn, query_emb, reserve, LIVE_VARIANT),
                    bm25(conn, query, reserve, LIVE_VARIANT),
                ]
        else:
            lists = [
                dense(conn, query_emb, CANDIDATE_K, LIVE_VARIANT),
                bm25(conn, query, CANDIDATE_K, LIVE_VARIANT),
            ]
        fused = rrf_fuse(lists)[:CANDIDATE_K]
        texts = chunk_texts(conn, fused, LIVE_VARIANT)

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
