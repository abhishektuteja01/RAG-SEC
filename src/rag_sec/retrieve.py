"""Standalone retrieval tool for the Day 8 agentic loop -- same dense+BM25/RRF+rerank
pipeline as scripts/arms/day5_run_arm3.py, extracted so a LangGraph node can call it as
a real tool (query in, ranked chunks out) instead of duplicating the eval-script's logic.

DECISIONS.md ARM3-1/ARM2-1: rerank model and RRF fusion choices are unchanged here --
this module only relocates the mechanism, it doesn't re-decide it.
"""

from functools import lru_cache

from rag_sec.company import resolve as resolve_companies
from rag_sec.config import EMBED_MODEL_NAME, RERANK_MODEL_NAME
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
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def retrieve(query: str, k: int = TOP_K, company_filter: bool = True, reserve: int = 0) -> list[dict]:
    """Dense+BM25/RRF candidates, reranked by the cross-encoder, top-k returned.

    Returns plain dicts (not a dataclass) so the same shape works as a LangGraph
    tool result (LLM-readable) and as eval input for later scoring.

    `company_filter` scopes candidate generation to whichever company the question names,
    resolved from the question text alone (`rag_sec.company`). Measured +8.1 points
    recall@50, 102 questions recovered against 2 lost -- DECISIONS.md RETR-26, the
    variant-clean re-measurement; RETR-5's +7.6/96/2 was taken on the contaminated pool.
    It is a parameter rather than unconditional so the pre-filter arms stay reproducible.

    NOT applied here: the entity-framing strip (`company.strip_entity_framing`), which is
    two thirds of the measured gain -- filter alone +0.042 recall@10 on test against
    +0.145 for filter+strip (RETR-31). It exists only in the offline payload builder, so
    this module currently implements the weaker of the two measured cells.

    `reserve` keeps that many candidate slots for unfiltered results, as insurance against
    the one failure mode that can delete gold: a question naming an acquired business or a
    counterparty instead of the filer ("the FIS Gaming Business" in a Global Payments
    filing). Measured at 0.0% of dev and 0.20% of train, so it defaults off.
    """
    embed_model = _embed_model()
    cross_encoder = _cross_encoder()
    query_emb = embed_model.encode(query, normalize_embeddings=True)
    tickers = resolve_companies(query) if company_filter else []

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
    pairs = [(query, texts[c]) for c in candidates]
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
