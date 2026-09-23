"""The retrieval path: dense + BM25/RRF candidates, reranked by a cross-encoder. Query in,
ranked chunks out. Candidate generation lives in `rag_sec.candidates`.
"""

import contextlib
import threading
import time
from functools import lru_cache, wraps

from rag_sec.candidates import (
    CANDIDATE_K,
    LIVE_VARIANT,
    READ_DEPTH_MAX,
    SERVING_READ_DEPTH,
    first_stage,
)
from rag_sec.company import resolve_with_reason, strip_entity_framing
from rag_sec.config import (
    EMBED_MODEL_NAME,
    EMBED_MODEL_REVISION,
    RERANK_MODEL_NAME,
    RERANK_MODEL_REVISION,
    pick_device,
)
from rag_sec.fiscal_year import extract_years
from rag_sec.store import get_conn

TOP_K = 10

# Per-stage timings + the pre-rerank candidate list from the most recent retrieve() call on
# THIS thread. Stashed rather than returned so the return value stays a plain list of chunks.
_last = threading.local()

# Torch's MPS backend is NOT thread-safe (see config.pick_device). Every local model call
# goes through this lock.
_gpu_lock = threading.Lock()


@lru_cache(maxsize=1)
def _device() -> str:
    """pick_device() re-imports torch on every call; the backend cannot change in a process."""
    device = pick_device()
    if device == "cpu":
        # torch's default intra-op thread count can come in low under a container runtime
        # (cgroup quota misread, inherited OMP_NUM_THREADS). Config only; scores unaffected.
        import os

        import torch

        torch.set_num_threads(os.cpu_count())
    return device


def last_call_stats() -> dict:
    """Timings and candidate list from this thread's most recent retrieve(). {} if none."""
    return getattr(_last, "stats", {})


def _model_cache(fn):
    """`lru_cache(maxsize=1)` plus a construction lock, i.e. build-once even under threads.

    lru_cache alone does NOT serialise concurrent misses (8 threads produced 8 constructions),
    and concurrent construction on MPS segfaults. Its own lock, not `_gpu_lock`, which is
    held around model calls and would self-deadlock. Double-checked so the warm path takes
    no mutex.
    """
    cached = lru_cache(maxsize=1)(fn)
    lock = threading.Lock()

    @wraps(fn)
    def get():
        if cached.cache_info().currsize:
            return cached()
        with lock:
            return cached()

    get.cache_clear = cached.cache_clear
    get.cache_info = cached.cache_info
    return get


@_model_cache
def _embed_model():
    from sentence_transformers import SentenceTransformer

    # device passed explicitly: otherwise SentenceTransformer auto-picks mps and ignores
    # RAG_SEC_DEVICE.
    return SentenceTransformer(EMBED_MODEL_NAME, revision=EMBED_MODEL_REVISION, device=_device())


@_model_cache
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    device = _device()
    # fp16 on cuda only: measured 3-4x faster than fp32 on a T4 with exact top-5 parity
    # (score deltas 1e-6 to 1e-3). Needs sentence-transformers>=6.0.0 -- older releases ran
    # the sigmoid in half precision and could reorder top candidates. Never tested on mps.
    kwargs = {"model_kwargs": {"torch_dtype": "float16"}} if device.startswith("cuda") else {}
    return CrossEncoder(RERANK_MODEL_NAME, revision=RERANK_MODEL_REVISION, device=device, **kwargs)


def _drain_mps(device: str) -> float:
    """Release MPS's cached blocks; returns the elapsed seconds.

    Without this, on a 16 GiB unified-memory Mac, embed and rerank latency climb question
    after question (rerank 27s -> 48s over ~10) while swap grows: the caching allocator never
    hands freed blocks back. `startswith`, so `mps:0` counts too.
    """
    t0 = time.perf_counter()
    if device.startswith("mps"):
        import torch

        torch.mps.empty_cache()
    return time.perf_counter() - t0


@contextlib.contextmanager
def _drain_on_error(device: str):
    """Drain on the way out of a FAILED retrieval too, so one error does not leave the
    allocator saturated for every later question."""
    try:
        yield
    except BaseException:
        _drain_mps(device)
        raise


def retrieve(
    query: str,
    k: int = TOP_K,
    company_filter: bool = True,
    strip_query: bool = True,
    reserve: int = 0,
    resolve_from: str | None = None,
    year_bias: bool = True,
    strip_dense: bool = True,
    read_depth: int = SERVING_READ_DEPTH,
    year_text_fusion: bool = True,
) -> list[dict]:
    """Dense+BM25/RRF candidates, reranked by the cross-encoder, top-k returned as dicts
    ({filing_stem, chunk_index, text, score}). The defaults are the best measured setup
    (test recall@10 0.831); the flags stay so each piece can be switched off and measured.

    `company_filter`: scope candidate generation to the company the question names,
    resolved from the question text alone (`rag_sec.company`). No match -> unfiltered.

    `strip_query`: rerank against the question with company/filing framing removed. Once
    every candidate is the right company, the name only rewards boilerplate-heavy chunks.
    The filter and the strip together are worth far more than either alone.

    `strip_dense`: embed that stripped question for the dense leg too. BM25 keeps the raw
    question, whose tokens decide which documents qualify at all.

    `year_bias`: add a small year-proximity bonus (from the filing's fiscal year) into RRF.
    Soft, never a filter: a missed year extraction just adds zero.

    `year_text_fusion` + `read_depth`: each leg reads `read_depth` rows; a third RRF list of
    candidates whose chunk TEXT names a year the question names decides which `CANDIDATE_K`
    of them survive to the reranker. Worth ~0 at depth 50, so the two go together. The
    rerank cost is unchanged (the pool stays 50). Known per-question cost: a gold chunk whose
    own text never restates the question's year is left out of that list and can drop out
    of the pool.

    `resolve_from`: text to resolve the company and years from, if not `query` (e.g. the
    original question when `query` is a rewrite). `reserve`: keep that many pool slots for
    unfiltered results; 0 (off) is what was measured.

    Rerank scores are attached to candidates in first-stage order and sorted here; a caller
    that reads raw candidate order gets a very different (worse) ranking.
    """
    if read_depth > READ_DEPTH_MAX:
        # store.HNSW_EF_SEARCH is derived from READ_DEPTH_MAX; a deeper read would come back
        # short without an error.
        raise ValueError(
            f"read_depth={read_depth} exceeds READ_DEPTH_MAX={READ_DEPTH_MAX}; raise it in "
            "candidates.py so store.HNSW_EF_SEARCH follows"
        )
    t = {}
    device = _device()
    with _drain_on_error(device):
        # ── resolve company, strip framing, extract years ──────────────────────────
        t0 = time.perf_counter()
        if company_filter:
            tickers, fallback_reason = resolve_with_reason(resolve_from or query)
        else:
            tickers, fallback_reason = [], None
        rerank_query = (
            strip_entity_framing(query, aliases_from=resolve_from or query)
            if strip_query
            else query
        )
        # `or year_text_fusion`: that feature reads the same years, so gating them on
        # year_bias alone would make it a silent no-op when year_bias is off.
        query_years = (extract_years(resolve_from or query)
                       if (year_bias or year_text_fusion) else [])
        # Unchanged from `query` when nothing resolves: strip_entity_framing returns it as is.
        dense_query = rerank_query if strip_dense else query
        t["resolve_s"] = time.perf_counter() - t0

        # ── embed ─────────────────────────────────────────────────────────────────
        # First-call model construction (~19s) is timed separately so embed_s stays a
        # per-question number.
        t0 = time.perf_counter()
        embed_model = _embed_model()
        cross_encoder = _cross_encoder()
        t["model_init_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        with _gpu_lock:
            compute0 = time.perf_counter()
            query_emb = embed_model.encode(dense_query, normalize_embeddings=True)
        t["embed_s"] = time.perf_counter() - t0
        t["embed_lock_wait_s"] = compute0 - t0

        # ── first stage: dense + BM25 (+ year list), fused with RRF ───────────────
        t0 = time.perf_counter()
        with get_conn() as conn:
            fused, texts, _n_lists = first_stage(
                conn, query_emb, query, tickers=tickers, variant=LIVE_VARIANT,
                read_depth=read_depth, pool_k=CANDIDATE_K, query_years=query_years,
                year_bias=year_bias, year_text_fusion=year_text_fusion, reserve=reserve,
            )
        t["search_s"] = time.perf_counter() - t0

        # ── rerank ────────────────────────────────────────────────────────────────
        candidates = [c for c in fused if c in texts]
        t0 = time.perf_counter()
        pairs = [(rerank_query, texts[c]) for c in candidates]
        # Batch pairs by length, then restore order: padding=True pads each batch to its
        # longest member, so mixing lengths wastes compute. Scores are unaffected.
        order_by_len = sorted(range(len(pairs)), key=lambda j: len(pairs[j][1]))
        sorted_pairs = [pairs[j] for j in order_by_len]
        wait0 = time.perf_counter()
        with _gpu_lock:
            compute0 = time.perf_counter()
            sorted_scores = cross_encoder.predict(sorted_pairs, batch_size=32) if pairs else []
        scores = [0.0] * len(pairs)
        for j, s in zip(order_by_len, sorted_scores):
            scores[j] = float(s)
        order = sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)[:k]
        t["rerank_s"] = time.perf_counter() - t0
        t["rerank_lock_wait_s"] = compute0 - wait0

        t["mps_empty_cache_s"] = _drain_mps(device)
        # model_init_s is one-off and the lock waits are already inside embed_s/rerank_s.
        t["total_s"] = (t["embed_s"] + t["resolve_s"] + t["search_s"] + t["rerank_s"]
                        + t["mps_empty_cache_s"])
        _last.stats = {
            "timings": t,
            "rerank_query": rerank_query,
            "dense_query": dense_query,
            "tickers": list(tickers),
            "fallback_reason": fallback_reason,
            # pre-rerank order, so recall@50 and the reranker's own contribution stay recoverable
            "candidates": [[c[0], int(c[1])] for c in candidates],
            "reranked": [[candidates[j][0], int(candidates[j][1]), float(scores[j])]
                         for j in sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)],
        }

        return [
            {
                "filing_stem": candidates[j][0],
                "chunk_index": candidates[j][1],
                "text": texts[candidates[j]],
                "score": float(scores[j]),
            }
            for j in order
        ]
