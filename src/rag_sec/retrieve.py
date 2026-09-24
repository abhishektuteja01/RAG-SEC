"""The retrieval path. Query in, ranked chunks out:

1. find the company the question names, and strip company and filing wording from it;
2. dense + BM25 candidates, fused with RRF (`rag_sec.candidates`);
3. rerank the pool with a cross-encoder and keep the top k.
"""

import contextlib
import threading
import time
from collections.abc import Callable
from functools import lru_cache, wraps

from rag_sec.candidates import first_stage
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

# Timings from the most recent retrieve() call on THIS thread. Stashed rather than returned
# so the return value stays a plain list of chunks.
_last = threading.local()

# Torch's MPS backend is not thread-safe (see config.pick_device). Every model call takes this.
_gpu_lock = threading.Lock()


@lru_cache(maxsize=1)
def _device() -> str:
    device = pick_device()
    if device == "cpu":
        # torch can pick a low thread count inside a container. Speed only; scores unaffected.
        import os

        import torch

        torch.set_num_threads(os.cpu_count())
    return device


def last_call_stats() -> dict:
    """Timings and query details from this thread's most recent retrieve(). {} if none."""
    return getattr(_last, "stats", {})


def _model_cache(fn):
    """Build the model once, even when several threads ask at the same time.

    `lru_cache` alone lets concurrent first calls each build a copy, and concurrent
    construction segfaults on MPS. Its own lock, not `_gpu_lock`, which is held around model
    calls. The warm path takes no lock.
    """
    cached = lru_cache(maxsize=1)(fn)
    lock = threading.Lock()

    @wraps(fn)
    def get():
        if cached.cache_info().currsize:
            return cached()
        with lock:
            return cached()

    get.loaded = lambda: bool(cached.cache_info().currsize)
    return get


@_model_cache
def _embed_model():
    from sentence_transformers import SentenceTransformer

    # device passed explicitly, or SentenceTransformer picks mps and ignores RAG_SEC_DEVICE.
    return SentenceTransformer(EMBED_MODEL_NAME, revision=EMBED_MODEL_REVISION, device=_device())


@_model_cache
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    device = _device()
    # fp16 on cuda only: 3-4x faster on a T4 with the same top-5. Never tested on mps.
    kwargs = {"model_kwargs": {"torch_dtype": "float16"}} if device.startswith("cuda") else {}
    return CrossEncoder(RERANK_MODEL_NAME, revision=RERANK_MODEL_REVISION, device=device, **kwargs)


def _drain_mps(device: str) -> float:
    """Release MPS's cached memory; returns the seconds it took.

    Without it, on a 16 GB Mac, latency climbs question after question as swap grows.
    """
    t0 = time.perf_counter()
    if device.startswith("mps"):
        import torch

        torch.mps.empty_cache()
    return time.perf_counter() - t0


@contextlib.contextmanager
def _drain_on_error(device: str):
    """Drain after a failed retrieval too, so one error does not slow every later question."""
    try:
        yield
    except BaseException:
        _drain_mps(device)
        raise


def retrieve(
    query: str, k: int = TOP_K, on_stage: Callable[[str, dict], None] | None = None
) -> list[dict]:
    """Top-k chunks for `query`, best first, as {filing_stem, chunk_index, text, score}.

    - Company filter: if the question names a company (read from the question text alone),
      both retrievers search only that company's filings. No match means no filter.
    - Stripped query: the company name and filing wording ("as reported in the 10-K") are
      removed for the dense search and the reranker. Once every candidate is the right
      company, the name only rewards boilerplate-heavy chunks. BM25 keeps the raw question.
    - Years named in the question nudge the fusion (see `candidates.first_stage`).

    Known cost: a right chunk whose own text never names the question's year can drop out
    of the pool.

    `on_stage(name, info)`, if given, is called as each stage starts (for progress display).
    It sees results only; it changes nothing that is computed.
    """
    t = {}
    report = on_stage or (lambda name, info: None)
    device = _device()
    with _drain_on_error(device):
        report("resolve", {})
        t0 = time.perf_counter()
        tickers, fallback_reason = resolve_with_reason(query)
        stripped = strip_entity_framing(query)  # unchanged when no company is found
        query_years = extract_years(query)
        t["resolve_s"] = time.perf_counter() - t0

        # First-call model loading is timed separately so embed_s stays a per-question number.
        if not (_embed_model.loaded() and _cross_encoder.loaded()):
            report("load_models", {})
        t0 = time.perf_counter()
        embed_model = _embed_model()
        cross_encoder = _cross_encoder()
        t["model_init_s"] = time.perf_counter() - t0
        report("embed", {"tickers": list(tickers)})
        t0 = time.perf_counter()
        with _gpu_lock:
            query_emb = embed_model.encode(stripped, normalize_embeddings=True)
        t["embed_s"] = time.perf_counter() - t0

        report("search", {})
        t0 = time.perf_counter()
        with get_conn() as conn:
            fused, texts = first_stage(conn, query_emb, query, tickers=tickers, query_years=query_years)
        t["search_s"] = time.perf_counter() - t0

        candidates = [c for c in fused if c in texts]
        report("rerank", {"candidates": len(candidates)})
        t0 = time.perf_counter()
        pairs = [(stripped, texts[c]) for c in candidates]
        # Batch pairs by length, then restore order: each batch pads to its longest member,
        # so mixing lengths wastes compute. Scores are unaffected.
        order_by_len = sorted(range(len(pairs)), key=lambda j: len(pairs[j][1]))
        sorted_pairs = [pairs[j] for j in order_by_len]
        with _gpu_lock:
            sorted_scores = cross_encoder.predict(sorted_pairs, batch_size=32) if pairs else []
        scores = [0.0] * len(pairs)
        for j, s in zip(order_by_len, sorted_scores):
            scores[j] = float(s)
        order = sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)[:k]
        t["rerank_s"] = time.perf_counter() - t0

        t["mps_empty_cache_s"] = _drain_mps(device)
        t["total_s"] = (t["embed_s"] + t["resolve_s"] + t["search_s"] + t["rerank_s"]
                        + t["mps_empty_cache_s"])
        _last.stats = {
            "timings": t,
            "stripped_query": stripped,
            "tickers": list(tickers),
            "fallback_reason": fallback_reason,
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
