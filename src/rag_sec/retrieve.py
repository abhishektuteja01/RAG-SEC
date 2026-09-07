"""The shipping retrieval path: dense + BM25/RRF + cross-encoder rerank, query in and
ranked chunks out. Same pipeline scripts/pipeline/05_arm3_rerank.py evaluates offline
(DECISIONS.md ARM2-1/ARM3-1/RETR-31); candidate generation itself lives in
`rag_sec.candidates`, shared with every offline arm rather than re-implemented (RETR-36).
"""

import contextlib
import threading
import time
from functools import lru_cache, wraps

from rag_sec.candidates import LIVE_VARIANT, bm25, chunk_texts, dense, rrf_fuse
from rag_sec.company import resolve as resolve_companies
from rag_sec.company import strip_entity_framing
from rag_sec.config import EMBED_MODEL_NAME, RERANK_MODEL_NAME, pick_device
from rag_sec.store import get_conn
from rag_sec.tracing import embedding, retriever, span

TOP_K = 10
CANDIDATE_K = 50

# Per-stage timings + the pre-rerank candidate list from the most recent retrieve() call on
# THIS thread. Stashed rather than returned so the signature stays what every offline arm and
# the LangGraph tool already call (RETR-36). Day 9 reads it after each call: spec.md:121 wants
# p50/p95 split embed/search/rerank, and recall@50 per iteration needs the candidates that
# reranking then drops. Thread-local because the Day 9 runner drives questions concurrently.
_last = threading.local()

# Torch's MPS backend is NOT thread-safe -- full account in config.pick_device(). Every local
# model call goes through this lock. Concurrency is still worth having: retrieval is ~2s of
# GPU against LLM calls of ~3-15s, and those stay parallel.
_gpu_lock = threading.Lock()


@lru_cache(maxsize=1)
def _device() -> str:
    """pick_device() re-imports torch and re-probes the backends on every call. Harmless
    once per model load, but span attributes put it on retrieve()'s hot path -- and span
    arguments are evaluated even when tracing is a no-op, so an untraced run would pay it
    too. The backend cannot change within a process, so caching it is free."""
    return pick_device()


def last_call_stats() -> dict:
    """Timings and candidate list from this thread's most recent retrieve(). {} if none."""
    return getattr(_last, "stats", {})


def _model_cache(fn):
    """`lru_cache(maxsize=1)` plus a construction lock, i.e. build-once even under threads.

    lru_cache alone does NOT serialise concurrent misses -- measured, 8 threads produced 8
    executions -- and construction happens outside `_gpu_lock`, so at the old --concurrency 8
    default eight CrossEncoder copies loaded onto MPS simultaneously. That is the SIGSEGV
    above, reproduced on 2026-09-04.

    Its own lock, not `_gpu_lock`: that one is a non-reentrant Lock held around every model
    call, so a construction path reached from inside it would self-deadlock.

    Double-checked, because this sits on retrieve()'s per-iteration path and the warm case
    must not take a mutex: the size check is a plain cache hit, and any thread that loses the
    race re-checks under the lock and gets the hit.
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

    # device passed explicitly: without it SentenceTransformer auto-picks mps and silently
    # ignores RAG_SEC_DEVICE, so half the pipeline would stay on the backend we're avoiding.
    return SentenceTransformer(EMBED_MODEL_NAME, device=_device())


@_model_cache
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    device = _device()
    return CrossEncoder(RERANK_MODEL_NAME, device=device)


def _drain_mps(device: str) -> float:
    """Release MPS's cached blocks. Returns the elapsed seconds so the caller can publish it.

    Without this, embed_s climbs 0.11s -> 7.7s and rerank_s 27s -> 48s over ~10 questions
    while swap grows 2.3 GB -> 8.4 GB, on an ONLY-process machine -- and it is not a leak in
    the usual sense: torch.mps.current_allocated_memory() stays pinned at 4542 MB throughout,
    so nothing is retained; the caching allocator simply never returns freed blocks to a
    16 GiB unified-memory system that needs them back. Measured both ways over 25 questions
    (AGENT-24, scripts/archive/mps_leak_probe.py).

    `startswith`, not `==`: RAG_SEC_DEVICE is returned verbatim by `pick_device()`, and a
    perfectly valid `mps:0` would otherwise turn the drain off silently and regress latency
    to the saturated curve with no error and no log line.

    Imported here because config.py keeps torch lazy for the chunking/eval paths that never
    touch a GPU.
    """
    t0 = time.perf_counter()
    if device.startswith("mps"):
        import torch

        torch.mps.empty_cache()
    return time.perf_counter() - t0


@contextlib.contextmanager
def _drain_on_error(device: str):
    """Drain on the way out of a FAILED retrieval.

    The success path drains inline and times it into `mps_empty_cache_s`. An exception --
    from `cross_encoder.predict`, `chunk_texts`, or the DB -- would otherwise skip the drain
    entirely, and `07_arm6_loop.py` catches per-question errors and keeps going, so one
    transient failure mid-pass silently restores the saturation curve for every question
    after it, inflating the stage latencies that pass then publishes. No timing is recorded
    here: a failed call has no per-question latency worth publishing.
    """
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
    scripts/pipeline/05_arm3_rerank.py measured. Alone it is +0.015, but
    with the filter it is +0.145: once every candidate is the right company, the company
    name only rewards whichever chunk repeats the most boilerplate (RETR-6).

    `resolve_from` is the text the company filter resolves against, defaulting to `query`.
    It exists because an agent rewrites the query between iterations and its rewrites drop
    the company's name, at which point `resolve_companies` returns nothing, the filter
    quietly takes the unfiltered branch across all 799 filings, and the results still look
    entirely normal -- measured at 59.6% of the loop's later iterations (`AGENT-25`). Callers
    that hold the ORIGINAL question should pass it here, so a rewrite cannot disable a
    retrieval feature by choosing different words. Default `None` keeps every non-agent
    caller on exactly the previous behaviour.

    `reserve` keeps that many candidate slots for unfiltered results, as insurance against
    the one failure mode that can delete gold: a question naming an acquired business or a
    counterparty instead of the filer ("the FIS Gaming Business" in a Global Payments
    filing). Measured at 0.0% of dev and 0.20% of train, so it defaults off.
    """
    t = {}
    device = _device()
    with _drain_on_error(device), \
            retriever("retrieve-chunks", input=query, k=k, company_filter=company_filter,
                      strip_query=strip_query, reserve=reserve,
                      # recorded so a trace states whether AGENT-25's fix was active: without
                      # it, pre- and post-fix traces are indistinguishable on the one
                      # attribute that changed
                      resolve_from=resolve_from) as root:
        with embedding("embed-query", model=EMBED_MODEL_NAME, input=query, device=device) as sp:
            # First-call model construction is timed SEPARATELY, not inside embed_s: it lands
            # in whichever question runs first and is ~19.0s against 0.114s for the identical
            # warm encode, and spec.md:121 publishes embed_s as a p50/p95. It is recorded
            # rather than dropped so the warm-up stays visible.
            # NOTE embed_s therefore changed meaning here: pre-fix numbers are not comparable.
            t0 = time.perf_counter()
            embed_model = _embed_model()
            cross_encoder = _cross_encoder()
            t["model_init_s"] = time.perf_counter() - t0
            # _gpu_lock wait is split out from GPU compute: both land inside embed_s/rerank_s,
            # so at concurrency >1 a slow stage is otherwise indistinguishable from a queued
            # one -- and the reranker holding this lock is ~94% of a run's wall clock.
            t0 = time.perf_counter()
            with _gpu_lock:
                compute0 = time.perf_counter()
                query_emb = embed_model.encode(query, normalize_embeddings=True)
                compute_s = time.perf_counter() - compute0
            t["embed_s"] = time.perf_counter() - t0
            t["embed_lock_wait_s"] = compute0 - t0
            # dim, not the vector: 1024 floats per iteration is noise, but a wrong width
            # is the one silent failure here -- it makes dense search return nothing useful
            sp.set(output={"dim": len(query_emb)}, lock_wait_s=t["embed_lock_wait_s"],
                   compute_s=compute_s)

        # input is what we RESOLVE from, not the search query: tracing `query` here while
        # resolving from `resolve_from` is what would make AGENT-25's own 59.6% analysis
        # unreproducible, since it pairs this span's input against its `tickers` output.
        with span("resolve-company", input=resolve_from or query) as sp:
            t0 = time.perf_counter()
            tickers = resolve_companies(resolve_from or query) if company_filter else []
            rerank_query = strip_entity_framing(query) if strip_query else query
            t["resolve_s"] = time.perf_counter() - t0
            sp.set(output={"tickers": list(tickers), "rerank_query": rerank_query},
                   stripped=rerank_query != query)

        with retriever("search-candidates", input=query, candidate_k=CANDIDATE_K,
                       filtered=bool(tickers), reserve=reserve) as sp:
            t0 = time.perf_counter()
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
            t["search_s"] = time.perf_counter() - t0
            # ids only: the chunk text this stage fetched is what the answer generation's
            # input already carries in full, so repeating it here just inflates every trace
            sp.set(output=[[c[0], int(c[1])] for c in fused], lists_fused=len(lists))

        candidates = [c for c in fused if c in texts]
        # `rerank_model`, not `model`: only generation/embedding observations have a model
        # field, and the SDK accepts the kwarg on the others and then drops it silently.
        with retriever("rerank-candidates", input=rerank_query, rerank_model=RERANK_MODEL_NAME,
                       device=device, pairs=len(candidates), k=k) as sp:
            t0 = time.perf_counter()
            pairs = [(rerank_query, texts[c]) for c in candidates]
            wait0 = time.perf_counter()
            with _gpu_lock:
                compute0 = time.perf_counter()
                scores = cross_encoder.predict(pairs, batch_size=32) if pairs else []
                compute_s = time.perf_counter() - compute0
            order = sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)[:k]
            t["rerank_s"] = time.perf_counter() - t0
            t["rerank_lock_wait_s"] = compute0 - wait0
            top = [[candidates[j][0], int(candidates[j][1]), float(scores[j])] for j in order]
            sp.set(output=top, lock_wait_s=t["rerank_lock_wait_s"], compute_s=compute_s)

        t["mps_empty_cache_s"] = _drain_mps(device)
        # Counted in total_s, unlike model_init_s: a caller pays this on every question, so
        # excluding it would publish a per-question latency the system does not actually
        # deliver. The lock-wait keys stay out -- they are already inside embed_s/rerank_s.
        t["total_s"] = (t["embed_s"] + t["resolve_s"] + t["search_s"] + t["rerank_s"]
                        + t["mps_empty_cache_s"])
        _last.stats = {
            "timings": t,
            "rerank_query": rerank_query,
            "tickers": list(tickers),
            # pre-rerank order, so recall@50 and the reranker's own contribution stay recoverable
            "candidates": [[c[0], int(c[1])] for c in candidates],
            "reranked": [[candidates[j][0], int(candidates[j][1]), float(scores[j])]
                         for j in sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)],
        }
        root.set(output=top, returned=len(order), total_s=t["total_s"])

        return [
            {
                "filing_stem": candidates[j][0],
                "chunk_index": candidates[j][1],
                "text": texts[candidates[j]],
                "score": float(scores[j]),
            }
            for j in order
        ]
