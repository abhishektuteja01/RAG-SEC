"""The shipping retrieval path: dense + BM25/RRF + cross-encoder rerank, query in and
ranked chunks out. Same pipeline scripts/pipeline/05_arm3_rerank.py evaluates offline
(DECISIONS.md ARM2-1/ARM3-1/RETR-31); candidate generation itself lives in
`rag_sec.candidates`, shared with every offline arm rather than re-implemented (RETR-36).
"""

import contextlib
import threading
import time
from collections import Counter
from functools import lru_cache, wraps

from rag_sec.candidates import (  # noqa: F401
    CANDIDATE_K,
    LIVE_VARIANT,
    READ_DEPTH,
    READ_DEPTH_MAX,
    SERVING_READ_DEPTH,
    bm25,
    chunk_texts,
    first_stage,
    dense,
    rrf_fuse,
    rrf_fuse_year_biased,
)
from rag_sec.company import resolve_with_reason
from rag_sec.company import strip_entity_framing
from rag_sec.config import EMBED_MODEL_NAME, RERANK_MODEL_NAME, pick_device
from rag_sec.fiscal_year import extract_years
from rag_sec.store import get_conn
from rag_sec.tracing import embedding, retriever, span

TOP_K = 10

# Per-stage timings + the pre-rerank candidate list from the most recent retrieve() call on
# THIS thread. Stashed rather than returned so the signature stays what every offline arm and
# the LangGraph tool already call (RETR-36). Day 9 reads it after each call: latency is
# published as p50/p95 split embed/search/rerank, and recall@50 per iteration needs the
# candidates that reranking then drops. Thread-local because the Day 9 runner drives questions concurrently.
_last = threading.local()

# Torch's MPS backend is NOT thread-safe -- full account in config.pick_device(). Every local
# model call goes through this lock. Concurrency is still worth having: retrieval is ~2s of
# GPU against LLM calls of ~3-15s, and those stay parallel.
_gpu_lock = threading.Lock()

# How often company_filter's fallback-to-unfiltered-search path fires, broken out by reason
# (research.md sec5b / sec7 item 2). Observability only: nothing here changes what
# `resolve_with_reason` returns or which candidates get searched.
_fallback_lock = threading.Lock()
_fallback_reasons: Counter[str] = Counter()


def fallback_reason_counts() -> dict[str, int]:
    """Snapshot of `_fallback_reasons` -- read after a run, never gated on."""
    with _fallback_lock:
        return dict(_fallback_reasons)


@lru_cache(maxsize=1)
def _device() -> str:
    """pick_device() re-imports torch and re-probes the backends on every call. Harmless
    once per model load, but span attributes put it on retrieve()'s hot path -- and span
    arguments are evaluated even when tracing is a no-op, so an untraced run would pay it
    too. The backend cannot change within a process, so caching it is free."""
    device = pick_device()
    if device == "cpu":
        # Graviton3 has no hyperthreads, so physical == logical core count -- explicit only
        # because torch's own default intraop thread count can come in lower than that under
        # a container runtime (cgroup CPU quota misread, or an inherited OMP_NUM_THREADS), and
        # there is no way to see which happened short of setting it ourselves. Config only:
        # does not touch a score, so nothing here needs re-validating against gold labels.
        import os

        import torch

        torch.set_num_threads(os.cpu_count())
    return device


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
    # fp16 on cuda only: DEPLOY-21 measured a further 3.14-4.22x on top of fp32-GPU
    # (g5g.xlarge and g4dn.xlarge respectively), exact top-5 parity both times, deltas
    # 1e-6 to 1e-3. Needs sentence-transformers>=6.0.0 (pinned in pyproject.toml) -- an
    # older release computed the CrossEncoder sigmoid in half precision and could silently
    # reorder top candidates. Not extended to cpu/mps: bf16-on-cpu is a separate, already
    # -measured lever (DNNL_DEFAULT_FPMATH_MODE, DEPLOY-20), and mps fp16 was never tested.
    kwargs = {"model_kwargs": {"torch_dtype": "float16"}} if device.startswith("cuda") else {}
    return CrossEncoder(RERANK_MODEL_NAME, device=device, **kwargs)


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
    year_bias: bool = True,
    strip_dense: bool = True,
    read_depth: int = SERVING_READ_DEPTH,
    year_text_fusion: bool = True,
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
    the company's name, at which point `resolve_with_reason` returns nothing, the filter
    quietly takes the unfiltered branch across all 799 filings, and the results still look
    entirely normal -- measured at 59.6% of the loop's later iterations (`AGENT-25`). Callers
    that hold the ORIGINAL question should pass it here, so a rewrite cannot disable a
    retrieval feature by choosing different words. Default `None` keeps every non-agent
    caller on exactly the previous behaviour.

    `reserve` keeps that many candidate slots for unfiltered results, as insurance against
    the one failure mode that can delete gold: a question naming an acquired business or a
    counterparty instead of the filer ("the FIS Gaming Business" in a Global Payments
    filing). Measured at 0.0% of dev and 0.20% of train, so it defaults off.

    `year_bias` blends an additive year-proximity nudge into RRF fusion, targeting sibling-
    year confusion (same company, wrong fiscal year -- the single largest slice of first-
    stage misses, `RETR-3`). Soft, not a filter: a missed/absent year extraction just adds
    a zero bonus, it never removes a candidate. Confirmed with a real HPC rerank pass, not
    just the candidate-pool proxy: recall@10 dev 0.760->0.791 (+3.1pt), test 0.747->0.771
    (+2.4pt), both baselines matching the published `RETR-39` numbers exactly (`RETR-40`).
    Defaults on; the flag stays so earlier arms remain reproducible from this same code path.

    `strip_dense` (N18d) embeds the STRIPPED query for the dense leg instead of the raw one,
    leaving BM25 on the raw question. `read_depth` is how many rows each leg asks for before
    fusion cuts back to `CANDIDATE_K`; `year_text_fusion` (P10) adds a third RRF list of
    candidates whose chunk TEXT names a fiscal year the question names, which is what decides
    which `read_depth` candidates survive into the pool. P10 is worth nothing at depth 50 --
    there is nothing extra to choose between -- so the measured arm is all three together:
    `strip_dense=True, read_depth=200, year_text_fusion=True`.

    All three default ON since `DEPLOY-25`, and the pool -- so the rerank bill -- is unchanged
    either way. They were off while the +5.8pt behind them had only been measured offline;
    what unblocked the flip was the missing half, a latency measurement on the serving host.
    Paired over 110 test questions in the deployed container, turning all three on costs
    `total_s` a mean -0.011s, 95% CI [-0.051, +0.031] -- indistinguishable from zero. The only
    effect that clears zero is +0.031s of `search_s` [+0.019, +0.043], which is the
    `extract_years` scan over the pre-cut union, partly paid for by `chunk_texts_exact` being
    CHEAPER than the per-filing fetch it replaces (10.5ms vs 26.9ms filtered).

    `year_text_fusion` is NOT free of accuracy risk per question, only on aggregate: against
    `deployed` it wins 109 test questions and **loses 15**. One demonstrated pathway for a
    loss is a gold chunk whose own TEXT never restates the question's year -- it is excluded
    from the third list while most of the union qualifies, and the demotion pushes it past the
    50-cut (`finqa_test_1`: base rank 20 -> 58, out of the pool, answer 14.46 correct ->
    INSUFFICIENT). That pathway does NOT explain the losses in general: across the 15, only
    3 of 19 gold chunks fail the year test, so the other losses are ordinary RRF reordering.
    Do not quote it as the characterisation of the failure mode; it is one worked example.
    Kept because +1.7pt over `RETR-51` is measured on 1545 scored test questions; `DEPLOY-25`.

    Depth 200 is free because HNSW visits `ef_search` candidates regardless of `LIMIT`: dense
    measured 55.6ms at depth 50 against 50.9ms at 200. That it returns a FULL 200 rows was
    verified by counting them, not inferred from the LIMIT -- a short read would have made
    this feature look free while doing nothing, which is `RETR-50`'s defect exactly.

    Note the rerank scores are zipped onto first-stage order, so a replay that forgets to sort
    reads 0.5668 and looks like a catastrophic regression (`AGENT-16`).
    """
    if read_depth > READ_DEPTH_MAX:
        # store.HNSW_EF_SEARCH is derived from READ_DEPTH_MAX, so a deeper read is one the
        # HNSW index was not configured to fill: `dense()` would quietly return short rather
        # than error, which is RETR-50 exactly. Fail loudly instead of retrieving less.
        raise ValueError(
            f"read_depth={read_depth} exceeds READ_DEPTH_MAX={READ_DEPTH_MAX}; raise it in "
            "candidates.py so store.HNSW_EF_SEARCH follows, then re-run scripts/checks/short_limit.py"
        )
    t = {}
    device = _device()
    with _drain_on_error(device), \
            retriever("retrieve-chunks", input=query, k=k, company_filter=company_filter,
                      strip_query=strip_query, reserve=reserve, year_bias=year_bias,
                      strip_dense=strip_dense, read_depth=read_depth,
                      year_text_fusion=year_text_fusion,
                      # recorded so a trace states whether AGENT-25's fix was active: without
                      # it, pre- and post-fix traces are indistinguishable on the one
                      # attribute that changed
                      resolve_from=resolve_from) as root:
        # input is what we RESOLVE from, not the search query: tracing `query` here while
        # resolving from `resolve_from` is what would make AGENT-25's own 59.6% analysis
        # unreproducible, since it pairs this span's input against its `tickers` output.
        with span("resolve-company", input=resolve_from or query) as sp:
            t0 = time.perf_counter()
            if company_filter:
                tickers, fallback_reason = resolve_with_reason(resolve_from or query)
            else:
                tickers, fallback_reason = [], None
            if fallback_reason:
                with _fallback_lock:
                    _fallback_reasons[fallback_reason] += 1
            # aliases_from, not `query`: the name is looked up in the ORIGINAL question for
            # the same reason the two calls around this one use resolve_from (AGENT-35). The
            # rewritten query is still what gets stripped and searched.
            rerank_query = (
                strip_entity_framing(query, aliases_from=resolve_from or query)
                if strip_query
                else query
            )
            # resolve_from, not query: same reason resolve_with_reason uses it -- an agent's
            # rewritten query can drop a year the same way AGENT-25 found it drops the
            # company name, and this must not silently go quiet.
            # `or year_text_fusion`: P10 reads the same extracted years, so gating them on
            # year_bias alone would make `year_text_fusion=True, year_bias=False` a SILENT
            # no-op -- the feature would run, find no years, and change nothing, with no error.
            query_years = (extract_years(resolve_from or query)
                           if (year_bias or year_text_fusion) else [])
            # N18d. `rerank_query` is already computed on every call and, until now, spent on
            # the cross-encoder alone while both first-stage legs took the raw question. Once
            # the company filter has fired, every candidate IS that company, so its name is a
            # token they all share: it cannot discriminate between them, but a dense query is
            # ONE fixed-length vector and every token still steers its direction. Removing it
            # points the vector at the part that separates chunks. BM25 deliberately keeps the
            # raw question -- that pairing is what was measured, and in OR-mode those tokens
            # decide which documents qualify at all, so stripping them there costs reach.
            # A no-op when nothing resolves: `strip_entity_framing` returns the question
            # unchanged, which is why this only ever removes a token every candidate shares.
            dense_query = rerank_query if strip_dense else query
            t["resolve_s"] = time.perf_counter() - t0
            sp.set(output={"tickers": list(tickers), "rerank_query": rerank_query,
                            "query_years": query_years, "fallback_reason": fallback_reason},
                   stripped=rerank_query != query, dense_stripped=dense_query != query)

        with embedding("embed-query", model=EMBED_MODEL_NAME, input=dense_query,
                       device=device) as sp:
            # First-call model construction is timed SEPARATELY, not inside embed_s: it lands
            # in whichever question runs first and is ~19.0s against 0.114s for the identical
            # warm encode, and embed_s is published as a p50/p95. It is recorded
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
                query_emb = embed_model.encode(dense_query, normalize_embeddings=True)
                compute_s = time.perf_counter() - compute0
            t["embed_s"] = time.perf_counter() - t0
            t["embed_lock_wait_s"] = compute0 - t0
            # dim, not the vector: 1024 floats per iteration is noise, but a wrong width
            # is the one silent failure here -- it makes dense search return nothing useful
            sp.set(output={"dim": len(query_emb)}, lock_wait_s=t["embed_lock_wait_s"],
                   compute_s=compute_s)

        with retriever("search-candidates", input=query, candidate_k=CANDIDATE_K,
                       read_depth=read_depth, year_text_fusion=year_text_fusion,
                       filtered=bool(tickers), reserve=reserve) as sp:
            t0 = time.perf_counter()
            with get_conn() as conn:
                # candidates.first_stage, not a copy of the loop here: the offline arms build
                # the same pool, and when this lived inline they drifted -- see its docstring.
                fused, texts, n_lists = first_stage(
                    conn, query_emb, query, tickers=tickers, variant=LIVE_VARIANT,
                    read_depth=read_depth, pool_k=CANDIDATE_K, query_years=query_years,
                    year_bias=year_bias, year_text_fusion=year_text_fusion, reserve=reserve,
                )
            t["search_s"] = time.perf_counter() - t0
            # ids only: the chunk text this stage fetched is what the answer generation's
            # input already carries in full, so repeating it here just inflates every trace
            sp.set(output=[[c[0], int(c[1])] for c in fused], lists_fused=n_lists)

        candidates = [c for c in fused if c in texts]
        # `rerank_model`, not `model`: only generation/embedding observations have a model
        # field, and the SDK accepts the kwarg on the others and then drops it silently.
        with retriever("rerank-candidates", input=rerank_query, rerank_model=RERANK_MODEL_NAME,
                       device=device, pairs=len(candidates), k=k) as sp:
            t0 = time.perf_counter()
            pairs = [(rerank_query, texts[c]) for c in candidates]
            # Sort by pair length before batching, unsort the scores after: `predict`'s
            # batch_size=32 groups pairs in whatever order `candidates` handed them, and
            # HuggingFace's padding=True pads every member of a batch to its longest --
            # so one long outlier in an otherwise-short batch inflates every pair beside
            # it for free. Sorting first only changes which pairs share a batch, never a
            # pair's own (query, text) content, so scores are unaffected -- this is
            # reordering compute, not re-scoring anything.
            order_by_len = sorted(range(len(pairs)), key=lambda j: len(pairs[j][1]))
            sorted_pairs = [pairs[j] for j in order_by_len]
            wait0 = time.perf_counter()
            with _gpu_lock:
                compute0 = time.perf_counter()
                sorted_scores = cross_encoder.predict(sorted_pairs, batch_size=32) if pairs else []
                compute_s = time.perf_counter() - compute0
            scores = [0.0] * len(pairs)
            for j, s in zip(order_by_len, sorted_scores):
                scores[j] = float(s)
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
            # what the DENSE leg actually embedded: with strip_dense on it is no longer
            # `query`, and a stored candidate dump is not reproducible without it
            "dense_query": dense_query,
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
