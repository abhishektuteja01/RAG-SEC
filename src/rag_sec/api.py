"""HTTP API: `POST /ask` retrieves with `retrieve()`'s defaults (the best measured setup) and
answers with Gemini. `GET /` serves a one-page chat UI. `/health` and `/ready` for the host.

Run: uv run --env-file .env uvicorn rag_sec.api:app --workers 1
One worker on purpose: two would load two copies of both models, and concurrent model
construction segfaults on Apple MPS.
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_sec.answer import dedupe_chunks, generate_answer, parse_answer
from rag_sec.retrieve import TOP_K, last_call_stats, retrieve
from rag_sec.store import get_conn

MAX_K = 50  # the reranker never sees more than the 50-candidate pool anyway
STATIC_DIR = Path(__file__).parent / "static"
STAGES = ("resolve_s", "embed_s", "search_s", "rerank_s", "mps_empty_cache_s", "total_s")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=TOP_K, ge=1, le=MAX_K)
    include_text: bool = False


class Citation(BaseModel):
    filing_stem: str
    chunk_index: int
    score: float
    text: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    value: float | None
    citations: list[Citation]
    latency_s: float
    stage_latency: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model construction is ~19s. Pay it at boot so no user request carries it, and so a
    # failed model load fails the deploy, not a request. RAG_SEC_WARM=0 skips it.
    if os.environ.get("RAG_SEC_WARM", "1") == "1":
        retrieve("warmup", k=1)
    yield


app = FastAPI(title="rag-sec", version="0.2.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """Liveness only -- no DB, no models."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: Postgres reachable AND the corpus is the expected one (`get_conn` runs
    store.preflight), so a wrong or half-loaded corpus fails here, not in answers."""
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"not ready: {type(e).__name__}: {e}")
    return {"status": "ready"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    chunks = dedupe_chunks(retrieve(req.question, k=req.k))
    # A whitelist: last_call_stats() also holds the full candidate list, and the lock-wait
    # keys are already inside embed_s/rerank_s, and model_init_s is one-off warm-up.
    timings = last_call_stats().get("timings", {})
    stages = {k: timings[k] for k in STAGES if k in timings}
    if not chunks:
        raise HTTPException(status_code=404, detail="no candidates retrieved")

    t_gen = time.perf_counter()
    text = generate_answer(req.question, chunks)
    stages["generation_s"] = time.perf_counter() - t_gen
    value, _ = parse_answer(text)

    return AskResponse(
        question=req.question,
        answer=text,
        value=value,
        citations=[
            Citation(
                filing_stem=c["filing_stem"],
                chunk_index=c["chunk_index"],
                score=c["score"],
                text=c["text"] if req.include_text else None,
            )
            for c in chunks
        ],
        latency_s=time.perf_counter() - t0,
        stage_latency=stages,
    )
