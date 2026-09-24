"""HTTP API: `POST /ask` retrieves with `retrieve()` and answers with Gemini. `POST /ask/stream`
does the same as server-sent events: progress, then sources, then the answer as it is written.
`GET /` serves a one-page chat UI. `/health` and `/ready` for the host.

Run: uv run --env-file .env uvicorn rag_sec.api:app --workers 1
One worker on purpose: two would load two copies of both models, and concurrent model
construction segfaults on Apple MPS.
"""

import json
import os
import queue
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

from rag_sec.answer import generate_answer, parse_answer, stream_answer
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
    value_status: str  # parse_answer's reason: ok, refused, no_number, no_answer_line, empty
    citations: list[Citation]
    latency_s: float
    stage_latency: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both models at boot, so no user request pays for it and a failed load fails the
    # start, not a request. RAG_SEC_WARM=0 skips it.
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


def _citations(req: AskRequest, chunks: list[dict]) -> list[Citation]:
    return [
        Citation(
            filing_stem=c["filing_stem"],
            chunk_index=c["chunk_index"],
            score=c["score"],
            text=c["text"] if req.include_text else None,
        )
        for c in chunks
    ]


def _answer(
    req: AskRequest,
    on_stage: Callable[[str, dict], None] | None = None,
    on_sources: Callable[[list[Citation]], None] | None = None,
    on_text: Callable[[str], None] | None = None,
) -> AskResponse:
    """`on_text` given: Gemini streams, each piece is passed on as it arrives, and
    `generation_first_text_s` is timed. Otherwise one blocking call, as /ask always did."""
    report = on_stage or (lambda name, info: None)
    t0 = time.perf_counter()
    chunks = retrieve(req.question, k=req.k, on_stage=on_stage)
    # model_init_s is left out: it is one-off warm-up.
    timings = last_call_stats().get("timings", {})
    stages = {k: timings[k] for k in STAGES if k in timings}
    if not chunks:
        raise HTTPException(status_code=404, detail="no candidates retrieved")

    citations = _citations(req, chunks)
    if on_sources:
        on_sources(citations)

    report("generate", {})
    t_gen = time.perf_counter()
    try:
        if on_text:
            pieces = []
            for piece in stream_answer(req.question, chunks):
                if not pieces:
                    stages["generation_first_text_s"] = time.perf_counter() - t_gen
                pieces.append(piece)
                on_text(piece)
            text = "".join(pieces)
        else:
            text = generate_answer(req.question, chunks)
    except genai_errors.APIError as e:
        if e.code == 429:
            raise HTTPException(status_code=429, detail="Gemini rate limit reached. The free "
                                "tier allows a few questions a day: try later or use a paid key.")
        raise
    stages["generation_s"] = time.perf_counter() - t_gen
    value, value_status = parse_answer(text)

    return AskResponse(
        question=req.question,
        answer=text,
        value=value,
        value_status=value_status,
        citations=citations,
        latency_s=time.perf_counter() - t0,
        stage_latency=stages,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return _answer(req)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Server-sent events, in order: `stage` as each step starts; `sources` (the citations,
    before Gemini starts); `text` pieces of the answer as Gemini writes them; then `answer`
    (the /ask body) or `error` ({status, detail}). The work runs on its own thread so events
    can be sent while it is still going."""
    events: queue.Queue = queue.Queue()

    def work():
        try:
            resp = _answer(
                req,
                on_stage=lambda name, info: events.put(_sse("stage", {"stage": name, **info})),
                on_sources=lambda cs: events.put(
                    _sse("sources", {"citations": [c.model_dump() for c in cs]})),
                on_text=lambda piece: events.put(_sse("text", {"text": piece})),
            )
            events.put(_sse("answer", resp.model_dump()))
        except HTTPException as e:
            events.put(_sse("error", {"status": e.status_code, "detail": e.detail}))
        except Exception as e:
            events.put(_sse("error", {"status": 500, "detail": f"{type(e).__name__}: {e}"}))
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()
    return StreamingResponse(iter(events.get, None), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})
