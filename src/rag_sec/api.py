"""HTTP serving layer for the shipped arm: Arm 3 + company filter + query strip.

Not Arm 6. The loop is an experiment (`AGENT-15`), and its own early signal says it adds
no retrieval; `retrieve()`'s defaults already ARE the winning `filtered_stripped` cell,
so serving it needs no arm selection here.

The answer call imports `_ANSWER_PROMPT`/`_answer_llm`/`_evidence_text` from `agent`, the
same three `scripts/eval/agent_run.py:static_baseline` imports. That is deliberate: it
makes the deployed system identical to the arm the published 0.747 was measured on, with
retrieval computed live instead of replayed from `retr7_rr_dev_scores.jsonl`.
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from rag_sec.agent import _ANSWER_PROMPT, _answer_llm, _dedupe_chunks, _evidence_text
from rag_sec.answer_eval import parse_reason
from rag_sec.retrieve import TOP_K, last_call_stats, retrieve
from rag_sec.store import get_conn

MAX_K = 50  # a request cannot ask the reranker for more work than the eval ever measured


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
    # Model construction is ~19s and lands in whichever request runs first otherwise
    # (retrieve.py times it separately for exactly that reason). Pay it at boot so no
    # user request carries it, and so a failed model load fails the deploy, not a request.
    if os.environ.get("RAG_SEC_WARM", "1") == "1":
        retrieve("warmup", k=1)
    yield


app = FastAPI(title="rag-sec", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness only -- no DB, no models. The container is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: Postgres reachable AND the corpus is the one the numbers were measured
    on. `get_conn` runs store.preflight, so a wrong or half-loaded corpus fails here
    rather than silently serving degraded retrieval.
    """
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"not ready: {type(e).__name__}: {e}")
    return {"status": "ready"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    chunks = _dedupe_chunks(retrieve(req.question, k=req.k))
    # `_s` keys only: last_call_stats() also carries the full candidate list, which is the
    # retriever's internals and would be several hundred KB on a k=50 response.
    stages = {k: v for k, v in last_call_stats().items() if k.endswith("_s")}
    if not chunks:
        raise HTTPException(status_code=404, detail="no candidates retrieved")

    prompt = _ANSWER_PROMPT.format(question=req.question, evidence=_evidence_text(chunks))
    response = _answer_llm().invoke([HumanMessage(content=prompt)])
    value, _ = parse_reason(response.text)

    return AskResponse(
        question=req.question,
        answer=response.text,
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
