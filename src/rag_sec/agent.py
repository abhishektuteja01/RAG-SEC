"""Day 8 -- LangGraph agentic loop: plan -> retrieve -> judge -> (loop back to plan, or
answer). See spec.md Day 8, DECISIONS.md AGENT-1/AGENT-2 for the model choices this
builds on (plan: Gemini 3.7 Flash at thinking_level="low" -- it rejects "minimal" with a
400; judge: Gemini 3.1 Flash-Lite at "minimal").

Unlike Arms 1-4 (one-shot retrieval), this issues a fresh sub-query each iteration and
lets a separate cheap model (`judge`) decide when enough evidence has been gathered --
targets the multi-hop questions Fin-RATE flags as breaking one-shot retrieval.
"""

import json
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from rag_sec.retrieve import retrieve as _retrieve

MAX_ITERATIONS = 4  # hard stop regardless of judge verdict -- bounds cost per spec.md's
# Day 9 budget; DECISIONS.md AGENT-1 sizes cost for a *fixed* number of calls per question


@tool
def retrieve_tool(query: str) -> list[dict]:
    """Search SEC filing chunks (dense + BM25 + reranked) for text relevant to `query`."""
    return _retrieve(query, k=10)


class AgentState(TypedDict):
    question: str
    messages: Annotated[list, add_messages]
    retrieved_chunks: Annotated[list[dict], operator.add]
    iteration: int
    judge_verdicts: Annotated[list[str], operator.add]  # one per iteration, not just the latest --
    # a smoke-test trajectory printout needs to see judge's call at every loop, not just
    # the one that ended it
    final_answer: str
    usage: Annotated[list[dict], operator.add]  # one entry per LLM call: {node, model, input_tokens,
    # output_tokens, total_tokens} -- per-call, not aggregated, so cost/latency can be broken
    # down by node (spec.md Day 9 wants full trajectory metrics, not just a final total)


def _evidence_text(chunks: list[dict]) -> str:
    # Loop iterations often re-retrieve the same chunk (later sub-queries stay on
    # topic) -- dedupe by (filing_stem, chunk_index) so judge/answer aren't billed
    # input tokens for the identical text twice.
    seen = set()
    deduped = []
    for c in chunks:
        key = (c["filing_stem"], c["chunk_index"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return "\n\n".join(f"[{c['filing_stem']} chunk {c['chunk_index']}] {c['text']}" for c in deduped)


def _record_usage(node: str, model: str, response) -> dict:
    meta = getattr(response, "usage_metadata", None) or {}
    # cache_read tracks Gemini's automatic prefix caching (no opt-in needed, confirmed
    # active for both gemini-3.1-flash-lite and gemini-3.7-flash) -- plan's growing
    # message history and judge's growing evidence block are each a literal shared
    # prefix across loop iterations, so cache hits should appear from iteration 2+ on
    # any question that loops, confirming it's actually firing and not just documented.
    # Min threshold is 4096 tokens for 3.7 Flash -- iteration 1 calls (~150-300 tokens
    # measured) are too small to ever hit; only loop iterations can benefit.
    cache_read = (meta.get("input_token_details") or {}).get("cache_read", 0)
    return {
        "node": node,
        "model": model,
        "input_tokens": meta.get("input_tokens", 0),
        "cached_input_tokens": cache_read,
        "output_tokens": meta.get("output_tokens", 0),
        "total_tokens": meta.get("total_tokens", 0),
    }


def _plan_llm():
    # Gemini 3.7 Flash only accepts low/medium/high -- "minimal" 400s (confirmed against
    # ai.google.dev/gemini-api/docs/thinking, 2026-08-31). "low" is the cheapest available
    # level for this model, not a stand-in for AGENT-1's "minimal" intent.
    return ChatGoogleGenerativeAI(model="gemini-3.7-flash", thinking_level="low").bind_tools(
        [retrieve_tool], tool_choice="any"
    )


def _judge_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", thinking_level="minimal")


def _answer_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.7-flash", thinking_level="low")


def plan_node(state: AgentState) -> dict:
    if not state["messages"]:
        seed = [
            SystemMessage(
                content=(
                    "You answer questions about SEC filings using the retrieve_tool. "
                    "Always call retrieve_tool with one focused search query -- on the "
                    "first call, search for the question itself; on later calls, search "
                    "for whatever piece of evidence is still missing given what's already "
                    "been retrieved."
                )
            ),
            HumanMessage(content=state["question"]),
        ]
        response = _plan_llm().invoke(seed)
        return {
            "messages": seed + [response],
            "iteration": 1,
            "usage": [_record_usage("plan", "gemini-3.7-flash", response)],
        }

    # Full accumulated message history (including prior AIMessage tool-call turns)
    # is passed back unmodified -- Gemini 3's "thought signatures" 4xx if a prior
    # AIMessage is stripped or reconstructed instead of replayed as-is.
    response = _plan_llm().invoke(state["messages"])
    return {
        "messages": [response],
        "iteration": state["iteration"] + 1,
        "usage": [_record_usage("plan", "gemini-3.7-flash", response)],
    }


def retrieve_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    assert isinstance(last, AIMessage) and last.tool_calls, "plan_node must always emit a tool call"
    call = last.tool_calls[0]
    results = _retrieve(call["args"]["query"], k=10)
    tool_message = ToolMessage(content=json.dumps(results), tool_call_id=call["id"])
    return {"messages": [tool_message], "retrieved_chunks": results}


_JUDGE_PROMPT = """Question: {question}

Retrieved evidence so far:
{evidence}

Is this enough evidence to answer the question fully and correctly? Reply with exactly
one word: "sufficient" or "insufficient"."""


def judge_node(state: AgentState) -> dict:
    evidence = _evidence_text(state["retrieved_chunks"])
    prompt = _JUDGE_PROMPT.format(question=state["question"], evidence=evidence)
    response = _judge_llm().invoke([HumanMessage(content=prompt)])
    # response.content can be a list of content parts, not a plain string -- .text
    # normalizes both shapes (confirmed against installed langchain-core, 2026-08-31).
    verdict = "finish" if "insufficient" not in response.text.lower() else "loop"
    return {
        "judge_verdicts": [verdict],
        "usage": [_record_usage("judge", "gemini-3.1-flash-lite", response)],
    }


def route_after_judge(state: AgentState) -> str:
    if state["iteration"] >= MAX_ITERATIONS:
        return "finish"
    return state["judge_verdicts"][-1]


_ANSWER_PROMPT = """Question: {question}

Evidence gathered:
{evidence}

Answer the question using only this evidence. If the evidence is insufficient, say so
plainly rather than guessing."""


def answer_node(state: AgentState) -> dict:
    evidence = _evidence_text(state["retrieved_chunks"])
    prompt = _ANSWER_PROMPT.format(question=state["question"], evidence=evidence)
    response = _answer_llm().invoke([HumanMessage(content=prompt)])
    return {
        "final_answer": response.text,
        "usage": [_record_usage("answer", "gemini-3.7-flash", response)],
    }


def build_graph(checkpointer):
    builder = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("judge", judge_node)
    builder.add_node("answer", answer_node)

    builder.set_entry_point("plan")
    builder.add_edge("plan", "retrieve")
    builder.add_edge("retrieve", "judge")
    builder.add_conditional_edges("judge", route_after_judge, {"loop": "plan", "finish": "answer"})
    builder.add_edge("answer", END)

    return builder.compile(checkpointer=checkpointer)
