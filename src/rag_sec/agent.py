"""LangGraph agentic loop: plan -> retrieve -> judge -> (loop, or answer). Unlike Arms 1-4's
one-shot retrieval it issues a fresh sub-query per iteration and lets a cheap `judge` model
decide when the evidence is enough. spec.md Day 8, DECISIONS.md AGENT-1/AGENT-2.
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

# Hard stop regardless of judge verdict: AGENT-1 sizes cost for a fixed number of calls.
MAX_ITERATIONS = 4


# Bound to `plan` for its schema only -- `plan` emits the tool call and `retrieve_node`
# executes it, so this body never runs inside the graph. k stays at retrieve()'s own
# default (AGENT-7 checked recall@3/5/10 before keeping 10).
@tool
def retrieve_tool(query: str) -> list[dict]:
    """Search SEC filing chunks (dense + BM25 + reranked) for text relevant to `query`."""
    return _retrieve(query)


class AgentState(TypedDict):
    question: str
    messages: Annotated[list, add_messages]
    retrieved_chunks: Annotated[list[dict], operator.add]
    iteration: int
    # one verdict per iteration, not just the last: a trajectory printout wants every call
    judge_verdicts: Annotated[list[str], operator.add]
    final_answer: str
    # one entry per LLM call, unaggregated, so cost/latency break down by node (spec.md Day 9)
    usage: Annotated[list[dict], operator.add]


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
    # Gemini's prefix caching is automatic, and both plan's message history and judge's
    # evidence block are literal shared prefixes across iterations -- so recording
    # cache_read is how we tell it actually fires rather than merely being documented.
    # 3.7 Flash's 4096-token minimum means only loop iterations can ever hit.
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
    results = _retrieve(call["args"]["query"])
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
