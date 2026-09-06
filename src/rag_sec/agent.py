"""LangGraph agentic loop: plan -> retrieve -> judge -> (loop, or answer). Unlike Arms 1-4's
one-shot retrieval it issues a fresh sub-query per iteration and lets a cheap `judge` model
decide when the evidence is enough. spec.md Day 8, DECISIONS.md AGENT-1/AGENT-2.
"""

import json
import operator
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from rag_sec.retrieve import last_call_stats as _last_call_stats
from rag_sec.retrieve import retrieve as _retrieve
from rag_sec.tracing import generation

# Hard stop regardless of judge verdict: AGENT-1 sizes cost for a fixed number of calls.
MAX_ITERATIONS = 4


# Bound to `plan` for its schema only -- `plan` emits the tool call and `retrieve_node`
# executes it, so this body never runs inside the graph. k stays at retrieve()'s own
# default (AGENT-7 checked recall@3/5/10 before keeping 10).
@tool
def retrieve_tool(query: str) -> list[dict]:
    """Search SEC filing chunks (dense + BM25 + reranked) for text relevant to `query`."""
    # Deliberately not executable. This body cannot pass `resolve_from`, because a tool call
    # carries only the planner's rewritten query and not the original question -- so running
    # it would silently restore the AGENT-25 bug it took a trace read to find. Swapping a
    # prebuilt ToolNode in for `retrieve_node` would do exactly that, quietly; this makes it
    # fail loudly instead.
    raise RuntimeError(
        "retrieve_tool is bound for its schema only; retrieve_node executes retrieval so it "
        "can pass resolve_from=state['question'] (AGENT-25)"
    )


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
    # one entry per retrieve call: per-stage timings + the pre-rerank candidates. Kept per
    # iteration (not just the last) so recall@k is scoreable at every step of the trajectory.
    retrieval_stats: Annotated[list[dict], operator.add]


def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
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
    return deduped


def _evidence_text(chunks: list[dict]) -> str:
    # idempotent on an already-deduped list, so callers that need the count can dedupe
    # first and pass the result here without the dedupe rule living in two places
    return "\n\n".join(
        f"[{c['filing_stem']} chunk {c['chunk_index']}] {c['text']}" for c in _dedupe_chunks(chunks)
    )


def _record_usage(node: str, model: str, response, elapsed: float = 0.0, gen=None,
                  keep_text: bool = False) -> dict:
    meta = getattr(response, "usage_metadata", None) or {}
    # Gemini's prefix caching is automatic, and both plan's message history and judge's
    # evidence block are literal shared prefixes across iterations -- so recording
    # cache_read is how we tell it actually fires rather than merely being documented.
    # 3.7 Flash's 4096-token minimum means only loop iterations can ever hit.
    cache_read = (meta.get("input_token_details") or {}).get("cache_read", 0)
    rec = {
        "node": node,
        "model": model,
        # spec.md:121 wants p50/p95 per stage; without this the generate stage has no latency
        "latency_s": round(elapsed, 3),
        "input_tokens": meta.get("input_tokens", 0),
        "cached_input_tokens": cache_read,
        "output_tokens": meta.get("output_tokens", 0),
        "total_tokens": meta.get("total_tokens", 0),
    }
    if keep_text:
        # plan's and judge's completions are otherwise discarded (the verdict collapses to
        # one word), which leaves "why did the judge call this evidence sufficient?"
        # unanswerable from the results file at any effort. Both are short.
        rec["response_text"] = response.text
    if gen is not None:
        # same counters into the span as into the row -- extracted once, above
        gen.set_usage(**{k: rec[k] for k in ("input_tokens", "cached_input_tokens",
                                             "output_tokens", "total_tokens")})
        gen.set(output=response.text)
    return rec


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
    # `medium`, not `low`, to match the static arms' answer calls (COST-34) -- otherwise an
    # Arm 6 vs Arm 3+filter+strip comparison moves the loop and the thinking level at once.
    return ChatGoogleGenerativeAI(model="gemini-3.7-flash", thinking_level="medium")


def plan_node(state: AgentState) -> dict:
    seed = []
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
    # Full accumulated message history (including prior AIMessage tool-call turns)
    # is passed back unmodified -- Gemini 3's "thought signatures" 4xx if a prior
    # AIMessage is stripped or reconstructed instead of replayed as-is.
    prompt = seed or state["messages"]
    iteration = 1 if seed else state["iteration"] + 1

    # one observation, not a span wrapping a generation: the wrapper carried no work of its
    # own, and every extra layer is a node a dashboard has to know about
    with generation(
        "plan-query", model="gemini-3.7-flash", thinking_level="low", iteration=iteration,
        input=[{"role": m.type, "content": m.content} for m in prompt],
    ) as gen:
        t0 = time.perf_counter()
        response = _plan_llm().invoke(prompt)
        el = time.perf_counter() - t0
        usage = _record_usage("plan", "gemini-3.7-flash", response, el, gen=gen, keep_text=True)
        # guarded, not indexed: a missing tool call is retrieve_node's assert to raise,
        # and tracing must not be what turns it into an IndexError here instead
        if response.tool_calls:
            # the planner's actual output is the tool call; `response.text` is empty on a
            # tool-only turn, so _record_usage's output would otherwise read null. This
            # shape is the one Langfuse renders as a tool-call card rather than raw JSON.
            c = response.tool_calls[0]
            gen.set(output={"role": "assistant", "tool_calls": [{
                "id": c["id"], "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}]})
    return {
        # seed is empty after the first iteration, so the reducer then adds only the new turn
        "messages": seed + [response],
        "iteration": iteration,
        "usage": [usage],
    }


def retrieve_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    assert isinstance(last, AIMessage) and last.tool_calls, "plan_node must always emit a tool call"
    call = last.tool_calls[0]
    # untraced here on purpose: retrieve() opens its own `retrieve-chunks` retriever, and a
    # wrapper span around it put two identical observations in the tree at every iteration
    # `resolve_from` is the ORIGINAL question, not the planner's query: the planner's later
    # rewrites drop the company name (and invent a `filing_stem:` syntax nothing parses), so
    # resolving from them silently disabled the company filter on 59.6% of later iterations
    # and searched all 799 filings (AGENT-25). The rewrite still drives the search itself --
    # that is the loop's whole point -- it just no longer decides which company we are in.
    results = _retrieve(call["args"]["query"], resolve_from=state["question"])
    stats = dict(_last_call_stats())
    stats["query"] = call["args"]["query"]
    tool_message = ToolMessage(content=json.dumps(results), tool_call_id=call["id"])
    return {"messages": [tool_message], "retrieved_chunks": results, "retrieval_stats": [stats]}


_JUDGE_PROMPT = """Question: {question}

Retrieved evidence so far:
{evidence}

Is this enough evidence to answer the question fully and correctly? Reply with exactly
one word: "sufficient" or "insufficient"."""


def judge_node(state: AgentState) -> dict:
    deduped = _dedupe_chunks(state["retrieved_chunks"])
    prompt = _JUDGE_PROMPT.format(question=state["question"], evidence=_evidence_text(deduped))
    # `generation`, not the semantically closer `evaluator`: only generation and embedding
    # observations carry model/usage_details, and these calls are ~22% of the arm's spend --
    # typing them as evaluator would delete that from every per-node cost figure.
    # `iteration` alongside the verdict: without both on the same observation Langfuse cannot
    # join them, and cap-termination rate and per-iteration judge accuracy stop being
    # expressible as dashboard widgets at all.
    with generation("judge-sufficiency", model="gemini-3.1-flash-lite",
                    thinking_level="minimal", input=prompt,
                    iteration=state["iteration"],
                    n_evidence_chunks=len(deduped)) as gen:
        t0 = time.perf_counter()
        response = _judge_llm().invoke([HumanMessage(content=prompt)])
        el = time.perf_counter() - t0
        usage = _record_usage("judge", "gemini-3.1-flash-lite", response, el, gen=gen,
                              keep_text=True)
        # response.content can be a list of content parts, not a plain string -- .text
        # normalizes both shapes (confirmed against installed langchain-core, 2026-08-31).
        verdict = "finish" if "insufficient" not in response.text.lower() else "loop"
        gen.set(verdict=verdict)
    return {"judge_verdicts": [verdict], "usage": [usage]}


def route_after_judge(state: AgentState) -> str:
    if state["iteration"] >= MAX_ITERATIONS:
        return "finish"
    return state["judge_verdicts"][-1]


# The trailing format block is COST-23's `FORMAT_LINE`, restated here rather than imported so
# `rag_sec` keeps no dependency on `scripts/`. It is not cosmetic: `answer_eval.parse_reason`
# matches ONLY an `ANSWER:` line, so without it every answer scores `no_answer_line` and both
# arms read 0.0% (measured on 19 rows, 2026-09-04). Its three clauses each fix a scoring bug
# COST-23 hit -- units, an explicit refusal token, and a single-line requirement so
# non-compliance is measured rather than silently absorbed.
_ANSWER_PROMPT = """Question: {question}

Evidence gathered:
{evidence}

Answer the question using only this evidence. If the evidence is insufficient, say so
plainly rather than guessing.

Give the numeric value in the same units as the evidence -- do not expand thousands or
millions. End your response with a single line:
ANSWER: <number>
or, if the evidence does not contain the answer:
ANSWER: INSUFFICIENT"""


def answer_node(state: AgentState) -> dict:
    deduped = _dedupe_chunks(state["retrieved_chunks"])
    prompt = _ANSWER_PROMPT.format(question=state["question"], evidence=_evidence_text(deduped))
    # no keep_text: the completion is already the row's `final_answer`
    with generation("generate-answer", model="gemini-3.7-flash", thinking_level="medium",
                    input=prompt, n_evidence_chunks=len(deduped)) as gen:
        t0 = time.perf_counter()
        response = _answer_llm().invoke([HumanMessage(content=prompt)])
        el = time.perf_counter() - t0
        usage = _record_usage("answer", "gemini-3.7-flash", response, el, gen=gen)
    return {"final_answer": response.text, "usage": [usage]}


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
