"""!! NOT FREE, NOT DRY. Despite the name "smoke test", every run of this file makes live,
paid Gemini calls (about $0.04 a run) and writes LangGraph checkpoint rows into the
Postgres database. There is no dry-run flag and no confirmation prompt -- importing and
calling it does the spending. Read the numbers below before you run it.

Filed here, not with the compression scripts, because it exercises the deferred agentic
loop (`src/rag_sec/agent.py`, `AGENT-*`), not the cost/compression work those are about.
It was previously filed next to them in the since-removed `scripts/compression/`, among
seven unrelated files; they now live in `scripts/archive/` as the `slice_*` scripts.

Manual smoke test for Day 8's agentic loop (src/rag_sec/agent.py) -- runs a handful
of dev-split questions through the compiled graph, one thread_id each, and prints the
trajectory (each iteration's search query, top retrieved chunk stems, judge verdict) so
a human can eyeball whether the loop is doing something sensible before Day 9's full run.

Since run end-to-end on 3 dev questions with a live key and DB: all 3 answered correctly,
none looped (OBS-10, which priced the loop off that sample). No Gemini spend on the graph
since. The full dev pass it was written to precede has never been run -- Day 8 turned into
retrieval work instead.
"""

import json
import time
import uuid

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres import PostgresSaver

from rag_sec.agent import build_graph
from rag_sec.eval import load_matched_questions
from rag_sec.store import get_conn_string

N_QUESTIONS = 3


def summarize_trajectory(messages: list) -> list[tuple[str, list[str]]]:
    """Pairs each plan-issued query with the top chunk stems the retrieve node got back."""
    trajectory = []
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            query = msg.tool_calls[0]["args"]["query"]
            chunks = json.loads(messages[i + 1].content)
            stems = [f"{c['filing_stem']}#{c['chunk_index']}" for c in chunks[:3]]
            trajectory.append((query, stems))
    return trajectory


def main() -> None:
    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    picks = dev.head(N_QUESTIONS)

    # This script is for repeated manual dev-loop testing, not the real Day 9 eval pass --
    # each run gets a fresh thread_id (question id + a run-specific nonce) so a prior
    # crashed/buggy attempt's checkpointed state never leaks into a later run (see
    # DECISIONS.md: reusing bare question ids as thread_id across script runs caused
    # finqa_dev_0's checkpoint history to double and its trajectory to include a stray,
    # never-judged extra query). The real eval script must NOT do this -- it needs thread_id
    # == question id exactly, unchanged across runs, so a killed full-dev-set pass can
    # resume via Postgres checkpointing instead of restarting -- the whole point of
    # checkpointing the loop at all.
    run_id = uuid.uuid4().hex[:8]

    with PostgresSaver.from_conn_string(get_conn_string()) as checkpointer:
        checkpointer.setup()
        graph = build_graph(checkpointer)

        for _, row in picks.iterrows():
            print(f"\n=== {row['id']} ===")
            print(row["question"])

            config = {"configurable": {"thread_id": f"{row['id']}-smoketest-{run_id}"}}
            initial_state = {
                "question": row["question"],
                "messages": [],
                "retrieved_chunks": [],
                "iteration": 0,
                "judge_verdicts": [],
                "final_answer": "",
                "usage": [],
                # every key AgentState declares, matching 07_arm6_loop.run_one exactly: two
                # entry points building different initial states is how a node ends up
                # reading a key one of them never seeded.
                "retrieval_stats": [],
            }
            start = time.perf_counter()
            result = graph.invoke(initial_state, config)
            elapsed = time.perf_counter() - start

            for step, (query, stems) in enumerate(summarize_trajectory(result["messages"]), start=1):
                verdict = result["judge_verdicts"][step - 1] if step - 1 < len(result["judge_verdicts"]) else "n/a"
                print(f"  iter {step}: query={query!r} top_chunks={stems} judge={verdict}")

            print(f"  iterations used: {result['iteration']}")
            print(f"  final answer: {result['final_answer']}")
            print(f"  wall clock: {elapsed:.2f}s")
            for u in result["usage"]:
                print(
                    f"    [{u['node']:6s}] {u['model']:22s} in={u['input_tokens']:5d} "
                    f"(cached={u['cached_input_tokens']:5d}) out={u['output_tokens']:5d}"
                )
            total_in = sum(u["input_tokens"] for u in result["usage"])
            total_out = sum(u["output_tokens"] for u in result["usage"])
            print(f"  total tokens: in={total_in} out={total_out} ({len(result['usage'])} LLM calls)")


if __name__ == "__main__":
    main()
