"""Arm 6 (the LangGraph loop) over a random dev sample -- spec.md Day 9's measured run.

!! NOT FREE. Live Gemini calls at `standard` tier. n=300 costs ~$8 (COST-36 rates).
Resumable: every completed question id is skipped on re-run, so a kill costs nothing.

Designed as a ONE-OFF that also feeds later days, because re-running is real money:
  Day 9  trajectory, tokens, dollars, wall clock, sufficiency-judge accuracy (spec.md:112-117)
  Day 10 per-node Langfuse spans, live: prompts, completions and absolute timestamps exist
         only there -- this file keeps durations (`stage_latency`, `usage`) (spec.md:387)
  Day 11 answers + the chunks that produced them, for citation grounding (spec.md:106)
  Day 13 p50/p95 per stage: embed / search / rerank / generate (spec.md:121)
Model ids and both retrieval flags are written into every row, per spec.md:136 -- a results
file that can't name the models that produced it is not reproducible.

Arm 6 runs on dev only. Test is deliberately untouched: the loop is not the shipped system,
and spending the holdout on a hypothesis that hasn't cleared dev is backwards (COST-30).
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from langchain_core.messages import HumanMessage

from rag_sec.agent import (
    _ANSWER_PROMPT,
    _answer_llm,
    _evidence_text,
    _record_usage,
    MAX_ITERATIONS,
    build_graph,
)
from rag_sec.eval import gold_relevant_chunk_ids, load_matched_questions
from rag_sec.config import pick_device
from rag_sec.candidates import LIVE_VARIANT, chunk_texts
from rag_sec.retrieve import CANDIDATE_K, TOP_K
from rag_sec.store import get_conn
from rag_sec.store import get_conn_string
from rag_sec.tracing import (_FLUSH_TIMEOUT_FINAL_S, flush_tracing, generation,
                             init_tracing, question_trace, trace_id_for)

SEED = 42
ARM = "arm6_loop_vs_static"
# Arm discriminators. Each question emits ONE TRACE PER ARM, grouped by a session whose id is
# the question id: Langfuse aggregates cost over a trace, so a single trace holding both arms
# would report only their sum -- and the difference between them is what this run measures.
ARM_LOOP = "loop"
ARM_STATIC = "static"
OUT_DEFAULT = "data/day9_arm6_dev_results.jsonl"
_write_lock = threading.Lock()


def _trace_context(row) -> dict:
    """Trace-level context, identical on both arms so one dashboard filter covers both.

    Tags are the dimensions a dashboard splits by and are immutable once set; per-question
    values stay metadata, because a tag per question id would mean a new tag every row.
    """
    return {
        "question": row["question"],
        "tags": [f"split:{row['split']}", f"source:{row['subset_source'].lower()}"],
        "split": row["split"],
        "subset_source": row["subset_source"],
        "company_cik": int(row["company_cik"]),
        "report_year": int(row["report_year"]),
        "experiment": ARM,
    }


def _trajectory(result: dict) -> list[dict]:
    """One entry per loop iteration: the query planned, what came back, the verdict."""
    out, msgs, stats = [], result["messages"], result.get("retrieval_stats", [])
    step = 0
    for i, m in enumerate(msgs):
        if not (isinstance(m, AIMessage) and m.tool_calls):
            continue
        chunks = json.loads(msgs[i + 1].content)
        st = stats[step] if step < len(stats) else {}
        verdicts = result.get("judge_verdicts", [])
        out.append({
            "iteration": step + 1,
            "query": m.tool_calls[0]["args"]["query"],
            "rerank_query": st.get("rerank_query"),
            "tickers": st.get("tickers", []),
            "judge_verdict": verdicts[step] if step < len(verdicts) else None,
            "top_k": [[c["filing_stem"], c["chunk_index"], c["score"]] for c in chunks],
            "candidates_prerank": st.get("candidates", []),
            # full reranked order with scores, not just the top 10 the tool returned: MRR is
            # undefined from a truncated list whenever the first relevant chunk sits below k.
            "reranked_all": st.get("reranked", []),
            "stage_latency": st.get("timings", {}),
        })
        step += 1
    return out


STATIC_SCORES = "data/retr7_rr_dev_scores.jsonl"
STATIC_CELL = "filtered_stripped"


def load_static_rankings(path: str = STATIC_SCORES) -> dict:
    """Published Arm 3 + filter + strip rankings, id -> (candidates sorted by DESCENDING
    rerank score, latency).

    The file is NOT stored in rank order: `rerank_hpc.py:132-135` zips the rerank scores onto
    the FIRST-STAGE RRF candidate order, so the score is merely attached (0/1235 dev cells are
    in descending order). Sorting on load is what `rerank_score.py:62` does, i.e. what
    published RETR-39; reading the file as stored made this baseline a first-stage ranking
    instead -- recall@10 0.552 vs 0.739 on the 197-question sample. Keep the two in step.
    """
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ranked = sorted(r["cells"][STATIC_CELL], key=lambda x: -x[2])
            out[r["id"]] = (ranked, r.get("latency_s"))
    return out


def static_baseline(row, rankings) -> dict:
    """Arm 3 + filter + strip on the same question: one answer call over the PUBLISHED ranking.

    Retrieval is NOT re-run. `data/retr7_rr_dev_scores.jsonl` already holds the reranked
    `filtered_stripped` list for every dev question on the post-RETR-7 corpus -- that is the
    exact arm RETR-39 published, so reusing it makes the baseline literally the published one
    instead of a re-run that could drift. It also saves ~28s of reranking per question.

    Only the answer call is new, and it imports `_ANSWER_PROMPT`/`_answer_llm` from the loop
    so the two arms cannot differ in prompt wording or thinking level: the single change
    between them is the loop itself (spec.md:132).
    """
    ranked, ret_latency = rankings[row["id"]]
    top = ranked[:TOP_K]
    keys = [(stem, idx) for stem, idx, _ in top]
    with get_conn() as conn:
        texts = chunk_texts(conn, keys, LIVE_VARIANT)
    chunks = [{"filing_stem": st, "chunk_index": ix, "text": texts[(st, ix)]}
              for st, ix in keys if (st, ix) in texts]

    prompt = _ANSWER_PROMPT.format(question=row["question"], evidence=_evidence_text(chunks))
    # Same observation name as the loop's answer call, deliberately: the two arms sit in
    # separate traces carrying different `arm:` tags, so one dashboard metric now splits by
    # arm. Naming them apart is what would leave nothing to split.
    with question_trace(row["id"], ARM_STATIC, **_trace_context(row)) as tr:
        with generation("generate-answer", model="gemini-3.7-flash",
                        thinking_level="medium", input=prompt) as gen:
            t1 = time.perf_counter()
            resp = _answer_llm().invoke([HumanMessage(content=prompt)])
            el = time.perf_counter() - t1
            usage = _record_usage("answer", "gemini-3.7-flash", resp, el, gen=gen)
        tr.set(output=resp.text)
    return {
        "source": f"{STATIC_SCORES}:{STATIC_CELL} (RETR-39, retrieval not re-run)",
        "final_answer": resp.text,
        "top_k": [[st, ix, sc] for st, ix, sc in top],
        "reranked_all": ranked,
        "retrieval_latency_s": ret_latency,
        "usage": [usage],
        "wall_clock_s": round(el, 3),
    }


def run_one(graph, checkpointer, row, rankings) -> dict:
    qid = row["id"]
    state = {"question": row["question"], "messages": [], "retrieved_chunks": [],
             "iteration": 0, "judge_verdicts": [], "final_answer": "", "usage": [],
             "retrieval_stats": []}
    cfg = {"configurable": {"thread_id": qid}}
    started = time.perf_counter()
    try:
        # The loop's trace CLOSES here, before static_baseline runs. It used to wrap it, which
        # put the static arm's ~2-3s answer call inside the loop root's ~80s span and so
        # overstated Arm 6's latency on every dashboard reading the root's duration.
        with question_trace(qid, ARM_LOOP, **_trace_context(row)) as tr:
            try:
                # thread_id == question id exactly, never a nonce (AGENT-5), so the run is
                # inspectable per question. But resume happens at the FILE level -- any id
                # reaching here is absent from the output, i.e. it never finished -- so its
                # checkpoint is a partial mid-question state, and resuming that is what breaks:
                # LangGraph replays a plan turn whose ToolMessage was never written, and Gemini
                # 400s with "Requests ending with a model turn are not supported". Measured, not
                # theorised: two ids with 5 stale checkpoints each failed 2/2, and the identical
                # question passed on a fresh thread id. So wipe before starting.
                # Inside the try: a transient Postgres blip here must cost one row, not the pass.
                checkpointer.delete_thread(qid)
                t0 = time.perf_counter()
                result = graph.invoke(state, cfg)
                wall = time.perf_counter() - t0
                # trace-level output: it is what the tracing table shows and what an evaluator
                # or a dataset experiment reads off this trace
                tr.set(output=result["final_answer"])
            except Exception as e:
                # set explicitly rather than relying on the re-raise reaching _observation:
                # the row's `error` string and the trace's status_message must be the same text.
                tr.set(level="ERROR", status_message=f"{type(e).__name__}: {e}")
                raise
        # outside the loop trace on purpose: these can raise too, and a failure here means the
        # loop itself finished, so marking its trace ERROR would misreport a successful loop.
        traj = _trajectory(result)
        gold = gold_relevant_chunk_ids(row)
        static = static_baseline(row, rankings)
    except Exception as e:
        # an escape here propagates through fut.result() and kills a run that is hours long.
        # One bad question must cost one row, not the pass.
        # loop trace only: the static arm's trace may never have been opened, and a row
        # pointing at a trace that does not exist is worse than a row without the key
        return {"id": qid, "error": f"{type(e).__name__}: {e}",
                "trace_id": trace_id_for(qid, ARM_LOOP), "session_id": qid,
                "wall_clock_s": round(time.perf_counter() - started, 3)}
    return {
        "id": qid,
        "split": row["split"],
        "subset_source": row["subset_source"],
        "question": row["question"],
        # both gold answer fields: they disagree on 10.1% of dev and the scorer needs each (COST-21)
        "program_answer": row.get("program_answer"),
        "original_answer": row.get("original_answer"),
        "company_cik": int(row["company_cik"]),
        "report_year": int(row["report_year"]),
        "gold_chunk_ids": gold,
        # the only join from a row to its traces. Day 10 reads the worst failures out of
        # this file and then opens each one's trace; without it that is a manual search,
        # and Hobby's 30-day retention means the id outlives the trace it points at.
        # `trace_id` keeps its name and its meaning (the loop arm); the static arm's is
        # additive, and `session_id` is the Langfuse view holding both.
        "trace_id": trace_id_for(qid, ARM_LOOP),
        "trace_id_static": trace_id_for(qid, ARM_STATIC),
        "session_id": qid,
        "final_answer": result["final_answer"],
        "iterations": result["iteration"],
        # "terminated BECAUSE of the cap", not "reached the cap": route_after_judge tests the
        # cap BEFORE the verdict, so a question that reached the last iteration and then got a
        # `finish` verdict would have stopped anyway. Counting it here overstates a published rate.
        "hit_iteration_cap": (result["iteration"] >= MAX_ITERATIONS
                              and result["judge_verdicts"][-1:] == ["loop"]),
        "judge_verdicts": result["judge_verdicts"],
        "trajectory": traj,
        "usage": result["usage"],
        "wall_clock_s": round(wall, 3),
        "static_baseline": static,
        "config": {"arm": ARM, "top_k": TOP_K, "candidate_k": CANDIDATE_K,
                   "max_iterations": MAX_ITERATIONS, "company_filter": True, "strip_query": True,
                   "plan_model": "gemini-3.7-flash/low", "judge_model": "gemini-3.1-flash-lite/minimal",
                   "answer_model": "gemini-3.7-flash/medium", "service_tier": "standard",
                   # stage latencies are device-dependent and spec.md:121 publishes them, so
                   # the device belongs in the row, not just in whoever ran it's memory
                   "device": pick_device()},
    }


def main() -> None:
    init_tracing()
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=300)
    # 1, not 8: the local embedder and reranker are the only shared GPU state and torch's MPS
    # backend segfaults when two threads touch them at once (see retrieve.py's _gpu_lock).
    # The lock serialises the calls, but the models are ~94% of a question's wall clock, so
    # above 1 the workers mostly queue -- and the default of 8 is what took the machine down.
    ap.add_argument("--concurrency", type=int, default=1,
                    help="worker threads (default: %(default)s; >1 is unsafe, see above)")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    df = load_matched_questions()
    pool_df = df[df["split"] == args.split]
    order = pool_df.sample(frac=1.0, random_state=SEED)
    picks = order.head(min(args.n, len(order)))

    # Resume keeps SUCCESSES only. An errored row must not count as done or a transient
    # failure is permanent; the file is rewritten without them so a retry cannot leave two
    # rows for one id (which would make the analysis depend on which one it read last).
    done, kept, dropped = set(), [], 0
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "error" in rec or rec["id"] in done:
                    dropped += 1
                    continue
                done.add(rec["id"])
                kept.append(line)
        if dropped:
            with open(args.out, "w") as f:
                f.writelines(kept)
            print(f"dropped {dropped} errored/duplicate row(s) from {args.out}; they will be retried")
    todo = [r for _, r in picks.iterrows() if r["id"] not in done]
    print(f"{args.split}: {len(picks)} sampled (seed {SEED}), {len(done)} already done, {len(todo)} to run")
    if not todo:
        print("nothing to do")
        return

    # one connection per worker plus headroom; min_size explicit because the pool's own
    # default (4) silently exceeds max_size at low concurrency and raises at construction
    pool = ConnectionPool(get_conn_string(), min_size=1, max_size=args.concurrency + 2,
                          kwargs={"autocommit": True})
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    graph = build_graph(checkpointer)
    rankings = load_static_rankings()

    t0, n_err = time.perf_counter(), 0
    try:
        with open(args.out, "a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(run_one, graph, checkpointer, r, rankings): r["id"] for r in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                n_err += "error" in rec
                with _write_lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                el = time.perf_counter() - t0
                if "error" in rec:
                    print(f"  {i}/{len(todo)} {rec['id']:<18} ERROR {rec['error'][:70]}", flush=True)
                else:
                    calls = len(rec["usage"])
                    print(f"  {i}/{len(todo)} {rec['id']:<18} iters={rec['iterations']} "
                          f"calls={calls} wall={rec['wall_clock_s']:>6.1f}s "
                          f"verdicts={','.join(rec['judge_verdicts'])}", flush=True)
                if i % 10 == 0 or i == len(todo):
                    print(f"     -- {el/60:.1f}min elapsed, ~{el/i*(len(todo)-i)/60:.1f}min left, "
                          f"{n_err} errors", flush=True)
    finally:
        pool.close()
        # spans are batched, so a Ctrl-C or a crash mid-run otherwise drops the tail
        # the generous budget: this fires once after ~4h and holds the only copy of the
        # un-exported tail, so giving up on it early would lose what it exists to save
        flush_tracing(_FLUSH_TIMEOUT_FINAL_S)
    print(f"wrote {args.out} ({n_err} errors) -- re-run the same command to retry any errors")


if __name__ == "__main__":
    main()
