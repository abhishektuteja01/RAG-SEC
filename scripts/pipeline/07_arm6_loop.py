"""Pipeline phase 07 — Arm 6: the LangGraph agentic loop, and its scoring.

THE ONLY PHASE THAT SPENDS MONEY ON EVERY RUN. `run` makes live Gemini calls:
**~$4.10 for 200 questions, ~$0.0205/question** (the measured 2026-09-04/05 pass). It is
therefore gated behind `--allow-paid-run` and refuses to start without it. `analyze` reads
only and is free.

WHAT ARM 6 IS: plan -> retrieve -> judge -> (loop | answer), capped at 4 iterations,
Postgres-checkpointed (`AGENT-4`). Every question is ALSO answered by the static shipped arm
(Arm 3 + company filter + query strip) over the PUBLISHED ranking, in the same process, so
the two arms are paired on question and answer prompt and the only difference is the loop.

PRODUCES
    data/day9_arm6_dev_results.jsonl   one JSON row per question: loop trajectory, judge
                                       verdicts, per-stage latency, token usage, both
                                       arms' answers, and both Langfuse trace ids.
                                       Append-only and resumable per question id.
    analyze                            prints Day 9's metrics; writes nothing.

READS
    the matched question set via rag_sec.eval.load_matched_questions
    data/retr7_rr_dev_scores.jsonl     the published RETR-39 `filtered_stripped` ranking,
                                       read through rag_sec.eval.load_ranking (see TRAPS)
    Postgres `chunks` (variant 'A') for the static arm's evidence text, and Postgres again
    as LangGraph's checkpoint store
    analyze reads only the results file above plus the question set.

DECISIONS.md ROWS THIS BACKS
    AGENT-1   plan = Gemini 3.7 Flash / low, judge = Gemini 3.1 Flash-Lite / minimal,
              answer = 3.7 Flash / medium. The ids are written into every row.
    AGENT-4   the graph shape and the 4-iteration cap; retrieve() reused from Arm 3.
    AGENT-5   `thread_id` == question id exactly, never a nonce, so a run is inspectable
              per question.
    AGENT-8   `cached_input_tokens` is tracked per call, and priced separately here.
    AGENT-10  torch's MPS backend is not thread-safe: SIGSEGV with no traceback and zero
              rows written. RUN SERIAL. The reproducing figure on record is concurrency
              ">1" -- SESSION.md said 4, the code comment said 2, no log survives, so
              nothing more precise than ">1" may be quoted.
    AGENT-15  spec.md:383's multi-document subset is EMPTY on this benchmark (max 1
              filing per question in all three splits), so this phase runs the trajectory
              half and publishes the negative.
    AGENT-16  the static baseline was not the arm it claimed to be. See TRAPS -- this is
              the single most important line in this file.
    AGENT-17  the pre-restart audit: `--concurrency` defaulted to 8 (now 1 and refused
              above it), tracing cannot kill the run, `delete_thread` sits inside the try,
              `hit_iteration_cap` no longer conflates the cap with a judge finish, and the
              union row dedupes.
    AGENT-19  the headline at n=200: answer accuracy 69.2% (loop) vs 61.6% (static),
              discordant 14-1, McNemar exact two-sided p=0.00098, while the loop's
              iteration-1 recall@10 is WORSE (0.703 vs 0.736). Both directions at once
              are the finding, not a contradiction.
    AGENT-21  four yes/no-gold questions are scored wrong in BOTH arms. `analyze` prints
              the sensitivity replay beside the headline, never instead of it.
    AGENT-22  the union retrieval row is NOT comparable to the static row. See TRAPS.
    AGENT-24  `torch.mps.empty_cache()` per retrieval, inherited from
              rag_sec.retrieve.retrieve(). Without it stage latencies drift 2.3x inside
              one process, and spec.md:121 publishes those latencies.
    AGENT-25  the company filter silently switches OFF for 59.6% of later-iteration
              queries. Fixed AFTER this run -- so every number here predates the fix.
    COST-21   two gold answer fields that disagree on 10.1% of dev; both are stored per
              row and `rag_sec.answer_eval` scores against either.
    COST-30   Arm 6 runs on DEV ONLY. The holdout is not spent on a hypothesis that has
              not cleared dev.
    COST-36   the budget envelope. Cost is no longer a design constraint, so the gate
              below exists to prevent an ACCIDENTAL spend, not to ration a scarce budget.
    OBS-10    where the loop's cost goes, on 3 piloted questions: answer 47%, plan 36%,
              judge 18%, at 3.6x the static arm. The full 200-question run's own split is
              answer 64% / judge 19% / plan 17% at 1.77x -- `analyze` prints it, and the
              two disagree because a 3-question pilot has a different iteration mix.
              Quote the 200-question figure; OBS-10's shape (answer is the biggest slice,
              plan replays the whole history) is what survives.
    OBS-13    the Langfuse dashboard only agrees with this file inside the run's own
              window: 2026-09-04 20:10 -> 2026-09-05 02:00.

WHEN THIS ACTUALLY RAN (calendar dates, not "Day N")
    2026-09-04 20:10   the paid Arm 6 pass started (after AGENT-16 stopped an earlier
                       attempt 18 questions in, and AGENT-17's audit)
    2026-09-05 01:55   finished: 200/200 questions, 0 errors, 339.6 min, serial, on mains
                       power. data/day9_arm6_dev_results.jsonl carries that timestamp.
    EVERY FIGURE FROM THAT RUN PREDATES THE AGENT-25 COMPANY-FILTER FIX. The filter was
    off for 59.6% of the loop's later-iteration queries, so the loop was retrieving with
    less than it now has. **69.2% answer accuracy is a FLOOR for the loop, not its
    ceiling** -- and AGENT-21's yes/no replay (71.5%) says the same thing from the
    scoring side. Do not quote 69.2% as what the loop can do.

TRAPS
  * MONEY. `run` needs `--allow-paid-run` or it refuses before constructing a model, a
    pool or a graph. ~$4.10/200 questions. There is no dry-run that costs less; the
    cheap thing to do is re-run `analyze` on the file already on disk.
  * SERIAL ONLY, AND ON MAINS POWER. `--concurrency` is 1 and values above 1 are
    REFUSED, not warned about: concurrent MPS model construction segfaults the machine
    with no traceback and no rows written (AGENT-10 / AGENT-17). Mains power is not
    superstition -- battery throttling moves the very per-stage latencies spec.md:121
    publishes, so a run on battery produces unpublishable timings (AGENT-20 / AGENT-24).
  * THE STATIC RANKING IS NOT STORED IN RANK ORDER. `data/retr7_rr_dev_scores.jsonl`
    holds each cell in FIRST-STAGE RRF order with the rerank score merely attached as the
    third element -- 0 of 1235 dev cells are score-descending. Reading it as stored made
    the baseline "Arm 2 + company filter" instead of the shipped arm, scoring recall@10
    **0.552 against the correct 0.739** on the 197-question sample, and biased the one
    question this phase exists to answer (AGENT-16). This file therefore reads it ONLY
    through `rag_sec.eval.load_ranking`, which sorts on load, with
    `rag_sec.eval.SHIPPED_CELL`. `scripts/checks/static_ranking_order.py` is the CI gate
    that locks both names. Do not open that file with `json.loads` here.
  * THE UNION RETRIEVAL ROW IS NOT COMPARABLE TO THE STATIC ROW (AGENT-22). `analyze`
    scores "all iters (union)" at k=len(got) -- every chunk the loop saw across up to 4
    iterations, so up to 40 slots -- while the static row is recall@10. The printed pair
    0.756 vs 0.736 is a 40-slot budget against a 10-slot one and reads as a loop win; the
    like-for-like pair is iteration-1 0.703 vs static 0.736, a loop LOSS. The union number
    is kept because it answers a different question ("did looping ever surface gold, at
    any depth"), and `analyze` now labels it with its own k and prints the like-for-like
    pair underneath. Separately, the union is built in iteration order, so its first ten
    elements are always iteration 1's top-10 and union nDCG@10 is IDENTICALLY iter-1
    nDCG@10 -- that match is arithmetic, not corroboration, and carries no information.
  * `analyze` also predates nothing and is free, so it is safe to point at a PARTIAL
    results file mid-run.
"""

import argparse
import json
import statistics as st
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402
from psycopg_pool import ConnectionPool  # noqa: E402

from rag_sec.agent import (  # noqa: E402
    MAX_ITERATIONS,
    _ANSWER_PROMPT,
    _answer_llm,
    _evidence_text,
    _record_usage,
    build_graph,
)
from rag_sec.answer_eval import (  # noqa: E402
    answer_line,
    gold_is_scoreable,
    gold_values,
    is_correct,
    parse_reason,
)
from rag_sec.candidates import LIVE_VARIANT, chunk_texts  # noqa: E402
from rag_sec.config import pick_device  # noqa: E402
from rag_sec.eval import (  # noqa: E402
    SHIPPED_CELL,
    gold_relevant_chunk_ids,
    load_matched_questions,
    load_ranking,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from rag_sec.retrieve import CANDIDATE_K, TOP_K  # noqa: E402
from rag_sec.store import get_conn, get_conn_string  # noqa: E402
from rag_sec.tracing import (  # noqa: E402
    _FLUSH_TIMEOUT_FINAL_S,
    flush_tracing,
    generation,
    init_tracing,
    question_trace,
    trace_id_for,
)

# ─── CONSTANTS ──────────────────────────────────────────────────────────────────
DATA_DIR = _ROOT / "data"

# Sampling seed. Fixed, and shared with scripts/archive/latency_measure.py and
# scripts/archive/mps_leak_probe.py, so a latency probe and this run see the SAME questions
# in the same order -- AGENT-20/AGENT-24's stage-latency comparisons depend on that.
SEED = 42

# Sample size. 200 is what the published run scored (AGENT-19); the original default was
# 300 and is kept, because lowering it would silently change what `run` means without
# changing the number anyone quotes. Resume makes the difference cheap either way.
N_DEFAULT = 300

ARM = "arm6_loop_vs_static"

# Arm discriminators. Each question emits ONE TRACE PER ARM, grouped by a session whose id
# is the question id: Langfuse aggregates cost over a trace, so a single trace holding both
# arms would report only their sum -- and the difference between them is what this measures.
ARM_LOOP = "loop"
ARM_STATIC = "static"

RESULTS_DEFAULT = DATA_DIR / "day9_arm6_dev_results.jsonl"

# The paired baseline's source. The published RETR-39 `filtered_stripped` ranking, read
# through rag_sec.eval.load_ranking (which SORTS ON LOAD) with rag_sec.eval.SHIPPED_CELL.
# Both names are imported, never re-spelled: scripts/checks/static_ranking_order.py locks
# this exact call, and reading the file as stored was AGENT-16 (recall@10 0.552 vs 0.739).
STATIC_SCORES = DATA_DIR / "retr7_rr_dev_scores.jsonl"
STATIC_CELL = SHIPPED_CELL

# Measured cost of the published pass, for --help and for the money gate's refusal text.
# From SESSION.md's Day 9 line: $4.10 across 200 questions. Stated per question as well,
# because `run -n` scales linearly -- one loop iteration more or less moves it, but not by
# an order of magnitude (OBS-10: answer 47%, plan 36%, judge 18%).
RUN_COST_TOTAL_USD = 4.10
RUN_COST_N = 200
RUN_COST_PER_Q_USD = 0.0205

# Serial, full stop. 1 is not a default to tune: concurrent CrossEncoder/SentenceTransformer
# construction on MPS segfaults the process (AGENT-10, reproduced at concurrency ">1" and no
# more precisely than that), and lru_cache does not serialise concurrent misses (AGENT-17
# measured 8 threads -> 8 executions). Serial costs nothing anyway: ~94% of a question's
# wall clock is the reranker, which retrieve._gpu_lock serialises regardless.
CONCURRENCY = 1

# Per-million-token prices, USD: (input, output, cached_input). Cached input is ~90% off and
# fires from iteration 2 on any looping question (AGENT-8), so pricing it at the input rate
# would overstate the loop's cost -- which is the number the whole arm is judged on.
PRICE = {
    "gemini-3.7-flash": (0.75, 3.75, 0.075),
    "gemini-3.1-flash-lite": (0.25, 1.50, 0.025),
}

# AGENT-21: gold is numeric 1.0/0.0 on four dev questions whose QUESTION is yes/no, and the
# model answers the word. `rag_sec.answer_eval` requires a number, so both arms score them
# wrong while being right. The mapping lives HERE and not in `answer_eval` on purpose: the
# shipped scorer stays exactly as it was when the run was measured, so the headline is the
# measured one and this is a reported sensitivity beside it, not a metric redefined after
# seeing its own results.
_YESNO = {"yes": 1.0, "no": 0.0}

# Retrieval stage keys printed in this order. model_init_s is one-off warm-up, not
# per-question work, and is excluded from total_s -- printed anyway so it stays visible
# instead of hiding inside embed_s as it once did.
STAGE_KEYS = ("embed_s", "search_s", "rerank_s", "model_init_s")

_write_lock = threading.Lock()


def _rel(path: Path) -> str:
    """Repo-relative, for --help text. Absolute paths in a --help make the help unreadable
    and machine-specific; the constants themselves stay absolute so cwd cannot matter."""
    return str(Path(path).relative_to(_ROOT))


# ─── STEP 1: the money gate ─────────────────────────────────────────────────────
def _refuse_unless_opted_in(args: argparse.Namespace) -> None:
    """`run` spends real money on every invocation, so it must be impossible to start by
    accident. Called FIRST, before any model, pool, graph or ranking is constructed, so a
    refusal costs nothing but a process start.

    Unlike phase 06's summary cache there is no cheap path here: every question is at least
    three live Gemini calls (plan, judge, answer) and there is nothing to hit in a cache.
    COST-36 closed cost as a design constraint, so this gate is about ACCIDENTS, not budget.
    """
    if args.allow_paid_run:
        return
    raise SystemExit(
        "REFUSING TO SPEND: `run` makes live Gemini calls on every question.\n"
        f"  The published pass cost ~${RUN_COST_TOTAL_USD:.2f} for {RUN_COST_N} questions "
        f"(~${RUN_COST_PER_Q_USD:.4f}/question), and -n {args.n} would be about "
        f"${args.n * RUN_COST_PER_Q_USD:.2f}.\n"
        f"  The finished 200-question run is already on disk: {RESULTS_DEFAULT}\n"
        "  Scoring it is free -- `analyze` reads only and never calls an API.\n"
        "  Pass --allow-paid-run if you really mean to buy another pass, and read the "
        "module docstring's TRAPS first (serial only, mains power)."
    )


def _refuse_unless_serial(concurrency: int) -> None:
    """Refused, not warned. AGENT-10's failure mode is SIGSEGV inside torch's Metal shader
    library: the process dies rather than raising, so no retry or except can catch it and a
    paid pass loses every row it had not yet flushed. A warning cannot prevent that.
    """
    if concurrency == 1:
        return
    raise SystemExit(
        f"REFUSING TO RUN: --concurrency {concurrency}. Arm 6 is serial-only.\n"
        "  Torch's MPS backend is not thread-safe and concurrent model construction "
        "SIGSEGVs in\n"
        "  MetalShaderLibrary::exec_unary_kernel with no traceback and zero rows written "
        "(AGENT-10,\n"
        "  reproduced at concurrency >1; the exact figure is not on record and must not be "
        "invented).\n"
        "  Serial costs nothing: ~94% of a question's wall clock is the reranker, which\n"
        "  retrieve._gpu_lock serialises anyway, so extra workers only queue on the lock."
    )


# ─── STEP 2: the loop arm — one question through the LangGraph graph ─────────────
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
        stat = stats[step] if step < len(stats) else {}
        verdicts = result.get("judge_verdicts", [])
        out.append({
            "iteration": step + 1,
            "query": m.tool_calls[0]["args"]["query"],
            "rerank_query": stat.get("rerank_query"),
            "tickers": stat.get("tickers", []),
            "judge_verdict": verdicts[step] if step < len(verdicts) else None,
            "top_k": [[c["filing_stem"], c["chunk_index"], c["score"]] for c in chunks],
            "candidates_prerank": stat.get("candidates", []),
            # full reranked order with scores, not just the top 10 the tool returned: MRR
            # is undefined from a truncated list whenever the first relevant chunk sits
            # below k.
            "reranked_all": stat.get("reranked", []),
            "stage_latency": stat.get("timings", {}),
        })
        step += 1
    return out


# ─── STEP 3: the paired static arm — the PUBLISHED ranking, sorted on load ──────
def load_static_rankings(path: Path = STATIC_SCORES) -> dict:
    """Published Arm 3 + filter + strip rankings, id -> (candidates sorted by DESCENDING
    rerank score, retrieval latency).

    The file is NOT stored in rank order: `rerank_hpc.py:132-135` zips the rerank scores
    onto the FIRST-STAGE RRF candidate order, so the score is merely attached (0/1235 dev
    cells are in descending order). `load_ranking` sorts on load, which is what published
    RETR-39; reading the file as stored made this baseline a first-stage ranking instead --
    recall@10 0.552 vs 0.739 on the 197-question sample (AGENT-16).

    This one call is what `scripts/checks/static_ranking_order.py` locks, down to the two
    imported names and the two flags. Keep them in step.
    """
    return load_ranking(str(path), STATIC_CELL, with_score=True, with_latency=True)


def static_baseline(row, rankings) -> dict:
    """Arm 3 + filter + strip on the same question: one answer call over the PUBLISHED
    ranking.

    Retrieval is NOT re-run. `data/retr7_rr_dev_scores.jsonl` already holds the reranked
    `filtered_stripped` list for every dev question on the post-RETR-7 corpus -- that is
    the exact arm RETR-39 published, so reusing it makes the baseline literally the
    published one instead of a re-run that could drift. It also saves ~28s of reranking per
    question.

    Only the answer call is new, and it imports `_ANSWER_PROMPT`/`_answer_llm` from the loop
    so the two arms cannot differ in prompt wording or thinking level: the single change
    between them is the loop itself (spec.md:132).
    """
    ranked, ret_latency = rankings[row["id"]]
    top = ranked[:TOP_K]
    keys = [(stem, idx) for stem, idx, _ in top]
    with get_conn() as conn:
        texts = chunk_texts(conn, keys, LIVE_VARIANT)
    chunks = [{"filing_stem": st_, "chunk_index": ix, "text": texts[(st_, ix)]}
              for st_, ix in keys if (st_, ix) in texts]

    prompt = _ANSWER_PROMPT.format(question=row["question"],
                                   evidence=_evidence_text(chunks))
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
        "source": f"{_rel(STATIC_SCORES)}:{STATIC_CELL} (RETR-39, retrieval not re-run)",
        "final_answer": resp.text,
        "top_k": [[st_, ix, sc] for st_, ix, sc in top],
        "reranked_all": ranked,
        "retrieval_latency_s": ret_latency,
        "usage": [usage],
        "wall_clock_s": round(el, 3),
    }


def run_one(graph, checkpointer, row, rankings) -> dict:
    """Both arms for one question, into one results row. Never raises: an escape here
    propagates through fut.result() and kills a run that is hours long, so one bad question
    must cost one row, not the pass.
    """
    qid = row["id"]
    state = {"question": row["question"], "messages": [], "retrieved_chunks": [],
             "iteration": 0, "judge_verdicts": [], "final_answer": "", "usage": [],
             "retrieval_stats": []}
    cfg = {"configurable": {"thread_id": qid}}
    started = time.perf_counter()
    try:
        # The loop's trace CLOSES here, before static_baseline runs. It used to wrap it,
        # which put the static arm's ~2-3s answer call inside the loop root's ~80s span and
        # so overstated Arm 6's latency on every dashboard reading the root's duration.
        with question_trace(qid, ARM_LOOP, **_trace_context(row)) as tr:
            try:
                # thread_id == question id exactly, never a nonce (AGENT-5), so the run is
                # inspectable per question. But resume happens at the FILE level -- any id
                # reaching here is absent from the output, i.e. it never finished -- so its
                # checkpoint is a partial mid-question state, and resuming that is what
                # breaks: LangGraph replays a plan turn whose ToolMessage was never
                # written, and Gemini 400s with "Requests ending with a model turn are not
                # supported". Measured, not theorised: two ids with 5 stale checkpoints
                # each failed 2/2, and the identical question passed on a fresh thread id.
                # So wipe before starting. Inside the try: a transient Postgres blip here
                # must cost one row, not the pass (AGENT-17).
                checkpointer.delete_thread(qid)
                t0 = time.perf_counter()
                result = graph.invoke(state, cfg)
                wall = time.perf_counter() - t0
                # trace-level output: it is what the tracing table shows and what an
                # evaluator or a dataset experiment reads off this trace
                tr.set(output=result["final_answer"])
            except Exception as e:
                # set explicitly rather than relying on the re-raise reaching
                # _observation: the row's `error` string and the trace's status_message
                # must be the same text.
                tr.set(level="ERROR", status_message=f"{type(e).__name__}: {e}")
                raise
        # outside the loop trace on purpose: these can raise too, and a failure here means
        # the loop itself finished, so marking its trace ERROR would misreport a success.
        traj = _trajectory(result)
        gold = gold_relevant_chunk_ids(row)
        static = static_baseline(row, rankings)
    except Exception as e:
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
        # both gold answer fields: they disagree on 10.1% of dev and the scorer needs
        # each (COST-21)
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
        # "terminated BECAUSE of the cap", not "reached the cap": route_after_judge tests
        # the cap BEFORE the verdict, so a question that reached the last iteration and
        # then got a `finish` verdict would have stopped anyway. Counting it here
        # overstates a published rate (AGENT-17).
        "hit_iteration_cap": (result["iteration"] >= MAX_ITERATIONS
                              and result["judge_verdicts"][-1:] == ["loop"]),
        "judge_verdicts": result["judge_verdicts"],
        "trajectory": traj,
        "usage": result["usage"],
        "wall_clock_s": round(wall, 3),
        "static_baseline": static,
        "config": {
            "arm": ARM, "top_k": TOP_K, "candidate_k": CANDIDATE_K,
            "max_iterations": MAX_ITERATIONS, "company_filter": True,
            "strip_query": True,
            "plan_model": "gemini-3.7-flash/low",
            "judge_model": "gemini-3.1-flash-lite/minimal",
            "answer_model": "gemini-3.7-flash/medium", "service_tier": "standard",
            # stage latencies are device-dependent and spec.md:121 publishes them, so the
            # device belongs in the row, not just in whoever ran it's memory
            "device": pick_device(),
        },
    }


def cmd_run(args: argparse.Namespace) -> None:
    """The paid pass. Gated, serial, resumable, dev only (COST-30)."""
    _refuse_unless_opted_in(args)
    _refuse_unless_serial(args.concurrency)
    init_tracing()

    out_path = args.out
    df = load_matched_questions()
    pool_df = df[df["split"] == args.split]
    order = pool_df.sample(frac=1.0, random_state=SEED)
    picks = order.head(min(args.n, len(order)))

    # Resume keeps SUCCESSES only. An errored row must not count as done or a transient
    # failure is permanent; the file is rewritten without them so a retry cannot leave two
    # rows for one id (which would make the analysis depend on which one it read last).
    done: set[str] = set()
    kept: list[str] = []
    dropped = 0
    if out_path.exists():
        with open(out_path) as f:
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
            with open(out_path, "w") as f:
                f.writelines(kept)
            print(f"dropped {dropped} errored/duplicate row(s) from {out_path}; "
                  "they will be retried")
    todo = [r for _, r in picks.iterrows() if r["id"] not in done]
    print(f"{args.split}: {len(picks)} sampled (seed {SEED}), {len(done)} already done, "
          f"{len(todo)} to run")
    print(f"ESTIMATED SPEND ~${len(todo) * RUN_COST_PER_Q_USD:.2f} "
          f"(~${RUN_COST_PER_Q_USD:.4f}/question, measured)")
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
        with open(out_path, "a") as fh, \
                ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(run_one, graph, checkpointer, r, rankings): r["id"]
                    for r in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                n_err += "error" in rec
                with _write_lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                el = time.perf_counter() - t0
                if "error" in rec:
                    print(f"  {i}/{len(todo)} {rec['id']:<18} "
                          f"ERROR {rec['error'][:70]}", flush=True)
                else:
                    print(f"  {i}/{len(todo)} {rec['id']:<18} "
                          f"iters={rec['iterations']} calls={len(rec['usage'])} "
                          f"wall={rec['wall_clock_s']:>6.1f}s "
                          f"verdicts={','.join(rec['judge_verdicts'])}", flush=True)
                if i % 10 == 0 or i == len(todo):
                    print(f"     -- {el / 60:.1f}min elapsed, "
                          f"~{el / i * (len(todo) - i) / 60:.1f}min left, "
                          f"{n_err} errors", flush=True)
    finally:
        pool.close()
        # spans are batched, so a Ctrl-C or a crash mid-run otherwise drops the tail. The
        # generous budget: this fires once after ~4h and holds the only copy of the
        # un-exported tail, so giving up on it early would lose what it exists to save.
        flush_tracing(_FLUSH_TIMEOUT_FINAL_S)
    print(f"wrote {out_path} ({n_err} errors) -- re-run the same command to retry errors")


# ─── STEP 4: score the run (free — reads only, never calls an API) ──────────────
def cost_of(usage) -> float:
    """USD for a list of recorded LLM calls, pricing cached input separately (AGENT-8)."""
    tot = 0.0
    for u in usage:
        pin, pout, pcache = PRICE[u["model"]]
        fresh = u["input_tokens"] - u["cached_input_tokens"]
        tot += (fresh * pin + u["cached_input_tokens"] * pcache
                + u["output_tokens"] * pout) / 1e6
    return tot


def dedupe(pairs):
    """First-seen order, no repeats. Loop iterations re-retrieve the same chunk (3/21 pilot
    rows), and ndcg_at_k would then credit one gold chunk twice and inflate nDCG. On the
    pilot every repeat sat past position 10, so its nDCG@10 is unmoved -- this guards the
    case where a repeat lands inside k. Recall and MRR need no dedupe: set membership, and
    first hit (AGENT-17)."""
    return list(dict.fromkeys(pairs))


def pct(x, n) -> str:
    return f"{x}/{n} = {x / n:.1%}" if n else "n/a"


def _percentile(values, q) -> float:
    """Nearest-rank percentile. Deliberately not statistics.quantiles: these samples are
    small and per-stage, and an interpolated p95 would invent a latency no question had."""
    v = sorted(values)
    return v[min(int(q * len(v)), len(v) - 1)]


def _pred(text: str, yesno: bool) -> float | None:
    v, _ = parse_reason(text)
    if v is not None or not yesno:
        return v
    body = answer_line(text)
    if body is None:
        return None
    return _YESNO.get(body.strip().strip(".").casefold())


def _answer_accuracy(rows, yesno: bool) -> None:
    label = "answer accuracy, paired"
    if yesno:
        label += " -- AGENT-21 yes/no sensitivity (NOT the headline)"
    print(f"\n-- {label} --")
    n10 = n01 = both = neither = skipped = 0
    for r in rows:
        if not gold_is_scoreable(r["program_answer"], r["original_answer"]):
            skipped += 1
            continue
        g = gold_values(r["program_answer"], r["original_answer"])
        a = is_correct(_pred(r["final_answer"], yesno), g)
        b = is_correct(_pred(r["static_baseline"]["final_answer"], yesno), g)
        both += a and b
        neither += (not a) and (not b)
        n10 += a and not b
        n01 += b and not a
    scored = both + neither + n10 + n01
    if scored:
        print(f"  scored {scored} ({skipped} excluded, COST-21 unscoreable gold)")
        print(f"  arm6 {(both + n10) / scored:.1%}   static {(both + n01) / scored:.1%}")
        print(f"  discordant: arm6-only {n10}, static-only {n01}  "
              "(McNemar pairs; needs n to interpret)")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Day 9's metrics for a results file. Reads only -- free, and safe on a partial file."""
    rows = [json.loads(line) for line in open(args.results) if line.strip()]
    rows = [r for r in rows if "error" not in r]
    df = load_matched_questions().set_index("id")
    # gold chunk ids are indices WITHIN the gold filing, so a retrieved (stem, idx) only
    # counts when the stem matches -- the convention arm1/arm2/rerank_score all use. The
    # stem is recovered by joining on id rather than stored, so the results file stays small.
    stem_of = {r["id"]: Path(df.loc[r["id"], "chunk_file"]).stem for r in rows}
    n = len(rows)
    print(f"=== Arm 6, {n} questions, {args.results} ===")
    cfg = rows[0]["config"]
    print(f"device={cfg['device']} max_iter={cfg['max_iterations']} "
          f"answer={cfg['answer_model']}")
    print("NOTE: the published run predates the AGENT-25 company-filter fix, so its answer "
          "accuracy is a FLOOR.\n")

    # ---- 1. trajectory (spec.md:112) ----
    iters = [r["iterations"] for r in rows]
    caps = sum(r["hit_iteration_cap"] for r in rows)
    first_loop = sum(1 for r in rows if r["judge_verdicts"][0] == "loop")
    dist = ", ".join(f"{k}:{iters.count(k)}" for k in sorted(set(iters)))
    print("-- trajectory --")
    print(f"iterations: mean {st.mean(iters):.2f}  dist {{{dist}}}")
    print(f"hit cap: {pct(caps, n)}    first-iteration 'insufficient': "
          f"{pct(first_loop, n)}  (COST-30 predicted 15.1%)")

    # ---- 2. cost (spec.md:113) ----
    loop_c = [cost_of(r["usage"]) for r in rows]
    stat_c = [cost_of(r["static_baseline"]["usage"]) for r in rows]
    bynode: dict[str, float] = {}
    for r in rows:
        for u in r["usage"]:
            bynode[u["node"]] = bynode.get(u["node"], 0.0) + cost_of([u])
    tin = sum(u["input_tokens"] for r in rows for u in r["usage"])
    cch = sum(u["cached_input_tokens"] for r in rows for u in r["usage"])
    print("\n-- cost --")
    print(f"arm6 ${st.mean(loop_c):.4f}/q   static ${st.mean(stat_c):.4f}/q   "
          f"ratio {st.mean(loop_c) / st.mean(stat_c):.2f}x")
    print("by node: " + "  ".join(
        f"{k} ${v / n:.4f} ({v / sum(bynode.values()):.0%})"
        for k, v in sorted(bynode.items())))
    print(f"prefix cache: {cch:,}/{tin:,} input tokens = {cch / tin:.1%}  (AGENT-8)")

    # ---- 3. latency (spec.md:121) ----
    stages: dict[str, list] = {}
    for r in rows:
        for t in r["trajectory"]:
            for k, v in t["stage_latency"].items():
                stages.setdefault(k, []).append(v)
    llm: dict[str, list] = {}
    for r in rows:
        for u in r["usage"]:
            llm.setdefault(u["node"], []).append(u["latency_s"])
    print("\n-- latency per stage (s) --")
    for k in STAGE_KEYS:
        if k in stages:
            print(f"  {k:<10} p50 {_percentile(stages[k], .5):>6.2f}  "
                  f"p95 {_percentile(stages[k], .95):>6.2f}")
    for k in sorted(llm):
        print(f"  {k:<10} p50 {_percentile(llm[k], .5):>6.2f}  "
              f"p95 {_percentile(llm[k], .95):>6.2f}   (LLM)")
    w = [r["wall_clock_s"] for r in rows]
    print(f"  {'question':<10} p50 {_percentile(w, .5):>6.1f}  "
          f"p95 {_percentile(w, .95):>6.1f}")

    # ---- 4. sufficiency-judge accuracy (spec.md:115) ----
    print("\n-- sufficiency-judge accuracy --")
    cells = {"stop_with_gold": 0, "stop_without_gold": 0,
             "loop_with_gold": 0, "loop_without_gold": 0}
    for r in rows:
        gold = {(stem_of[r["id"]], i) for i in r["gold_chunk_ids"]}
        seen: set = set()
        for t in r["trajectory"]:
            seen |= {(a, b) for a, b, _ in t["top_k"]}
            have = bool(gold & seen)
            v = t["judge_verdict"]
            if v == "finish":
                cells["stop_with_gold" if have else "stop_without_gold"] += 1
            elif v == "loop":
                cells["loop_with_gold" if have else "loop_without_gold"] += 1
    tot = sum(cells.values())
    correct = cells["stop_with_gold"] + cells["loop_without_gold"]
    print(f"  verdicts scored: {tot}   judge correct: {pct(correct, tot)}")
    print(f"  stopped WITH gold (right)      {cells['stop_with_gold']:>4}")
    print(f"  stopped WITHOUT gold (early)   {cells['stop_without_gold']:>4}  "
          "<- answers a question it cannot")
    print(f"  looped WITHOUT gold (right)    {cells['loop_without_gold']:>4}")
    print(f"  looped WITH gold (wasted)      {cells['loop_with_gold']:>4}  "
          "<- pays for evidence it already had")

    # ---- 5. retrieval (spec.md 2.1) ----
    # THE k COLUMN IS NOT CONSTANT DOWN THIS TABLE AND THAT IS THE POINT (AGENT-22). The
    # union row is scored at k=len(got) -- every chunk the loop saw across up to 4
    # iterations, so up to 40 slots -- because it answers "did looping ever surface gold, at
    # any depth". It therefore CANNOT be compared with the two recall@10 rows: the published
    # pair 0.756 (union) vs 0.736 (static) is a 40-slot budget against a 10-slot one. The
    # like-for-like pair is iteration-1 vs static, and it is a loop LOSS. The k each row
    # used is printed so the table cannot be misread off the page. Also by construction:
    # the union is built in iteration order, so its first ten elements are always iteration
    # 1's top-10 and its nDCG@10 is IDENTICALLY iter-1's -- that column carries no
    # information for the union row and is printed only to keep the rows aligned.
    print("\n-- retrieval: does looping add gold? --")
    print("   (recall k differs by row -- see the k= column; AGENT-22)")
    getters = (
        ("arm6 iter-1 only", "iter1",
         lambda r: [(a, b) for a, b, _ in r["trajectory"][0]["top_k"]]),
        ("arm6 all iters (union)", "union",
         lambda r: dedupe((a, b) for t in r["trajectory"] for a, b, _ in t["top_k"])),
        ("static (RETR-39)", "static",
         lambda r: [(a, b) for a, b, _ in r["static_baseline"]["top_k"]]),
    )
    means = {}
    for label, key, getter in getters:
        r10, nd, mr, ks = [], [], [], []
        for r in rows:
            rel = [(stem_of[r["id"]], i) for i in r["gold_chunk_ids"]]
            if not rel:
                continue
            got = getter(r)
            # k=len(got) for the union row ONLY, preserving what the published number
            # computed; every other row is a true recall@10.
            k = len(got) if key == "union" else 10
            ks.append(k)
            r10.append(recall_at_k(got, rel, k))
            nd.append(ndcg_at_k(got, rel, 10))
            mr.append(mrr(got, rel))
        means[key] = st.mean(r10)
        kdesc = "10" if key != "union" else f"len(got), max {max(ks)}"
        print(f"  {label:<24} recall {st.mean(r10):.3f} (k={kdesc})   "
              f"nDCG@10 {st.mean(nd):.3f}   MRR {st.mean(mr):.3f}  (n={len(r10)})")
    print(f"  LIKE-FOR-LIKE, both at k=10: arm6 iter-1 {means['iter1']:.3f} vs static "
          f"{means['static']:.3f} -- this is the comparable pair (AGENT-19).")
    print(f"  NOT COMPARABLE: union {means['union']:.3f} vs static {means['static']:.3f} "
          "-- up to 40 slots against 10 (AGENT-22). Do not quote it as a loop win.")
    print("  union nDCG@10 == iter-1 nDCG@10 by construction (union is in iteration "
          "order); it is arithmetic, not corroboration.")

    # ---- 6. answer accuracy, paired (spec.md 2.1 / COST-21) ----
    for yesno in (False, True):
        _answer_accuracy(rows, yesno)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "run", help="THE PAID PASS. Live Gemini calls. Needs --allow-paid-run.",
        description=(
            "Step 1-3: run the LangGraph loop and the paired static arm over a random dev "
            "sample, one JSON row per question.\n\n"
            f"MONEY: this spends on EVERY question. The published pass cost "
            f"~${RUN_COST_TOTAL_USD:.2f} for {RUN_COST_N} questions "
            f"(~${RUN_COST_PER_Q_USD:.4f}/question) across three Gemini calls per "
            "iteration (plan, judge, answer) plus one for the static arm. There is no "
            "cache to hit and no cheaper mode. It REFUSES TO START without "
            "--allow-paid-run.\n\n"
            f"The finished 200-question run is already on disk as "
            f"{_rel(RESULTS_DEFAULT)}; scoring it with `analyze` is free.\n\n"
            "OPERATIONAL RULES, both load-bearing:\n"
            "  * SERIAL. --concurrency is 1 and anything higher is refused: concurrent "
            "MPS model construction SIGSEGVs the process with no traceback and no rows "
            "written (AGENT-10, at concurrency >1).\n"
            "  * ON MAINS POWER. Battery throttling moves the per-stage latencies "
            "spec.md:121 publishes, so a run on battery produces unpublishable "
            "timings.\n\n"
            "Dev only (COST-30). Resumable: completed ids are skipped, errored rows are "
            "dropped and retried."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--allow-paid-run", action="store_true",
                   help=f"required. Permits live Gemini calls: ~$"
                        f"{RUN_COST_PER_Q_USD:.4f}/question, "
                        f"~${RUN_COST_TOTAL_USD:.2f} for {RUN_COST_N}")
    p.add_argument("-n", type=int, default=N_DEFAULT,
                   help="questions to sample (default: %(default)s; the published run "
                        f"scored {RUN_COST_N})")
    p.add_argument("--concurrency", type=int, default=CONCURRENCY,
                   help="worker threads (default: %(default)s; anything else is REFUSED, "
                        "see AGENT-10)")
    p.add_argument("--out", type=Path, default=RESULTS_DEFAULT,
                   help=f"append-only results file (default: {_rel(RESULTS_DEFAULT)})")
    p.add_argument("--split", default="dev",
                   help="(default: %(default)s; test is deliberately untouched, COST-30)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser(
        "analyze", help="score a results file into Day 9's metrics (free)",
        description=(
            "Step 4: trajectory, cost, per-stage p50/p95, sufficiency-judge accuracy, "
            "retrieval and paired answer accuracy.\n\nReads only -- no API, no GPU, no "
            "database writes -- so it is free to re-run and safe to point at a PARTIAL "
            "results file mid-pass.\n\n"
            "TWO THINGS IT PRINTS THAT MUST NOT BE MISQUOTED:\n"
            "  * The union retrieval row is scored at k=len(got) (up to 40 chunks) while "
            "the static row is recall@10, so the two are NOT comparable (AGENT-22). The "
            "like-for-like pair is printed underneath.\n"
            "  * The whole run predates the AGENT-25 company-filter fix, so 69.2% answer "
            "accuracy is a FLOOR for the loop, not its ceiling. AGENT-21's yes/no replay "
            "is printed beside the headline, never instead of it."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=RESULTS_DEFAULT,
                   help=f"(default: {_rel(RESULTS_DEFAULT)})")
    p.set_defaults(fn=cmd_analyze)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
