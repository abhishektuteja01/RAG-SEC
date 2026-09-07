"""The 20 worst failures per arm from the Arm 6 run (spec.md:2.2 rule 5), written as
markdown to be read cold. Reads only -- no API call, no retrieval, no Postgres -- so it is
free and safe to re-run while `agent_run.py` is still appending rows.

"Worst" is a CHOICE, not a measurement. Ranking failures by wrongness alone is useless here
because every failure is equally wrong (a number either matches gold or it does not), and it
would mix three unrelated bugs into one list. So failures are ordered by HOW CLOSE the gold
evidence got to the answer model, worst first:

  1. reasoning  -- gold chunk was in the top-k the model was given, and the answer is still
                   wrong. Worst, because nothing upstream can be blamed: retrieval and
                   reranking both did their job and the answer stage burned the evidence.
  2. rerank     -- gold chunk was in the 50 candidates but never in the top-k. Fixable at the
                   reranker without touching first-stage recall.
  3. retrieval  -- gold chunk never entered the candidate pool. The answer stage never had a
                   chance, so it says nothing about the model.

Within a class, ties break on the gold's best rerank rank ascending (gold ranked #1 and still
answered wrong is a sharper failure than gold ranked #10), then iterations descending (more
compute burned for the same nothing), then id ascending so the file does not churn between
runs. Any other defensible ordering would produce a different list; this one is picked because
its top entries are the ones we can act on.

Two files, one per arm, per spec.md's "per arm" -- a merged file would sort a loop failure
against a static failure on a key that means different things in each arm's pipeline.
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag_sec import eval as E
from rag_sec.answer_eval import (SCALES, gold_is_scoreable, gold_values, is_correct,
                                 parse_reason)

CLASS_ORDER = {"reasoning": 0, "rerank": 1, "retrieval": 2, "unlabeled": 3}
CLASS_WHY = {
    "reasoning": "gold chunk WAS in the top-k -- answer stage failed on evidence it had",
    "rerank": "gold chunk was in the 50 candidates but the reranker never put it in the top-k",
    "retrieval": "gold chunk never entered the candidate pool -- first-stage recall miss",
    "unlabeled": "no gold_chunk_ids for this question, so the failure cannot be located",
}
INF = 10**9


def load_rows(path: str) -> list[dict]:
    """`agent_run.py:257` rewrites the file on resume dropping errored and duplicate rows, so
    ids are unique on disk; the dict keyed by id only guards against reading mid-rewrite.
    A JSONDecodeError is the last line being half-flushed by the live run, not corruption."""
    rows = {}
    for line in open(path):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in r:  # transient failure, not a retrieval/answer failure -- runner retries it
            continue
        rows[r["id"]] = r
    return sorted(rows.values(), key=lambda r: r["id"])


def best_rank(ranked, gold: set) -> int | None:
    """1-based position of the highest-ranked gold chunk, None if absent."""
    for i, item in enumerate(ranked, 1):
        if (item[0], item[1]) in gold:
            return i
    return None


def why_wrong(pred, reason: str, golds: list[float]) -> str:
    if reason != "ok":
        return {"refused": "declined to answer (ANSWER line says insufficient/unknown)",
                "no_answer_line": "answer did not follow the ANSWER: format, so nothing was scored",
                "no_number": "ANSWER line carried no parseable number",
                "empty": "empty answer"}[reason]
    g = " or ".join(f"{v:g}" for v in golds) or "none"
    # SCALES is quoted so the reader knows the miss is not a units artifact -- the scorer
    # already accepted x100/x1000/x1e6 before calling this wrong.
    return f"answered {pred:g}, gold {g}; no factor in {tuple(SCALES)} brings them within 1%"


def arm_view(row: dict, arm: str, gold: set) -> dict:
    """Per-arm ranking view. `reranked_all` holds the same 50 candidates as
    `candidates_prerank`, so pool membership can be read off either; the first-stage rank is
    also reported for the loop arm because a large gap between it and the rerank rank IS the
    reranker's fault.

    The rerank rank is computed on `reranked_all` re-sorted by score, not on its stored order.
    Arm 6 writes it already sorted, but the static arm's rankings come from
    `retr7_rr_dev_scores.jsonl`, whose cells are stored in first-stage order with rerank
    scores attached -- `rerank_score.py:62` sorts before scoring, `agent_run.py:118` does not.
    So for the static arm "delivered" (what the answer model saw) and "rerank rank" differ,
    and both are printed rather than one silently standing in for the other."""
    if arm == "static":
        s = row["static_baseline"]
        pool = [sorted(s["reranked_all"], key=lambda x: -x[2])]
        prerank = []
        topk = [s["top_k"]]
        answer = s["final_answer"]
    else:
        pool = [sorted(t["reranked_all"], key=lambda x: -x[2]) for t in row["trajectory"]]
        prerank = [t["candidates_prerank"] for t in row["trajectory"]]
        topk = [t["top_k"] for t in row["trajectory"]]
        answer = row["final_answer"]
    ranks_topk = [best_rank(t, gold) for t in topk]
    ranks_pool = [best_rank(p, gold) for p in pool]
    ranks_pre = [best_rank(p, gold) for p in prerank]
    hit_topk = [r for r in ranks_topk if r]
    hit_pool = [r for r in ranks_pool if r]
    cls = "unlabeled" if not gold else "reasoning" if hit_topk else "rerank" if hit_pool else "retrieval"
    return {"answer": answer, "topk": topk, "cls": cls,
            "rank_topk": min(hit_topk) if hit_topk else None,
            "rank_pool": min(hit_pool) if hit_pool else None,
            "rank_pre": min([r for r in ranks_pre if r], default=None)}


def fmt_ranked(ranked, gold: set, n: int) -> str:
    out = []
    for stem, idx, score in ranked[:n]:
        mark = " <-GOLD" if (stem, idx) in gold else ""
        out.append(f"{stem}#{idx} {score:.3f}{mark}")
    return "; ".join(out)


def trace_line(row: dict, arm: str) -> str:
    key = "trace_id_static" if arm == "static" else "trace_id"
    tid, sid = row.get(key), row.get("session_id")
    if not tid:
        # finqa_dev_753 predates tracing; an empty link would look like a broken trace.
        return "**Langfuse**: not traced (row predates tracing)"
    return f"**Langfuse**: trace_id `{tid}`" + (f"  session_id `{sid}`" if sid else "  (no session_id)")


def render(row: dict, view: dict, gold: set, arm: str, rank: int, golds: list[float]) -> str:
    pred, reason = parse_reason(view["answer"])
    gold_txt = ", ".join(f"{s}#{i}" for s, i in sorted(gold)) or "(none labelled)"
    L = [f"## {rank}. `{row['id']}` — {view['cls']}",
         f"*{CLASS_WHY[view['cls']]}*", "",
         f"**Question**: {row['question']}", "",
         f"**Gold**: program_answer `{row['program_answer']}` | original_answer "
         f"`{row['original_answer']}` | gold chunks {gold_txt}",
         f"**Scored wrong because**: {why_wrong(pred, reason, golds)}", ""]

    where = [f"delivered at top-k position {view['rank_topk']}" if view["rank_topk"] else "absent from top-k",
             f"rerank rank {view['rank_pool']}" if view["rank_pool"] else "absent from the 50 candidates"]
    if arm == "arm6" and view["rank_pre"]:
        where.append(f"first-stage rank {view['rank_pre']}")
    L += [f"**Gold chunk was**: {', '.join(where)}", ""]

    for i, t in enumerate(view["topk"], 1):
        label = f"**Retrieved (iter {i})**" if arm == "arm6" else "**Retrieved**"
        L.append(f"{label}: {fmt_ranked(t, gold, 10)}")
    L.append("")

    if arm == "arm6":
        judge_texts = [u.get("response_text", "") for u in row["usage"] if u["node"] == "judge"]
        for i, t in enumerate(row["trajectory"]):
            raw = judge_texts[i] if i < len(judge_texts) else ""
            L.append(f"**Iter {i + 1}** query `{t['query']}` | rerank_query `{t['rerank_query']}` | "
                     f"tickers {t['tickers']} | judge `{t['judge_verdict']}` raw `{raw.strip() or '(empty)'}`")
        L.append(f"**Stopped on**: {'iteration cap' if row['hit_iteration_cap'] else 'judge said finish'} "
                 f"after {row['iterations']} iteration(s)")
        L.append("")

    L += [trace_line(row, arm), "", "**Answer given**:", "```",
          view["answer"].strip() or "(empty)", "```", "", "---", ""]
    return "\n".join(L)


def write_arm(path: Path, arm: str, rows: list[dict], stems: dict, top: int, src: str) -> tuple[int, dict]:
    fails, skipped = [], 0
    for r in rows:
        if not gold_is_scoreable(r["program_answer"], r["original_answer"]):
            skipped += 1  # COST-21: no defensible target, so it is not a failure either way
            continue
        golds = gold_values(r["program_answer"], r["original_answer"])
        gold = {(stems[r["id"]], i) for i in r["gold_chunk_ids"]}
        v = arm_view(r, arm, gold)
        if is_correct(parse_reason(v["answer"])[0], golds):
            continue
        fails.append((r, v, gold, golds))
    fails.sort(key=lambda f: (CLASS_ORDER[f[1]["cls"]], f[1]["rank_pool"] or INF,
                              -f[0]["iterations"], f[0]["id"]))
    counts = {}
    for _, v, _, _ in fails:
        counts[v["cls"]] = counts.get(v["cls"], 0) + 1
    cfg = rows[0]["config"]
    head = [f"# Worst failures — {arm}", "",
            f"Source `{src}` ({len(rows)} rows read, {skipped} excluded as unscoreable gold "
            f"per COST-21), {len(fails)} failures, showing {min(top, len(fails))}.",
            f"Config: top_k={cfg['top_k']} candidate_k={cfg['candidate_k']} "
            f"max_iterations={cfg['max_iterations']} answer={cfg['answer_model']} "
            f"judge={cfg['judge_model']} plan={cfg['plan_model']}",
            f"Classes among all {len(fails)} failures: "
            + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"), "",
            *(["**Read the ranks carefully for this arm**: the delivered top-k is the stored "
               "order of `retr7_rr_dev_scores.jsonl`, which is first-stage order with rerank "
               "scores attached, not rerank order (`rerank_score.py:62` sorts before scoring, "
               "`agent_run.py:118` does not). So a gold chunk can be delivered at position 7 "
               "while its rerank rank is 2.", ""] if arm == "static" else []),
            "Ordering is a choice, not a measure: reasoning failures (gold was in the top-k) "
            "before rerank failures (gold was in the 50) before retrieval failures (gold was "
            "never fetched); then best gold rank, then iterations, then id. See the module "
            "docstring of `scripts/archive/worst_failures.py`.", "", "---", ""]
    body = [render(r, v, g, arm, i, gs) for i, (r, v, g, gs) in enumerate(fails[:top], 1)]
    path.write_text("\n".join(head) + "\n".join(body) + ("_No failures._\n" if not fails else ""))
    return len(fails), counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/day9_arm6_dev_results.jsonl")
    ap.add_argument("--out-prefix", default="data/day9_worst_failures")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    rows = load_rows(args.results)
    if not rows:
        print(f"no scoreable rows in {args.results}")
        return
    df = E.load_matched_questions().set_index("id")
    # gold_chunk_ids are indices WITHIN the gold filing, so a retrieved (stem, idx) only
    # counts when the stem matches -- same convention as agent_analyze.py:49.
    stems = {r["id"]: Path(df.loc[r["id"], "chunk_file"]).stem for r in rows}

    for arm in ("arm6", "static"):
        p = Path(f"{args.out_prefix}_{arm}.md")
        n, counts = write_arm(p, arm, rows, stems, args.top, args.results)
        print(f"{arm:<7} {n} failures of {len(rows)} rows -> {p}  "
              + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"))


if __name__ == "__main__":
    main()
