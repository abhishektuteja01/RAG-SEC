"""Day 9 metrics for the Arm 6 run. Reads only -- never calls an API, so it is free to
re-run as rows arrive and can be pointed at a partial file mid-pass.

Covers spec.md:112-117 (trajectory + sufficiency-judge accuracy), 2.1's retrieval and answer
layers, and 2.1's operational layer (p50/p95 per stage). The paired Arm 6 vs Arm 3+filter+strip
comparison uses `static_baseline`, written by the same run against the published RETR-39
ranking, so both arms see the same questions and the same answer prompt.
"""

import argparse
import json
import statistics as st
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag_sec import eval as E
from rag_sec.answer_eval import answer_line, gold_is_scoreable, gold_values, is_correct, parse_reason

PRICE = {"gemini-3.7-flash": (0.75, 3.75, 0.075), "gemini-3.1-flash-lite": (0.25, 1.50, 0.025)}


def cost_of(usage) -> float:
    tot = 0.0
    for u in usage:
        pin, pout, pcache = PRICE[u["model"]]
        fresh = u["input_tokens"] - u["cached_input_tokens"]
        tot += (fresh * pin + u["cached_input_tokens"] * pcache + u["output_tokens"] * pout) / 1e6
    return tot


def dedupe(pairs):
    """First-seen order, no repeats. Loop iterations re-retrieve the same chunk (3/21 pilot
    rows), and ndcg_at_k would then credit one gold chunk twice and inflate nDCG. On the pilot
    every repeat sat past position 10, so its nDCG@10 is unmoved -- this guards the case where
    a repeat lands inside k. Recall and MRR need no dedupe: set membership, and first hit."""
    return list(dict.fromkeys(pairs))


def pct(x, n):
    return f"{x}/{n} = {x / n:.1%}" if n else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/day9_arm6_dev_results.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.results) if l.strip()]
    rows = [r for r in rows if "error" not in r]
    df = E.load_matched_questions().set_index("id")
    # gold chunk ids are indices WITHIN the gold filing, so a retrieved (stem, idx) only counts
    # when the stem matches -- the convention arm1/arm2/rerank_score all use. The stem is
    # recovered by joining on id rather than stored, so the results file stays small.
    stem_of = {r["id"]: Path(df.loc[r["id"], "chunk_file"]).stem for r in rows}
    n = len(rows)
    print(f"=== Arm 6, {n} questions, {args.results} ===")
    dev = rows[0]["config"]
    print(f"device={dev['device']} max_iter={dev['max_iterations']} answer={dev['answer_model']}\n")

    # ---- 1. trajectory (spec.md:112) ----
    iters = [r["iterations"] for r in rows]
    caps = sum(r["hit_iteration_cap"] for r in rows)
    first_loop = sum(1 for r in rows if r["judge_verdicts"][0] == "loop")
    print("-- trajectory --")
    print(f"iterations: mean {st.mean(iters):.2f}  dist {{{', '.join(f'{k}:{iters.count(k)}' for k in sorted(set(iters)))}}}")
    print(f"hit cap: {pct(caps, n)}    first-iteration 'insufficient': {pct(first_loop, n)}  (COST-30 predicted 15.1%)")

    # ---- 2. cost (spec.md:113) ----
    loop_c = [cost_of(r["usage"]) for r in rows]
    stat_c = [cost_of(r["static_baseline"]["usage"]) for r in rows]
    bynode = {}
    for r in rows:
        for u in r["usage"]:
            bynode[u["node"]] = bynode.get(u["node"], 0.0) + cost_of([u])
    tin = sum(u["input_tokens"] for r in rows for u in r["usage"])
    cch = sum(u["cached_input_tokens"] for r in rows for u in r["usage"])
    print("\n-- cost --")
    print(f"arm6 ${st.mean(loop_c):.4f}/q   static ${st.mean(stat_c):.4f}/q   ratio {st.mean(loop_c)/st.mean(stat_c):.2f}x")
    print("by node: " + "  ".join(f"{k} ${v/n:.4f} ({v/sum(bynode.values()):.0%})" for k, v in sorted(bynode.items())))
    print(f"prefix cache: {cch:,}/{tin:,} input tokens = {cch/tin:.1%}  (AGENT-8)")

    # ---- 3. latency (spec.md:121) ----
    def p(v, q):
        v = sorted(v)
        return v[min(int(q * len(v)), len(v) - 1)]
    stages = {}
    for r in rows:
        for t in r["trajectory"]:
            for k, v in t["stage_latency"].items():
                stages.setdefault(k, []).append(v)
    llm = {}
    for r in rows:
        for u in r["usage"]:
            llm.setdefault(u["node"], []).append(u["latency_s"])
    print("\n-- latency per stage (s) --")
    # model_init_s is one-off warm-up, not per-question work, and is excluded from total_s --
    # printed anyway so it stays visible rather than hiding inside embed_s as it used to.
    for k in ("embed_s", "search_s", "rerank_s", "model_init_s"):
        if k in stages:
            print(f"  {k:<10} p50 {p(stages[k],.5):>6.2f}  p95 {p(stages[k],.95):>6.2f}")
    for k in sorted(llm):
        print(f"  {k:<10} p50 {p(llm[k],.5):>6.2f}  p95 {p(llm[k],.95):>6.2f}   (LLM)")
    w = [r["wall_clock_s"] for r in rows]
    print(f"  {'question':<10} p50 {p(w,.5):>6.1f}  p95 {p(w,.95):>6.1f}")

    # ---- 4. sufficiency-judge accuracy (spec.md:115) ----
    print("\n-- sufficiency-judge accuracy --")
    cells = {"stop_with_gold": 0, "stop_without_gold": 0, "loop_with_gold": 0, "loop_without_gold": 0}
    for r in rows:
        gold = {(stem_of[r["id"]], i) for i in r["gold_chunk_ids"]}
        seen = set()
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
    print(f"  stopped WITHOUT gold (early)   {cells['stop_without_gold']:>4}  <- answers a question it cannot")
    print(f"  looped WITHOUT gold (right)    {cells['loop_without_gold']:>4}")
    print(f"  looped WITH gold (wasted)      {cells['loop_with_gold']:>4}  <- pays for evidence it already had")

    # ---- 5. retrieval (spec.md 2.1) ----
    print("\n-- retrieval: does looping add gold? --")
    for label, getter in (("arm6 iter-1 only", lambda r: [(a, b) for a, b, _ in r["trajectory"][0]["top_k"]]),
                          ("arm6 all iters (union)", lambda r: dedupe((a, b) for t in r["trajectory"] for a, b, _ in t["top_k"])),
                          ("static (RETR-39)", lambda r: [(a, b) for a, b, _ in r["static_baseline"]["top_k"]])):
        r10, nd, mr = [], [], []
        for r in rows:
            rel = [(stem_of[r["id"]], i) for i in r["gold_chunk_ids"]]
            if not rel:
                continue
            got = getter(r)
            r10.append(E.recall_at_k(got, rel, 10 if "union" not in label else len(got)))
            nd.append(E.ndcg_at_k(got, rel, 10))
            mr.append(E.mrr(got, rel))
        print(f"  {label:<24} recall {st.mean(r10):.3f}   nDCG@10 {st.mean(nd):.3f}   MRR {st.mean(mr):.3f}  (n={len(r10)})")

    # ---- 6. answer accuracy, paired (spec.md 2.1 / COST-21) ----
    for yesno in (False, True):
        _answer_accuracy(rows, yesno)


# AGENT-21: gold is numeric 1.0/0.0 on four dev questions whose *question* is yes/no, and the
# model answers the word. `answer_eval` requires a number, so both arms score them wrong while
# being right. This maps the word to the number -- and lives HERE, not in `answer_eval`, on
# purpose: the shipped scorer stays exactly as it was when the run was measured, so the
# headline is the measured one and this is a reported sensitivity beside it, not a metric
# redefined after seeing its own results.
_YESNO = {"yes": 1.0, "no": 0.0}


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
        print(f"  arm6 {(both+n10)/scored:.1%}   static {(both+n01)/scored:.1%}")
        print(f"  discordant: arm6-only {n10}, static-only {n01}  (McNemar pairs; needs n to interpret)")


if __name__ == "__main__":
    main()
