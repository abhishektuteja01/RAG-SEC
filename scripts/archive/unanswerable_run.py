"""Runs the unanswerable set through the shipped arm and scores refusal correctness:
when the answer genuinely is not in the corpus, does the system say so instead of guessing?

WHAT IS BEING MEASURED. Every question in T2-RAGBench has an answer in the corpus, so
nothing in this project has ever tested the opposite case: evidence that genuinely is not
there. A retriever always returns its top-k -- there is no empty result -- so the answer
model is handed ten plausible-looking chunks about the right sort of thing and asked a
question none of them answers. Refusing there is a behaviour, not a default.

SCORING is `answer_eval.parse_reason`, imported rather than re-derived. A question is
answered CORRECTLY when the model produces no number:
    refused          the ANSWER line explicitly declines -- the intended behaviour
    no_number        declined without matching the refusal vocabulary; still not a fabrication
    ok               a number came out. On this set that is a HALLUCINATION, by construction.
    no_answer_line   format not followed; counted separately, it is a different failure

WHY THE PER-CATEGORY SPLIT IS THE POINT. The five categories fail at different stages, and
one aggregate rate would hide that. `out_of_corpus_company` and `cross_company_premise`
should be catchable before generation -- the company resolver finds a name with no filings --
whereas `out_of_scope_metric` retrieves perfectly well and can only be caught by the answer
model. A high aggregate refusal rate carried entirely by the first group would say nothing
about the second.

Cost: one answer call per question, ~$0.012 each at COST-34's medium rates -- roughly $0.55
for all 47. Retrieval is local (GPU/CPU), so run it serially on a quiet machine like any
other pass here (AGENT-10/AGENT-17).

The live set is 47. `data/unanswerable_results.jsonl` still has 48 rows because it predates
retiring `unans_034` to `data/archive/unans_034_retired.jsonl`; the stored run was never
re-paid. Filter by the live question ids rather than trusting the row count.

Usage:
    uv run scripts/archive/unanswerable_run.py
    uv run scripts/archive/unanswerable_run.py --limit 5      # smoke test first
    uv run scripts/archive/unanswerable_run.py --score-only   # re-score an existing file
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from langchain_core.messages import HumanMessage  # noqa: E402
from tqdm import tqdm  # noqa: E402

from rag_sec.agent import _ANSWER_PROMPT, _answer_llm, _dedupe_chunks, _evidence_text  # noqa: E402
from rag_sec.answer_eval import parse_reason  # noqa: E402
from rag_sec.company import resolve as resolve_company  # noqa: E402
from rag_sec.retrieve import retrieve  # noqa: E402

QUESTIONS = _ROOT / "data" / "unanswerable_questions.jsonl"
RESULTS = _ROOT / "data" / "unanswerable_results.jsonl"
GOOD = ("refused", "no_number")  # no number produced = correct on this set


def run(rows: list[dict], out: Path) -> list[dict]:
    # Resumable in the same way agent_run.py is: an interrupted pass keeps its successes,
    # because the answer calls cost money and the reranker costs wall clock.
    done = {}
    if out.exists():
        for ln in out.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                done[r["id"]] = r

    results = []
    with open(out, "a") as f:
        for row in tqdm(rows, desc="asking"):
            if row["id"] in done:
                results.append(done[row["id"]])
                continue
            t0 = time.perf_counter()
            chunks = _dedupe_chunks(retrieve(row["question"]))
            prompt = _ANSWER_PROMPT.format(question=row["question"],
                                           evidence=_evidence_text(chunks))
            resp = _answer_llm().invoke([HumanMessage(content=prompt)])
            value, reason = parse_reason(resp.text)
            rec = {
                **row,
                # Recorded because it is the cheap pre-generation signal: an empty resolve
                # means the filter had no company to scope to, which is the case a
                # retrieval-side refusal would key on.
                "resolved_companies": resolve_company(row["question"]),
                "n_chunks": len(chunks),
                "retrieved": [[c["filing_stem"], c["chunk_index"]] for c in chunks],
                "answer": resp.text,
                "value": value,
                "reason": reason,
                "correct": reason in GOOD,
                "wall_s": time.perf_counter() - t0,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            results.append(rec)
    return results


def report(results: list[dict]) -> None:
    n = len(results)
    ok = sum(r["correct"] for r in results)
    print(f"\nrefusal correctness: {ok}/{n} = {ok / n:.1%}\n")

    print(f"{'category':<24} {'n':>3} {'refused':>8} {'rate':>7}  {'hallucinated':>12}")
    by_cat = collections.defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        good = sum(x["correct"] for x in rs)
        hall = sum(x["reason"] == "ok" for x in rs)
        print(f"{cat:<24} {len(rs):>3} {good:>8} {good / len(rs):>6.0%}  {hall:>12}")

    print("\nreasons:", dict(collections.Counter(r["reason"] for r in results)))

    # The diagnostic that decides whether a cheap pre-generation guard is worth building.
    unresolved = [r for r in results if not r["resolved_companies"]]
    if unresolved:
        good = sum(r["correct"] for r in unresolved)
        print(f"\nquestions whose company resolved to nothing: {len(unresolved)}, "
              f"{good} refused -- these are the ones a retrieval-side guard could catch "
              f"before any LLM call")

    for r in results:
        if r["reason"] == "ok":
            print(f"\nHALLUCINATION {r['id']} [{r['category']}] -> {r['value']}\n  "
                  f"{r['question']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in args.questions.read_text().splitlines() if ln.strip()]
    if args.limit:
        rows = rows[: args.limit]

    if args.score_only:
        if not args.out.exists():
            print(f"error: {args.out} missing", file=sys.stderr)
            return 1
        results = [json.loads(ln) for ln in args.out.read_text().splitlines() if ln.strip()]
        # The QUESTIONS file defines the set; the results file is an append-only record of
        # what was asked, including questions since retired (EVAL-2 dropped `unans_034`,
        # which turned out to be answerable). Filtering here rather than deleting the row
        # keeps the measurement on disk while the score reflects the current set -- a
        # retired question must not go on being scored, in either direction.
        keep = {r["id"] for r in rows}
        retired = [r for r in results if r["id"] not in keep]
        results = [r for r in results if r["id"] in keep]
        if retired:
            print(f"note: {len(retired)} result row(s) not in the current set, excluded: "
                  + ", ".join(r["id"] for r in retired))
    else:
        results = run(rows, args.out)

    report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
