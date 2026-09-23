"""Score retrieval recall@10 on the T2-RAGBench questions whose filing is in the corpus.

Two modes, one scorer:

  --replay   Re-score the stored rerank output in data/arms_scores.jsonl (cell `n18d_p10`,
             the serving setup). No GPU, no database, no API key. Expected: test recall@10
             0.831 (n=1545), dev 0.847.
  --live     Run the real retrieve() for each question (needs the Postgres corpus and the
             two models) and score its top 50 the same way.

Gold labels come from rag_sec.eval: computed from data/chunks/ when present, otherwise read
from the tracked data/gold_chunk_ids.json. The question set is downloaded from Hugging Face
on first use, so the first run needs network.

Usage (from the repo root):
    uv run scripts/evaluate.py --replay [--split test|dev] [--out results.json]
    uv run --env-file .env scripts/evaluate.py --live [--split dev|test] [--limit N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec.eval import (  # noqa: E402
    _filing_stem,
    gold_relevant_chunk_ids,
    load_matched_questions,
    load_ranking,
    mean_and_stderr,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

REPLAY_SCORES = _ROOT / "data" / "arms_scores.jsonl"
REPLAY_CELL = "n18d_p10"  # the serving setup: filter + strip + year nudge + stripped dense + year list
METRICS = ("recall_10", "recall_50", "ndcg_10", "mrr")


def score(rows, rankings: dict) -> dict:
    """Mean ± stderr of each metric over questions that have a ranking and a gold label.

    `rankings` maps question id -> [(filing_stem, chunk_index), ...], best first. Questions
    with no ranking or no gold label are counted, never silently dropped.
    """
    per = {m: [] for m in METRICS}
    skipped_missing = skipped_no_gold = 0
    for _, row in rows.iterrows():
        got = rankings.get(row["id"])
        if got is None:
            skipped_missing += 1
            continue
        stem = _filing_stem(row)
        rel = [(stem, i) for i in gold_relevant_chunk_ids(row)]
        if not rel:
            skipped_no_gold += 1
            continue
        per["recall_10"].append(recall_at_k(got, rel, 10))
        per["recall_50"].append(recall_at_k(got, rel, 50))
        per["ndcg_10"].append(ndcg_at_k(got, rel, 10))
        per["mrr"].append(mrr(got, rel))
    out = {"n_scored": len(per["recall_10"]), "n_questions": len(rows),
           "skipped_missing": skipped_missing, "skipped_no_gold": skipped_no_gold}
    for m in METRICS:
        mean, se = mean_and_stderr(per[m])
        out[m] = {"mean": mean, "stderr": se}
    return out


def replay(rows) -> dict:
    ranked = load_ranking(REPLAY_SCORES, REPLAY_CELL)
    return score(rows, ranked)


def live(rows) -> dict:
    from rag_sec.candidates import CANDIDATE_K
    from rag_sec.retrieve import last_call_stats, retrieve

    rankings, totals = {}, []
    for i, (_, row) in enumerate(rows.iterrows(), 1):
        # k = the whole 50-candidate pool, so recall@50 is scoreable too.
        hits = retrieve(row["question"], k=CANDIDATE_K)
        rankings[row["id"]] = [(h["filing_stem"], int(h["chunk_index"])) for h in hits]
        totals.append(last_call_stats()["timings"]["total_s"])
        print(f"[{i}/{len(rows)}] {row['id']}  {totals[-1]:.2f}s", flush=True)
    result = score(rows, rankings)
    result["retrieval_total_s_mean"] = sum(totals) / len(totals) if totals else None
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", action="store_true", help="re-score data/arms_scores.jsonl")
    mode.add_argument("--live", action="store_true", help="run retrieve() per question")
    ap.add_argument("--split", choices=("dev", "test"), default="test")
    ap.add_argument("--limit", type=int, help="first N questions of the split (live smoke test)")
    ap.add_argument("--out", type=Path, help="also write the result as JSON")
    args = ap.parse_args()

    df = load_matched_questions()
    rows = df[df["split"] == args.split].reset_index(drop=True)
    if args.limit:
        rows = rows.head(args.limit)

    t0 = time.perf_counter()
    result = replay(rows) if args.replay else live(rows)
    result.update(mode="replay" if args.replay else "live", split=args.split,
                  wall_s=time.perf_counter() - t0)

    r10 = result["recall_10"]
    print(f"\n{result['mode']} {args.split}: recall@10 {r10['mean']:.3f} ± {r10['stderr']:.3f} "
          f"(n={result['n_scored']} of {result['n_questions']}; skipped "
          f"{result['skipped_missing']} without a ranking, "
          f"{result['skipped_no_gold']} without a gold label)")
    print("  " + "   ".join(f"{m} {result[m]['mean']:.3f}" for m in METRICS[1:]))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
