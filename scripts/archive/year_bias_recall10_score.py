"""Laptop leg 3/3 of the year-bias recall@10 check. Reads only -- no GPU, no database,
no spend, free to re-run. Reconstructs each condition's own top-10 from the HPC leg's
shared per-candidate scores (`base_pool`/`biased_pool` are membership lists into the same
union `candidates` the HPC leg reranked once), then scores both against gold.

PRODUCES
    printed recall@10 (± stderr) for base vs year-biased, and the delta.

READS
    data/year_bias_recall10_{split}_pools.json     (the prepare payload minus texts and rerank_query)
    data/year_bias_recall10_{split}_scores.jsonl    (HPC leg: per-candidate scores)
    rag_sec.eval.load_matched_questions            (gold labels)

Usage:
    python scripts/archive/year_bias_recall10_score.py --split dev
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag_sec.eval import (  # noqa: E402
    _filing_stem,
    gold_relevant_chunk_ids,
    load_matched_questions,
    mean_and_stderr,
    recall_at_k,
)

_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "data"
TOP_K = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    args = ap.parse_args()

    payload = json.loads((DATA_DIR / f"year_bias_recall10_{args.split}_pools.json").read_text())
    by_id = {q["id"]: q for q in payload["questions"]}

    scores_path = DATA_DIR / f"year_bias_recall10_{args.split}_scores.jsonl"
    score_of_by_id: dict[str, dict] = {}
    with open(scores_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            score_of_by_id[rec["id"]] = {(s[0], s[1]): s[2] for s in rec["scores"]}

    df = load_matched_questions()
    rows = df[df["split"] == args.split].reset_index(drop=True)

    base_hits, biased_hits = [], []
    n_scored = skipped_no_scores = skipped_no_gold = 0
    for _, row in rows.iterrows():
        qid = row["id"]
        if qid not in by_id or qid not in score_of_by_id:
            skipped_no_scores += 1
            continue
        gold_idx = gold_relevant_chunk_ids(row)
        if not gold_idx:
            skipped_no_gold += 1
            continue
        stem = _filing_stem(row)
        gold = [(stem, gi) for gi in gold_idx]

        q = by_id[qid]
        score_of = score_of_by_id[qid]
        base_pool = [tuple(p) for p in q["base_pool"]]
        biased_pool = [tuple(p) for p in q["biased_pool"]]

        base_top10 = sorted(base_pool, key=lambda p: score_of.get(p, -1e9), reverse=True)[:TOP_K]
        biased_top10 = sorted(biased_pool, key=lambda p: score_of.get(p, -1e9), reverse=True)[:TOP_K]

        n_scored += 1
        base_hits.append(recall_at_k(base_top10, gold, TOP_K))
        biased_hits.append(recall_at_k(biased_top10, gold, TOP_K))

    bm, bse = mean_and_stderr(base_hits)
    ym, yse = mean_and_stderr(biased_hits)
    alpha = payload["questions"][0]["alpha"] if payload["questions"] else "?"
    print(f"\n[{args.split}] questions scored: {n_scored}   alpha={alpha}"
          f"   (skipped: {skipped_no_scores} missing scores, {skipped_no_gold} no gold)\n")
    print(f"  recall@10, baseline (filtered_stripped)     : {bm:.3f} ± {bse:.3f}")
    print(f"  recall@10, + year_bias                      : {ym:.3f} ± {yse:.3f}"
          f"   ({ym - bm:+.3f})")


if __name__ == "__main__":
    main()
