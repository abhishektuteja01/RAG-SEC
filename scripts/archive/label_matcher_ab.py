"""RETR-35: what the coverage-based Layer-1 matcher (RETR-34) does to every published number.

Runs the legacy filter-by-threshold matcher and the new minimal-coverage matcher over the
same questions and the same retrieval ordering already on disk. No GPU, no API: only the
labels change, and labels are used at scoring time only.

The legacy function is copied in verbatim rather than imported, so both rules run in one
process and the comparison cannot drift with the working tree.

Usage:
    python scripts/archive/label_matcher_ab.py --split dev
"""

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec import eval as E  # noqa: E402

SCORES = {"dev": Path("data/day8_retr16v2_dev_scores.jsonl"),
          "test": Path("data/day8_retr18_test_scores.jsonl")}
CELL = "filtered_stripped"


def legacy_table_row_relevant_chunks(cells, candidates):
    """Verbatim copy of the pre-RETR-34 rule: every chunk over the bar becomes gold."""
    gold_nums = E._numeric_tokens(E._normalize_words(" ".join(str(c) for c in cells)))
    if not gold_nums:
        return set()
    chunk_nums_list = [E._numeric_tokens(E._normalize_words(text)) for _, text in candidates]
    doc_freq = Counter()
    for nums in chunk_nums_list:
        doc_freq.update(nums)
    distinctive = {g for g in gold_nums if doc_freq[g] <= E.ROW_NUMBER_MAX_DOC_FREQ}
    if not distinctive:
        return set()
    hits = set()
    for (cid, _), chunk_nums in zip(candidates, chunk_nums_list):
        if len(distinctive & chunk_nums) / len(distinctive) >= E.ROW_NUMBER_MATCH_THRESHOLD:
            hits.add(cid)
    return hits


def load_order(path: Path):
    return E.load_ranking(path, CELL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    order = load_order(SCORES[args.split])
    df = E.load_matched_questions()
    sub = df[df["split"] == args.split]
    new_fn = E._table_row_relevant_chunks

    res = {}
    per_q = {}
    for name, fn in (("legacy", legacy_table_row_relevant_chunks), ("coverage", new_fn)):
        E._table_row_relevant_chunks = fn
        E._load_chunks.cache_clear()
        sizes, rec10, rec50, ndcg, mrr, empty = [], [], [], [], [], 0
        for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"{args.split}/{name}"):
            qid = row["id"]
            if qid not in order:
                continue
            stem = E._filing_stem(row)
            gold = [(stem, gi) for gi in E.gold_relevant_chunk_evidence(row)]
            if not gold:
                empty += 1
                continue
            cand = order[qid]
            sizes.append(len(gold))
            rec10.append(E.recall_at_k(cand, gold, 10))
            rec50.append(E.recall_at_k(cand, gold, 50))
            ndcg.append(E.ndcg_at_k(cand, gold, 10))
            mrr.append(E.mrr(cand, gold))
            per_q.setdefault(qid, {})[name] = {"n_gold": len(gold), "recall10": rec10[-1]}
        res[name] = {
            "n": len(sizes), "no_gold": empty,
            "mean_gold_chunks": st.mean(sizes), "median_gold_chunks": st.median(sizes),
            "max_gold_chunks": max(sizes),
            "recall@10": st.mean(rec10), "recall@50": st.mean(rec50),
            "nDCG@10": st.mean(ndcg), "MRR": st.mean(mrr),
        }
    E._table_row_relevant_chunks = new_fn

    print(f"\n{args.split}: {res['legacy']['n']} scored questions\n")
    print(f"{'metric':20}{'legacy':>10}{'coverage':>11}{'delta':>10}")
    for k in ("mean_gold_chunks", "median_gold_chunks", "max_gold_chunks",
              "recall@10", "recall@50", "nDCG@10", "MRR"):
        a, b = res["legacy"][k], res["coverage"][k]
        print(f"{k:20}{a:>10.3f}{b:>11.3f}{b-a:>+10.3f}")
    print(f"{'questions w/o gold':20}{res['legacy']['no_gold']:>10}{res['coverage']['no_gold']:>11}"
          f"{res['coverage']['no_gold']-res['legacy']['no_gold']:>+10}")
    if args.out:
        args.out.write_text(json.dumps({"split": args.split, "summary": res, "per_question": per_q}, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
