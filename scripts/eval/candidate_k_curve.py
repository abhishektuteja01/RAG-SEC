#!/usr/bin/env python
"""What does cutting CANDIDATE_K actually cost in recall?

Cutting K is the live latency route (DEPLOY-11), and until now its price has been
asserted rather than measured. This measures it EXACTLY, with no model and no GPU.

WHY NO RERANKER RUN IS NEEDED. `retr7_rr_dev_scores.jsonl` stores, per question, all 50
candidates in FIRST-STAGE (RRF) order with the reranker's score attached to each -- the
property `scripts/checks/static_ranking_order.py` locks, and the one AGENT-16 got wrong.
Serving at CANDIDATE_K=K' would hand the reranker the first K' of that same list and take
the top 10 by score. Both halves are already on disk, so slicing the list at K' and
re-sorting reproduces the served ranking bit for bit. This is a replay, not an estimate.

WHAT IT CANNOT SEE. The first stage itself is unchanged -- same RRF, same 50 retrieved.
So this prices the reranker's shrinking input, which is the whole latency lever, and says
nothing about retrieving fewer candidates in Postgres (which is not where the time goes:
search is 0.241s against rerank's 32.60s).

Usage:
    uv run python scripts/eval/candidate_k_curve.py
    uv run python scripts/eval/candidate_k_curve.py --out data/deploy14_candidate_k_curve.json

--out exists because DEPLOY-14 is a load-bearing negative result that lived only as prose
in DECISIONS.md. INFRA-10 made the same fix for three other print-only scripts: a number
whose producer wrote nothing to disk is the shape of the unsourced 9.4x.
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec.eval import (  # noqa: E402
    _filing_stem,
    gold_relevant_chunk_ids,
    load_matched_questions,
    mean_and_stderr,
    ndcg_at_k,
    recall_at_k,
)

CELL = "filtered_stripped"  # the shipped arm
TOP_K = 10                  # what the answer stage sees; unchanged by this lever
KS = (50, 40, 30, 25, 20, 15, 10)


def curve(split: str) -> dict:
    scores = _ROOT / "data" / f"retr7_rr_{split}_scores.jsonl"
    published = {}
    with open(scores) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                published[r["id"]] = r["cells"][CELL]

    df = load_matched_questions()
    rows = []
    for _, row in df[df["split"] == split].iterrows():
        cell = published.get(row["id"])
        if not cell:
            continue
        gold = {(_filing_stem(row), i) for i in gold_relevant_chunk_ids(row)}
        if gold:
            rows.append((cell, gold))

    print(f"\n-- {split}, n={len(rows)}, cell={CELL}, top_k={TOP_K} --")
    print(f"  {'K':>4}  {'recall@10':>10}  {'+/-':>6}  {'nDCG@10':>8}  {'vs K=50':>8}  {'ceiling':>8}")
    base = None
    curve_rows = []
    for k in KS:
        rec, nd, ceil = [], [], []
        for cell, gold in rows:
            pool = cell[:k]                                    # first-stage order, sliced
            ranked = [(s, i) for s, i, _ in sorted(pool, key=lambda c: -c[2])]
            rec.append(recall_at_k(ranked, gold, TOP_K))
            nd.append(ndcg_at_k(ranked, gold, TOP_K))
            # What the reranker COULD have reached with a perfect ordering of this pool --
            # separates "the pool lost the gold" from "the reranker mis-ranked it".
            ceil.append(recall_at_k([(s, i) for s, i, _ in pool], gold, k))
        r, r_se = mean_and_stderr(rec)
        n, n_se = mean_and_stderr(nd)
        c, _ = mean_and_stderr(ceil)
        base = base if base is not None else r
        print(f"  {k:>4}  {r:>10.3f}  {r_se:>6.3f}  {n:>8.3f}  {r - base:>+8.3f}  {c:>8.3f}")
        curve_rows.append(
            {
                "k": k,
                "recall_at_10": r,
                "recall_at_10_stderr": r_se,
                "ndcg_at_10": n,
                "ndcg_at_10_stderr": n_se,
                "delta_recall_vs_k50": r - base,
                "pool_ceiling_at_k": c,
            }
        )

    return {
        "n": len(rows),
        "source": str(scores.relative_to(_ROOT)),
        "curve": curve_rows,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out",
        type=Path,
        help="write the curve as JSON so DEPLOY-14 has a producer output on disk",
    )
    args = ap.parse_args()

    result = {
        "cell": CELL,
        "top_k": TOP_K,
        "ks": list(KS),
        "splits": {s: curve(s) for s in ("dev", "test")},
    }

    if args.out:
        out = args.out if args.out.is_absolute() else _ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {out.relative_to(_ROOT)}")
