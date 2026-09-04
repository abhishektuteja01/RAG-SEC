"""Stage 3 (laptop): score the RETR-16 2x2 and fill in the Arm 3 row.

Cells (DECISIONS.md RETR-16):
    unfiltered_raw       existing Arm 3 baseline, read from day6_arm4_A_rerank_scores.jsonl
    filtered_raw         + company filter                      -> isolates RETR-5
    filtered_stripped    + company filter + query cleanup      -> the deployment candidate
    unfiltered_stripped  + query cleanup only                  -> isolates RETR-6

Metrics come from eval.py, not a second copy, so these numbers sit on the same scale as
the baseline table. recall@50 is reported but is a property of the candidate pool, not the
reranker -- it can only differ between filtered and unfiltered cells.

Usage:
    python scripts/retrieval/rerank_score.py
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.eval import (  # noqa: E402
    _filing_stem,
    gold_relevant_chunk_ids,
    load_matched_questions,
    mean_and_stderr,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

SCORES = Path("data/day8_retr16v2_dev_scores.jsonl")  # variant-clean re-run (RETR-24)
# dev's fourth cell was scored back on Day 6 and is merged in from here. test has no such
# prior run, so its payload carries all four cells and no merge is needed (RETR-30).
BASELINE = Path("data/day6_arm4_A_rerank_scores.jsonl")
CELLS = ("unfiltered_raw", "filtered_raw", "unfiltered_stripped", "filtered_stripped")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the scored table to JSON as well as printing it")
    ap.add_argument(
        "--baseline",
        type=Path,
        help="file supplying unfiltered_raw; omit when the scores file already has that cell",
    )
    args = ap.parse_args()
    ranked: dict[str, dict[str, list]] = {}
    with open(args.scores) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ranked[r["id"]] = {
                    c: [(s, i) for s, i, _sc in sorted(v, key=lambda x: -x[2])]
                    for c, v in r["cells"].items()
                }
    # Resolved AFTER loading, so a scores file that already carries unfiltered_raw is never
    # silently overwritten by the cached one. That cache (day6_arm4_A) was scored against
    # pre-RETR-7 chunk text, so after a re-index merging it would mix two corpora inside one
    # 2x2 and mis-attribute the ablation. An explicit --baseline still wins.
    baseline = args.baseline or (BASELINE if args.split == "dev" else None)
    if baseline and not args.baseline and ranked and all("unfiltered_raw" in v for v in ranked.values()):
        print(f"scores file already has unfiltered_raw for all {len(ranked)} questions "
              f"-- NOT merging the cached baseline {BASELINE}")
        baseline = None

    if baseline:
        merged = 0
        with open(baseline) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["id"] in ranked:  # already in rerank order in this file
                        ranked[r["id"]]["unfiltered_raw"] = [(s, i) for s, i, _v in r["reranked"]]
                        merged += 1
        print(f"merged unfiltered_raw for {merged} questions from {baseline}")

    df = load_matched_questions()
    dev = df[df["split"] == args.split].reset_index(drop=True)

    per: dict[str, dict[str, list[float]]] = {c: {k: [] for k in ("recall_10", "recall_50", "ndcg_10", "mrr")} for c in CELLS}
    n = 0
    # Counted, not silently skipped: a question dropped for a missing cell would otherwise
    # shrink the denominator invisibly and make cells incomparable.
    skipped_cells = skipped_gold = 0
    for _, row in tqdm(dev.iterrows(), total=len(dev), desc="scoring"):
        qid = row["id"]
        if qid not in ranked or any(c not in ranked[qid] for c in CELLS):
            skipped_cells += 1
            continue
        stem = _filing_stem(row)
        rel = [(stem, i) for i in gold_relevant_chunk_ids(row)]
        if not rel:
            skipped_gold += 1
            continue
        n += 1
        for c in CELLS:
            got = ranked[qid][c]
            per[c]["recall_10"].append(recall_at_k(got, rel, 10))
            per[c]["recall_50"].append(recall_at_k(got, rel, 50))
            per[c]["ndcg_10"].append(ndcg_at_k(got, rel, 10))
            per[c]["mrr"].append(mrr(got, rel))

    print(f"\nquestions scored: {n} of {len(dev)} {args.split}"
          f"   (skipped: {skipped_cells} missing a cell, {skipped_gold} with no gold label)\n")
    hdr = f"{'cell':<22}" + "".join(f"{k:>18}" for k in ("recall@10", "recall@50", "nDCG@10", "MRR"))
    print(hdr)
    print("-" * len(hdr))
    base, table = {}, {}
    for c in CELLS:
        cells = []
        for k in ("recall_10", "recall_50", "ndcg_10", "mrr"):
            m, se = mean_and_stderr(per[c][k])
            table.setdefault(c, {})[k] = {"mean": m, "stderr": se}
            if c == "unfiltered_raw":
                base[k] = m
                cells.append(f"{m:.3f} ± {se:.3f}".rjust(18))
            else:
                cells.append(f"{m:.3f} ({m - base[k]:+.3f})".rjust(18))
        print(f"{c:<22}" + "".join(cells))
    print("\n(baseline row shows ± stderr; other rows show the delta against it)")

    # Stderr is stored for every cell, not just the baseline the table prints it for: the
    # deltas are what get quoted, and a delta needs both cells' spread to be defensible.
    if args.out:
        args.out.write_text(json.dumps({
            "split": args.split, "scores": str(args.scores), "n_scored": n,
            "skipped_missing_cell": skipped_cells, "skipped_no_gold": skipped_gold,
            "cells": table,
        }, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
