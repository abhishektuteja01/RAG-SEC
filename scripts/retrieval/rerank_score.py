"""Day 8, stage 3 (laptop): score the RETR-16 2x2 and fill in the Arm 3 row.

Cells (DECISIONS.md RETR-16):
    unfiltered_raw       existing Arm 3 baseline, read from day6_arm4_A_rerank_scores.jsonl
    filtered_raw         + company filter                      -> isolates RETR-5
    filtered_stripped    + company filter + query cleanup      -> the deployment candidate
    unfiltered_stripped  + query cleanup only                  -> isolates RETR-6

Metrics come from eval.py, not a second copy, so these numbers sit on the same scale as
the baseline table. recall@50 is reported but is a property of the candidate pool, not the
reranker -- it can only differ between filtered and unfiltered cells.

Usage:
    python scripts/arms/day8_finalize_retr16.py
"""

import argparse
import json
import math
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.eval import (  # noqa: E402
    _filing_stem,
    gold_relevant_chunk_ids,
    load_matched_questions,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

SCORES = Path("data/day8_retr16v2_dev_scores.jsonl")  # variant-clean re-run (RETR-24)
# dev's fourth cell was scored back on Day 6 and is merged in from here. test has no such
# prior run, so its payload carries all four cells and no merge is needed (RETR-30).
BASELINE = Path("data/day6_arm4_A_rerank_scores.jsonl")
CELLS = ("unfiltered_raw", "filtered_raw", "unfiltered_stripped", "filtered_stripped")


def mean_stderr(v: list[float]) -> tuple[float, float]:
    v = [x for x in v if not math.isnan(x)]
    n = len(v)
    m = sum(v) / n
    var = sum((x - m) ** 2 for x in v) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--split", default="dev")
    ap.add_argument(
        "--baseline",
        type=Path,
        help="file supplying unfiltered_raw; omit when the scores file already has that cell",
    )
    args = ap.parse_args()
    baseline = args.baseline or (BASELINE if args.split == "dev" else None)

    ranked: dict[str, dict[str, list]] = {}
    with open(args.scores) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ranked[r["id"]] = {
                    c: [(s, i) for s, i, _sc in sorted(v, key=lambda x: -x[2])]
                    for c, v in r["cells"].items()
                }
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
    base = {}
    for c in CELLS:
        cells = []
        for k in ("recall_10", "recall_50", "ndcg_10", "mrr"):
            m, se = mean_stderr(per[c][k])
            if c == "unfiltered_raw":
                base[k] = m
                cells.append(f"{m:.3f} ± {se:.3f}".rjust(18))
            else:
                cells.append(f"{m:.3f} ({m - base[k]:+.3f})".rjust(18))
        print(f"{c:<22}" + "".join(cells))
    print("\n(baseline row shows ± stderr; other rows show the delta against it)")


if __name__ == "__main__":
    main()
