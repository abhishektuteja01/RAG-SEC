"""Why does retrieval miss? Splits every dev question into one of four buckets so
the next chunking change is made on evidence instead of a guess.

The reranker only reorders the 50 candidates it is handed, so its ceiling is recall@50
against its actual recall@10 -- on the shipped ordering, 0.776 vs 0.736, 4.0 points of
headroom. Everything else never had the gold in the pool at all, and that is decided
upstream by chunking and indexing. This script measures which.

Buckets, in the order a question is tested:
  hit@10         gold chunk reached the final top 10. Success, no action.
  rerank_miss    gold was among the 50 but not promoted into the top 10. Reranker's fault;
                 bounded by the recall@50-minus-recall@10 headroom above.
  candidate_miss gold chunk exists in the filing, but dense+BM25/RRF never surfaced it.
                 Retrieval/indexing's fault. Widening the pool may fix it -- the gold's
                 rank is unknown here, which is exactly what a DB re-run would answer.
  no_gold_chunk  no chunk anywhere in the filing matches the gold at all. Chunking's
                 fault (split across a boundary, or a table separated from its caption)
                 or a labeling gap. The most actionable bucket, and the only one that
                 justifies a re-chunk + re-embed cycle.

Reuses eval.py's own matcher (`gold_relevant_chunk_ids`) rather than a second copy -- a
divergent matcher would let this triage and the recall table disagree for reasons that
have nothing to do with retrieval.

Read-only: no DB, no API, no writes to the corpus. Emits a JSON of failure cases for
qualitative follow-up.

Defaults to the shipped ordering (company filter + query strip, `RETR-16v2`). The Day 6
ordering the original triage used is still reachable via --scores for comparison.

Usage:
    python scripts/archive/failure_triage.py [--n N]
    python scripts/archive/failure_triage.py --scores data/day6_arm4_A_rerank_scores.jsonl \
        --out data/day8_failure_cases_day6.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.eval import (  # noqa: E402
    _filing_stem,
    _gold_evidence_resolved,
    _load_chunks,
    gold_relevant_chunk_evidence,
    load_matched_questions,
    load_ranking,
)

SCORES = Path("data/day8_retr16v2_dev_scores.jsonl")   # shipped ordering (RETR-16v2)
LEGACY_SCORES = Path("data/day6_arm4_A_rerank_scores.jsonl")  # pre-RETR-24, variant-contaminated
CELL = "filtered_stripped"
OUT_CASES = Path("data/day8_failure_cases_retr16v2.json")
TOP_K = 10


def _load_rerank_order(path: Path, cell: str) -> dict[str, list[tuple[str, int]]]:
    """This caller genuinely takes either shape: --scores defaults to the modern cells file
    but LEGACY_SCORES (the Day 6 Arm 4-A file) is a documented alternative, and the two
    return the same `{id: ranking}` here. So sniff the first record and dispatch explicitly
    -- a cell name for the modern shape, `cell=None` for the legacy no-cells one. Passing
    the wrong one of those raises in the loader, so a bad sniff cannot pass silently.
    Never ALL_CELLS: that returns a dict of cells, not the ranking this caller wants.
    """
    with open(path) as f:
        first = next((json.loads(ln) for ln in f if ln.strip()), {})
    return load_ranking(path, cell if "cells" in first else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--cell", default=CELL, help="ablation cell, multi-cell score files only")
    ap.add_argument("--out", type=Path, default=OUT_CASES)
    args = ap.parse_args()

    order = _load_rerank_order(args.scores, args.cell)
    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    if args.n:
        dev = dev.head(args.n)
    resolved_all = _gold_evidence_resolved()

    buckets = Counter()
    gold_ranks: list[int] = []          # rank within the 50, for rerank_miss
    layers = defaultdict(Counter)       # which matcher layer fired, per bucket
    n_gold_chunks = defaultdict(list)
    cases = []
    skipped = 0

    for _, row in tqdm(dev.iterrows(), total=len(dev), desc="triaging"):
        qid = row["id"]
        if qid not in order:
            skipped += 1
            continue
        stem = _filing_stem(row)
        evidence = gold_relevant_chunk_evidence(row)
        gold = {(stem, gi) for gi in evidence}

        cand = order[qid]
        top10, pool = set(cand[:TOP_K]), set(cand)

        if not gold:
            bucket = "no_gold_chunk"
        elif gold & top10:
            bucket = "hit@10"
        elif gold & pool:
            bucket = "rerank_miss"
        else:
            bucket = "candidate_miss"

        buckets[bucket] += 1
        n_gold_chunks[bucket].append(len(gold))
        for ev in evidence.values():
            layers[bucket][ev["layer"]] += 1

        if bucket == "rerank_miss":
            gold_ranks.append(min(cand.index(g) + 1 for g in gold if g in pool))

        if bucket == "hit@10":
            continue

        chunks = _load_chunks(row["chunk_file"])
        r = resolved_all.get(qid) or {}
        cases.append(
            {
                "id": qid,
                "bucket": bucket,
                "question": row["question"],
                "answer": str(row.get("answer", "")),
                "filing_stem": stem,
                "n_chunks_in_filing": len(chunks),
                "gold_table_rows": r.get("table_rows", []),
                "gold_sentences": r.get("sentences", []),
                "gold_chunk_indices": sorted(gi for _s, gi in gold),
                "gold_chunk_texts": [chunks[gi]["text"] for _s, gi in sorted(gold)],
                "gold_rank_in_50": (
                    min(cand.index(g) + 1 for g in gold if g in pool) if gold & pool else None
                ),
                "top10_retrieved": [
                    {"stem": s, "chunk_index": i, "text": _load_chunks(f"{s}.json")[i]["text"]}
                    for s, i in cand[:TOP_K]
                ],
            }
        )

    total = sum(buckets.values())
    print(f"\ndev questions triaged: {total}" + (f"  ({skipped} skipped: no rerank record)" if skipped else ""))
    print(f"\n{'bucket':<16} {'n':>6} {'share':>8}   {'mean gold chunks':>17}   matcher layers")
    for b in ("hit@10", "rerank_miss", "candidate_miss", "no_gold_chunk"):
        n = buckets[b]
        if not n:
            continue
        g = n_gold_chunks[b]
        mean_g = f"{sum(g)/len(g):.1f}" if g and sum(g) else "-"
        top_layers = ", ".join(f"{k} {v}" for k, v in layers[b].most_common(3)) or "-"
        print(f"{b:<16} {n:>6} {100*n/total:>7.1f}%   {mean_g:>17}   {top_layers}")

    if gold_ranks:
        gold_ranks.sort()
        pct = lambda p: gold_ranks[min(int(p * len(gold_ranks)), len(gold_ranks) - 1)]  # noqa: E731
        print(f"\nrerank_miss -- where the gold actually sat in the 50 (best gold chunk):")
        print(f"  median {pct(0.5)}   p75 {pct(0.75)}   p90 {pct(0.9)}   max {gold_ranks[-1]}")
        for cut in (15, 20, 25, 50):
            n = sum(1 for r in gold_ranks if r <= cut)
            print(f"  would be recovered by taking top-{cut} instead of top-10: {n}"
                  f" ({100*n/total:.1f}pt of overall recall)")

    args.out.write_text(json.dumps(cases, indent=1))
    print(f"\nwrote {len(cases)} failure cases -> {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
