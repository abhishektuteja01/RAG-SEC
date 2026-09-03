"""Day 8: why does retrieval miss? Splits every dev question into one of four buckets so
the next chunking change is made on evidence instead of a guess.

The reranker only reorders the 50 candidates it is handed, so its ceiling is recall@50
(0.685) against its actual recall@10 (0.609) -- 7.6 points of headroom. The other 31.5%
never had the gold in the pool at all, and that is decided upstream by chunking and
indexing. This script measures which.

Buckets, in the order a question is tested:
  hit@10         gold chunk reached the final top 10. Success, no action.
  rerank_miss    gold was among the 50 but not promoted into the top 10. Reranker's fault;
                 bounded by the 7.6 points above.
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

Usage:
    python scripts/diagnostics/day8_retrieval_failure_triage.py [--n N]
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
)

RERANK_SCORES = Path("data/day6_arm4_A_rerank_scores.jsonl")
OUT_CASES = Path("data/day8_failure_cases.json")
TOP_K = 10


def _load_rerank_order() -> dict[str, list[tuple[str, int]]]:
    order = {}
    with open(RERANK_SCORES) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                order[rec["id"]] = [(s, i) for s, i, _v in rec["reranked"]]
    return order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    args = ap.parse_args()

    order = _load_rerank_order()
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

    OUT_CASES.write_text(json.dumps(cases, indent=1))
    print(f"\nwrote {len(cases)} failure cases -> {OUT_CASES} ({OUT_CASES.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
