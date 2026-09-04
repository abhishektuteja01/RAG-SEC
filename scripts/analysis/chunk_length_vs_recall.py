"""Ad-hoc analysis: does recall@10/nDCG@10 degrade for longer gold-relevant chunks?

Reuses the existing Arm 1 dev results (data/day3_arm1_dev_results.json) rather than
re-running retrieval. For each dev question, looks up the token length of its gold-
relevant chunk(s), buckets questions by that length, and compares metrics per bucket.
This is a diagnostic for whether BGE-M3's single-CLS pooling (no MCLS) is degrading on
our longer chunks -- see chunking.py's MAX_CHUNK_TOKENS=1500 ceiling and the MCLS
discussion in DECISIONS.md.
"""

import json
import math
from pathlib import Path

from rag_sec.eval import CHUNKS_DIR, _load_chunks, gold_relevant_chunk_ids, load_matched_questions

RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "day3_arm1_dev_results.json"

BUCKETS = [(0, 500, "short (<500)"), (500, 1000, "medium (500-1000)"), (1000, 10_000, "long (1000+)")]


def bucket_for(tokens: int) -> str:
    for lo, hi, label in BUCKETS:
        if lo <= tokens < hi:
            return label
    return "unknown"


def mean(values: list[float]) -> float | None:
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else None


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())
    per_question = results["per_question"]

    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    by_id = {row["id"]: row for _, row in dev.iterrows()}

    bucketed: dict[str, list[dict]] = {label: [] for _, _, label in BUCKETS}
    skipped = 0

    for q in per_question:
        row = by_id.get(q["id"])
        if row is None:
            skipped += 1
            continue
        relevant_idx = gold_relevant_chunk_ids(row)
        if not relevant_idx:
            skipped += 1
            continue
        chunks = _load_chunks(row["chunk_file"], CHUNKS_DIR)
        gold_tokens = [chunks[i]["n_tokens"] for i in relevant_idx if i < len(chunks)]
        if not gold_tokens:
            skipped += 1
            continue
        avg_gold_tokens = sum(gold_tokens) / len(gold_tokens)
        bucketed[bucket_for(avg_gold_tokens)].append(q)

    print(f"n questions: {len(per_question)}, skipped (no gold chunk found): {skipped}\n")
    print(f"{'bucket':<20}{'n':>5}{'recall@10':>12}{'recall@50':>12}{'nDCG@10':>12}{'MRR':>10}")
    for _, _, label in BUCKETS:
        qs = bucketed[label]
        if not qs:
            print(f"{label:<20}{0:>5}")
            continue
        r10 = mean([q["recall_10"] for q in qs])
        r50 = mean([q["recall_50"] for q in qs])
        nd = mean([q["ndcg_10"] for q in qs])
        m = mean([q["mrr"] for q in qs])
        print(f"{label:<20}{len(qs):>5}{r10:>12.3f}{r50:>12.3f}{nd:>12.3f}{m:>10.3f}")


if __name__ == "__main__":
    main()
