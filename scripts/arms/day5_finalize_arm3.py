"""Day 5, Arm 3, stage 3 (laptop): combine the HPC's reranked orderings with local
gold-relevance labels to compute recall@10/50, nDCG@10, MRR, and rerank latency --
same shape as Arm 1/2's results files for direct comparability. No DB, no GPU needed:
gold labels come from `data/chunks/*.json` (see `rag_sec.eval`), not Postgres.

Usage:
    python day5_finalize_arm3.py [rerank_scores.jsonl]
"""

import json
import math
import sys
from pathlib import Path

from rag_sec.config import RERANK_MODEL_NAME
from rag_sec.eval import gold_relevant_chunk_ids, load_matched_questions, mrr, ndcg_at_k, recall_at_k

RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "day5_arm3_dev_results.json"
FAILURES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "day5_arm3_dev_failures.md"


def load_scores(path: Path) -> dict[str, dict]:
    scores = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                scores[row["id"]] = row
    return scores


def mean_and_stderr(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if not math.isnan(v)]
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    stderr = math.sqrt(variance / n) if n > 0 else float("nan")
    return mean, stderr


def main() -> None:
    scores_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent / "data" / "rerank_scores.jsonl"
    scores = load_scores(scores_path)

    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    dev = dev[dev["id"].isin(scores)].reset_index(drop=True)
    print(f"Scoring {len(dev)}/{len(scores)} scored questions (dev split)")

    per_question = []
    for _, row in dev.iterrows():
        s = scores[row["id"]]
        retrieved = [tuple(c) for c in s["reranked"]]

        filing_stem = Path(row["chunk_file"]).stem
        relevant = [(filing_stem, i) for i in gold_relevant_chunk_ids(row)]

        per_question.append(
            {
                "id": row["id"],
                "question": row["question"],
                "chunk_file": row["chunk_file"],
                "n_relevant": len(relevant),
                "recall_10": recall_at_k(retrieved, relevant, 10),
                "recall_50": recall_at_k(retrieved, relevant, 50),
                "ndcg_10": ndcg_at_k(retrieved, relevant, 10),
                "mrr": mrr(retrieved, relevant),
                "rerank_latency_s": s["latency_s"],
                "top_5_retrieved": retrieved[:5],
            }
        )

    metrics = {}
    for key in ["recall_10", "recall_50", "ndcg_10", "mrr"]:
        mean, stderr = mean_and_stderr([q[key] for q in per_question])
        metrics[key] = {"mean": mean, "stderr": stderr}
        print(f"{key}: {mean:.3f} +/- {stderr:.3f}")

    latencies = sorted(q["rerank_latency_s"] for q in per_question)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    print(f"rerank latency (top-50 candidates, V100): p50={p50*1000:.0f}ms, p95={p95*1000:.0f}ms")

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "rerank_model": RERANK_MODEL_NAME,
                "rerank_device": "Tesla V100-PCIE-32GB (Northeastern Explorer HPC)",
                "top_k": 50,
                "n": len(dev),
                "metrics": metrics,
                "rerank_latency_ms": {"p50": p50 * 1000, "p95": p95 * 1000},
                "per_question": per_question,
            },
            indent=2,
        )
    )
    print(f"Results written to {RESULTS_PATH}")

    worst = sorted(per_question, key=lambda q: (q["recall_10"] if not math.isnan(q["recall_10"]) else 0))[:20]
    with open(FAILURES_PATH, "w") as f:
        f.write("# Arm 3 (Arm 2 hybrid + local cross-encoder rerank) — 20 worst failures on dev split\n\n")
        for w in worst:
            f.write(f"## {w['id']} (recall@10={w['recall_10']:.2f}, filing={w['chunk_file']})\n")
            f.write(f"Q: {w['question']}\n\n")
            f.write(f"Top 5 retrieved: {w['top_5_retrieved']}\n\n")
    print(f"Worst failures written to {FAILURES_PATH}")


if __name__ == "__main__":
    main()
