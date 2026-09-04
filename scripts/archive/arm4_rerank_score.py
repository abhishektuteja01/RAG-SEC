"""Day 6, Arm 4, stage 3 (laptop): combine the HPC's reranked orderings for one variant
with DB-backed gold-relevance labels to compute recall@10/50, nDCG@10, MRR -- same shape
as Arm 3's results files for direct comparability.

Unlike Arm 3's day5_finalize_arm3.py, this DOES need Postgres: relevance labeling for B/C
comes from `eval.gold_relevant_chunk_evidence_db`, which reads the variant-filtered
candidate pool straight from the `chunks` table (DECISIONS.md ARM4-5), not from the old
`data/chunks/*.json` files (those only ever held Strategy A).

Usage:
    python day6_finalize_arm4.py --variant A [rerank_scores.jsonl]
    python day6_finalize_arm4.py --variant B [rerank_scores.jsonl]
    python day6_finalize_arm4.py --variant C [rerank_scores.jsonl]
"""

import argparse
import json
import math
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag_sec.config import RERANK_MODEL_NAME
from rag_sec.eval import gold_relevant_chunk_ids_db, load_matched_questions, mrr, ndcg_at_k, recall_at_k
from rag_sec.store import get_conn

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=["A", "B", "C"])
    parser.add_argument("scores_path", nargs="?", default=None)
    args = parser.parse_args()
    variant = args.variant

    scores_path = Path(args.scores_path) if args.scores_path else DATA_DIR / f"day6_arm4_{variant}_rerank_scores.jsonl"
    results_path = DATA_DIR / f"day6_arm4_{variant}_dev_results.json"
    failures_path = DATA_DIR / f"day6_arm4_{variant}_dev_failures.md"

    scores = load_scores(scores_path)

    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    dev = dev[dev["id"].isin(scores)].reset_index(drop=True)
    print(f"Scoring variant={variant}: {len(dev)}/{len(scores)} scored questions (dev split)")

    per_question = []
    with get_conn() as conn:
        for _, row in dev.iterrows():
            s = scores[row["id"]]
            retrieved = [tuple(c) for c in s["reranked"]]

            filing_stem = f"{row['company_symbol']}_{int(row['report_year'])}_{int(row['company_cik'])}"
            relevant = [(filing_stem, ci, v) for ci, v in gold_relevant_chunk_ids_db(row, conn, variant)]

            per_question.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "filing_stem": filing_stem,
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

    results_path.write_text(
        json.dumps(
            {
                "variant": variant,
                "rerank_model": RERANK_MODEL_NAME,
                "rerank_device": "Tesla V100-SXM2-32GB (Northeastern Explorer HPC)",
                "top_k": 50,
                "n": len(dev),
                "metrics": metrics,
                "rerank_latency_ms": {"p50": p50 * 1000, "p95": p95 * 1000},
                "per_question": per_question,
            },
            indent=2,
        )
    )
    print(f"Results written to {results_path}")

    worst = sorted(per_question, key=lambda q: (q["recall_10"] if not math.isnan(q["recall_10"]) else 0))[:20]
    with open(failures_path, "w") as f:
        f.write(f"# Arm 4 variant={variant} (Arm 2 hybrid + HPC cross-encoder rerank) — 20 worst failures on dev split\n\n")
        for w in worst:
            f.write(f"## {w['id']} (recall@10={w['recall_10']:.2f}, filing={w['filing_stem']})\n")
            f.write(f"Q: {w['question']}\n\n")
            f.write(f"Top 5 retrieved: {w['top_5_retrieved']}\n\n")
    print(f"Worst failures written to {failures_path}")


if __name__ == "__main__":
    main()
