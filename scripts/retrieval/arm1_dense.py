"""Arm 1: dense-only retrieval, scored on the dev split.

spec.md 2.2: tune on dev, touch test once at the end per arm. Arm 1 has no
hyperparameters to tune, but we still hold test back — first pass to confirm the
plumbing (embed -> pgvector -> harness) is measuring something real before it becomes
the number that goes in a comparison table.

No cik+year metadata pre-filter (DECISIONS.md ARM1-2): pure semantic search across the
whole corpus, so recall reflects the actual retrieval difficulty, not "did we already
know which filing to look in."
"""

import argparse
import json
import math
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer

from rag_sec.company import resolve as resolve_companies
from rag_sec.config import EMBED_MODEL_NAME
from rag_sec.eval import gold_relevant_chunk_ids, load_matched_questions, mean_and_stderr, mrr, ndcg_at_k, recall_at_k
from rag_sec.store import get_conn

TOP_K = 50
RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "day3_arm1_dev_results.json"
FAILURES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "day3_arm1_dev_failures.md"


def retrieve(conn, embedding, k: int, tickers: list[str] | None = None) -> list[tuple[str, int]]:
    where = "AND split_part(filing_stem, '_', 1) = ANY(%s)" if tickers else ""
    args = (tickers, embedding, k) if tickers else (embedding, k)
    rows = conn.execute(
        f"SELECT filing_stem, chunk_index FROM chunks WHERE variant = 'A' {where} ORDER BY embedding <=> %s LIMIT %s",
        args,
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    # Off by default so the already-published numbers stay reproducible from this script.
    # Filtered runs write to a *_companyfilter.json sidecar rather than overwriting them --
    # DECISIONS.md RETR-5.
    ap.add_argument("--company-filter", action="store_true",
                    help="scope candidates to the company named in the question (RETR-5)")
    opts = ap.parse_args()
    suffix = "_companyfilter" if opts.company_filter else ""
    results_path = RESULTS_PATH.with_name(RESULTS_PATH.stem + suffix + RESULTS_PATH.suffix)
    failures_path = FAILURES_PATH.with_name(FAILURES_PATH.stem + suffix + FAILURES_PATH.suffix)
    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    print(f"Running Arm 1 on {len(dev)} dev-split questions")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    per_question = []

    with get_conn() as conn:
        for _, row in dev.iterrows():
            query_emb = model.encode(row["question"], normalize_embeddings=True)
            tickers = resolve_companies(row["question"]) if opts.company_filter else []
            retrieved = retrieve(conn, query_emb, TOP_K, tickers)
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
                    "top_5_retrieved": retrieved[:5],
                }
            )

    metrics = {}
    for key in ["recall_10", "recall_50", "ndcg_10", "mrr"]:
        mean, stderr = mean_and_stderr([q[key] for q in per_question])
        metrics[key] = {"mean": mean, "stderr": stderr}
        print(f"{key}: {mean:.3f} +/- {stderr:.3f}")

    results_path.write_text(
        json.dumps({"model": EMBED_MODEL_NAME, "top_k": TOP_K, "n": len(dev), "metrics": metrics, "per_question": per_question}, indent=2)
    )
    print(f"Results written to {results_path}")

    worst = sorted(per_question, key=lambda q: (q["recall_10"] if not math.isnan(q["recall_10"]) else 0))[:20]
    with open(failures_path, "w") as f:
        f.write("# Arm 1 (dense, BGE-M3) — 20 worst failures on dev split\n\n")
        for w in worst:
            f.write(f"## {w['id']} (recall@10={w['recall_10']:.2f}, filing={w['chunk_file']})\n")
            f.write(f"Q: {w['question']}\n\n")
            f.write(f"Top 5 retrieved: {w['top_5_retrieved']}\n\n")
    print(f"Worst failures written to {failures_path}")


if __name__ == "__main__":
    main()
