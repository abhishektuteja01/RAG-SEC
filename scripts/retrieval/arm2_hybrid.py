"""Arm 2: hybrid retrieval -- dense (Arm 1) + BM25 (pg_search), fused with
Reciprocal Rank Fusion. Same dev split, same metrics, same TOP_K as Arm 1
(arm1_dense.py) so the two results are directly comparable -- one change
(add a second ranked list + fuse it), per spec.md 2.2.

DECISIONS.md INFRA-4/ARM2-1: real BM25 via pg_search, not tsvector/ts_rank (spec.md's
explicit trap -- no length normalization or term saturation, degrades on long docs).
"""

import argparse
import json
import math
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer

from rag_sec.candidates import LIVE_VARIANT, RRF_K, bm25, dense, rrf_fuse
from rag_sec.company import resolve as resolve_companies
from rag_sec.config import EMBED_MODEL_NAME
from rag_sec.eval import gold_relevant_chunk_ids, load_matched_questions, mean_and_stderr, mrr, ndcg_at_k, recall_at_k
from rag_sec.store import get_conn

TOP_K = 50
CANDIDATE_K = 50  # how many each of dense/BM25 contributes to the fused pool
RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "day4_arm2_dev_results.json"
FAILURES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "day4_arm2_dev_failures.md"


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
    print(f"Running Arm 2 (hybrid: dense + BM25, RRF fusion) on {len(dev)} dev-split questions")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    per_question = []

    with get_conn() as conn:
        for _, row in dev.iterrows():
            query_emb = model.encode(row["question"], normalize_embeddings=True)
            tickers = resolve_companies(row["question"]) if opts.company_filter else []
            dense_hits = dense(conn, query_emb, CANDIDATE_K, LIVE_VARIANT, tickers)
            bm25_hits = bm25(conn, row["question"], CANDIDATE_K, LIVE_VARIANT, tickers)
            retrieved = rrf_fuse([dense_hits, bm25_hits])[:TOP_K]

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
        json.dumps(
            {
                "model": EMBED_MODEL_NAME,
                "candidate_k": CANDIDATE_K,
                "rrf_k": RRF_K,
                "top_k": TOP_K,
                "n": len(dev),
                "metrics": metrics,
                "per_question": per_question,
            },
            indent=2,
        )
    )
    print(f"Results written to {results_path}")

    worst = sorted(per_question, key=lambda q: (q["recall_10"] if not math.isnan(q["recall_10"]) else 0))[:20]
    with open(failures_path, "w") as f:
        f.write("# Arm 2 (hybrid: dense + BM25, RRF) — 20 worst failures on dev split\n\n")
        for w in worst:
            f.write(f"## {w['id']} (recall@10={w['recall_10']:.2f}, filing={w['chunk_file']})\n")
            f.write(f"Q: {w['question']}\n\n")
            f.write(f"Top 5 retrieved: {w['top_5_retrieved']}\n\n")
    print(f"Worst failures written to {failures_path}")


if __name__ == "__main__":
    main()
