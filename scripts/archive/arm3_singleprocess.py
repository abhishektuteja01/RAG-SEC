"""ARCHIVED -- Day 5's single-process Arm 3 reference run (old filename in `git log --follow`).

What it did: ran the whole of Arm 3 in one laptop process -- dense + BM25/RRF fusion over
the dev split, then re-ordered the fused top-50 with the `bge-reranker-v2-m3` cross-encoder.

Provenance: DECISIONS.md `ARM3-1` (why this reranker) and `ARM3-2` (why this route was
abandoned). It produced no published number. Its outputs
`data/day5_arm3_cpu_dev_results.json` / `_failures.md` are not on disk, because a full CPU
run was never finished.

Replaced by: the split-job trio, now `scripts/retrieval/rerank_prepare.py` ->
`rerank_hpc.py` -> `rerank_score.py`. The recorded Arm 3 numbers came from this file's
Day 5 siblings (`arm3_rerank_*.py`, also archived here), not from here.

Safe to run today? Yes with a small `-n`, and only then. Needs Postgres and downloads both
models. A full dev run on CPU is not viable (`ARM3-2`: ~110 s/question plateau, ~25 h). It
writes only its own `*_cpu_dev_*` files, so it cannot clobber a published result.

--- original header, kept verbatim ---

Day 5, Arm 3 -- single-machine reference pipeline: Arm 2's hybrid retrieval (dense +
BM25 + RRF, same as arm2_hybrid.py), reranked in-process with a local cross-encoder.
Same dev split, same metrics as Arms 1/2, one change per spec.md 2.2 (rerank the fused
top-50).

This is a REFERENCE implementation, not the one used to produce the recorded Arm 3
baseline. A CPU run of the full dev split (n=1235) is not viable -- DECISIONS.md
ARM3-2 measured per-question latency climbing to a ~110s plateau under normal laptop
load, i.e. ~1 day of wall clock. The actual baseline was produced by the split-job
pipeline (arm3_rerank_prepare.py -> arm3_rerank_hpc.py -> arm3_rerank_score.py)
on an HPC GPU node instead. Use this script for a quick local sanity check (small `-n`)
or if you do have a CUDA GPU on the machine running it; use the split-job pipeline for
a full run.

DECISIONS.md ARM3-1: bge-reranker-v2-m3, not Qwen3-Reranker -- one forward pass, one
score, vs. an autoregressive decode per candidate.
"""

import argparse
import json
import math
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_sec.config import EMBED_MODEL_NAME, RERANK_MODEL_NAME, pick_device
from rag_sec.eval import gold_relevant_chunk_ids, load_matched_questions, mean_and_stderr, mrr, ndcg_at_k, recall_at_k
from rag_sec.store import get_conn

TOP_K = 50
CANDIDATE_K = 50
RRF_K = 60
RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "day5_arm3_cpu_dev_results.json"
FAILURES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "day5_arm3_cpu_dev_failures.md"


def retrieve_dense(conn, embedding, k: int) -> list[tuple[str, int]]:
    rows = conn.execute(
        "SELECT filing_stem, chunk_index FROM chunks WHERE variant = 'A' ORDER BY embedding <=> %s LIMIT %s",
        (embedding, k),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def retrieve_bm25(conn, query_text: str, k: int) -> list[tuple[str, int]]:
    rows = conn.execute(
        """SELECT filing_stem, chunk_index, paradedb.score(id) AS s
           FROM chunks
           WHERE id @@@ paradedb.match('text', %s) AND variant = 'A'
           ORDER BY s DESC LIMIT %s""",
        (query_text, k),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def rrf_fuse(ranked_lists: list[list[tuple[str, int]]], k: int = RRF_K) -> list[tuple[str, int]]:
    scores: dict[tuple[str, int], float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


def fetch_texts(conn, pairs: list[tuple[str, int]]) -> dict[tuple[str, int], str]:
    if not pairs:
        return {}
    stems = list({p[0] for p in pairs})
    rows = conn.execute(
        "SELECT filing_stem, chunk_index, text FROM chunks WHERE variant = 'A' AND filing_stem = ANY(%s)",
        (stems,),
    ).fetchall()
    lookup = {(r[0], r[1]): r[2] for r in rows}
    return {p: lookup[p] for p in pairs if p in lookup}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=None, help="limit to the first N dev questions (sanity check)")
    args = parser.parse_args()

    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    if args.n:
        dev = dev.head(args.n)
    print(f"Running Arm 3 (CPU reference) on {len(dev)} dev-split questions")

    device = pick_device()
    print(f"Using device: {device}")
    if device == "cpu" and len(dev) > 50:
        print("WARNING: CPU reranking is slow (DECISIONS.md ARM3-2) -- consider -n for a quick check")

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    cross_encoder = CrossEncoder(RERANK_MODEL_NAME, device=device)
    per_question = []

    with get_conn() as conn:
        for _, row in dev.iterrows():
            query_emb = embed_model.encode(row["question"], normalize_embeddings=True)
            dense = retrieve_dense(conn, query_emb, CANDIDATE_K)
            bm25 = retrieve_bm25(conn, row["question"], CANDIDATE_K)
            fused = rrf_fuse([dense, bm25])[:TOP_K]
            texts = fetch_texts(conn, fused)
            candidates = [c for c in fused if c in texts]

            pairs = [(row["question"], texts[c]) for c in candidates]
            scores = cross_encoder.predict(pairs, batch_size=32) if pairs else []
            order = sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)
            retrieved = [candidates[j] for j in order]

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

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "embed_model": EMBED_MODEL_NAME,
                "rerank_model": RERANK_MODEL_NAME,
                "rerank_device": device,
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
    print(f"Results written to {RESULTS_PATH}")

    worst = sorted(per_question, key=lambda q: (q["recall_10"] if not math.isnan(q["recall_10"]) else 0))[:20]
    with open(FAILURES_PATH, "w") as f:
        f.write("# Arm 3 (CPU reference) — 20 worst failures on dev split\n\n")
        for w in worst:
            f.write(f"## {w['id']} (recall@10={w['recall_10']:.2f}, filing={w['chunk_file']})\n")
            f.write(f"Q: {w['question']}\n\n")
            f.write(f"Top 5 retrieved: {w['top_5_retrieved']}\n\n")
    print(f"Worst failures written to {FAILURES_PATH}")


if __name__ == "__main__":
    main()
