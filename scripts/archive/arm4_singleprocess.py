"""ARCHIVED -- Day 6's single-process Arm 4 reference run (old filename in `git log --follow`).

What it did: Arm 3's pipeline run once per table-indexing variant (A/B/C) in a single
process, keying every chunk by `(filing_stem, chunk_index, variant)` because A and B/C each
number a filing's chunks from zero.

Provenance: DECISIONS.md `ARM4-1`/`ARM4-2` (what B and C are), `ARM4-3` (scoped to gold
tables), `ARM4-4` (the chunk-boundary drift bug), `ARM4-10` (A beat both). Outputs:
`data/day6_arm4_{A,B,C}_cpu_dev_results.json` / `_failures.md` -- 5-question CPU warm-ups
only. These are **not** the published Arm 4 numbers and read as ~0.9 scores; do not quote
them.

Replaced by: `arm4_rerank_{prepare,hpc,score}.py` in this folder for Arm 4 itself. The live
pipeline is `scripts/retrieval/rerank_*.py`, which queries `variant = 'A'` only.

Safe to run today? Only with a small `-n`, and it needs Postgres plus both models. It reads
B/C rows on purpose -- that is legitimate here and nowhere else, since those rows are what
`RETR-24` showed contaminating Arms 1-3 when a query forgets its `variant` predicate.

--- original header, kept verbatim ---

Day 6, Arm 4 -- single-machine reference pipeline: same dense + BM25/RRF + reranker
pipeline as Arm 3 (arm3_singleprocess.py), run separately per table-indexing variant (A/B/C)
via `--variant`. Same dev split, same metrics, same reranker -- the only thing that
changes between runs is which chunk represents each gold table.

REFERENCE implementation, not the one used to produce the recorded numbers -- full-dev-set
CPU reranking isn't viable (DECISIONS.md ARM3-2). Use `-n` for a quick local check; the
real runs go through the HPC split-job pipeline (arm4_rerank_prepare.py ->
arm4_rerank_hpc.py -> arm4_rerank_score.py), once per variant.

Chunk identity here is (filing_stem, chunk_index, variant), not (filing_stem, chunk_index)
like Arms 1-3 -- Strategy A and B/C each number a filing's chunks from 0 independently, so
a bare (filing_stem, chunk_index) pair collides across variants (DECISIONS.md ARM4-*). A
run's candidate pool is `variant='A'` rows not superseded for this variant, plus this
variant's own rows -- see `eval.gold_relevant_chunk_evidence_db` for the matching WHERE
clause used on the relevance-labeling side, so retrieval and scoring never disagree about
what's in play.
"""

import argparse
import json
import math
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_sec.config import EMBED_MODEL_NAME, RERANK_MODEL_NAME, pick_device
from rag_sec.eval import gold_relevant_chunk_ids_db, load_matched_questions, mean_and_stderr, mrr, ndcg_at_k, recall_at_k
from rag_sec.store import get_conn

TOP_K = 50
CANDIDATE_K = 50
RRF_K = 60
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

ChunkId = tuple[str, int, str]  # (filing_stem, chunk_index, variant)


def retrieve_dense(conn, embedding, variant: str, k: int) -> list[ChunkId]:
    rows = conn.execute(
        """SELECT filing_stem, chunk_index, variant FROM chunks
           WHERE (variant = 'A' AND NOT (%s = ANY(excluded_by_variant))) OR variant = %s
           ORDER BY embedding <=> %s LIMIT %s""",
        (variant, variant, embedding, k),
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def retrieve_bm25(conn, query_text: str, variant: str, k: int) -> list[ChunkId]:
    rows = conn.execute(
        """SELECT filing_stem, chunk_index, variant, paradedb.score(id) AS s
           FROM chunks
           WHERE id @@@ paradedb.match('text', %s)
             AND ((variant = 'A' AND NOT (%s = ANY(excluded_by_variant))) OR variant = %s)
           ORDER BY s DESC LIMIT %s""",
        (query_text, variant, variant, k),
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def rrf_fuse(ranked_lists: list[list[ChunkId]], k: int = RRF_K) -> list[ChunkId]:
    scores: dict[ChunkId, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


def fetch_texts(conn, triples: list[ChunkId]) -> dict[ChunkId, str]:
    if not triples:
        return {}
    stems = list({t[0] for t in triples})
    rows = conn.execute(
        "SELECT filing_stem, chunk_index, variant, text FROM chunks WHERE filing_stem = ANY(%s)",
        (stems,),
    ).fetchall()
    lookup = {(r[0], r[1], r[2]): r[3] for r in rows}
    return {t: lookup[t] for t in triples if t in lookup}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=["A", "B", "C"])
    parser.add_argument("-n", type=int, default=None, help="limit to the first N dev questions (sanity check)")
    args = parser.parse_args()
    variant = args.variant

    results_path = DATA_DIR / f"day6_arm4_{variant}_cpu_dev_results.json"
    failures_path = DATA_DIR / f"day6_arm4_{variant}_cpu_dev_failures.md"

    # Full 1235-question dev split, same universe as Arms 1-3 (DECISIONS.md ARM4-9: ~224
    # of these have no matched gold table for any variant and will contribute NaN to every
    # metric here regardless of `variant` -- kept in for direct comparability, not dropped).
    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    if args.n:
        dev = dev.head(args.n)
    print(f"Running Arm 4 variant={variant} (CPU reference) on {len(dev)} dev-split questions")

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
            dense = retrieve_dense(conn, query_emb, variant, CANDIDATE_K)
            bm25 = retrieve_bm25(conn, row["question"], variant, CANDIDATE_K)
            fused = rrf_fuse([dense, bm25])[:TOP_K]
            texts = fetch_texts(conn, fused)
            candidates = [c for c in fused if c in texts]

            pairs = [(row["question"], texts[c]) for c in candidates]
            scores = cross_encoder.predict(pairs, batch_size=32) if pairs else []
            order = sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)
            retrieved = [candidates[j] for j in order]

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
                "variant": variant,
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
    print(f"Results written to {results_path}")

    worst = sorted(per_question, key=lambda q: (q["recall_10"] if not math.isnan(q["recall_10"]) else 0))[:20]
    with open(failures_path, "w") as f:
        f.write(f"# Arm 4 variant={variant} (CPU reference) — 20 worst failures on dev split\n\n")
        for w in worst:
            f.write(f"## {w['id']} (recall@10={w['recall_10']:.2f}, filing={w['filing_stem']})\n")
            f.write(f"Q: {w['question']}\n\n")
            f.write(f"Top 5 retrieved: {w['top_5_retrieved']}\n\n")
    print(f"Worst failures written to {failures_path}")


if __name__ == "__main__":
    main()
