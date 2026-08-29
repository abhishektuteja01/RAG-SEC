"""Day 5, Arm 3, stage 1 (laptop): run Arm 2's retrieval (dense + BM25 + RRF) for every
dev-split question and dump each question's fused top-50 candidates -- with chunk text
inlined -- to a single JSON file. This is the only stage that needs Postgres.

DECISIONS.md ARM3-2 (pending): CPU-only cross-encoder reranking of 837 questions x 50
candidates projected at ~25h wall clock (observed: per-question latency climbed from
~27s to a ~110s plateau over the first 15 questions, laptop under memory pressure from
other running apps) -- not viable. Split the job instead of tunneling a live DB
connection to the HPC node: this script prepares a self-contained payload so the GPU
side (`day5_hpc_rerank.py`) needs no DB access and can't be broken by a dropped
connection mid-run.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_sec.config import EMBED_MODEL_NAME
from rag_sec.eval import load_matched_questions
from rag_sec.store import get_conn

TOP_K = 50
CANDIDATE_K = 50
RRF_K = 60
PAYLOAD_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "day5_rerank_payload.json"


def retrieve_dense(conn, embedding, k: int) -> list[tuple[str, int]]:
    rows = conn.execute(
        "SELECT filing_stem, chunk_index FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (embedding, k),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def retrieve_bm25(conn, query_text: str, k: int) -> list[tuple[str, int]]:
    rows = conn.execute(
        """SELECT filing_stem, chunk_index, paradedb.score(id) AS s
           FROM chunks
           WHERE id @@@ paradedb.match('text', %s)
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
        "SELECT filing_stem, chunk_index, text FROM chunks WHERE filing_stem = ANY(%s)",
        (stems,),
    ).fetchall()
    lookup = {(r[0], r[1]): r[2] for r in rows}
    return {p: lookup[p] for p in pairs if p in lookup}


def main() -> None:
    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    print(f"Preparing rerank payload for {len(dev)} dev-split questions")

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    payload = []

    with get_conn() as conn:
        for _, row in tqdm(dev.iterrows(), total=len(dev)):
            query_emb = embed_model.encode(row["question"], normalize_embeddings=True)
            dense = retrieve_dense(conn, query_emb, CANDIDATE_K)
            bm25 = retrieve_bm25(conn, row["question"], CANDIDATE_K)
            fused = rrf_fuse([dense, bm25])[:TOP_K]
            texts = fetch_texts(conn, fused)

            payload.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "chunk_file": row["chunk_file"],
                    "candidates": [
                        [stem, idx, texts[(stem, idx)]] for stem, idx in fused if (stem, idx) in texts
                    ],
                }
            )

    PAYLOAD_PATH.write_text(json.dumps(payload))
    print(f"Wrote {len(payload)} questions' candidates to {PAYLOAD_PATH}")
    print("Copy this file to the HPC node, e.g.:")
    print(f"  scp {PAYLOAD_PATH} <user>@explorer.northeastern.edu:~/rerank_payload.json")


if __name__ == "__main__":
    main()
