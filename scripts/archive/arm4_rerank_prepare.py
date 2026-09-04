"""ARCHIVED -- Day 6, Arm 4, stage 1 of the split job (old filename in `git log --follow`).

What it did: stage 1 of Arm 4 on the laptop, once per variant -- ran that variant's
retrieval and dumped the fused top-50 with text inlined, candidates carrying a `variant`
field so identities cannot collide across A/B/C.

Provenance: DECISIONS.md `ARM4-3`, `ARM4-4`. Outputs:
`data/day6_arm4_{A,B,C}_rerank_payload.json`. B and C have been deleted; **the A payload
(230 MB) is still on disk and is not dead weight from Arm 4 alone** -- `COST-14` reused it
as the byte-identical 50-candidate pool for the first slice payload, which is why the
compression arms and the whole-chunk baseline saw the same candidates.

Replaced by: `scripts/retrieval/rerank_prepare.py`.

Safe to run today? Yes with Postgres; minutes per variant. Re-running `--variant A`
overwrites the 230 MB payload `COST-14` used. The content should come out identical, but
there is no reason to find out.

--- original header, kept verbatim ---

Day 6, Arm 4, stage 1 (laptop): run this variant's retrieval (dense + BM25 + RRF,
same as arm4_singleprocess.py) for every dev-split question and dump each question's fused
top-50 candidates -- with chunk text inlined -- to a self-contained JSON payload. This is
the only stage that needs Postgres; the GPU side (arm4_rerank_hpc.py) needs no DB access.

Candidates carry (filing_stem, chunk_index, variant) triples, not (filing_stem,
chunk_index) pairs like Arm 3's arm3_rerank_prepare.py -- Strategy A and B/C each
number a filing's chunks from 0 independently, so a bare pair collides across variants
(DECISIONS.md ARM4-*). Candidate pool per variant matches arm4_singleprocess.py's retrieval
WHERE clause: `variant='A'` rows not superseded for this variant, plus this variant's own
rows -- so scoring later (arm4_rerank_score.py) sees exactly what retrieval saw here.

Run once per variant:
    python arm4_rerank_prepare.py --variant A
    python arm4_rerank_prepare.py --variant B
    python arm4_rerank_prepare.py --variant C
"""

import argparse
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
    parser.add_argument("-n", type=int, default=None, help="limit to the first N dev questions (sanity/timing check)")
    args = parser.parse_args()
    variant = args.variant

    payload_path = DATA_DIR / f"day6_arm4_{variant}_rerank_payload.json"

    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    if args.n:
        dev = dev.head(args.n)
    print(f"Preparing Arm 4 variant={variant} rerank payload for {len(dev)} dev-split questions")

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    payload = []

    with get_conn() as conn:
        for _, row in tqdm(dev.iterrows(), total=len(dev)):
            query_emb = embed_model.encode(row["question"], normalize_embeddings=True)
            dense = retrieve_dense(conn, query_emb, variant, CANDIDATE_K)
            bm25 = retrieve_bm25(conn, row["question"], variant, CANDIDATE_K)
            fused = rrf_fuse([dense, bm25])[:TOP_K]
            texts = fetch_texts(conn, fused)

            payload.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "filing_stem": f"{row['company_symbol']}_{int(row['report_year'])}_{int(row['company_cik'])}",
                    "candidates": [
                        [stem, idx, v, texts[(stem, idx, v)]] for stem, idx, v in fused if (stem, idx, v) in texts
                    ],
                }
            )

    payload_path.write_text(json.dumps(payload))
    print(f"Wrote {len(payload)} questions' candidates to {payload_path}")
    print("Copy this file to the HPC node via the transfer node (DECISIONS.md ARM3-2 -- the")
    print("login node throttles/kills large transfers), e.g.:")
    print(f"  scp {payload_path} <user>@xfer.discovery.neu.edu:~/rerank_payload_{variant}.json")


if __name__ == "__main__":
    main()
