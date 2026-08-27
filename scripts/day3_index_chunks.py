"""Day 3, Arm 1: embed every chunk with BGE-M3 and load into pgvector.

Loads data/chunks/*.json into the `chunks` table, skipping any filing_stem already
present so an interrupted run can be resumed without re-embedding (BGE-M3 encode is
the expensive part). Delete a filing's rows manually first if you want to force a
re-embed of it (e.g. after a re-chunk).
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer

from rag_sec.store import HNSW_INDEX_SQL, get_conn, init_schema

CHUNKS_DIR = Path(__file__).resolve().parent.parent / "data" / "chunks"
EMBED_BATCH_SIZE = 32
MODEL_NAME = "BAAI/bge-m3"


def main() -> None:
    init_schema()
    model = SentenceTransformer(MODEL_NAME)

    paths = sorted(CHUNKS_DIR.glob("*.json"))
    print(f"Embedding chunks from {len(paths)} filings")

    with get_conn() as conn:
        done_stems = {r[0] for r in conn.execute("SELECT DISTINCT filing_stem FROM chunks").fetchall()}
        print(f"{len(done_stems)}/{len(paths)} filings already indexed, skipping those")

        for i, path in enumerate(paths, 1):
            stem = path.stem
            if stem in done_stems:
                continue
            chunks = json.loads(path.read_text())
            texts = [c["text"] for c in chunks]
            if not texts:
                continue
            embeddings = model.encode(
                texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True
            )
            rows = [
                (stem, idx, c.get("heading"), c.get("n_tokens"), c["text"], emb)
                for idx, (c, emb) in enumerate(zip(chunks, embeddings))
            ]
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks (filing_stem, chunk_index, heading, n_tokens, text, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    rows,
                )
            conn.commit()
            print(f"[{i}/{len(paths)}] {stem}: {len(rows)} chunks embedded")

        print("Building HNSW index...")
        conn.execute(HNSW_INDEX_SQL)
        conn.commit()

    print("Done.")


if __name__ == "__main__":
    main()
