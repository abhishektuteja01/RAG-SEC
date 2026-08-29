"""Day 5, corpus-growth embedding, stage 3 (laptop): join the HPC's embeddings back
with local chunk text/metadata and insert into Postgres. Mirrors day3_index_chunks.py's
insert shape exactly, just fed from a file instead of computing embeddings locally.

Usage:
    python day5_load_embeddings.py embed_results.jsonl
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag_sec.store import HNSW_INDEX_SQL, get_conn, init_schema

CHUNKS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chunks"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python day5_load_embeddings.py <embed_results.jsonl>")
        sys.exit(1)

    results_path = Path(sys.argv[1])

    by_stem: dict[str, dict[int, list[float]]] = {}
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                by_stem.setdefault(row["filing_stem"], {})[row["chunk_index"]] = row["embedding"]

    print(f"Loaded embeddings for {len(by_stem)} filings")

    init_schema()
    with get_conn() as conn:
        done_stems = {r[0] for r in conn.execute("SELECT DISTINCT filing_stem FROM chunks").fetchall()}

        for i, (stem, emb_by_idx) in enumerate(by_stem.items(), 1):
            if stem in done_stems:
                continue
            chunks = json.loads((CHUNKS_DIR / f"{stem}.json").read_text())
            rows = []
            for idx, c in enumerate(chunks):
                if idx not in emb_by_idx:
                    print(f"WARNING: missing embedding for {stem} chunk {idx}, skipping filing")
                    rows = []
                    break
                rows.append((stem, idx, c.get("heading"), c.get("n_tokens"), c["text"], emb_by_idx[idx]))
            if not rows:
                continue
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks (filing_stem, chunk_index, heading, n_tokens, text, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    rows,
                )
            conn.commit()
            print(f"[{i}/{len(by_stem)}] {stem}: {len(rows)} chunks inserted")

        print("Rebuilding HNSW index...")
        conn.execute(HNSW_INDEX_SQL)
        conn.commit()

    print("Done.")


if __name__ == "__main__":
    main()
