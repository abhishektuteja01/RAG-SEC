"""Day 5, corpus-growth embedding, stage 1 (laptop): find every filing in data/chunks/
not yet in Postgres (the 195 new filings from day4_ingest_next200.py's completion run)
and dump their chunk texts to a payload file for GPU embedding on HPC.

Same split-job reasoning as day5_prepare_rerank_payload.py: embedding needs no DB
access on the GPU side, so a dropped HPC connection can't corrupt anything.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag_sec.store import get_conn

CHUNKS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chunks"
PAYLOAD_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "day5_embed_payload.json"


def main() -> None:
    with get_conn() as conn:
        done_stems = {r[0] for r in conn.execute("SELECT DISTINCT filing_stem FROM chunks").fetchall()}

    paths = sorted(CHUNKS_DIR.glob("*.json"))
    todo = [p for p in paths if p.stem not in done_stems]
    print(f"{len(done_stems)}/{len(paths)} filings already indexed, {len(todo)} to embed")

    payload = []
    for path in todo:
        chunks = json.loads(path.read_text())
        for idx, c in enumerate(chunks):
            payload.append({"filing_stem": path.stem, "chunk_index": idx, "text": c["text"]})

    PAYLOAD_PATH.write_text(json.dumps(payload))
    print(f"Wrote {len(payload)} chunks (from {len(todo)} filings) to {PAYLOAD_PATH}")


if __name__ == "__main__":
    main()
