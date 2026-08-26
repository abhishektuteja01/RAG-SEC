"""Day 2: chunk all parsed filings in data/parsed/ into data/chunks/.

Reads each data/parsed/*.json (list of text/table blocks from rag_sec.parsing),
packs them into token-budgeted chunks via rag_sec.chunking, and writes
data/chunks/<stem>.json plus a per-filing stats log.
"""

import dataclasses
import json
import time
from pathlib import Path

from rag_sec.chunking import chunk_blocks
from rag_sec.parsing import TableBlock, TextBlock

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
LOG_PATH = DATA_DIR / "day2_chunk_log.jsonl"


def load_blocks(path: Path) -> list:
    data = json.loads(path.read_text())
    blocks = []
    for d in data:
        if "rows" in d:
            blocks.append(TableBlock(rows=d["rows"]))
        else:
            blocks.append(TextBlock(text=d["text"], is_title=d["is_title"]))
    return blocks


def main() -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(PARSED_DIR.glob("*.json"))
    print(f"Chunking {len(paths)} filings")

    with open(LOG_PATH, "w") as log_file:
        for i, path in enumerate(paths, 1):
            stem = path.stem
            t0 = time.monotonic()
            blocks = load_blocks(path)
            chunks = chunk_blocks(blocks)
            dt = time.monotonic() - t0

            out_path = CHUNKS_DIR / f"{stem}.json"
            out_path.write_text(json.dumps([dataclasses.asdict(c) for c in chunks], ensure_ascii=False))

            sizes = [c.n_tokens for c in chunks]
            log_file.write(
                json.dumps(
                    {
                        "stem": stem,
                        "n_chunks": len(chunks),
                        "median_tokens": sorted(sizes)[len(sizes) // 2] if sizes else 0,
                        "min_tokens": min(sizes) if sizes else 0,
                        "max_tokens": max(sizes) if sizes else 0,
                        "chunk_s": round(dt, 2),
                    }
                )
                + "\n"
            )
            log_file.flush()
            print(f"[{i}/{len(paths)}] {stem}: {len(chunks)} chunks, {dt:.1f}s")

    print(f"Done. Log at {LOG_PATH}")


if __name__ == "__main__":
    main()
