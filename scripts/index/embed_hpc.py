"""Day 5, corpus-growth embedding, stage 2 (HPC GPU node): embed every chunk in the
payload with BGE-M3. Self-contained -- no DB, no `rag_sec` import -- and checkpointed
the same way as day5_hpc_rerank.py: writes results incrementally, skips ids already
done on restart.

Usage:
    python day5_hpc_embed.py embed_payload.json embed_results.jsonl
"""

import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"  # duplicated, not imported from rag_sec.config -- this script
# runs on the HPC node with no rag_sec package installed, by design (ARM3-2)
BATCH_SIZE = 64  # larger than the laptop's 32 -- GPU has the headroom


def load_done_keys(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                done.add(f"{row['filing_stem']}::{row['chunk_index']}")
    return done


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python day5_hpc_embed.py <payload.json> <output.jsonl>")
        sys.exit(1)

    payload_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    chunks = json.loads(payload_path.read_text())
    done_keys = load_done_keys(output_path)
    remaining = [c for c in chunks if f"{c['filing_stem']}::{c['chunk_index']}" not in done_keys]
    print(f"{len(chunks)} total, {len(done_keys)} already done, {len(remaining)} remaining")

    if not remaining:
        print("Nothing to do.")
        return

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # force safetensors -- avoids transformers' torch.load-based legacy .bin loading
    # path, which refuses to run under torch<2.6 (CVE-2025-32434) and our HPC node's
    # cu121 wheel index tops out at torch 2.5.1
    model = SentenceTransformer(MODEL_NAME, device=device, model_kwargs={"use_safetensors": True})

    with open(output_path, "a") as out:
        for i in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[i : i + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            embeddings = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)
            for c, emb in zip(batch, embeddings):
                out.write(
                    json.dumps({"filing_stem": c["filing_stem"], "chunk_index": c["chunk_index"], "embedding": emb.tolist()})
                    + "\n"
                )
            out.flush()
            print(f"[{min(i + BATCH_SIZE, len(remaining))}/{len(remaining)}]")

    print(f"Done. Results in {output_path}")


if __name__ == "__main__":
    main()
