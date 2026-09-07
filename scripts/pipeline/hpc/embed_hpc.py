"""Pipeline phase 03, stage 2 — embed a payload with BGE-M3 on a GPU node.

THIS FILE RUNS ON THE CLUSTER, NOT HERE. RUNBOOK.md scp's it to the login/xfer host and
invokes it by BARE NAME from the home directory:

    scp scripts/pipeline/hpc/embed_hpc.py $NEU@xfer.discovery.neu.edu:~/
    cd ~ && python -u embed_hpc.py retr7_embed_payload.json retr7_embed_results.jsonl

So its basename is load-bearing and must stay `embed_hpc.py`.

*** IT DELIBERATELY IMPORTS NOTHING FROM `rag_sec`, AND HAS NO sys.path BOOTSTRAP.
*** The cluster has no install of this project (DECISIONS.md ARM3-2 / INFRA-6), so the
*** model name and the device-selection ladder are DUPLICATED here on purpose. Making this
*** file import the library would break the only route this project has to a GPU. Do not
*** "fix" the duplication.

PRODUCES
    <output.jsonl> — one line per chunk: {"filing_stem", "chunk_index", "embedding"}.
    Written and flushed incrementally, and restart-safe: ids already in the file are
    skipped, so a requeued or timed-out job resumes instead of starting over.

READS
    <payload.json> — a list of {"filing_stem", "chunk_index", "text"}. Both of phase 03's
    cluster legs (`new-filings --prepare` and `changed-chunks --prepare`) emit exactly this
    shape, which is why one unchanged script serves both (INFRA-12).

DECISIONS.md ROWS THIS BACKS
    ARM3-2 / INFRA-6   GPU work belongs on the cluster and needs no DB access, so a
                       dropped connection there cannot corrupt anything; and no rag_sec
                       install exists on the node.
    INFRA-12           the RETR-7/RETR-8 re-index reused this script unchanged.
    INFRA-13           measured throughput: 47,312 chunks in ~72 min ≈ 11 chunks/s on the
                       HPC V100 at BATCH_SIZE=64. The counter-anchor is this laptop's MPS
                       at ~1.4 chunks/s, ~8x slower.

WHEN THIS ACTUALLY RAN
    2026-08-28   the corpus-growth pass (26,737 chunks, from data/day5_embed_payload.json
                 — a plan-number filename, written 2026-08-28 17:20).
    2026-09-04   the RETR-7/RETR-8 re-index: job 9949705, allocation
                 05:07:11 -> 06:23:31, all 47,312 chunks written by 06:21.

TRAPS
  * Restart safety depends on APPENDING to the same output path. Point it at a fresh file
    and it re-embeds everything; point it at a file from a DIFFERENT payload and those ids
    are silently treated as done. Phase 03's --load stage is what catches that: it
    recomputes the work set and refuses if any chunk has no embedding.
  * `use_safetensors=True` is not cosmetic. It avoids transformers' torch.load-based legacy
    .bin path, which refuses to run under torch<2.6 (CVE-2025-32434), and the HPC node's
    cu121 wheel index tops out at torch 2.5.1.
  * torch is imported inside main(), after the "nothing to do" exit, so a no-op resume does
    not pay the import or touch the GPU.
"""

import argparse
import json
from pathlib import Path

# ─── CONSTANTS ──────────────────────────────────────────────────────────────────
# Duplicated from rag_sec.config, NOT imported -- see the docstring. If the project's
# embedding model ever changes, this line must be changed by hand in lockstep, because
# nothing can check it from here.
MODEL_NAME = "BAAI/bge-m3"

# 64, larger than the local leg's 32: the V100 has the headroom, and this is the value
# INFRA-13's 11 chunks/s was measured at. Changing it invalidates that figure.
BATCH_SIZE = 64

# Resume key. filing_stem and chunk_index jointly identify a chunk; "::" cannot occur in a
# stem (which is TICKER_YEAR_CIK), so the join is unambiguous.
KEY_SEP = "::"


def load_done_keys(output_path: Path) -> set[str]:
    """Ids already written, so a restarted job skips them."""
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                done.add(f"{row['filing_stem']}{KEY_SEP}{row['chunk_index']}")
    return done


def pick_device() -> str:
    """cuda > mps > cpu. Duplicated from rag_sec.config.pick_device() -- this script runs
    on the HPC node with no rag_sec package, by design. mps matters only when it is run
    locally on Apple Silicon (INFRA-6)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("payload", help="payload.json: [{filing_stem, chunk_index, text}, ...]")
    ap.add_argument("output", help="output.jsonl, APPENDED to; already-done ids are skipped")
    args = ap.parse_args()

    payload_path = Path(args.payload)
    output_path = Path(args.output)

    # ─── STEP 1: work out what is left to do ──────────────────────────────────
    chunks = json.loads(payload_path.read_text())
    done_keys = load_done_keys(output_path)
    remaining = [
        c for c in chunks
        if f"{c['filing_stem']}{KEY_SEP}{c['chunk_index']}" not in done_keys
    ]
    print(f"{len(chunks)} total, {len(done_keys)} already done, {len(remaining)} remaining")

    if not remaining:
        print("Nothing to do.")
        return

    # ─── STEP 2: load the model on the best available device ──────────────────
    from sentence_transformers import SentenceTransformer

    device = pick_device()
    print(f"Using device: {device}")
    model = SentenceTransformer(
        MODEL_NAME, device=device, model_kwargs={"use_safetensors": True}
    )

    # ─── STEP 3: embed in batches, flushing after each one ────────────────────
    with open(output_path, "a") as out:
        for i in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[i : i + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            embeddings = model.encode(
                texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
                show_progress_bar=False,
            )
            for c, emb in zip(batch, embeddings):
                out.write(
                    json.dumps({"filing_stem": c["filing_stem"],
                                "chunk_index": c["chunk_index"],
                                "embedding": emb.tolist()})
                    + "\n"
                )
            out.flush()
            print(f"[{min(i + BATCH_SIZE, len(remaining))}/{len(remaining)}]")

    print(f"Done. Results in {output_path}")


if __name__ == "__main__":
    main()
