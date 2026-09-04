"""Day 6, Arm 4, stage 2 (HPC GPU node): score every question's candidates with a local
cross-encoder reranker. Deliberately self-contained -- no `rag_sec` package import, no
DB connection -- so it only needs `torch`, `sentence-transformers`, `tqdm` on the HPC
side and can't be broken by a dropped connection back to the laptop's Postgres.

Same design as Arm 3's day5_hpc_rerank.py (checkpointed resume, cross-question batching
into QUESTIONS_PER_CHUNK-sized predict() calls -- see that file's docstring for why).
Only difference: each candidate is a 4-element [filing_stem, chunk_index, variant, text]
instead of Arm 3's 3-element [filing_stem, chunk_index, text] -- Strategy A and B/C each
number a filing's chunks from 0 independently, so `variant` has to travel with every
candidate to keep identities from colliding across variants (DECISIONS.md ARM4-*). This
script itself doesn't care which variant a payload is for -- run it unmodified, once per
payload file.

Usage:
    python day6_hpc_rerank.py rerank_payload_A.json rerank_scores_A.jsonl
    python day6_hpc_rerank.py rerank_payload_B.json rerank_scores_B.jsonl
    python day6_hpc_rerank.py rerank_payload_C.json rerank_scores_C.jsonl

Input: JSON list of {id, question, filing_stem, candidates: [[filing_stem, chunk_index, variant, text], ...]}
    (produced by day6_prepare_rerank_payload.py on the laptop side).
Output: JSONL, one line per question: {id, reranked: [[filing_stem, chunk_index, variant], ...], latency_s}
    (latency_s is the chunk's total wall time divided evenly across its questions --
    an average, not a true per-question measurement, since they're scored together)
"""

import json
import sys
import time
from pathlib import Path

from sentence_transformers import CrossEncoder
from tqdm import tqdm

RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # duplicated, not imported from
# rag_sec.config -- this script runs on the HPC node with no rag_sec package, by design
QUESTIONS_PER_CHUNK = 20  # flatten this many questions' candidate pairs into one predict() call
BATCH_SIZE = 128  # CrossEncoder's internal batch size across the flattened pairs


def load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
    return done


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python day6_hpc_rerank.py <payload.json> <output.jsonl>")
        sys.exit(1)

    payload_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    questions = json.loads(payload_path.read_text())
    done_ids = load_done_ids(output_path)
    remaining = [q for q in questions if q["id"] not in done_ids]
    print(f"{len(questions)} total, {len(done_ids)} already done, {len(remaining)} remaining")

    if not remaining:
        print("Nothing to do.")
        return

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    cross_encoder = CrossEncoder(RERANK_MODEL_NAME, device=device)

    with open(output_path, "a") as out:
        for i in tqdm(range(0, len(remaining), QUESTIONS_PER_CHUNK)):
            chunk = remaining[i : i + QUESTIONS_PER_CHUNK]

            pairs = []
            spans = []  # (question_idx_in_chunk, start, end) into `pairs`
            for qi, q in enumerate(chunk):
                start = len(pairs)
                pairs.extend((q["question"], c[3]) for c in q["candidates"])
                spans.append((qi, start, len(pairs)))

            t0 = time.perf_counter()
            scores = cross_encoder.predict(pairs, batch_size=BATCH_SIZE) if pairs else []
            elapsed = time.perf_counter() - t0
            per_question_latency = elapsed / len(chunk)

            for qi, start, end in spans:
                q = chunk[qi]
                candidates = [(c[0], c[1], c[2]) for c in q["candidates"]]
                q_scores = scores[start:end]
                order = sorted(range(len(candidates)), key=lambda j: q_scores[j], reverse=True)
                reranked = [candidates[j] for j in order]
                out.write(
                    json.dumps({"id": q["id"], "reranked": reranked, "latency_s": per_question_latency}) + "\n"
                )
            out.flush()

    print(f"Done. Results in {output_path}")


if __name__ == "__main__":
    main()
