"""Stage 2 (HPC GPU node): score the RETR-16 2x2 with the local cross-encoder.

Cells (candidates x query form), from one payload
(`rerank_prepare.py`):

    unfiltered_raw       as today                                     -> baseline
    filtered_raw         + company filter                             -> isolates RETR-5
    unfiltered_stripped  + entity framing stripped from the query     -> isolates RETR-6
    filtered_stripped    + both                                       -> deployment candidate

Which cells a question needs travels WITH the question. **dev** skips `unfiltered_raw`: it
already has a clean one in `data/day6_arm4_A_rerank_scores.jsonl`, and re-scoring it would
only create a way for the two to disagree. **test** has no baseline at all, so it gets the
full 2x2 -- that is what lets the RETR-22 interaction be confirmed out of sample
(DECISIONS.md RETR-18).

Same contract as day5/day6/day8_slice HPC scripts: deliberately self-contained (no
`rag_sec` import, no DB), checkpointed by question id so a killed job resumes, and
cross-question batching so `predict()` sees large batches rather than 50 pairs at a time.
The stripped query is read from the payload, not recomputed here -- that logic lives on the
laptop where `rag_sec.company` is importable.

A question with no resolved company has `cands_filtered == cands_unfiltered` by
construction (the fallback `retrieve()` uses). Its filtered cells are still scored, so every
cell covers every question and the arms stay directly comparable.

`RETR16_BATCH` overrides BATCH_SIZE. 256 is worth trying on a 32GB V100: it is numerically
identical (same fp32 math, more rows in flight) so it cannot shift scores the way fp16
would, and `CrossEncoder.predict` sorts by length, so an over-large batch OOMs within
seconds rather than hours in.

Usage:
    python rerank_hpc.py day8_retr16_rerank_payload.json day8_retr16_scores.jsonl

Output: JSONL, one line per question:
    {id, cells: {cell_name: [[stem, chunk_index, score], ...], ...}, latency_s}
    Raw scores, not just an ordering, so downstream can re-cut at any k without re-scoring.
    latency_s is the batch's wall time split evenly across its questions -- an average.
"""

import json
import os
import sys
import time
from pathlib import Path

from sentence_transformers import CrossEncoder
from tqdm import tqdm

RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # duplicated, not imported from
# rag_sec.config -- this script runs on the HPC node with no rag_sec package, by design
QUESTIONS_PER_CHUNK = 20  # matches Arm 3; 3-4 cells each means ~150-200 pairs per question
BATCH_SIZE = int(os.environ.get("RETR16_BATCH", "128"))
# 256 is worth trying on a 32GB V100: it is numerically identical (same fp32 math, more
# rows in flight) so it cannot shift scores the way fp16 would, and CrossEncoder.predict
# sorts by length so an over-large batch OOMs within seconds rather than hours in.

# cell name -> (which candidate list, which question field). Which cells each question
# needs is carried per-question in the payload: dev already has a clean unfiltered_raw
# baseline and skips it, test has none and gets the full 2x2 (DECISIONS.md RETR-16/18).
CELL_SPEC = {
    "unfiltered_raw": ("cands_unfiltered", "question"),
    "filtered_raw": ("cands_filtered", "question"),
    "unfiltered_stripped": ("cands_unfiltered", "question_stripped"),
    "filtered_stripped": ("cands_filtered", "question_stripped"),
}


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
        print("Usage: python rerank_hpc.py <payload.json> <output.jsonl>")
        sys.exit(1)
    payload_path, output_path = Path(sys.argv[1]), Path(sys.argv[2])

    payload = json.loads(payload_path.read_text())
    texts, questions = payload["texts"], payload["questions"]
    done = load_done_ids(output_path)
    remaining = [q for q in questions if q["id"] not in done]
    print(f"{len(questions)} total, {len(done)} already done, {len(remaining)} remaining")
    print(f"{len(texts)} unique chunks carried in the payload")
    if not remaining:
        print("Nothing to do.")
        return

    import torch

    # cuda > mps > cpu, duplicated from rag_sec.config.pick_device() -- this script runs
    # on the HPC node with no rag_sec package, by design. mps matters only when it is
    # run locally on Apple Silicon (INFRA-6).
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    cross_encoder = CrossEncoder(RERANK_MODEL_NAME, device=device)

    with open(output_path, "a") as out:
        for i in tqdm(range(0, len(remaining), QUESTIONS_PER_CHUNK)):
            batch = remaining[i : i + QUESTIONS_PER_CHUNK]

            pairs: list[tuple[str, str]] = []
            spans: list[tuple[int, str, list, int, int]] = []
            for qi, q in enumerate(batch):
                for cell in q["cells"]:
                    cand_key, q_key = CELL_SPEC[cell]
                    cands = [c for c in q[cand_key] if f"{c[0]}|{c[1]}" in texts]
                    start = len(pairs)
                    pairs.extend((q[q_key], texts[f"{c[0]}|{c[1]}"]) for c in cands)
                    spans.append((qi, cell, cands, start, len(pairs)))

            t0 = time.perf_counter()
            scores = cross_encoder.predict(pairs, batch_size=BATCH_SIZE) if pairs else []
            per_question_latency = (time.perf_counter() - t0) / len(batch)

            cells_by_q: dict[int, dict] = {qi: {} for qi in range(len(batch))}
            for qi, cell, cands, start, end in spans:
                cells_by_q[qi][cell] = [
                    [c[0], c[1], float(s)] for c, s in zip(cands, scores[start:end])
                ]
            for qi, q in enumerate(batch):
                out.write(
                    json.dumps(
                        {
                            "id": q["id"],
                            "split": q.get("split", "dev"),
                            "cells": cells_by_q[qi],
                            "latency_s": per_question_latency,
                        }
                    )
                    + "\n"
                )
            out.flush()

    print(f"Done. Results in {output_path}")


if __name__ == "__main__":
    main()
