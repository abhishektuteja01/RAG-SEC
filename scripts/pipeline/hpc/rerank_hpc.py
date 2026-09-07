"""Pipeline phase 05, cluster leg (runs ON the HPC GPU node) — cross-encoder rerank.

PRODUCES
    <output>.jsonl   one JSON line per question:
        {id, split, cells: {cell_name: [[stem, chunk_index, score], ...]}, latency_s}
        Raw scores, not just an ordering, so downstream can re-cut at any k without
        re-scoring. `latency_s` is the batch's wall time split evenly across its
        questions -- an average, not a per-question measurement.
    The live artifacts are data/retr7_rr_{dev,test}_scores.jsonl (2026-09-04).

READS
    <payload>.json   written by `scripts/pipeline/05_arm3_rerank.py prepare` and scp'd
                     to the cluster home directory. Carries the candidate pools, the
                     shared `texts` map, and the per-question `cells` list.

DECISIONS.md ROWS THIS BACKS
    ARM3-2    why reranking is a split job at all: a laptop CPU pass plateaued at
              ~110s/question (~25h) and a live SSH tunnel to the laptop's Postgres dies
              with the connection. Laptop prepares, GPU node reranks with no DB, laptop
              scores. Transfers go via the `xfer` host, never the login node.
    INFRA-6   device selection is cuda > mps > cpu, INLINED here rather than imported.
    RETR-16   the 2x2: candidates (unfiltered|filtered) x query (raw|stripped).
    RETR-18   test gets the full 2x2 so RETR-22's interaction is confirmed out of sample.
    RETR-39   the published headline this leg produced: dev filter+strip recall@10 0.760,
              test 0.747.

WHEN THIS ACTUALLY RAN (calendar dates, not "Day N")
    2026-08-28   the Arm 3 cross-encoder HPC pass (its Day-5/6 ancestor,
                 scripts/archive/arm3_rerank_hpc.py)
    2026-08-30   those results rescored on the laptop
    2026-09-04   the current retr7_* passes, dev and test, after the RETR-7/RETR-8
                 re-index

TRAPS
  * THE BASENAME IS PART OF THE CONTRACT. `scripts/retrieval/rerank_hpc.sbatch` does
    `cd ~` and then `python -u rerank_hpc.py` by bare name, after this file has been
    scp'd to the cluster home directory. Renaming it breaks the submitted job, not this
    repo. Same for argv: exactly two positionals, `payload.json results.jsonl`.
  * IT DELIBERATELY IMPORTS NOTHING FROM `rag_sec`, and so it does NOT carry the
    `sys.path.insert(_ROOT/"src")` preamble every other pipeline phase has. The cluster
    has no `rag_sec` install and no database (ARM3-2 / INFRA-6), so the model name and
    the device pick are duplicated from `rag_sec.config` on purpose. Adding an import
    here would only fail on the node, hours into a booking.
  * The stripped query is read from the payload, never recomputed here -- that logic
    lives in `rag_sec.company`, which is not importable on the node.
  * Scores are ZIPPED ONTO THE FIRST-STAGE RRF ORDER, so a cell as stored is NOT in
    rank order (0/1235 dev cells are). Every reader must sort, which is why
    `rag_sec.eval.load_ranking` sorts on load and `scripts/checks/static_ranking_order.py`
    gates it. Reading a cell as stored was bug AGENT-16: recall@10 0.552 vs 0.739.
  * The usage line below still names `day8_retr16*` files while the live artifacts are
    `retr7_*`. Reported, not silently changed -- the sbatch's own hardcoded filenames
    (`day8_retr18_test_*`) are the same drift.
  * `RETR16_BATCH` overrides BATCH_SIZE. 256 is safe to try on a 32GB V100: same fp32
    math, so it cannot shift scores the way fp16 would, and `CrossEncoder.predict` sorts
    by length, so an over-large batch OOMs within seconds rather than hours in.
  * A question with no resolved company has `cands_filtered == cands_unfiltered` by
    construction (the same fallback `retrieve()` uses). Its filtered cells are still
    scored, so every cell covers every question and the arms stay comparable.

Usage (ON the cluster, from ~):
    python -u rerank_hpc.py day8_retr16_rerank_payload.json day8_retr16_scores.jsonl
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
