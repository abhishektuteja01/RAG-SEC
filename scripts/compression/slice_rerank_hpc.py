"""Day 8, stage 2 (HPC GPU node): score every question's candidate *slices* with the local
cross-encoder. See DECISIONS.md COST-7/COST-11.

Same contract as day5/day6_hpc_rerank.py -- deliberately self-contained (no `rag_sec`
import, no DB), checkpointed by question id so a killed job resumes, cross-question
batching into QUESTIONS_PER_CHUNK predict() calls.

Two differences from the Arm 3/4 rerankers:
  * a candidate is a slice, identified by [filing_stem, chunk_index, atom_index,
    piece_index] -- chunk_index alone no longer identifies a scoreable unit.
  * raw scores are written out, not just the ordering. Budget-based selection needs to walk
    slices in score order and stop on a token budget, and the downstream sweep re-runs that
    walk at several budgets without re-scoring.

There are ~8.9x more pairs than an Arm 3 rerank (50 chunks -> ~445 slices per question) but
each pair is far shorter: measured slice length is median 124 tokens, p99 256, max 1500.

BATCH_SIZE stays at Arm 3's 128 despite the shorter inputs. CrossEncoder.predict sorts by
length internally and pads each batch to its longest member, so the first batches are all
max-length: peak memory is batch_size x ~1500 tokens regardless of the median. 128 keeps
that at ~200k tokens/batch, the same peak Arm 3 already ran successfully on a V100-32GB.
256 would double it. Because the sort puts the longest pairs first, an over-large batch
OOMs within seconds of starting rather than hours in -- so raising it is cheap to test.

Usage:
    python day8_hpc_slice_rerank.py day8_slice_payload_t150.json day8_slice_scores_t150.jsonl

Input:  JSON list of {id, question, slices: [[stem, chunk_index, atom_i, piece_i, text], ...]}
Output: JSONL, one line per question:
        {id, target, scores: [[stem, chunk_index, atom_i, piece_i, score], ...], latency_s}
        (latency_s is the batch's wall time divided evenly across its questions -- an
        average, not a true per-question measurement, since they are scored together)
"""

import json
import re
import sys
import time
from pathlib import Path

from sentence_transformers import CrossEncoder
from tqdm import tqdm

RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # duplicated, not imported from
# rag_sec.config -- this script runs on the HPC node with no rag_sec package, by design
QUESTIONS_PER_CHUNK = 8  # fewer questions per predict() than Arm 3's 20: ~6x the pairs each
BATCH_SIZE = 128


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
        print("Usage: python day8_hpc_slice_rerank.py <payload.json> <output.jsonl>")
        sys.exit(1)

    payload_path, output_path = Path(sys.argv[1]), Path(sys.argv[2])

    # Slices travel as (atom_index, piece_index) positions, not text, so the payload stays
    # one copy of the corpus. Stage 3 rebuilds the same slicing to resolve them -- which is
    # only correct at the same target. Carry it through explicitly rather than trusting two
    # scripts to agree on a default: a mismatch would resolve every position to the wrong
    # text, silently and with no error.
    # `_t<N>` need not be the last field -- payloads are also tagged by pool
    # (..._t150_filtered_stripped.json). Require exactly one match rather than taking the
    # first: two would make the target ambiguous, which is the silent-wrong-text case above.
    found = set(re.findall(r"_t(\d+)(?=_|\.json$)", payload_path.name))
    if len(found) != 1:
        print(f"Cannot read one slice target from filename {payload_path.name!r}; expected ..._t<N>[_tag].json")
        sys.exit(1)
    target = int(found.pop())
    print(f"Slice target: {target} tokens")
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

            pairs, spans = [], []
            for qi, q in enumerate(chunk):
                start = len(pairs)
                pairs.extend((q["question"], sl[4]) for sl in q["slices"])
                spans.append((qi, start, len(pairs)))

            t0 = time.perf_counter()
            scores = cross_encoder.predict(pairs, batch_size=BATCH_SIZE) if pairs else []
            per_question_latency = (time.perf_counter() - t0) / len(chunk)

            for qi, start, end in spans:
                q = chunk[qi]
                out.write(
                    json.dumps(
                        {
                            "id": q["id"],
                            "target": target,
                            "scores": [
                                [sl[0], sl[1], sl[2], sl[3], float(s)]
                                for sl, s in zip(q["slices"], scores[start:end])
                            ],
                            "latency_s": per_question_latency,
                        }
                    )
                    + "\n"
                )
            out.flush()

    print(f"Done. Results in {output_path}")


if __name__ == "__main__":
    main()
