"""Year-bias recall@10 check, cluster leg (runs ON the HPC GPU node) -- cross-encoder
rerank of the union candidate pool per question. Mirrors `hpc/rerank_hpc.py` exactly
(ARM3-2): no `rag_sec` import, no database, model name and device pick duplicated from
`rag_sec.config` / `rag_sec.retrieve` on purpose, because the cluster node has neither
installed.

PRODUCES
    <output>.jsonl   one line per question:
        {id, split, scores: [[stem, chunk_index, score], ...], latency_s}
    Scores cover every candidate in the payload's UNION pool (`candidates`), not just one
    condition's top-k -- the laptop `score` leg reconstructs BOTH the base and biased
    top-10 from this one shared score dict, so this leg reranks each question once, not
    twice, even though it is answering a two-condition question.

READS
    <payload>.json   written by `year_bias_recall10_prepare.py` and scp'd to the cluster
                     home directory. Carries `candidates` (the union pool to score) and
                     `rerank_query` per question, plus the shared `texts` map.

TRAPS (same as hpc/rerank_hpc.py, repeated here since this file travels alone)
  * THE BASENAME IS PART OF THE CONTRACT: `year_bias_recall10.sbatch` invokes this by
    bare name from `~` after it is scp'd there. Renaming it breaks the submitted job.
  * Deliberately no `sys.path` / `rag_sec` import -- the node has neither.
  * `RETR16_BATCH` env var overrides BATCH_SIZE, same knob as rerank_hpc.py.

Usage (ON the cluster, from ~):
    python -u year_bias_recall10_hpc.py year_bias_recall10_dev_payload.json year_bias_recall10_dev_scores.jsonl
"""

import json
import os
import sys
import time
from pathlib import Path

from sentence_transformers import CrossEncoder
from tqdm import tqdm

RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # duplicated, not imported -- see TRAPS
QUESTIONS_PER_CHUNK = 20  # matches rerank_hpc.py
BATCH_SIZE = int(os.environ.get("RETR16_BATCH", "128"))


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
        print("Usage: python year_bias_recall10_hpc.py <payload.json> <output.jsonl>")
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
            spans: list[tuple[int, list, int, int]] = []
            for qi, q in enumerate(batch):
                cands = [c for c in q["candidates"] if f"{c[0]}|{c[1]}" in texts]
                start = len(pairs)
                pairs.extend((q["rerank_query"], texts[f"{c[0]}|{c[1]}"]) for c in cands)
                spans.append((qi, cands, start, len(pairs)))

            t0 = time.perf_counter()
            scores = cross_encoder.predict(pairs, batch_size=BATCH_SIZE) if pairs else []
            per_question_latency = (time.perf_counter() - t0) / len(batch)

            scores_by_q: dict[int, list] = {}
            for qi, cands, start, end in spans:
                scores_by_q[qi] = [
                    [c[0], c[1], float(s)] for c, s in zip(cands, scores[start:end])
                ]
            for qi, q in enumerate(batch):
                out.write(json.dumps({
                    "id": q["id"],
                    "split": q.get("split", "dev"),
                    "scores": scores_by_q[qi],
                    "latency_s": per_question_latency,
                }) + "\n")
            out.flush()

    print(f"Done. Results in {output_path}")


if __name__ == "__main__":
    main()
