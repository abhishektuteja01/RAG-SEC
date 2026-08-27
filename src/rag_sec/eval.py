"""Eval harness: gold relevance labeling + retrieval metrics.

T2-RAGBench gives each question a single annotated source page (`context`), not a
chunk ID into our independently-parsed, independently-chunked corpus (see DECISIONS.md
#11-12). Relevance is derived by text-overlap: a chunk counts as gold-relevant for a
question if it contains most of that page's shingles, not by matching the numeric
answer (rejected — original_answer vs. narrative text can differ by a scale factor,
e.g. "380" vs "$3.8 million", producing false negatives).
"""

import math
import os
import re
from functools import lru_cache

import pandas as pd

from rag_sec.dataset import load_t2_ragbench

CHUNKS_DIR = "data/chunks"
SHINGLE_N = 5
RELEVANCE_THRESHOLD = 0.3  # calibrated on a 200-row sample: gold context often spans
# two adjacent chunks (page split differs from our chunk boundary), so genuine matches
# commonly score 0.4-0.5 containment, not 1.0 — 0.3 drops the zero-hit rate from 28.5%
# (threshold 0.5) to 3%, and the remaining 3% have best-score <0.3 (real corpus gaps,
# e.g. wrong fiscal-year filing ingested — see DECISIONS.md follow-up), not false negatives


def _shingles(text: str, n: int = SHINGLE_N) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


@lru_cache(maxsize=None)
def _chunk_file_index(chunks_dir: str = CHUNKS_DIR) -> dict[tuple[int, int], str]:
    """Maps (cik, report_year) -> chunk filename, parsed from TICKER_YEAR_CIK.json."""
    index = {}
    for fname in os.listdir(chunks_dir):
        ticker, year, cik = fname[:-5].rsplit("_", 2)
        index[(int(cik), int(year))] = fname
    return index


@lru_cache(maxsize=None)
def _load_chunks(fname: str, chunks_dir: str = CHUNKS_DIR) -> tuple[dict, ...]:
    import json

    with open(os.path.join(chunks_dir, fname)) as f:
        return tuple(json.load(f))


def load_matched_questions(chunks_dir: str = CHUNKS_DIR) -> pd.DataFrame:
    """FinQA + ConvFinQA rows whose filing is in our ingested corpus (DECISIONS.md #12).

    TAT-DQA excluded — no CIK, deferred per decision #12.
    """
    df = pd.concat(
        [load_t2_ragbench("FinQA"), load_t2_ragbench("ConvFinQA")], ignore_index=True
    )
    index = _chunk_file_index(chunks_dir)
    df = df.dropna(subset=["company_cik", "report_year"])
    df["chunk_file"] = df.apply(
        lambda r: index.get((int(r["company_cik"]), int(r["report_year"]))), axis=1
    )
    return df[df["chunk_file"].notna()].reset_index(drop=True)


def gold_relevant_chunk_ids(
    row: pd.Series, chunks_dir: str = CHUNKS_DIR, threshold: float = RELEVANCE_THRESHOLD
) -> list[int]:
    """Indices (into that filing's chunk list) of chunks overlapping the gold context."""
    gold_sh = _shingles(row["context"])
    if not gold_sh:
        return []
    chunks = _load_chunks(row["chunk_file"], chunks_dir)
    relevant = []
    for i, chunk in enumerate(chunks):
        chunk_sh = _shingles(chunk["text"])
        if not chunk_sh:
            continue
        containment = len(gold_sh & chunk_sh) / len(gold_sh)
        if containment >= threshold:
            relevant.append(i)
    return relevant


def recall_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    if not relevant_ids:
        return float("nan")
    hit = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hit / len(relevant_ids)


def mrr(retrieved_ids: list[int], relevant_ids: list[int]) -> float:
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return float("nan")
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved_ids[:k], start=1)
        if cid in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else float("nan")
