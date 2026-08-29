"""Eval harness: gold relevance labeling + retrieval metrics.

T2-RAGBench gives each question a single annotated source page (`context`), not a
chunk ID into our independently-parsed, independently-chunked corpus (see DECISIONS.md
DATA-3/DATA-4). Relevance is derived by text-overlap: a chunk counts as gold-relevant for a
question if it contains most of that page's shingles, not by matching the numeric
answer (rejected — original_answer vs. narrative text can differ by a scale factor,
e.g. "380" vs "$3.8 million", producing false negatives).
"""

import math
import os
import re
from collections import Counter
from functools import lru_cache

import pandas as pd

from rag_sec.dataset import load_t2_ragbench

CHUNKS_DIR = "data/chunks"
SHINGLE_N = 5
RELEVANCE_THRESHOLD = 0.3  # calibrated on a 200-row sample: gold context often spans
# two adjacent chunks (page split differs from our chunk boundary), so genuine matches
# commonly score 0.4-0.5 containment, not 1.0 — 0.3 drops the zero-hit rate from 28.5%
# (threshold 0.5) to 3%, and the remaining 3% have best-score <0.3 (real corpus gaps,
# e.g. wrong fiscal-year filing ingested — see DECISIONS.md DATA-5), not false negatives

NUMBER_RE = re.compile(r"\d+\.\d+")
NUMERIC_RELEVANCE_THRESHOLD = 0.5  # see DECISIONS.md DATA-8 — rescues answer-bearing table
# chunks that shingle overlap misses because a number appears a different number of
# times in the dataset's gold text vs. our table serialization (e.g. "-17.1 ( 17.1 )"
# in gold vs. "(17.1)" in our chunk), which shifts every 5-gram window straddling that
# gap even though the informational content is identical

MIN_NUMERIC_EVIDENCE = 3  # see DECISIONS.md DATA-9 — below this, the numeric-overlap ratio
# is unreliable: with only 1-2 unique numbers in the gold text, any chunk that happens
# to share one recurring boilerplate figure (e.g. a stock-option exercise price reused
# across a dozen unrelated per-executive tables) clears NUMERIC_RELEVANCE_THRESHOLD by
# coincidence, not because it's the answer chunk


def _numbers(text: str) -> set[str]:
    """Decimal figures only (not bare integers like years or page numbers) — these are
    the financial-statement values that matter, and matching them as a set makes this
    signal immune to the token-count misalignment that breaks shingle windows."""
    return set(NUMBER_RE.findall(text))


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
    """FinQA + ConvFinQA rows whose filing is in our ingested corpus (DECISIONS.md DATA-3).

    TAT-DQA excluded — no CIK, deferred per decision DATA-3.
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


def _doc_freqs(chunks: tuple[dict, ...], extractor) -> tuple[Counter, list[set]]:
    """token -> number of chunks (within this one filing) containing it, plus each
    chunk's own token set (so callers don't re-extract it)."""
    df = Counter()
    per_chunk = []
    for chunk in chunks:
        toks = extractor(chunk["text"])
        per_chunk.append(toks)
        df.update(toks)
    return df, per_chunk


def _idf(token, df: Counter, n_chunks: int) -> float:
    # A gold token never seen in any chunk of this filing (e.g. wrong-fiscal-year gold
    # text) is treated as maximally common (floor weight) rather than dividing by zero.
    doc_freq = df.get(token, n_chunks)
    return math.log(1 + n_chunks / doc_freq)


def gold_relevant_chunk_evidence(
    row: pd.Series,
    chunks_dir: str = CHUNKS_DIR,
    threshold: float = RELEVANCE_THRESHOLD,
    numeric_threshold: float = NUMERIC_RELEVANCE_THRESHOLD,
) -> dict[int, dict]:
    """Per-chunk relevance evidence for a question (see DECISIONS.md DATA-9).

    A chunk is relevant if EITHER weighted shingle containment clears `threshold` (the
    general case — mainly prose) OR weighted numeric containment clears
    `numeric_threshold` (a fallback for table/figure chunks where shingle overlap
    systematically undercounts — see DECISIONS.md DATA-8). Overlap is IDF-weighted within
    the filing: a shingle/number that recurs across many of the filing's own chunks
    (boilerplate, repeated tables) counts for less than one that appears in only one or
    two chunks — this stops a single shared number from flooding the relevant set with
    unrelated chunks (DECISIONS.md DATA-9). The numeric fallback is skipped entirely when
    the gold text has fewer than MIN_NUMERIC_EVIDENCE unique numbers, since the ratio is
    unreliable with too little evidence to weight in the first place.

    Returns {chunk_id: {"sh_score", "num_score", "gold_sh_evidence", "gold_num_evidence"}}
    so callers can judge match confidence, not just a bare relevant/not-relevant label.
    """
    gold_sh = _shingles(row["context"])
    gold_nums = _numbers(row["context"])
    if not gold_sh:
        return {}
    chunks = _load_chunks(row["chunk_file"], chunks_dir)
    n_chunks = len(chunks)

    sh_df, chunk_shingles = _doc_freqs(chunks, _shingles)
    gold_sh_weight = {t: _idf(t, sh_df, n_chunks) for t in gold_sh}
    gold_sh_total = sum(gold_sh_weight.values()) or 1.0

    use_numeric = len(gold_nums) >= MIN_NUMERIC_EVIDENCE
    if use_numeric:
        num_df, chunk_numbers = _doc_freqs(chunks, _numbers)
        gold_num_weight = {t: _idf(t, num_df, n_chunks) for t in gold_nums}
        gold_num_total = sum(gold_num_weight.values()) or 1.0

    evidence = {}
    for i in range(n_chunks):
        sh_matched = sum(gold_sh_weight[t] for t in gold_sh & chunk_shingles[i])
        sh_score = sh_matched / gold_sh_total

        num_score = 0.0
        if use_numeric:
            num_matched = sum(gold_num_weight[t] for t in gold_nums & chunk_numbers[i])
            num_score = num_matched / gold_num_total

        if sh_score >= threshold or num_score >= numeric_threshold:
            evidence[i] = {
                "sh_score": sh_score,
                "num_score": num_score,
                "gold_sh_evidence": len(gold_sh),
                "gold_num_evidence": len(gold_nums),
            }
    return evidence


def gold_relevant_chunk_ids(
    row: pd.Series, chunks_dir: str = CHUNKS_DIR, threshold: float = RELEVANCE_THRESHOLD
) -> list[int]:
    """Indices (into that filing's chunk list) of chunks overlapping the gold context."""
    return sorted(gold_relevant_chunk_evidence(row, chunks_dir, threshold).keys())


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
