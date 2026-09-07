"""Gold relevance labeling + retrieval metrics. T2-RAGBench annotates a source page, not a
chunk id into our own corpus, so relevance is inferred in three layers -- gold table row,
then gold sentence, then whole page. `_relevance_evidence` documents them; DECISIONS.md
GOLD-1 records why per-page matching was replaced (DATA-3/DATA-4/DATA-7..9).
"""

import difflib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Hashable, Sequence
from functools import lru_cache

import pandas as pd

from rag_sec.dataset import load_t2_ragbench

CHUNKS_DIR = "data/chunks"
DAY6_GOLD_TABLES_PATH = "data/day6_gold_tables.json"
DAY6_SUMMARIES_PATH = "data/day6_table_summaries.json"
GOLD_EVIDENCE_RESOLVED_PATH = "data/day7_gold_evidence_resolved.json"

_COMMA_IN_NUMBER_RE = re.compile(r"(?<=\d),(?=\d)")
_WORD_RE = re.compile(r"\d+\.\d+|[a-z0-9]+")  # decimal alternative tried first, so
# `384.3` tokenizes as one token, not `384` + `3` -- the latter would both fragment a
# real figure into two weaker signals and silently create a generic, noisy `3` token
_NUMERIC_TOKEN_RE = re.compile(r"^\d+(\.\d+)?$")
_YEAR_TOKEN_RE = re.compile(r"^(19|20)\d{2}$")
# A 1-2 digit number (a `(4)` footnote marker, a list position) recurs across unrelated
# chunks and produced false Layer-1 matches (finqa_dev_17/MSI). Years are excluded
# separately despite being 4 digits, because a filing's own fiscal year recurs in nearly
# every chunk of it (finqa_dev_321/AES: `2003` alone matched 130 chunks).
MIN_ROW_NUMBER_DIGITS = 3
# This constant empties `gold_nums` for whole classes of rows -- percentages, whole millions,
# day/case counts, and small decimals (`len('4.2'.replace('.',''))==2`) -- which is why 35 of
# `RETR-37`'s 66 cases fell through to Layer 3 with a usable pointer in hand. Locating those
# rows by their LABEL instead was built and reverted (`RETR-38`): dev recall@10 +0.0008, test
# +0.0000 over 9/1235 and 18/1546 changed labels. It rescues nothing because Layer 3 already
# labels those questions -- it only substitutes for Layer 3, so there is no unlabeled
# population to recover. Lowering the constant does not work either: at 1 the values are then
# rejected as non-distinctive anyway (`22` occurred in 102 chunks of one filing).

# Calibrated on a 200-row hand-checked sample spanning all three layers (GOLD-1): the min
# block sizes separate a verbatim match from coincidental phrase overlap, MAX_CLUSTER_GAP
# stops one stray long match from bridging everything between it and the real content, and
# CLUSTER_SHARE_THRESHOLD keeps a second genuine occurrence while rejecting a short one.
MIN_BLOCK_PAGE = 10
MIN_BLOCK_SENTENCE = 6
MAX_CLUSTER_GAP = 3
CLUSTER_SHARE_THRESHOLD = 0.15
# *Most*, not all, of a gold row's numbers -- tolerates one our table parser rounded or
# scaled differently, without admitting coincidental partial overlap (GOLD-1).
ROW_NUMBER_MATCH_THRESHOLD = 0.6
# A number recurring in more of the same filing's own chunks than this is not a fingerprint
# even at 3+ digits (finqa_dev_126: a `100%` column total matched 21 chunks).
ROW_NUMBER_MAX_DOC_FREQ = 5
# Exempting WIDE values from that cap was tried and reverted (RETR-38): a 5+ digit figure
# recurring in one filing really is a restatement rather than a collision, but the exemption
# admits the front-of-filing "selected financial data" summary -- and coverage then PREFERS
# it, because one chunk covering every figure is a smaller cover than the two chunks at the
# real location. 5 of 66 hand-verified cases went from correct to wrong. Coverage rejects a
# redundant candidate; a summary table is not redundant, it is a more efficient cover.


def _normalize_words(text: str) -> list[str]:
    """Lowercase + strip punctuation, collapsing comma thousands-separators first
    (`6,951` -> `6951`) -- our serialization and the dataset's otherwise tokenize the same
    figure to a different word count, breaking every match on it (GOLD-1)."""
    text = _COMMA_IN_NUMBER_RE.sub("", text.lower())
    return _WORD_RE.findall(text)


def _numeric_tokens(words: list[str]) -> set[str]:
    return {
        w
        for w in words
        if _NUMERIC_TOKEN_RE.match(w)
        and len(w.replace(".", "")) >= MIN_ROW_NUMBER_DIGITS
        and not _YEAR_TOKEN_RE.match(w)
    }


def _row_label_words(cells: list[str]) -> set[str]:
    """The gold row's own label -- its first cell. A table row is a label/value pair and the
    benchmark hands us both; matching only the values throws away half the evidence we were
    given, which is how a `shares outstanding` row matched a `weighted-average diluted
    shares` row carrying near-identical figures (RETR-34)."""
    if not cells:
        return set()
    return {w for w in _normalize_words(str(cells[0])) if not _NUMERIC_TOKEN_RE.match(w)}


def _table_row_relevant_chunks(cells: list[str], candidates: list[tuple]) -> set:
    """Layer 1: the smallest set of chunks that COVERS this gold row's distinctive numbers.

    Selection, not filtering (RETR-34). The benchmark names one row, and a row exists at one
    place in the filing, so the quantity to recover is a location. The previous rule admitted
    every chunk over a similarity bar, which is a different question and answered it badly:
    labels averaged 2.88 chunks where they were wrong (max 7), each extra chunk inflating the
    denominator of a recall score it had no business being in.

    Coverage generalises over how many chunks the evidence really occupies, without capping
    the count -- the document decides, not us:
      * row wholly inside one chunk -> the best chunk covers everything, nothing else can add
        anything, gold set is exactly 1
      * row split across n chunks    -> each holds figures the others lack, all n admitted
      * coincidental match           -> contributes only figures already covered, rejected,
                                        however well it scores

    Two guards, both the same "a single figure is not a fingerprint" reasoning that already
    justifies MIN_ROW_NUMBER_DIGITS. A chunk carrying exactly one of the row's numbers is
    admitted only if the row's *label* corroborates it -- otherwise one shared `500` makes a
    stranger chunk gold. And if the union still fails to recover most of the row, we have
    located nothing, so return empty rather than a pile of partial matches; that is
    ROW_NUMBER_MATCH_THRESHOLD applied to the union instead of to each chunk alone.
    """
    gold_nums = _numeric_tokens(_normalize_words(" ".join(str(c) for c in cells)))
    if not gold_nums:
        return set()

    chunk_nums_list = [_numeric_tokens(_normalize_words(text)) for _, text in candidates]
    doc_freq = Counter()
    for nums in chunk_nums_list:
        doc_freq.update(nums)
    distinctive = {g for g in gold_nums if doc_freq[g] <= ROW_NUMBER_MAX_DOC_FREQ}
    if not distinctive:
        return set()

    label = _row_label_words(cells)
    scored = []
    for (cid, text), chunk_nums in zip(candidates, chunk_nums_list):
        got = distinctive & chunk_nums
        if not got:
            continue
        # Same "most, not all" rule the numbers use, reused rather than a second constant.
        words = set(_normalize_words(text))
        label_ok = bool(label) and len(label & words) / len(label) >= ROW_NUMBER_MATCH_THRESHOLD
        if len(got) < 2 and not label_ok:
            continue
        scored.append((len(got), label_ok, cid, got))

    covered: set[str] = set()
    hits = set()
    for _n, _lab, cid, got in sorted(scored, key=lambda x: (-x[0], not x[1], x[2])):
        if got - covered:                      # earns its place only by adding evidence
            hits.add(cid)
            covered |= got
        if covered == distinctive:
            break
    if len(covered) / len(distinctive) < ROW_NUMBER_MATCH_THRESHOLD:
        return set()
    return hits


def _clustered_align_relevant_chunks(gold_words: list[str], candidates: list[tuple], min_block: int) -> set:
    """Layers 2/3: chunks overlapping the gold text's real position in the document, found
    by word-position alignment rather than per-chunk overlap scoring. `candidates` must be
    in document order (chunk order for the JSON path, `chunk_index` within variant='A' for
    the DB path -- ARM4-4).

    `difflib.SequenceMatcher`, not fixed-width shingles: it tolerates insertions, so one
    duplicated gold number token (DATA-8) doesn't misalign every window after it. Blocks
    under `min_block` words are dropped as boilerplate coincidence, and survivors are
    clustered by candidate-index proximity instead of spanning first-match-to-last --
    ~10% of sampled filings had a strong match 5+ chunks from the real content, which
    first-to-last would bridge. A cluster counts only if it covers
    `CLUSTER_SHARE_THRESHOLD` of the gold words.
    """
    if not gold_words:
        return set()

    doc_words: list[str] = []
    word_to_cid: list = []
    for cid, text in candidates:
        w = _normalize_words(text)
        doc_words.extend(w)
        word_to_cid.extend([cid] * len(w))
    if not doc_words:
        return set()

    sm = difflib.SequenceMatcher(None, doc_words, gold_words, autojunk=False)
    blocks = sorted((b for b in sm.get_matching_blocks() if b.size >= min_block), key=lambda b: b.a)
    if not blocks:
        return set()

    cid_order = {cid: i for i, (cid, _) in enumerate(candidates)}
    clusters: list[list] = [[blocks[0]]]
    for b in blocks[1:]:
        prev = clusters[-1][-1]
        prev_cid = word_to_cid[prev.a + prev.size - 1]
        this_cid = word_to_cid[b.a]
        if abs(cid_order[this_cid] - cid_order[prev_cid]) <= MAX_CLUSTER_GAP:
            clusters[-1].append(b)
        else:
            clusters.append([b])

    relevant = set()
    for cluster in clusters:
        if sum(b.size for b in cluster) / len(gold_words) >= CLUSTER_SHARE_THRESHOLD:
            for b in cluster:
                for wi in range(b.a, b.a + b.size):
                    relevant.add(word_to_cid[wi])
    return relevant


@lru_cache(maxsize=None)
def _chunk_file_index(chunks_dir: str = CHUNKS_DIR) -> dict[tuple[int, int], str]:
    """Maps (cik, report_year) -> chunk filename, parsed from TICKER_YEAR_CIK.json."""
    index = {}
    for fname in os.listdir(chunks_dir):
        if not fname.endswith(".json"):  # .DS_Store and friends would crash the unpack
            continue
        ticker, year, cik = fname[:-5].rsplit("_", 2)
        index[(int(cik), int(year))] = fname
    return index


@lru_cache(maxsize=None)
def _load_chunks(fname: str, chunks_dir: str = CHUNKS_DIR) -> tuple[dict, ...]:
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


@lru_cache(maxsize=None)
def _gold_evidence_resolved() -> dict:
    """question id -> {"table_rows": [[cell, ...], ...], "sentences": [str, ...]},
    resolved from the original FinQA/ConvFinQA `gold_inds` by `scripts/eval/
    resolve_gold_evidence.py` (DECISIONS.md GOLD-1). Empty dict (not an error) if
    the file hasn't been built yet -- every question then falls through to Layer 3."""
    if not os.path.exists(GOLD_EVIDENCE_RESOLVED_PATH):
        return {}
    return json.loads(open(GOLD_EVIDENCE_RESOLVED_PATH).read())


@lru_cache(maxsize=None)
def _gold_table_indices_by_question() -> dict:
    """question_id -> set of gold table_index values, from arm4_identify_gold_tables.py's
    own table-vs-table shingle match (independent of this module's chunk-vs-question one)."""
    if not os.path.exists(DAY6_GOLD_TABLES_PATH):
        return {}
    records = json.loads(open(DAY6_GOLD_TABLES_PATH).read())
    out: dict = {}
    for r in records:
        out.setdefault(r["question_id"], set()).add(r["table_index"])
    return out


@lru_cache(maxsize=None)
def _table_summaries() -> dict:
    """'{filing_stem}:{table_index}' -> Arm 4 Strategy C's cached LLM summary text."""
    if not os.path.exists(DAY6_SUMMARIES_PATH):
        return {}
    return json.loads(open(DAY6_SUMMARIES_PATH).read())


def _filing_stem(row: pd.Series) -> str:
    return f"{row['company_symbol']}_{int(row['report_year'])}_{int(row['company_cik'])}"


def _gold_summaries_for_row(row: pd.Series, stem: str | None = None) -> list[str]:
    """Arm 4 Strategy C's cached summary text for whichever gold table(s) this question
    matched (DECISIONS.md ARM4-5) -- used to relevance-match a summary chunk directly
    instead of by wording, since a paraphrase routinely fails shingle/numeric overlap."""
    stem = stem or _filing_stem(row)
    gold_table_idxs = _gold_table_indices_by_question().get(row["id"], set())
    summaries = _table_summaries()
    return [summaries[f"{stem}:{t}"] for t in gold_table_idxs if f"{stem}:{t}" in summaries]


def _relevance_evidence(
    resolved: dict | None,
    gold_context: str,
    candidates: list[tuple],
    gold_summaries: list[str],
) -> dict:
    """Shared scoring core behind both public entry points, so the JSON-file path (Arms
    1-3) and the per-variant DB path (Arm 4) can never drift apart. `candidates` is
    (id, text) in document order; `id` is whatever the caller wants back -- a bare index
    for JSON, a (chunk_index, variant) pair for the DB, where the index alone collides.

    Three layers (GOLD-1), each tried only if the previous found nothing:
      1. `resolved["table_rows"]` -- numeric match against a specific gold table row.
      2. `resolved["sentences"]` -- word alignment against a specific gold sentence.
      3. `gold_context` -- word alignment against the whole page, for the ~4.9% whose
         gold_inds never resolved.
    The gold-table-summary check (ARM4-5) is independent of all three: an LLM paraphrase
    fails wording-based matching by design, so it is matched by provenance instead.

    Returns {chunk_id: {"layer": ...}} -- callers only use `.keys()`.
    """
    if not candidates:
        return {}

    evidence: dict = {}

    def _mark(ids, layer):
        for cid in ids:
            evidence.setdefault(cid, {"layer": layer})

    if resolved:
        for cells in resolved.get("table_rows", []):
            _mark(_table_row_relevant_chunks(cells, candidates), "table_row")
        for sentence in resolved.get("sentences", []):
            gold_words = _normalize_words(sentence)
            _mark(_clustered_align_relevant_chunks(gold_words, candidates, MIN_BLOCK_SENTENCE), "sentence")

    if not evidence and gold_context:
        gold_words = _normalize_words(gold_context)
        _mark(_clustered_align_relevant_chunks(gold_words, candidates, MIN_BLOCK_PAGE), "context_cluster")

    for cid, text in candidates:
        if any(s in text for s in gold_summaries):
            evidence.setdefault(cid, {"layer": "gold_summary"})

    return evidence


def gold_relevant_chunk_evidence(row: pd.Series, chunks_dir: str = CHUNKS_DIR) -> dict[int, dict]:
    """Per-chunk relevance evidence for a question, over the JSON-file corpus (Arms 1-3,
    Strategy A only). See `_relevance_evidence` for the scoring rules."""
    chunks = _load_chunks(row["chunk_file"], chunks_dir)
    candidates = [(i, c["text"]) for i, c in enumerate(chunks)]
    resolved = _gold_evidence_resolved().get(row["id"])
    gold_summaries = _gold_summaries_for_row(row)
    return _relevance_evidence(resolved, row["context"], candidates, gold_summaries)


def gold_relevant_chunk_ids(row: pd.Series, chunks_dir: str = CHUNKS_DIR) -> list[int]:
    """Indices (into that filing's chunk list) of chunks overlapping the gold context."""
    return sorted(gold_relevant_chunk_evidence(row, chunks_dir).keys())


def gold_relevant_chunk_evidence_db(row: pd.Series, conn, variant: str) -> dict[tuple[int, str], dict]:
    """Per-chunk relevance evidence over Arm 4's per-variant pool: `variant='A'` rows not
    superseded for `variant`, plus `variant`'s own rows -- the same WHERE clause the Arm 4
    retrieval script uses, so scoring and retrieval see one candidate set. Keyed by
    (chunk_index, variant) because A and B/C each number chunks from 0 (ARM4-*).
    `ORDER BY chunk_index` is load-bearing: Layers 2/3 need document order.
    """
    stem = _filing_stem(row)
    rows = conn.execute(
        """SELECT chunk_index, variant, text FROM chunks
           WHERE filing_stem = %s
             AND ((variant = 'A' AND NOT (%s = ANY(excluded_by_variant))) OR variant = %s)
           ORDER BY chunk_index""",
        (stem, variant, variant),
    ).fetchall()
    candidates = [((r[0], r[1]), r[2]) for r in rows]
    resolved = _gold_evidence_resolved().get(row["id"])
    gold_summaries = _gold_summaries_for_row(row, stem)
    return _relevance_evidence(resolved, row["context"], candidates, gold_summaries)


def gold_relevant_chunk_ids_db(row: pd.Series, conn, variant: str) -> list[tuple[int, str]]:
    """(chunk_index, variant) pairs of chunks overlapping the gold context, from the DB's
    per-variant candidate pool -- see `gold_relevant_chunk_evidence_db`."""
    return sorted(gold_relevant_chunk_evidence_db(row, conn, variant).keys())


# Chunk ids are opaque to the metrics -- callers pass a bare index, a (stem, index) pair or
# a (stem, index, variant) triple depending on the arm; only hashing and equality matter.
def recall_at_k(retrieved_ids: Sequence[Hashable], relevant_ids: Sequence[Hashable], k: int) -> float:
    if not relevant_ids:
        return float("nan")
    hit = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hit / len(relevant_ids)


def mrr(retrieved_ids: Sequence[Hashable], relevant_ids: Sequence[Hashable]) -> float:
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[Hashable], relevant_ids: Sequence[Hashable], k: int) -> float:
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


def mean_and_stderr(values: list[float]) -> tuple[float, float]:
    """Sample mean and its standard error, ignoring NaN (ndcg is NaN when a question has no
    gold chunk). ddof=1 because these are a sample of questions, not the population."""
    values = [v for v in values if not math.isnan(v)]
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    stderr = math.sqrt(variance / n) if n > 0 else float("nan")
    return mean, stderr


# The shipped arm's cell name in the rerank score files. Lives here rather than in a script
# so the CI guard and the agent run cannot drift apart on which cell they mean.
SHIPPED_CELL = "filtered_stripped"


def load_ranking(
    path,
    cell: str | None = None,
    *,
    with_score: bool = False,
    with_latency: bool = False,
    sort: bool = True,
) -> dict:
    """Load a `*_scores.jsonl` rerank file into `{question_id: ranking}`.

    ONE loader, because there were nine near-copies of this and AGENT-16 was one of them
    silently omitting the sort -- which turned Arm 6's own baseline into a first-stage
    ranking and cost recall@10 0.552 against 0.739.

    TWO ON-DISK SHAPES, and the dispatch is on the RECORD, never on an argument:

      {"id":.., "cells": {name: [[stem, idx, rerank_score], ..]}}
          Written by `rerank_hpc.py`, which zips rerank scores onto the FIRST-STAGE RRF
          candidate order -- so the score is attached, not applied. 0/1235 dev cells are
          in score order as stored. This shape MUST be sorted on load.

      {"id":.., "reranked": [[stem, idx, variant_tag], ..]}
          The Day 6 file. Already ranked, and its third field is the variant tag, NOT a
          score. Sorting it would order by a string. This shape is never sorted.

    Dispatching on the record is the fix for the copy that keyed on `cell is None`: passing
    a cell name against a legacy-shaped file returned `{}` there, with no error.

    `sort=False` is for the two callers that legitimately want stored order -- slicing the
    first K candidates in first-stage order before re-sorting (`candidate_k_curve.py`,
    DEPLOY-13) and re-scoring the candidates from scratch (`onnx_rerank_parity.py`).
    """
    if cell is None and with_score:
        raise ValueError("with_score needs an explicit cell; all-cells mode returns ids only")

    out: dict = {}
    cell_seen = False
    cells_available: set[str] = set()

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            if "cells" in rec:
                cells_available.update(rec["cells"])
                if cell is None:
                    value = {
                        name: _project(entries, sort=sort, with_score=False)
                        for name, entries in rec["cells"].items()
                    }
                elif cell in rec["cells"]:
                    cell_seen = True
                    value = _project(rec["cells"][cell], sort=sort, with_score=with_score)
                else:
                    continue

            elif "reranked" in rec:
                if cell is not None:
                    raise ValueError(
                        f"{path}: legacy 'reranked' record {rec['id']!r} has no cells, but "
                        f"cell={cell!r} was requested. Pass cell=None for this file."
                    )
                if with_score:
                    raise ValueError(
                        f"{path}: legacy 'reranked' third field is a variant tag, not a "
                        "score, so with_score is meaningless here"
                    )
                # Already ranked -- see the docstring; `sort` is deliberately ignored.
                value = [(s, i) for s, i, _tag in rec["reranked"]]

            else:
                raise ValueError(
                    f"{path}: record {rec.get('id')!r} has neither 'cells' nor 'reranked'; "
                    f"keys were {sorted(rec)}. A 'scores' key means this is a SLICE file "
                    "(5-tuples of slice positions, not chunk rankings) -- different unit, "
                    "not loadable here."
                )

            out[rec["id"]] = (value, rec.get("latency_s")) if with_latency else value

    if cell is not None and not cell_seen:
        raise ValueError(
            f"{path}: cell {cell!r} appears in no record. Available: "
            f"{sorted(cells_available) or 'none'}"
        )
    return out


def _project(entries, *, sort: bool, with_score: bool) -> list[tuple]:
    ordered = sorted(entries, key=lambda e: -e[2]) if sort else list(entries)
    if with_score:
        return [(s, i, score) for s, i, score in ordered]
    return [(s, i) for s, i, _score in ordered]
