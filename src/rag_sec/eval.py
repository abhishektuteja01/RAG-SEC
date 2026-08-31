"""Eval harness: gold relevance labeling + retrieval metrics.

T2-RAGBench gives each question a single annotated source page (`context`), not a chunk ID
into our independently-parsed, independently-chunked corpus (DECISIONS.md DATA-3/DATA-4).

Relevance labeling is a layered design (DECISIONS.md GOLD-1), tried in order per question:
  1. `table_N` gold_inds (a specific table row, from the original FinQA/ConvFinQA datasets
     upstream of T2-RAGBench -- 95.1% of the corpus, `data/day7_gold_evidence_resolved.json`):
     exact numeric match against that row's own numbers -- a specific row's numeric
     combination is a near-unique fingerprint, no overlap-ratio tuning needed.
  2. `text_N` gold_inds (a specific sentence): word-position alignment against just that
     sentence.
  3. Fallback, whenever gold_inds didn't resolve (~4.9%): word-position alignment against
     the whole `context` page.
This replaced an IDF-weighted shingle+numeric-overlap labeler scored against the whole
`context` page for every chunk (DATA-7/DATA-8/DATA-9). That approach is not reused here:
auditing it found real, confirmed failure modes baked into whole-page matching --
Table-of-Contents entries and repeated financial-statement headers scoring as relevant
purely from page boilerplate, whole-integer tables invisible to a decimal-only numeric
regex, and paragraphs split across our chunk boundaries systematically under-scoring the
smaller half. Matching against a specific row or sentence instead of a whole page removes
the ambiguity those failures came from, rather than patching each one individually.
"""

import difflib
import json
import math
import os
import re
from collections import Counter
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
MIN_ROW_NUMBER_DIGITS = 3  # calibrated: a bare 1-2 digit number (a footnote marker
# `(4)`, a list position) recurs constantly across unrelated chunks and produces false
# Layer-1 matches -- confirmed on a real case (finqa_dev_17/MSI matched an unrelated
# debt-rating paragraph purely because it happened to contain a `(4)` footnote marker
# matching the gold row's `$4` cell). 3+ digit figures are specific enough to trust.
# Years (1900-2099) are excluded separately, not just by digit count: a filing's own
# fiscal year recurs in nearly every one of its chunks (dates, headers), so it's exactly
# as unreliable as a footnote marker despite being 4 digits -- confirmed on a real case
# (finqa_dev_321/AES: a row's only "distinctive" numbers were `2003` + a $ amount;
# without excluding the year, `2003` alone matched 130 of the filing's chunks).

# Calibrated on a 200-row hand-checked sample spanning all three layers (DECISIONS.md
# GOLD-1): min block sizes separate a genuine verbatim match from short coincidental
# phrase overlap (page headers, boilerplate); MAX_CLUSTER_GAP stops a single stray
# long-enough match (e.g. a Table-of-Contents entry echoing a section title many chunks
# before the real section) from bridging into everything in between being marked
# relevant; CLUSTER_SHARE_THRESHOLD still lets a second genuine occurrence of the same
# content elsewhere in the filing register as relevant, while rejecting an isolated
# short coincidence.
MIN_BLOCK_PAGE = 10
MIN_BLOCK_SENTENCE = 6
MAX_CLUSTER_GAP = 3
CLUSTER_SHARE_THRESHOLD = 0.15
ROW_NUMBER_MATCH_THRESHOLD = 0.6  # requires *most*, not all, of a gold row's own numbers
# to reappear in a chunk -- lets one number through that our table parser rounded/scaled
# differently without opening the door to coincidental partial overlap (DECISIONS.md GOLD-1)
ROW_NUMBER_MAX_DOC_FREQ = 5  # a number recurring in more than this many of the *same
# filing's* own chunks isn't a distinctive fingerprint even if it's 3+ digits and not a
# year -- confirmed on a real case (finqa_dev_126: a row's only qualifying number was a
# `100%` column total, which recurs across dozens of unrelated percentage tables in the
# same filing, matching 21 chunks). A genuinely evidence-specific number, or even a
# legitimately duplicated disclosure, should recur in only a handful of chunks at most.


def _normalize_words(text: str) -> list[str]:
    """Lowercase + strip punctuation, collapsing comma thousands-separators inside a
    number into one token first (`6,951` -> `6951`). Without this, our own chunk
    serialization (`$6,951`) and the dataset's (`6951`) tokenize to a different word
    count for the exact same figure, silently breaking any match on that number
    (DECISIONS.md GOLD-1)."""
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


def _table_row_relevant_chunks(cells: list[str], candidates: list[tuple]) -> set:
    """Layer 1: chunk ids whose own text contains most of this gold row's numbers.
    Position-independent (unlike Layers 2/3) -- a specific row's own numeric combination
    is distinctive enough that no word-alignment is needed to place it, provided the
    numbers themselves are actually distinctive within this filing (ROW_NUMBER_MAX_DOC_FREQ)."""
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

    hits = set()
    for (cid, _), chunk_nums in zip(candidates, chunk_nums_list):
        if len(distinctive & chunk_nums) / len(distinctive) >= ROW_NUMBER_MATCH_THRESHOLD:
            hits.add(cid)
    return hits


def _clustered_align_relevant_chunks(gold_words: list[str], candidates: list[tuple], min_block: int) -> set:
    """Layers 2/3: chunks overlapping the gold text's real position in the document,
    found by word-position alignment instead of independent per-chunk overlap scoring.
    `candidates` must be given in document order -- natural chunk order for the
    JSON-file path; `chunk_index` order within variant='A' for Arm 4's DB path
    (ARM4-4: 'A' numbers a filing's chunks continuously from 0 -- the closest available
    approximation for B/C's own standalone rows/summaries, which don't have a document
    position of their own).

    Uses `difflib.SequenceMatcher` rather than fixed-width shingle windows: it tolerates
    insertions/deletions, so one duplicated number token in the gold text (a known
    dataset quirk, DECISIONS.md DATA-8) doesn't misalign every window after it the way a
    fixed 5-gram window did. Matching blocks below `min_block` words are dropped (filters
    short boilerplate/title coincidences -- confirmed on a real case: a Table-of-Contents
    entry echoing a section header verbatim for 10-20+ words). Surviving blocks are
    grouped into clusters by candidate-index proximity (`MAX_CLUSTER_GAP`) rather than
    spanning from the first match to the last -- confirmed necessary: ~10% of a sampled
    set of filings had a strong match more than 5 chunks from the real content (the same
    TOC/boilerplate effect, just long enough to clear a naive floor); bridging
    first-to-last would mark everything in between as relevant. A cluster only counts if
    it accounts for at least `CLUSTER_SHARE_THRESHOLD` of the gold text's own words, so a
    second genuine occurrence of the same content elsewhere in the filing still
    registers, while an isolated short coincidence doesn't.
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


@lru_cache(maxsize=None)
def _gold_evidence_resolved() -> dict:
    """question id -> {"table_rows": [[cell, ...], ...], "sentences": [str, ...]},
    resolved from the original FinQA/ConvFinQA `gold_inds` by `scripts/eval/
    day7_resolve_gold_evidence.py` (DECISIONS.md GOLD-1). Empty dict (not an error) if
    the file hasn't been built yet -- every question then falls through to Layer 3."""
    if not os.path.exists(GOLD_EVIDENCE_RESOLVED_PATH):
        return {}
    return json.loads(open(GOLD_EVIDENCE_RESOLVED_PATH).read())


@lru_cache(maxsize=None)
def _gold_table_indices_by_question() -> dict:
    """question_id -> set of gold table_index values, from day6_identify_gold_tables.py's
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
    """Shared scoring core behind both `gold_relevant_chunk_evidence` (JSON-file corpus,
    Arms 1-3) and `gold_relevant_chunk_evidence_db` (DB, per-variant, Arm 4) so the two
    never drift apart. `candidates` is a list of (id, text) pairs, in document order (see
    `_clustered_align_relevant_chunks`) -- `id` is whatever the caller wants back (a bare
    index for the JSON path, a (chunk_index, variant) pair for the DB path, since
    chunk_index alone collides across variants -- DECISIONS.md ARM4-*).

    Three layers, tried in order (DECISIONS.md GOLD-1), falling through only if the
    current one finds nothing:
      1. `resolved["table_rows"]` -- exact numeric match against a specific gold table row.
      2. `resolved["sentences"]` -- word-position alignment against a specific sentence.
      3. `gold_context` -- word-position alignment against the whole page, used only when
         `resolved` is empty/None (gold_inds didn't resolve for this question, ~4.9% of
         the corpus) or produced no hits.
    The gold-table-summary check (ARM4-5) is independent of all three -- an LLM paraphrase
    routinely fails wording-based matching by design, so it's matched by direct
    provenance instead, same as before.

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
    """Per-chunk relevance evidence for a question, over the DB's per-variant candidate
    pool for Arm 4: `variant='A'` rows not superseded for `variant`, plus `variant`'s own
    rows (mirrors the WHERE clause the Arm 4 retrieval script itself uses, so scoring and
    retrieval always see the same candidate set). Keyed by (chunk_index, variant) pairs,
    not bare chunk_index, since A and B/C independently number chunks from 0 per filing
    (DECISIONS.md ARM4-*) -- a bare index would silently collide two unrelated chunks.
    `ORDER BY chunk_index` matters here (it didn't for the old per-chunk-independent
    scoring): Layers 2/3 need candidates in document order to detect clusters correctly.
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
