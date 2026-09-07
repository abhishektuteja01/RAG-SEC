"""Evidence compression (DSLR, arXiv:2407.03627): score the pieces of a retrieved chunk with
the reranker already loaded and print only what fits a token budget. Retrieval still returns
k=10, so recall is untouched -- only the prompt shrinks. DECISIONS.md COST-18..COST-26.

The unit is the `Atom` the chunker packed the chunk from, replayed from data/parsed/ rather
than re-derived by regex, so "keep `$1,740`, drop the `(in millions)` that scales it" stays
bounded by chunk packing instead of by a parser of our own.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rag_sec.chunking import (
    MULTI_HEADING,
    STRIP_TITLE_FURNITURE,
    Atom,
    Chunk,
    chunk_blocks_with_atoms,
    count_tokens,
)
from rag_sec.parsing import load_parsed_blocks

PARSED_DIR = Path("data/parsed")
CHUNKS_DIR = Path("data/chunks")

# Named in every replay failure below, because the two flags ARE the failure: set differently
# from the corpus on disk they reproduce only 52.5% of chunks (chunking.py, RETR-7/RETR-8).
_FLAG_STATE = (
    f"RAG_SEC_MULTI_HEADING={MULTI_HEADING}, "
    f"RAG_SEC_STRIP_TITLE_FURNITURE={STRIP_TITLE_FURNITURE}; both must match the corpus "
    "that was embedded (RETR-7/RETR-8)."
)


@lru_cache(maxsize=1024)
def _packed(stem: str) -> tuple[tuple[Chunk, tuple[Atom, ...]], ...]:
    """Replay of the packer for one filing -- it re-tokenizes every block, and a question's
    10 chunks usually come from two or three filings. Sized to hold the whole 799-filing
    corpus (~220MB of atoms): a batch pass touches hundreds per run, and a small cache
    thrashes badly enough to dominate runtime."""
    blocks = load_parsed_blocks(PARSED_DIR / f"{stem}.json")
    return tuple((chunk, tuple(atoms)) for chunk, atoms in chunk_blocks_with_atoms(blocks))


@lru_cache(maxsize=1024)
def _stored_texts(stem: str) -> tuple[str, ...]:
    """The chunk texts that were actually embedded and retrieved -- what the replay above
    has to reproduce to be talking about the same document."""
    with open(CHUNKS_DIR / f"{stem}.json") as f:
        return tuple(c["text"] for c in json.load(f))


def chunk_atoms(stem: str, chunk_index: int) -> tuple[Chunk, tuple[Atom, ...]]:
    """Atoms of one STORED chunk, by replay -- and the replay is checked, not trusted.
    `chunk_index` comes from the `chunks` table while the atoms come from re-running the
    packer over data/parsed/, and nothing else ties the two together: under mismatched
    chunking flags the packer emits a different chunk list and this would silently hand back
    another chunk's atoms. That is the project's recurring bug class, so it raises here."""
    packed = _packed(stem)
    stored = _stored_texts(stem)
    if not 0 <= chunk_index < min(len(packed), len(stored)):
        raise ValueError(
            f"chunk {chunk_index} is out of range for {stem}: replaying "
            f"{PARSED_DIR}/{stem}.json gave {len(packed)} chunks and "
            f"{CHUNKS_DIR}/{stem}.json holds {len(stored)}. {_FLAG_STATE}"
        )
    if packed[chunk_index][0].text != stored[chunk_index]:
        raise ValueError(
            f"replaying {PARSED_DIR}/{stem}.json does not reproduce chunk {chunk_index} of "
            f"{CHUNKS_DIR}/{stem}.json, so its atoms belong to a different chunk than the "
            f"one that was embedded and retrieved. {_FLAG_STATE}"
        )
    return packed[chunk_index]


def compress(atoms: list[Atom], scores: list[float], budget: int, heading: str | None) -> str:
    """Highest-scoring atoms up to `budget` tokens, re-emitted in document order.

    CURRENTLY UNUSED, kept deliberately: every measured arm calls `pack_by_score`, which
    emits in score order and carries no heading. This is the template for the fix that
    SESSION.md 2(a) proposes, not dead weight.

    Document order because DSLR found relevance-rank reassembly costs accuracy -- the reader
    loses the discourse thread. The heading is always kept and charged to the budget: it is
    a handful of tokens and it is what says which filing and section the numbers are from.
    """
    prefix = f"# {heading}\n\n" if heading else ""
    remaining = budget - count_tokens(prefix)
    kept: set[int] = set()
    for i in sorted(range(len(atoms)), key=lambda j: scores[j], reverse=True):
        # Not a break: a later, cheaper atom can still fit once an expensive one is
        # skipped, and a skipped table is exactly the case where the next-best atom
        # (often its caption) is the one worth keeping.
        if atoms[i].tokens <= remaining:
            kept.add(i)
            remaining -= atoms[i].tokens
    return prefix + "\n\n".join(atoms[i].text for i in sorted(kept))


# --- slicing --------------------------------------------------------------------
# An atom is too coarse to select on: a gold-bearing atom averages 684 tokens, 76% of its
# chunk, so a budget holds ~3 and the result is effectively recall@2 (0.42 against 0.61 at
# k=10). Slicing to a token target lets the same budget span far more of the retrieved set.
#
# Deliberately NOT chunking._split_table/_split_text: those run at ingest, and changing
# them changes chunk text, which invalidates the embeddings, the BM25 index and every
# scored arm. The duplication is the price of keeping the stored corpus frozen.

# Row count is a weak guard -- a real balance sheet legitimately stacks date, year, a
# two-line units note and a section header above its first figure. The token share is the
# guard that matters: it rejects the pathological case (one table put 18 of its 22 rows
# above the first figure) without discarding ordinary financial statements.
MAX_PREAMBLE_ROWS = 8
MAX_PREAMBLE_TOKEN_SHARE = 0.3

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_YEAR = re.compile(r"^(19|20)\d{2}$")
_LETTER = re.compile(r"[A-Za-z]")
# Accepts a sentence break with no following space: the HTML->text conversion drops it in
# 31.5% of prose atoms ("patterns.Working", "ships.Project"), and a \s+ boundary silently
# refuses to split those, leaving one oversized piece.
_SENTENCE = re.compile(r"(?<=[.!?])(?:\s+|(?=[A-Z]))")


def _is_figure(cell: str) -> bool:
    """A real financial figure: 3+ digits and not a year. Tested per number token, not per
    cell -- `September 24, 2011` holds a 4-digit run and would otherwise read as data,
    making a stacked date header look like the first data row and stranding the column
    labels. Same distinction eval.py draws at MIN_ROW_NUMBER_DIGITS / _YEAR_TOKEN_RE."""
    for tok in _NUM.findall(cell):
        digits = tok.replace(",", "").split(".")[0]
        if len(digits) >= 3 and not _YEAR.match(digits):
            return True
    return False


def _preamble_rows(lines: list[str]) -> int:
    """How many leading rows must be repeated into every row-group: stacked column headers,
    the scale line (`(In thousands)`), section labels (`Revenue:`).

    Returns -1 when no header can be identified with confidence -- either no data row was
    found at all, or the candidate preamble is implausibly large (one real table put 18 of
    its 22 rows above the first figure). Callers keep the table whole in that case: failing
    to slice costs compression, whereas fabricating a header boundary corrupts every slice.
    """
    total = sum(count_tokens(x) for x in lines) or 1
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split("|")]
        if _LETTER.search(cells[0]) and any(_is_figure(c) for c in cells[1:]):
            if i > MAX_PREAMBLE_ROWS:
                return -1
            if sum(count_tokens(x) for x in lines[:i]) > MAX_PREAMBLE_TOKEN_SHARE * total:
                return -1
            return i
    return -1


def _pack(units: list[str], target: int, joiner: str, preamble: str = "") -> list[str]:
    pre_tokens = count_tokens(preamble) if preamble else 0
    out: list[str] = []
    cur: list[str] = []
    cur_tokens = pre_tokens
    for unit in units:
        tokens = count_tokens(unit)
        if cur and cur_tokens + tokens > target:
            out.append(preamble + joiner.join(cur) if preamble else joiner.join(cur))
            cur, cur_tokens = [], pre_tokens
        cur.append(unit)
        cur_tokens += tokens
    if cur:
        out.append(preamble + joiner.join(cur) if preamble else joiner.join(cur))
    return out


def slice_atom(atom: Atom, target: int) -> list[str]:
    """Atom -> pieces of roughly `target` tokens. Prose splits on sentences; a table splits
    into row-groups with its full preamble repeated in each, so no slice ever carries
    numbers under unlabelled columns."""
    if atom.tokens <= target:
        return [atom.text]
    lines = atom.text.split("\n")
    # sec-parser types a footnote list as a TableBlock whenever the filer used a table for
    # layout ("(1) | Includes the impact of adopting guidance..."). Those have no header to
    # repeat and no columns to preserve, so slice them as the prose they are.
    if not atom.is_table or not any(_is_figure(c) for ln in lines for c in ln.split("|")[1:]):
        return _pack(_SENTENCE.split(" ".join(atom.text.split())), target, " ")
    start = _preamble_rows(lines)
    if start < 0:
        return [atom.text]
    preamble = "\n".join(lines[:start])
    return _pack(lines[start:], target, "\n", preamble + "\n" if preamble else "")


def pack_by_score(units: list[tuple[str, int]], budget: int) -> list[str]:
    """Fill a token budget from (text, tokens) units already in descending score order.

    Skip-don't-break: a cheaper unit further down still fits once an oversized one is passed
    over. So this is not monotone per question -- a larger budget can admit one big unit that
    crowds out several small ones, measured at 6/1235 between budgets 500 and 1500 (COST-20).

    Lives here, not in the sweep script, because COST-13 must reproduce byte-for-byte the
    prompt whose gold survival the sweep measured.

    Returns SCORE order with no `[stem chunk N]` provenance, unlike `compress` above, while
    the uncompressed control is chunk-ordered and labelled -- so COST-23 varies three things,
    not one. Open item, SESSION.md 2(a).
    """
    kept, remaining = [], budget
    for text, tokens in units:
        if tokens <= remaining:
            kept.append(text)
            remaining -= tokens
    return kept


@dataclass(frozen=True)
class Slice:
    """A packable piece with the provenance `pack_by_score` throws away."""

    text: str
    tokens: int
    stem: str
    chunk_index: int
    atom_i: int
    piece_i: int
    chunk_rank: int  # position in the retrieval ranking, NOT in this score-ordered list


def slice_heading(stem: str, chunk_index: int) -> str:
    """The label the *uncompressed* control has always carried (`answer_ab_prepare.py`)."""
    return f"[{stem} chunk {chunk_index}]"


def pack_grouped(slices: list[Slice], budget: int) -> str:
    """Same greedy selection as `pack_by_score`, but emitted the way the control is:
    grouped under a `[stem chunk N]` heading, chunks in retrieval-rank order, and slices in
    document order within a chunk.

    Why this exists (COST-27): `pack_by_score` returns score order with no provenance while
    its own control is chunk-ordered and labelled, so COST-23 varied three things at once.
    96% of compressed prompts hold two or more filing-years of the same company -- RETR-3's
    wrong-document problem restated inside one prompt -- and unlabelled, nothing tells the
    reader which year a number belongs to. 39.7% also split one table's rows apart.

    Headings are charged to the budget, so slices@1500 stays honestly 1500 tokens and the
    arm remains the one COST-18 priced. Measured cost of that: 3 questions in 300.

    `pack_by_score` is kept, not replaced: COST-13's payload on disk has to stay reproducible
    byte-for-byte.
    """
    # Chunks come out in retrieval-rank order, matching the uncompressed control. That rank
    # has to be carried in: a chunk's first appearance in this score-ordered list is the
    # chunk holding the best *slice*, which is a different ordering.
    chunk_rank = {(s.stem, s.chunk_index): s.chunk_rank for s in slices}

    kept: list[Slice] = []
    seen: set[tuple[str, int]] = set()
    remaining = budget
    for s in slices:
        key = (s.stem, s.chunk_index)
        cost = s.tokens + (0 if key in seen else count_tokens(slice_heading(*key) + "\n"))
        if cost <= remaining:  # skip-don't-break, as pack_by_score
            kept.append(s)
            seen.add(key)
            remaining -= cost

    out = []
    for key in sorted({(s.stem, s.chunk_index) for s in kept}, key=lambda k: chunk_rank[k]):
        body = sorted((s for s in kept if (s.stem, s.chunk_index) == key),
                      key=lambda s: (s.atom_i, s.piece_i))
        out.append(slice_heading(*key) + "\n" + "\n".join(s.text for s in body))
    return "\n\n".join(out)
