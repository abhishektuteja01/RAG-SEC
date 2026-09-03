"""Evidence compression: keep the highest-scoring atoms of a retrieved chunk, drop the rest.

Day 8 measured that ~89% of the agent's cost is one ~12,200-token evidence block sent
twice (judge, then answer), while the `gold_inds` text that actually answers a question
averages 38 tokens. Retrieval still returns k=10 chunks -- recall@10 is scored on chunk
ids and is untouched -- but only the atoms that earn their place get printed into the
prompt. This is DSLR (arXiv:2407.03627) with `bge-reranker-v2-m3`, the cross-encoder the
Arm 3 pipeline already loads, so scoring costs no API tokens.

The unit of selection is the `Atom` the chunker already packed the chunk from, replayed
from data/parsed/ -- not a regex re-derivation of the flattened chunk text. Atoms know
which spans are tables (never split) and which are prose, so the failure mode that
matters here -- keeping `$1,740` and dropping the `(in millions)` that scales it -- is
bounded by chunk packing rather than reintroduced by a parser of our own.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from rag_sec.chunking import Atom, Chunk, chunk_blocks_with_atoms, count_tokens
from rag_sec.parsing import load_parsed_blocks

PARSED_DIR = Path("data/parsed")


@lru_cache(maxsize=1024)
def _packed(stem: str) -> tuple[tuple[Chunk, tuple[Atom, ...]], ...]:
    """Replay of the packer for one filing. Cached because a question's 10 retrieved
    chunks routinely come from only two or three filings, and the replay re-tokenizes
    every block in the filing. Sized to hold the whole 799-filing corpus (~220MB of atoms):
    a batch job walking 50 candidates per question touches hundreds of filings per pass, and
    a small cache thrashes badly enough to dominate runtime."""
    blocks = load_parsed_blocks(PARSED_DIR / f"{stem}.json")
    return tuple((chunk, tuple(atoms)) for chunk, atoms in chunk_blocks_with_atoms(blocks))


def chunk_atoms(stem: str, chunk_index: int) -> tuple[Chunk, tuple[Atom, ...]]:
    return _packed(stem)[chunk_index]


def compress(atoms: list[Atom], scores: list[float], budget: int, heading: str | None) -> str:
    """Highest-scoring atoms up to `budget` tokens, re-emitted in document order.

    CURRENTLY UNUSED -- kept deliberately. `pack_by_score` below is what every measured arm
    actually calls, and it does neither of the two things this does: it emits in score order
    and carries no heading. Whether that costs accuracy is an open question logged in the
    session notes; this function is the template for the fix, not dead weight to delete.

    Document order, not score order: DSLR found reassembling by relevance rank costs
    accuracy, because the reader loses the discourse thread. The heading is always kept
    and charged to the budget -- it is what identifies which filing and section the
    numbers belong to, and it is a handful of tokens.
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
# An atom is too coarse to select on its own: a gold-bearing atom averages 684 tokens,
# 76% of its chunk, so a budget holds ~3 of them and the result is effectively recall@2
# (0.42 against 0.61 at k=10 -- measured, scripts/diagnostics). Slicing atoms to a token
# target instead lets the same budget span far more of the retrieved set.
#
# This deliberately does NOT reuse chunking._split_table/_split_text. Those run inside
# _blocks_to_atoms at ingest; changing them changes chunk text, which invalidates the
# embeddings, the BM25 index and every scored arm. The duplication is the price of
# keeping the stored corpus frozen, and is intentional rather than an oversight.

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

    Skip-don't-break: a cheaper unit further down still fits once an oversized one is
    passed over. Consequence worth knowing (`COST-20`): this is not monotone per question --
    a larger budget can admit one big unit that crowds out several small ones, measured at
    6/1235 questions between budgets 500 and 1500.

    Lives here rather than in the sweep script because COST-13 has to reproduce byte-for-byte
    the prompt whose gold survival the sweep measured; two copies of this rule would silently
    invalidate the survival flags the experiment's strata are built from.

    Returns units in SCORE order and carries no `[stem chunk N]` provenance -- unlike
    `compress` above, which does both. Callers join the result directly, so every measured
    compressed prompt is score-ordered and anonymous while its uncompressed control is
    chunk-ordered and labelled. Open item, see the session notes: that makes COST-23 a
    three-variable comparison rather than the one-variable one it is written up as.
    """
    kept, remaining = [], budget
    for text, tokens in units:
        if tokens <= remaining:
            kept.append(text)
            remaining -= tokens
    return kept
