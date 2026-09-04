"""Packs parsed blocks into token-budgeted chunks that never split a table mid-row.
Budgets come from T2-RAGBench's own evidence spans (median ~800 tokens, p90 ~1300); Arm 4's
per-table B/C variants are a post-packing splice -- DECISIONS.md CHUNK-2/ARM4-*.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Literal

from transformers import AutoTokenizer

from rag_sec.config import EMBED_MODEL_NAME
from rag_sec.parsing import Block, TableBlock, TextBlock

TableStrategy = Literal["A", "B", "C"]


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


# RETR-7/RETR-8, both reversible from the environment without a code edit: the only way to
# learn whether they helped is a re-index, and the comparison run has to be byte-identical
# to the old packer, not "close".
#
# They default ON as of the 2026-09-04 re-index (`RETR-39`). They shipped OFF, because
# `rag_sec.compress` replays this packer from data/parsed/ and requires it to reproduce the
# stored data/chunks/ byte-for-byte (scripts/checks/atom_replay.py asserts exactly that) --
# so before the re-index, ON would have made compression select from a different document
# than the one embedded. After it, OFF has that same effect in reverse: measured 2026-09-04,
# default-off replay matched only 52,342/99,654 chunks (52.52%), which is precisely the
# 52.5% RETR-7 left untouched. The default has to track whichever corpus is stored.
MULTI_HEADING = _flag("RAG_SEC_MULTI_HEADING", True)
STRIP_TITLE_FURNITURE = _flag("RAG_SEC_STRIP_TITLE_FURNITURE", True)

MIN_CHUNK_TOKENS = 200
TARGET_CHUNK_TOKENS = 900
MAX_CHUNK_TOKENS = 1500
# sec-parser types a long bold paragraph (exhibit-index entry, signature block) as a
# TitleElement. All 16 titles >100 tokens in the 100-filing sample were checked by hand and
# none was a real heading (CHUNK-3). Empirical, but the failure mode is soft: a genuine long
# heading would only lose its grouping role, not corrupt anything.
MAX_TITLE_TOKENS = 100

# RETR-8. sec-parser already drops PageHeaderElement/PageNumberElement (parsing.py), so
# everything here is furniture it typed as a *TitleElement* -- the element-type filter
# cannot catch it and a text test is the only option left. Structural classes, not a list
# of the strings we happened to see; each share is of the 122,524 title instances in the
# corpus. The false positives named below are real ones a looser rule caught.
_FURNITURE_TITLE = (
    # EDGAR typesetter metadata, e.g. ZEQ.=1,SEQ=1,EFW="2096490",CP="APPLE COMPUTER, INC."
    re.compile(r"(?:\bZEQ\b|\bSEQ\s*=|\bCP\s*=|\bEFW\s*=)"),          # 3.41%
    re.compile(r"^/s/"),                                              # 0.28% signature blocks
    re.compile(r"^[^A-Za-z]+$"),                                      # 0.70% phone numbers, rules
    # Page labels only. Deliberately strict: `[A-Z]{0,3}-?\d+` also swallows the exhibit
    # index (EX-1..EX-8) and accounting references (FIN 48, AB5000), which are content.
    re.compile(r"^(?:page\s+)?(?:[FSABC]|I{1,3}|IV|VI{0,3})?[-–]\s?\d{1,3}$", re.I),
    re.compile(r"^\d{1,3}$"),
)

# A title repeated this often inside ONE filing is a running header. The threshold is high
# on purpose: at >=3 the rule deletes real headings -- `income taxes` repeats >=3x in 69
# filings, `revenue recognition` in 19, `condensed consolidating statement of cash flows`
# in 18. At >=10 all 49 distinct strings in the corpus are furniture, every one of the form
# "table of contents <company> notes to consolidated financial statements - (continued)".
RUNNING_HEADER_MIN_REPEATS = 10


def _running_header_titles(blocks: list[Block]) -> set[str]:
    counts = Counter(
        " ".join(b.text.lower().split())
        for b in blocks
        if isinstance(b, TextBlock) and b.is_title
    )
    return {t for t, n in counts.items() if n >= RUNNING_HEADER_MIN_REPEATS}


def is_furniture_title(text: str, running_headers: frozenset[str] | set[str] = frozenset()) -> bool:
    """True if this title is page furniture rather than a section heading (RETR-8)."""
    stripped = text.strip()
    if not stripped:
        return True
    if " ".join(stripped.lower().split()) in running_headers:
        return True
    return any(rx.search(stripped) for rx in _FURNITURE_TITLE)


_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        # Budget against the actual embedding model's tokenizer (rag_sec.config) so
        # chunk sizes match what the embed step sees, not a generic proxy.
        _TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    return _TOKENIZER


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def _table_to_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(row) for row in rows)


def _split_table(rows: list[list[str]]) -> list[str]:
    """Splits an oversized table into row-groups under MAX_CHUNK_TOKENS, repeating the header row in each group."""
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    header_text = " | ".join(header)
    groups: list[str] = []
    current = [header_text]
    current_tokens = count_tokens(header_text)
    for row in body:
        row_text = " | ".join(row)
        row_tokens = count_tokens(row_text)
        if current_tokens + row_tokens > MAX_CHUNK_TOKENS and len(current) > 1:
            groups.append("\n".join(current))
            current = [header_text]
            current_tokens = count_tokens(header_text)
        current.append(row_text)
        current_tokens += row_tokens
    groups.append("\n".join(current))
    return groups


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_text(text: str) -> list[str]:
    """Splits an oversized text block into sentence-packed groups under MAX_CHUNK_TOKENS.

    Filers that never restate "Item N" as a body heading leave sec-parser no structural
    break, so it merges whole Items into one 18k-token element (JPM_2007) -- which without
    this would be a single unembeddable chunk, past BGE-M3's 8192 limit (CHUNK-3).
    """
    normalized = " ".join(text.split())
    sentences = _SENTENCE_SPLIT.split(normalized)

    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        tokens = count_tokens(sentence)
        if tokens > MAX_CHUNK_TOKENS:
            # no sentence boundary to lean on (e.g. a run-on list) -- hard
            # split by raw token windows as a last resort
            if current:
                groups.append(" ".join(current))
                current, current_tokens = [], 0
            ids = _tokenizer().encode(sentence, add_special_tokens=False)
            for start in range(0, len(ids), MAX_CHUNK_TOKENS):
                groups.append(_tokenizer().decode(ids[start : start + MAX_CHUNK_TOKENS]))
            continue
        if current_tokens + tokens > MAX_CHUNK_TOKENS and current:
            groups.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += tokens
    if current:
        groups.append(" ".join(current))
    return groups


@dataclass
class Atom:
    text: str
    tokens: int
    is_title: bool
    is_table: bool
    # forces its own chunk, bypassing the token-packer -- see DECISIONS.md ARM4-1/ARM4-2
    is_standalone: bool = False
    # 0-based position among TableBlocks only (None for text atoms) -- lets
    # chunk_blocks_with_variant splice one table's atoms out after packing (ARM4-4)
    table_idx: int | None = None
    # RETR-8: a title that is page furniture. Still `is_title`, so it keeps its
    # flush-triggering role and chunk boundaries do not move -- it just never becomes a
    # heading, and the section heading it used to clobber survives it.
    is_furniture: bool = False


def _table_rows_to_atoms_b(rows: list[list[str]]) -> list[Atom]:
    """Arm 4 Strategy B: one atom per data row, header row repeated in each -- DECISIONS.md ARM4-1."""
    if not rows:
        return []
    header_text = " | ".join(rows[0])
    atoms = []
    for row in rows[1:]:
        text = f"{header_text}\n{' | '.join(row)}"
        atoms.append(Atom(text=text, tokens=count_tokens(text), is_title=False, is_table=True, is_standalone=True))
    return atoms


def _table_rows_to_atoms_c(rows: list[list[str]], summarize_table: Callable[[list[list[str]]], str]) -> list[Atom]:
    """Arm 4 Strategy C: the raw table (unchanged from Strategy A's own serialization,
    split the same way if oversized) plus a standalone LLM-summary atom -- DECISIONS.md ARM4-2."""
    atoms = []
    table_text = _table_to_text(rows)
    tokens = count_tokens(table_text)
    if tokens > MAX_CHUNK_TOKENS:
        for part in _split_table(rows):
            atoms.append(Atom(text=part, tokens=count_tokens(part), is_title=False, is_table=True))
    else:
        atoms.append(Atom(text=table_text, tokens=tokens, is_title=False, is_table=True))
    summary = summarize_table(rows)
    atoms.append(Atom(text=summary, tokens=count_tokens(summary), is_title=False, is_table=True, is_standalone=True))
    return atoms


def _blocks_to_atoms(blocks: list[Block]) -> list[Atom]:
    """Always builds plain Strategy-A atoms (every table serialized whole) -- variant
    overrides are applied later, as a post-packing splice, not here. See
    chunk_blocks_with_variant and DECISIONS.md ARM4-4."""
    atoms = []
    table_idx = 0
    running_headers = _running_header_titles(blocks) if STRIP_TITLE_FURNITURE else frozenset()
    for block in blocks:
        if isinstance(block, TableBlock):
            idx = table_idx
            table_idx += 1
            table_text = _table_to_text(block.rows)
            tokens = count_tokens(table_text)
            if tokens > MAX_CHUNK_TOKENS:
                for part in _split_table(block.rows):
                    atoms.append(Atom(text=part, tokens=count_tokens(part), is_title=False, is_table=True, table_idx=idx))
            else:
                atoms.append(Atom(text=table_text, tokens=tokens, is_title=False, is_table=True, table_idx=idx))
        else:
            tokens = count_tokens(block.text)
            is_title = block.is_title and tokens <= MAX_TITLE_TOKENS
            if tokens > MAX_CHUNK_TOKENS:
                for part in _split_text(block.text):
                    atoms.append(Atom(text=part, tokens=count_tokens(part), is_title=False, is_table=False))
            else:
                furniture = (
                    STRIP_TITLE_FURNITURE
                    and is_title
                    and is_furniture_title(block.text, running_headers)
                )
                atoms.append(
                    Atom(
                        text=block.text,
                        tokens=tokens,
                        is_title=is_title,
                        is_table=False,
                        is_furniture=furniture,
                    )
                )
    return atoms


@dataclass
class Chunk:
    text: str
    n_tokens: int
    heading: str | None
    standalone: bool = False


def _pack_atoms(atoms: list[Atom]) -> list[tuple[Chunk, list[Atom]]]:
    """The token-budget packer. Returns each Chunk paired with the atoms that made it up, so
    chunk_blocks_with_variant can splice one table's atoms out afterwards rather than
    re-running this with different-sized atoms, which moves every later boundary (ARM4-4)."""
    packed: list[tuple[Chunk, list[Atom]]] = []
    # RETR-7: the heading each atom actually sits under, recorded when the atom is buffered
    # rather than read off `current_heading` at flush time. That single mutable variable was
    # the bug -- a short section's text was emitted under the *next* section's heading,
    # wrong for 49.7% of atoms, and unrelated to the true topic in 54.7% of those.
    packed_headings: list[tuple[str, ...]] = []
    # Token count each chunk WOULD have had under the pre-RETR-7 heading. The fold-back
    # below thresholds on a count that includes the heading line, so changing the heading
    # silently changes which tiny chunks get folded -- which moves boundaries, which moves
    # the chunk indices eval.py's gold labels are. Deciding on the legacy count keeps
    # boundaries identical by construction rather than by hoping the drift is small.
    packed_legacy_tokens: list[int] = []
    current: list[Atom] = []
    current_headings: list[str | None] = []
    current_tokens = 0
    current_heading: str | None = None
    # what `current_heading` would have been under the pre-fix rule, where every title --
    # furniture included -- overwrote it. Only used to size the fold-back decision, never
    # emitted. Without it, stripping furniture changes `current_heading`, which changes the
    # token count, which changes which chunks get folded, which moves boundaries.
    legacy_heading: str | None = None

    def render(headings: tuple[str, ...]) -> str | None:
        return "\n# ".join(headings) if headings else None

    def flush() -> None:
        nonlocal current, current_headings, current_tokens
        if not current:
            return
        if MULTI_HEADING:
            # every section this chunk actually spans, in order, deduped
            headings = tuple(dict.fromkeys(h for h in current_headings if h))
        else:
            headings = (current_heading,) if current_heading else ()
        heading = render(headings)
        body = "\n\n".join(a.text for a in current)
        text = f"# {heading}\n\n{body}" if heading else body
        tokens = count_tokens(text)
        legacy_text = f"# {legacy_heading}\n\n{body}" if legacy_heading else body
        legacy_tokens = tokens if heading == legacy_heading else count_tokens(legacy_text)
        # a tiny trailing chunk (e.g. one short leftover paragraph) is mostly
        # index noise -- fold it into the previous chunk instead, unless that
        # previous chunk is a standalone table row/summary (Arm 4 B/C), which
        # must stay isolated or it loses the point of being a small chunk
        if (
            packed
            and not packed[-1][0].standalone
            and legacy_tokens < MIN_CHUNK_TOKENS
            and packed_legacy_tokens[-1] + legacy_tokens <= MAX_CHUNK_TOKENS
        ):
            prev, prev_atoms = packed[-1]
            # the fold-back is where flushing on every title would have re-introduced the
            # bug: it merges across a section boundary, so the merged chunk has to own both
            # headings rather than silently keep the earlier one
            merged_atoms = prev_atoms + list(current)
            if MULTI_HEADING:
                merged = tuple(dict.fromkeys(packed_headings[-1] + headings))
                merged_heading = render(merged)
                merged_body = "\n\n".join(a.text for a in merged_atoms)
                merged_text = f"# {merged_heading}\n\n{merged_body}" if merged_heading else merged_body
                merged_chunk = Chunk(text=merged_text, n_tokens=count_tokens(merged_text), heading=merged_heading)
            else:
                merged = packed_headings[-1]
                merged_chunk = Chunk(
                    text=f"{prev.text}\n\n{body}", n_tokens=prev.n_tokens + tokens, heading=prev.heading
                )
            packed_headings[-1] = merged
            packed_legacy_tokens[-1] += legacy_tokens
            packed[-1] = (merged_chunk, merged_atoms)
        else:
            packed.append((Chunk(text=text, n_tokens=tokens, heading=heading), list(current)))
            packed_headings.append(headings)
            packed_legacy_tokens.append(legacy_tokens)
        current = []
        current_headings = []
        current_tokens = 0

    for i, atom in enumerate(atoms):
        if atom.is_title:
            # prefer a fresh chunk at section boundaries once we already
            # have a reasonable amount of content
            if current_tokens >= TARGET_CHUNK_TOKENS:
                flush()
            # RETR-8: furniture still reaches the flush check above -- suppressing that too
            # would move chunk boundaries, and the whole point of doing this without a
            # re-chunk is that boundaries stay put. It just never becomes the heading, so
            # the real section heading it used to overwrite survives it.
            legacy_heading = atom.text
            if not atom.is_furniture:
                current_heading = atom.text
            continue

        if atom.is_standalone:
            flush()
            text = f"# {current_heading}\n\n{atom.text}" if current_heading else atom.text
            chunk = Chunk(text=text, n_tokens=count_tokens(text), heading=current_heading, standalone=True)
            packed.append((chunk, [atom]))
            packed_headings.append((current_heading,) if current_heading else ())
            legacy_sa = f"# {legacy_heading}\n\n{atom.text}" if legacy_heading else atom.text
            packed_legacy_tokens.append(
                chunk.n_tokens if legacy_heading == current_heading else count_tokens(legacy_sa)
            )
            continue

        # a table's evidence usually only makes sense with the paragraph
        # right after it (per T2-RAGBench's own pre_text/table/post_text
        # bundling) -- keep them together past the soft target if it fits
        # under the hard ceiling
        lookahead_tokens = 0
        if atom.is_table and i + 1 < len(atoms) and not atoms[i + 1].is_title and not atoms[i + 1].is_table:
            lookahead_tokens = atoms[i + 1].tokens

        projected = current_tokens + atom.tokens
        keep_with_table = atom.is_table and projected + lookahead_tokens <= MAX_CHUNK_TOKENS

        if current and projected > MAX_CHUNK_TOKENS:
            flush()
        elif current and projected > TARGET_CHUNK_TOKENS and not keep_with_table:
            flush()

        current.append(atom)
        current_headings.append(current_heading)
        current_tokens += atom.tokens

    flush()
    return packed


def chunk_blocks_with_atoms(blocks: list[Block]) -> list[tuple[Chunk, list[Atom]]]:
    """Each chunk paired with the atoms it was packed from. `chunk_blocks` drops the
    atoms; rag_sec.compress needs them -- they are the structure-aware block boundaries
    (table vs prose, one atom per source TableBlock) that would otherwise have to be
    re-derived from the flattened chunk text by regex."""
    return _pack_atoms(_blocks_to_atoms(blocks))


def chunk_blocks(blocks: list[Block]) -> list[Chunk]:
    return [chunk for chunk, _ in chunk_blocks_with_atoms(blocks)]


def _build_chunk_from_atoms(atoms: list[Atom], heading: str | None) -> Chunk:
    body = "\n\n".join(a.text for a in atoms)
    text = f"# {heading}\n\n{body}" if heading else body
    return Chunk(text=text, n_tokens=count_tokens(text), heading=heading)


def chunk_blocks_with_variant(
    blocks: list[Block],
    table_variant_map: dict[int, TableStrategy],
    summarize_table: Callable[[list[list[str]]], str] | None = None,
) -> list[Chunk]:
    """Builds Strategy-A chunk boundaries first (`_pack_atoms`, identical to `chunk_blocks`),
    then splices each targeted table's atoms out of whichever chunk(s) they landed in and
    replaces them with its B/C atoms -- surrounding chunks, and any leading/trailing
    narrative sharing a chunk with the table, are left byte-identical to Strategy A because
    the packer runs exactly once, on plain atoms (DECISIONS.md ARM4-4).
    """
    if summarize_table is None and "C" in table_variant_map.values():
        raise ValueError("Strategy C needs a summarize_table callable (see rag_sec.summarize)")
    table_rows_by_idx = {i: b.rows for i, b in enumerate(blk for blk in blocks if isinstance(blk, TableBlock))}
    packed = _pack_atoms(_blocks_to_atoms(blocks))

    already_replaced: set[int] = set()
    output: list[Chunk] = []
    for chunk, chunk_atoms in packed:
        groups: list[tuple[int | None, list[Atom]]] = []
        for atom in chunk_atoms:
            key = atom.table_idx if atom.table_idx in table_variant_map else None
            if groups and groups[-1][0] == key:
                groups[-1][1].append(atom)
            else:
                groups.append((key, [atom]))

        if len(groups) == 1 and groups[0][0] is None:
            output.append(chunk)  # no targeted table in this chunk -- untouched
            continue

        for key, group_atoms in groups:
            if key is None:
                output.append(_build_chunk_from_atoms(group_atoms, chunk.heading))
                continue
            if key in already_replaced:
                continue  # this table's atoms were already emitted from an earlier chunk
            already_replaced.add(key)
            strategy = table_variant_map[key]
            rows = table_rows_by_idx[key]
            repl_atoms = (
                _table_rows_to_atoms_b(rows) if strategy == "B" else _table_rows_to_atoms_c(rows, summarize_table)
            )
            for atom in repl_atoms:
                text = f"# {chunk.heading}\n\n{atom.text}" if chunk.heading else atom.text
                output.append(Chunk(text=text, n_tokens=count_tokens(text), heading=chunk.heading, standalone=True))

    return output
