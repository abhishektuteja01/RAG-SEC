"""Chunking v1: packs parsed blocks into token-budgeted, structure-aware chunks.

Bounds derived from T2-RAGBench's own evidence spans (median 630 words / ~800
tokens, p90 990 words / ~1300 tokens) -- see DECISIONS.md. Tables are never
split mid-row; oversized tables are split by row-group with the header row
repeated in each group instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer

import re

from rag_sec.config import EMBED_MODEL_NAME
from rag_sec.parsing import Block, TableBlock

MIN_CHUNK_TOKENS = 200
TARGET_CHUNK_TOKENS = 900
MAX_CHUNK_TOKENS = 1500
# sec-parser sometimes misclassifies a long bold-formatted paragraph (e.g. an
# Item 15 exhibit-index entry, boilerplate cross-reference, or signature
# block) as a TitleElement. Checked every title >100 tokens across all 100
# sampled filings (16 total): none are genuine headings. Not a logical
# guarantee for filings outside this sample -- but the failure mode is soft
# either way (a real long heading would just lose its grouping role and
# become ordinary body text, not corrupted data).
MAX_TITLE_TOKENS = 100

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

    Some filers (e.g. JPM_2007) don't restate "Item N" as body headings, so
    sec-parser -- lacking any internal structural break -- merges whole
    swaths of body text (Item 1 through Item 4, in that case) into a single
    18k-token TextElement. Without this, that one atom would become one
    unembeddable oversized chunk (BGE-M3's max context is 8192 tokens).
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


def _blocks_to_atoms(blocks: list[Block]) -> list[Atom]:
    atoms = []
    for block in blocks:
        if isinstance(block, TableBlock):
            table_text = _table_to_text(block.rows)
            tokens = count_tokens(table_text)
            if tokens > MAX_CHUNK_TOKENS:
                for part in _split_table(block.rows):
                    atoms.append(Atom(text=part, tokens=count_tokens(part), is_title=False, is_table=True))
            else:
                atoms.append(Atom(text=table_text, tokens=tokens, is_title=False, is_table=True))
        else:
            tokens = count_tokens(block.text)
            is_title = block.is_title and tokens <= MAX_TITLE_TOKENS
            if tokens > MAX_CHUNK_TOKENS:
                for part in _split_text(block.text):
                    atoms.append(Atom(text=part, tokens=count_tokens(part), is_title=False, is_table=False))
            else:
                atoms.append(Atom(text=block.text, tokens=tokens, is_title=is_title, is_table=False))
    return atoms


@dataclass
class Chunk:
    text: str
    n_tokens: int
    heading: str | None


def chunk_blocks(blocks: list[Block]) -> list[Chunk]:
    atoms = _blocks_to_atoms(blocks)

    chunks: list[Chunk] = []
    current: list[Atom] = []
    current_tokens = 0
    current_heading: str | None = None

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        body = "\n\n".join(a.text for a in current)
        text = f"# {current_heading}\n\n{body}" if current_heading else body
        tokens = count_tokens(text)
        # a tiny trailing chunk (e.g. one short leftover paragraph) is mostly
        # index noise -- fold it into the previous chunk instead
        if chunks and tokens < MIN_CHUNK_TOKENS and chunks[-1].n_tokens + tokens <= MAX_CHUNK_TOKENS:
            prev = chunks[-1]
            chunks[-1] = Chunk(
                text=f"{prev.text}\n\n{body}",
                n_tokens=prev.n_tokens + tokens,
                heading=prev.heading,
            )
        else:
            chunks.append(Chunk(text=text, n_tokens=tokens, heading=current_heading))
        current = []
        current_tokens = 0

    for i, atom in enumerate(atoms):
        if atom.is_title:
            # prefer a fresh chunk at section boundaries once we already
            # have a reasonable amount of content
            if current_tokens >= TARGET_CHUNK_TOKENS:
                flush()
            current_heading = atom.text
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
        current_tokens += atom.tokens

    flush()
    return chunks
