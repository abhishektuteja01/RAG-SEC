"""Packs parsed blocks into token-budgeted chunks that never split a table mid-row.
Budgets come from T2-RAGBench's own evidence spans (median ~800 tokens, p90 ~1300); Arm 4's
per-table B/C variants are a post-packing splice -- DECISIONS.md CHUNK-2/ARM4-*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal

from transformers import AutoTokenizer

from rag_sec.config import EMBED_MODEL_NAME
from rag_sec.parsing import Block, TableBlock

TableStrategy = Literal["A", "B", "C"]

MIN_CHUNK_TOKENS = 200
TARGET_CHUNK_TOKENS = 900
MAX_CHUNK_TOKENS = 1500
# sec-parser types a long bold paragraph (exhibit-index entry, signature block) as a
# TitleElement. All 16 titles >100 tokens in the 100-filing sample were checked by hand and
# none was a real heading (CHUNK-3). Empirical, but the failure mode is soft: a genuine long
# heading would only lose its grouping role, not corrupt anything.
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
                atoms.append(Atom(text=block.text, tokens=tokens, is_title=is_title, is_table=False))
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
        # index noise -- fold it into the previous chunk instead, unless that
        # previous chunk is a standalone table row/summary (Arm 4 B/C), which
        # must stay isolated or it loses the point of being a small chunk
        if (
            packed
            and not packed[-1][0].standalone
            and tokens < MIN_CHUNK_TOKENS
            and packed[-1][0].n_tokens + tokens <= MAX_CHUNK_TOKENS
        ):
            prev, prev_atoms = packed[-1]
            packed[-1] = (
                Chunk(text=f"{prev.text}\n\n{body}", n_tokens=prev.n_tokens + tokens, heading=prev.heading),
                prev_atoms + list(current),
            )
        else:
            packed.append((Chunk(text=text, n_tokens=tokens, heading=current_heading), list(current)))
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

        if atom.is_standalone:
            flush()
            text = f"# {current_heading}\n\n{atom.text}" if current_heading else atom.text
            chunk = Chunk(text=text, n_tokens=count_tokens(text), heading=current_heading, standalone=True)
            packed.append((chunk, [atom]))
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
