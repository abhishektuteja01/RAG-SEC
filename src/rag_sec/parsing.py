"""Parses 10-K HTML into structured text/table blocks using sec-parser.

sec-parser ships only Edgar10QParser (no native 10-K support as of 0.58.1), so
this strips its 10-Q-specific top-section classification step and derives
Item boundaries ourselves via regex. See DECISIONS.md CHUNK-1 for why, and the
fallback (Docling + normalization) if a sec-parser version bump breaks the
internal classes this relies on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import bs4
import sec_parser as sp
from sec_parser.processing_steps import (
    IndividualSemanticElementExtractor,
    TopSectionManagerFor10Q,
    TopSectionTitleCheck,
)
from sec_parser.semantic_elements.semantic_elements import (
    EmptyElement,
    PageHeaderElement,
    PageNumberElement,
    SupplementaryText,
    TextElement,
)
from sec_parser.semantic_elements.table_element.table_element import TableElement
from sec_parser.semantic_elements.table_element.table_of_contents_element import (
    TableOfContentsElement,
)
from sec_parser.semantic_elements.title_element import TitleElement

_NOISE_TYPES = (EmptyElement, PageHeaderElement, PageNumberElement, TableOfContentsElement)

# Splits like "(619" / ")" or "$" / "6,635" across adjacent table cells are a
# source-HTML artifact (filing agents wrap each text run in its own <td>), not
# a parsing bug -- both Docling and pandas.read_html reproduce them identically
# from the raw HTML (verified on JPM_2007_19617). Reassemble before storing.
_OPEN_PAREN_NUM = re.compile(r"^\(-?[\d,]+\.?\d*$")
_CURRENCY_PREFIX = {"$", "-$"}

ITEM_HEADING_RE = re.compile(r"^\s*item\s+(\d+[a-z]?)[.:]?\s*(.*)$", re.IGNORECASE)


@dataclass
class TextBlock:
    text: str
    is_title: bool


@dataclass
class TableBlock:
    rows: list[list[str]]


Block = TextBlock | TableBlock


def _without_10q_section_classification() -> sp.Edgar10QParser:
    """Builds a parser with the 10-Q-only top-section step removed.

    Workaround from github.com/Elijas/sec-parser-exploration (2023-04),
    "02_other_sec_form_types.ipynb", Method 2 -- otherwise every 10-K item
    (7, 7A, 8, 9, 9A, Part III...) gets misclassified as
    InvalidTopSectionIn10Q.
    """

    def build_steps():
        all_steps = sp.Edgar10QParser().get_default_steps()
        steps = [s for s in all_steps if not isinstance(s, TopSectionManagerFor10Q)]

        def checks_without_top_section_title():
            all_checks = sp.Edgar10QParser().get_default_single_element_checks()
            return [c for c in all_checks if not isinstance(c, TopSectionTitleCheck)]

        return [
            IndividualSemanticElementExtractor(get_checks=checks_without_top_section_title)
            if isinstance(s, IndividualSemanticElementExtractor)
            else s
            for s in steps
        ]

    return sp.Edgar10QParser(get_steps=build_steps)


def _clean_table_row(cells: list[str]) -> list[str]:
    """Drops spacer cells and reassembles values split across per-run cell boundaries."""
    cells = [c.strip() for c in cells if c.strip()]
    merged: list[str] = []
    i = 0
    while i < len(cells):
        cell = cells[i]
        nxt = cells[i + 1] if i + 1 < len(cells) else None
        if cell in _CURRENCY_PREFIX and nxt is not None:
            merged.append(cell + nxt)
            i += 2
            continue
        if _OPEN_PAREN_NUM.match(cell) and nxt == ")":
            merged.append(cell + ")")
            i += 2
            continue
        merged.append(cell)
        i += 1
    return merged


def _table_to_rows(table_element: TableElement) -> list[list[str]]:
    soup = bs4.BeautifulSoup(table_element.get_source_code(), "lxml")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        cleaned = _clean_table_row(cells)
        if cleaned:
            rows.append(cleaned)
    return rows


def parse_filing(html: str) -> list[Block]:
    """Parses a 10-K's raw HTML into an ordered list of text/table blocks.

    Filters out page numbers, running headers, empty elements, and the
    boilerplate table-of-contents table -- these are all noise for chunking,
    not content.
    """
    parser = _without_10q_section_classification()
    elements = parser.parse(html)

    blocks: list[Block] = []
    for element in elements:
        if isinstance(element, _NOISE_TYPES):
            continue
        if isinstance(element, TableElement):
            rows = _table_to_rows(element)
            if rows:
                blocks.append(TableBlock(rows=rows))
        elif isinstance(element, (TextElement, TitleElement, SupplementaryText)):
            text = str(element.text).strip()
            if text:
                blocks.append(TextBlock(text=text, is_title=isinstance(element, TitleElement)))
    return blocks


def find_item_boundaries(blocks: list[Block]) -> list[tuple[int, str, str]]:
    """Finds (block_index, item_number, item_title) for each 'Item N. ...' heading.

    sec-parser doesn't classify 10-K item sections natively (only 10-Q), so
    we detect them ourselves from title blocks. Known gap: some filers (e.g.
    JPM_2007) never restate "Item N" as a body heading -- they use
    business-narrative headings (e.g. "EXECUTIVE OVERVIEW") instead, and
    "Item N" appears only once in the front-matter index. This returns an
    empty list for those filings rather than guessing; accepted as a known
    limitation for now (see DECISIONS.md CHUNK-1), not fixed via TOC
    cross-referencing.
    """
    boundaries = []
    for i, block in enumerate(blocks):
        if not isinstance(block, TextBlock) or not block.is_title:
            continue
        match = ITEM_HEADING_RE.match(block.text)
        if match:
            boundaries.append((i, match.group(1).upper(), match.group(2).strip()))
    return boundaries
