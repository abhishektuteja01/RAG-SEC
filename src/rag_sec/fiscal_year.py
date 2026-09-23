"""Fiscal-year signal for retrieval: extracts years mentioned in a query and blends them into
candidate ranking as an additive nudge, never a hard filter -- the extraction signal is only
75-82% accurate (figure's source untraced, see RETR-40), and a hard year filter at that accuracy deletes the right answer
on every miss. Chunk-side year comes from `filing_stem`'s own naming convention
(`TICKER_YEAR_CIK`, verified against all 799 filings), not a query-side guess.
"""

import re

_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_FY_SHORTHAND_RE = re.compile(r"\bFY\s?(\d{2})\b", re.IGNORECASE)
_DOLLAR_BEFORE_RE = re.compile(r"\$\s*$")
_UNIT_AFTER_RE = re.compile(
    r"^\s*(million|billion|thousand|%|percent|shares|units|bps|basis points)",
    re.IGNORECASE,
)


def extract_years(text: str) -> list[int]:
    """Years mentioned in free text. Bare 4-digit years, skipping ones immediately preceded
    by `$` or followed by a unit word (a dollar amount or share count that happens to fall in
    year range, not a year) -- measured at 1 false positive in ~5,000 real questions, so this
    guard alone is enough; a cue-word requirement ("in", "fiscal", ...) was tried and tested
    worse, dropping the true-year hit rate from 86% to 22%, since most real phrasing doesn't
    place a cue word next to the year. Also matches `FY19`-style shorthand (0.26% of
    questions), windowed at the standard pivot: >50 -> 19xx, <=50 -> 20xx.
    """
    years: set[int] = set()
    for m in _YEAR_RE.finditer(text):
        if _DOLLAR_BEFORE_RE.search(text[max(0, m.start() - 6):m.start()]):
            continue
        if _UNIT_AFTER_RE.match(text[m.end():m.end() + 20]):
            continue
        years.add(int(m.group()))
    for m in _FY_SHORTHAND_RE.finditer(text):
        yy = int(m.group(1))
        years.add(1900 + yy if yy > 50 else 2000 + yy)
    return sorted(years)


def chunk_year(filing_stem: str) -> int:
    """Fiscal year from a chunk's filing stem, `TICKER_YEAR_CIK` -- verified against all 799
    filings under `data/chunks`, always a clean 3-part stem with a 4-digit year in the middle.
    The last part is the CIK, not the year: `split('_')[1]`, not `[-1]`.
    """
    return int(filing_stem.split("_")[1])


def year_distance_bonus(filing_stem: str, query_years: list[int], alpha: float) -> float:
    """Additive nudge toward chunks whose fiscal year is close to a year the query mentions.
    Zero when the query mentions no year, so a missed extraction just fails to help -- it can
    never actively demote a candidate the way a hard filter would (RETR-5's rejected fix).
    """
    if not query_years:
        return 0.0
    distance = min(abs(chunk_year(filing_stem) - qy) for qy in query_years)
    return alpha / (1 + distance)
