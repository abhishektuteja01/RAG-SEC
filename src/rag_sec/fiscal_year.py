"""Years for retrieval: the years a question (or chunk) mentions, and a small ranking bonus
for filings near them. A bonus, never a filter: the extraction is not always right, and a
filter would delete the right answer on every miss. A filing's year comes from its name,
`TICKER_YEAR_CIK`.
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
    """Years mentioned in free text: bare 4-digit years, skipping ones right after `$` or
    right before a unit word (a dollar amount or share count, not a year). Also `FY19`-style
    shorthand: >50 -> 19xx, <=50 -> 20xx.
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
    """Fiscal year from a filing stem, `TICKER_YEAR_CIK`. The last part is the CIK."""
    return int(filing_stem.split("_")[1])


def year_distance_bonus(filing_stem: str, query_years: list[int], alpha: float) -> float:
    """Bonus for a filing whose year is close to a year the query mentions; 0 if none."""
    if not query_years:
        return 0.0
    distance = min(abs(chunk_year(filing_stem) - qy) for qy in query_years)
    return alpha / (1 + distance)
