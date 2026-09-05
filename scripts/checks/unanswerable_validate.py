"""Checks that the unanswerable set is actually unanswerable, for the 30 of 48 questions
where that claim is machine-checkable.

WHY THIS EXISTS. A refusal benchmark is worthless if a question turns out to be answerable:
the model refuses, gets marked correct, and the number is a lie. That is this project's
recurring bug class (RETR-24, AGENT-16) pointed at a label file -- an assumption about an
artifact nobody verified. So the claim gets a check.

WHAT IS CHECKED, per category:
  out_of_corpus_company   the named ticker has zero filings in data/chunks/
  cross_company_premise   the attributed company has zero filings, and the host company does
  out_of_corpus_year      the year is outside every filing's REPORTING REACH for that company

That last rule is the one worth reading. A year is NOT absent just because no filing carries
it in the filename. A 10-K reprints the prior year's comparatives, and filings of this era
carried a five-year Selected Financial Data table (Item 6) -- which is the same reprinting
RETR-3 blames for retrieval failures. So a 2010 filing puts 2005-2010 in reach, and asking
about Apple's 2009 revenue would have been answerable from the 2010 10-K. A year is only safe
if it is LATER than the company's last filing, or at least SPAN years EARLIER than its first.

NOT CHECKED, and deliberately: out_of_scope_metric and underspecified (18 questions) rest on
what a 10-K contains rather than on what is in the corpus. No structural check can settle
them. They are marked `verifiable: false` in the data file and need a human read.

Usage:
    python scripts/checks/unanswerable_validate.py
"""

import collections
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = _ROOT / "data" / "unanswerable_questions.jsonl"
CHUNKS = _ROOT / "data" / "chunks"
# Prior-year comparatives (1) plus Item 6's five-year Selected Financial Data table.
REPRINT_SPAN = 6


def corpus_years() -> dict[str, set[int]]:
    """ticker -> filing years, read from chunk filenames (TICKER_YEAR_CIK.json)."""
    out = collections.defaultdict(set)
    for p in CHUNKS.glob("*.json"):
        parts = p.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            out[parts[0]].add(int(parts[1]))
    return out


def main() -> int:
    if not QUESTIONS.exists():
        print(f"error: {QUESTIONS} missing", file=sys.stderr)
        return 1
    if not CHUNKS.is_dir():
        # Gitignored corpus. Skipped rather than failed so a fresh checkout still runs.
        print(f"note: {CHUNKS} absent, corpus half skipped")
        return 0

    universe = corpus_years()
    rows = [json.loads(ln) for ln in QUESTIONS.read_text().splitlines() if ln.strip()]
    failures = []
    checked = 0

    seen = collections.Counter(r["id"] for r in rows)
    for qid, n in seen.items():
        if n > 1:
            failures.append(f"duplicate id {qid} ({n}x)")

    for r in rows:
        b, cat = r["basis"], r["category"]
        if not r["verifiable"]:
            if b:
                failures.append(f"{r['id']}: marked unverifiable but carries a basis {b}")
            continue

        if "absent_ticker" in b:
            checked += 1
            if b["absent_ticker"] in universe:
                failures.append(f"{r['id']}: {b['absent_ticker']} HAS "
                                f"{len(universe[b['absent_ticker']])} filings -- answerable")
        if "host_ticker" in b:
            # The host must be present, or the question is unanswerable for the wrong reason
            # and stops testing cross-company attribution at all.
            if b["host_ticker"] not in universe:
                failures.append(f"{r['id']}: host {b['host_ticker']} absent from corpus, so "
                                f"this tests the wrong thing")
        if "year" in b:
            checked += 1
            tk, yr = b["ticker"], b["year"]
            years = universe.get(tk)
            if not years:
                failures.append(f"{r['id']}: {tk} absent, so this is not a year question")
            elif not (yr > max(years) or yr <= min(years) - REPRINT_SPAN):
                reach = f"{min(years) - REPRINT_SPAN + 1}-{max(years)}"
                failures.append(f"{r['id']}: {tk} {yr} is inside the reporting reach {reach} "
                                f"(filings {sorted(years)}) -- a comparative or the five-year "
                                f"table can answer it")

    by_cat = collections.Counter(r["category"] for r in rows)
    if failures:
        print(f"unanswerable set has {len(failures)} problem(s):\n", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(f"ok: {len(rows)} questions, {checked} claims machine-verified against "
          f"{sum(len(v) for v in universe.values())} filings")
    for c, n in sorted(by_cat.items()):
        v = sum(r["verifiable"] for r in rows if r["category"] == c)
        print(f"  {c:<24} {n:>3}  ({v} verified, {n - v} asserted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
