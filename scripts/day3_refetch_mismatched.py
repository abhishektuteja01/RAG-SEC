"""Day 3: find and re-fetch filings ingested under the wrong fiscal year.

DECISIONS.md #14: `edgar.py`'s old filingDate-window matching picked the wrong 10-K
for non-calendar-fiscal-year filers (Apple, Nike, Sysco, etc.). Fixed to match on
reportDate instead. This audits every filing currently in data/chunks/ against
EDGAR's own reportDate, then deletes and re-fetches/re-parses any mismatch found.

Re-run scripts/day2_chunk.py afterward to rebuild chunks for the whole corpus.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from day2_ingest import FILINGS_DIR, PARSED_DIR, process_one

from rag_sec.edgar import _get, find_10k_accession

CHUNKS_DIR = Path(__file__).resolve().parent.parent / "data" / "chunks"


def report_date_for_accession(cik: int, accession: str) -> str | None:
    data = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    filings = data["filings"]
    blocks = [filings["recent"]]
    for older in filings.get("files", []):
        blocks.append(_get(f"https://data.sec.gov/submissions/{older['name']}").json())
    for block in blocks:
        for a, rd in zip(block["accessionNumber"], block["reportDate"]):
            if a == accession:
                return rd
    return None


def find_mismatches() -> list[tuple[str, int, int]]:
    """Returns (symbol, cik, expected_year) for every filing whose actual reportDate
    year doesn't match the fiscal year encoded in its filename."""
    mismatches = []
    stems = sorted(p.stem for p in CHUNKS_DIR.glob("*.json"))
    for i, stem in enumerate(stems, 1):
        symbol, year, cik = stem.rsplit("_", 2)
        year, cik = int(year), int(cik)
        hit = find_10k_accession(cik, year)
        if hit is None:
            print(f"[{i}/{len(stems)}] {stem}: no 10-K found for this fiscal year")
            continue
        report_date = report_date_for_accession(cik, hit["accession_number"])
        if report_date is None:
            print(f"[{i}/{len(stems)}] {stem}: accession not found in submissions history")
            continue
        actual_year = int(report_date[:4])
        status = "ok" if actual_year == year else "MISMATCH"
        print(f"[{i}/{len(stems)}] {stem}: {status} (reportDate={report_date})")
        if actual_year != year:
            mismatches.append((symbol, cik, year))
    return mismatches


def refetch(symbol: str, cik: int, year: int) -> None:
    stem = f"{symbol}_{year}_{cik}"
    for old_path in list(FILINGS_DIR.glob(f"{stem}.*")) + [PARSED_DIR / f"{stem}.json"]:
        if old_path.exists():
            old_path.unlink()
            print(f"  removed stale {old_path.name}")
    process_one({"company_cik": cik, "report_year": year, "company_symbol": symbol})


def main() -> None:
    print("Auditing ingested filings against EDGAR reportDate...")
    mismatches = find_mismatches()
    print(f"\n{len(mismatches)} mismatched filings found")

    for i, (symbol, cik, year) in enumerate(mismatches, 1):
        print(f"[{i}/{len(mismatches)}] refetching {symbol}_{year}_{cik}")
        refetch(symbol, cik, year)

    if mismatches:
        print("Done. Now re-run: uv run python scripts/day2_chunk.py")
    else:
        print("Done. No mismatches — nothing to re-fetch.")


if __name__ == "__main__":
    main()
