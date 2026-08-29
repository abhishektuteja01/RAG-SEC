"""Fetches real 10-K filings from SEC EDGAR by CIK + fiscal year.

Used to reconstruct full filings (not just the single annotated page T2-RAGBench
ships), so parsing and retrieval are exercised at production scale, not toy snippets.
"""

import os
import time

import httpx

SEC_REQUEST_INTERVAL_S = 0.15  # stays under SEC's ~10 req/sec guidance


def _user_agent() -> str:
    email = os.environ.get("EDGAR_CONTACT_EMAIL")
    if not email:
        raise RuntimeError("EDGAR_CONTACT_EMAIL must be set (see .env.example) — SEC requires a real contact email.")
    return f"rag-sec research project ({email})"


def _get(url: str) -> httpx.Response:
    time.sleep(SEC_REQUEST_INTERVAL_S)
    resp = httpx.get(url, headers={"User-Agent": _user_agent()}, timeout=30.0)
    resp.raise_for_status()
    return resp


def _search_filings_block(block: dict, cik: int, fiscal_year: int) -> dict | None:
    for form, filing_date, accession, primary_doc, report_date in zip(
        block["form"],
        block["filingDate"],
        block["accessionNumber"],
        block["primaryDocument"],
        block["reportDate"],
    ):
        if form != "10-K":
            continue
        if report_date and int(report_date[:4]) == fiscal_year:
            return {
                "cik": cik,
                "accession_number": accession,
                "primary_document": primary_doc,
                "filing_date": filing_date,
            }
    return None


def find_10k_accession(cik: int, fiscal_year: int) -> dict | None:
    """Finds the 10-K filed for a given fiscal year.

    Matches on the filing's `reportDate` (period of report) year, not `filingDate`.
    A prior version matched filingDate in {fiscal_year, fiscal_year+1} to account for
    the filing lag after fiscal year-end — but for non-calendar-fiscal-year filers
    (e.g. Sysco, Apple, Nike), consecutive fiscal years' 10-Ks can both have filingDate
    in that window, and since EDGAR lists filings most-recent-first, the wrong
    (later) fiscal year's 10-K would be returned. Confirmed on the 100-filing corpus:
    26/100 had the wrong fiscal year under the old logic (DECISIONS.md DATA-5).

    The submissions API's "recent" block only covers roughly the company's last
    ~1,000 filings; older filings live in separate paginated JSON files listed
    under filings.files, so we fall back to those for older fiscal years.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    data = _get(url).json()
    filings = data["filings"]

    hit = _search_filings_block(filings["recent"], cik, fiscal_year)
    if hit is not None:
        return hit

    for older_file in filings.get("files", []):
        older_url = f"https://data.sec.gov/submissions/{older_file['name']}"
        older_block = _get(older_url).json()
        hit = _search_filings_block(older_block, cik, fiscal_year)
        if hit is not None:
            return hit

    return None


def filing_document_url(cik: int, accession_number: str, primary_document: str) -> str:
    accession_nodash = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_document}"


def download_filing(cik: int, accession_number: str, primary_document: str) -> bytes:
    url = filing_document_url(cik, accession_number, primary_document)
    return _get(url).content
