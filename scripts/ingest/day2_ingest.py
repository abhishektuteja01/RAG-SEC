"""Day 2: fetch ~100 real 10-K filings from EDGAR and parse each with sec-parser.

Samples unique (cik, report_year) pairs from T2-RAGBench across its three
subsets, downloads the real filing (not the dataset's single cut page), runs
it through sec-parser (see rag_sec.parsing and DECISIONS.md CHUNK-1 for why not
Docling), and logs fetch/parse outcomes + timing so we have a concrete list
of where the parser breaks (drives Day 6).
"""

import dataclasses
import json
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag_sec.dataset import load_t2_ragbench
from rag_sec.edgar import download_filing, find_10k_accession
from rag_sec.parsing import TableBlock, find_item_boundaries, parse_filing

N_SAMPLES = 100
SEED = 42

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
FILINGS_DIR = DATA_DIR / "filings"
PARSED_DIR = DATA_DIR / "parsed"
LOG_PATH = DATA_DIR / "day2_ingest_log.jsonl"


def sample_targets() -> list[dict]:
    df = load_t2_ragbench("all")
    df = df.dropna(subset=["company_cik", "report_year"])
    unique = df.drop_duplicates(subset=["company_cik", "report_year"])[
        ["company_cik", "report_year", "company_symbol", "subset_source"]
    ]
    # stratify roughly proportionally across the three subsets
    per_subset = max(1, N_SAMPLES // unique["subset_source"].nunique())
    sampled = (
        unique.groupby("subset_source", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_subset), random_state=SEED))
    )
    return sampled.to_dict("records")


def log_event(event: dict) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def process_one(target: dict) -> None:
    cik = int(target["company_cik"])
    year = int(target["report_year"])
    symbol = target["company_symbol"]
    stem = f"{symbol}_{year}_{cik}"

    t0 = time.monotonic()
    filing = find_10k_accession(cik, year)
    if filing is None:
        log_event({"stem": stem, "cik": cik, "year": year, "status": "no_10k_found"})
        return

    raw_path = FILINGS_DIR / f"{stem}{Path(filing['primary_document']).suffix}"
    if not raw_path.exists():
        content = download_filing(cik, filing["accession_number"], filing["primary_document"])
        raw_path.write_bytes(content)
    fetch_s = time.monotonic() - t0

    t1 = time.monotonic()
    try:
        html = raw_path.read_text(encoding="utf-8", errors="replace")
        blocks = parse_filing(html)
        items = find_item_boundaries(blocks)
        parse_s = time.monotonic() - t1
    except Exception as e:
        log_event(
            {
                "stem": stem,
                "cik": cik,
                "year": year,
                "status": "parse_failed",
                "fetch_s": round(fetch_s, 2),
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        return

    parsed_path = PARSED_DIR / f"{stem}.json"
    parsed_path.write_text(
        json.dumps([dataclasses.asdict(b) for b in blocks], ensure_ascii=False)
    )

    log_event(
        {
            "stem": stem,
            "cik": cik,
            "year": year,
            "status": "ok",
            "raw_bytes": raw_path.stat().st_size,
            "fetch_s": round(fetch_s, 2),
            "parse_s": round(parse_s, 2),
            "n_blocks": len(blocks),
            "n_tables": sum(1 for b in blocks if isinstance(b, TableBlock)),
            "n_items_found": len(items),
        }
    )


def main() -> None:
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    targets = sample_targets()
    print(f"Sampled {len(targets)} (cik, year) targets")

    for i, target in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {target['company_symbol']} {int(target['report_year'])}")
        try:
            process_one(target)
        except Exception as e:
            log_event(
                {
                    "cik": int(target["company_cik"]),
                    "year": int(target["report_year"]),
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

    print(f"Done. Log at {LOG_PATH}")


if __name__ == "__main__":
    main()
