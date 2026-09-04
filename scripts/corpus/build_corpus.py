"""Build the corpus: fetch every T2-RAGBench filing from EDGAR, parse it, chunk it.

Replaces day2_ingest.py + day4_ingest_next200.py + day2_chunk.py, which existed only
because the corpus was grown in stages (DECISIONS.md DATA-6) and each stage got its own
sampled script. There is nothing to sample: 799 is the exact count of unique
(cik, report_year) pairs across FinQA+ConvFinQA, i.e. the whole pool, so this takes all
of it. No seed, no N_SAMPLES -- the two knobs that let DATA-6's accidental duplicate run
happen, and that made the documented rebuild stop at ~300 filings.

Every stage skips its own output, so one run reaches 799 and re-running is a no-op.
  --rechunk   re-chunk from data/parsed/ (after a chunking.py change)
  --reparse   re-parse from data/filings/ (after a parsing.py change); implies --rechunk
  --limit N   first N filings in stem order, for a smoke test on a fresh machine
"""

import argparse
import dataclasses
import json
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag_sec.chunking import chunk_blocks
from rag_sec.dataset import load_t2_ragbench
from rag_sec.edgar import download_filing, find_10k_accession
from rag_sec.parsing import (
    TableBlock,
    find_item_boundaries,
    load_parsed_blocks,
    parse_filing,
)

EXPECTED_FILINGS = 799  # DATA-6, confirmed by direct count

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
FILINGS_DIR = DATA_DIR / "filings"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
LOG_PATH = DATA_DIR / "ingest_log.jsonl"


def load_targets() -> list[dict]:
    """The full pool. Same expression the staged scripts used -- TAT-DQA rows carry no
    company_cik/report_year and drop out here, which is what leaves FinQA+ConvFinQA's 799."""
    df = load_t2_ragbench("all")
    df = df.dropna(subset=["company_cik", "report_year"])
    unique = df.drop_duplicates(subset=["company_cik", "report_year"])[
        ["company_cik", "report_year", "company_symbol"]
    ]
    targets = [
        {"cik": int(r["company_cik"]), "year": int(r["report_year"]), "symbol": r["company_symbol"]}
        for r in unique.to_dict("records")
    ]
    for t in targets:
        t["stem"] = f"{t['symbol']}_{t['year']}_{t['cik']}"
    return sorted(targets, key=lambda t: t["stem"])


def log_event(event: dict) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def process_one(target: dict, rechunk: bool, reparse: bool) -> str:
    stem, cik, year = target["stem"], target["cik"], target["year"]
    chunks_path = CHUNKS_DIR / f"{stem}.json"
    parsed_path = PARSED_DIR / f"{stem}.json"

    if chunks_path.exists() and not rechunk and not reparse:
        return "skipped"

    t0 = time.monotonic()
    raw_candidates = list(FILINGS_DIR.glob(f"{stem}.*"))
    if raw_candidates:
        raw_path = raw_candidates[0]
    else:
        filing = find_10k_accession(cik, year)
        if filing is None:
            log_event({"stem": stem, "cik": cik, "year": year, "status": "no_10k_found"})
            return "no_10k_found"
        raw_path = FILINGS_DIR / f"{stem}{Path(filing['primary_document']).suffix}"
        raw_path.write_bytes(download_filing(cik, filing["accession_number"], filing["primary_document"]))
    fetch_s = time.monotonic() - t0

    t1 = time.monotonic()
    try:
        if parsed_path.exists() and not reparse:
            blocks = load_parsed_blocks(parsed_path)
            n_items = None
        else:
            html = raw_path.read_text(encoding="utf-8", errors="replace")
            blocks = parse_filing(html)
            n_items = len(find_item_boundaries(blocks))
            parsed_path.write_text(json.dumps([dataclasses.asdict(b) for b in blocks], ensure_ascii=False))
        parse_s = time.monotonic() - t1
    except Exception as e:
        log_event(
            {
                "stem": stem, "cik": cik, "year": year, "status": "parse_failed",
                "fetch_s": round(fetch_s, 2), "error": str(e), "traceback": traceback.format_exc(),
            }
        )
        return "parse_failed"

    t2 = time.monotonic()
    chunks = chunk_blocks(blocks)
    chunk_s = time.monotonic() - t2
    chunks_path.write_text(json.dumps([dataclasses.asdict(c) for c in chunks], ensure_ascii=False))

    sizes = [c.n_tokens for c in chunks]
    log_event(
        {
            "stem": stem, "cik": cik, "year": year, "status": "ok",
            "raw_bytes": raw_path.stat().st_size,
            "fetch_s": round(fetch_s, 2), "parse_s": round(parse_s, 2), "chunk_s": round(chunk_s, 2),
            "n_blocks": len(blocks),
            "n_tables": sum(1 for b in blocks if isinstance(b, TableBlock)),
            "n_items_found": n_items,
            "n_chunks": len(chunks),
            "median_tokens": sorted(sizes)[len(sizes) // 2] if sizes else 0,
            "max_tokens": max(sizes) if sizes else 0,
        }
    )
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rechunk", action="store_true", help="re-chunk from data/parsed/")
    ap.add_argument("--reparse", action="store_true", help="re-parse from data/filings/ (implies --rechunk)")
    ap.add_argument("--limit", type=int, help="first N filings in stem order (smoke test)")
    args = ap.parse_args()

    for d in (FILINGS_DIR, PARSED_DIR, CHUNKS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    targets = load_targets()
    if len(targets) != EXPECTED_FILINGS:
        print(f"WARNING: pool is {len(targets)} filings, expected {EXPECTED_FILINGS} (DATA-6). "
              "The dataset changed -- every published corpus number is against 799.")
    if args.limit:
        targets = targets[: args.limit]
    print(f"{len(targets)} filings to process")

    counts: dict[str, int] = {}
    for i, target in enumerate(targets, 1):
        try:
            status = process_one(target, rechunk=args.rechunk, reparse=args.reparse)
        except Exception as e:
            log_event(
                {
                    "stem": target["stem"], "cik": target["cik"], "year": target["year"],
                    "status": "error", "error": str(e), "traceback": traceback.format_exc(),
                }
            )
            status = "error"
        counts[status] = counts.get(status, 0) + 1
        if status != "skipped":
            print(f"[{i}/{len(targets)}] {target['stem']} {status}")

    print(f"Done. {counts}")
    print(f"{len(list(CHUNKS_DIR.glob('*.json')))} filings chunked. Log at {LOG_PATH}")


if __name__ == "__main__":
    main()
