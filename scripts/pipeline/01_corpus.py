"""Pipeline phase 01 — build the corpus: EDGAR fetch -> parse -> chunk.

PRODUCES
    data/filings/            raw 10-K HTML, one file per (ticker, year, cik)
    data/parsed/             sec-parser block lists, one JSON per filing
    data/chunks/             ~900-word chunks, one JSON per filing
    data/ingest_log.jsonl    one append-only record per filing (timings, counts, errors)

READS
    T2-RAGBench via rag_sec.dataset.load_t2_ragbench("all") — for the target list only.
    EDGAR over the network (rag_sec.edgar). Needs a User-Agent; see .env / README.

DECISIONS.md ROWS THIS BACKS
    DATA-6    799 is the exact count of unique (cik, report_year) pairs across
              FinQA+ConvFinQA, i.e. the whole pool. There is nothing to sample.
    INFRA-8   this script replaced day2_ingest.py + day4_ingest_next200.py +
              day2_chunk.py, which only existed because the corpus was grown in stages
              and each stage got its own sampled script. The seed/N_SAMPLES knobs those
              carried are what let DATA-6's accidental duplicate run happen, and made the
              documented rebuild stop at ~300 filings.
    ARM4-2    stored chunks predate chunking.py's `standalone` field, so --rechunk
              rewrites all 799 files to add one key (content-identical).
    RETR-7/8  the two chunking flags that default ON; the corpus on disk must match them.

WHEN THIS ACTUALLY RAN (calendar dates, not "Day N" — a dayN_ filename says nothing
about when the work happened)
    2026-08-24 -> 08-26   first ingest + parse + chunk pass
    2026-08-27 -> 08-28   corpus grown to the full 799 filings
    2026-09-04            re-chunked under RETR-7/RETR-8 (`--rechunk`), which is what the
                          live data/chunks/ is. Note data/day2_*_log.jsonl and
                          data/day4_ingest_log.jsonl are the ORIGINAL staged scripts'
                          logs, kept for provenance; this script writes ingest_log.jsonl.

TRAPS
  * Every stage skips its own output, so one run reaches 799 and re-running is a no-op.
    That also means a chunking.py or parsing.py change is SILENTLY IGNORED unless you
    pass --rechunk / --reparse.
  * --rechunk over the live corpus overwrites data/chunks/. If you intend to re-embed
    afterwards, take the backup FIRST — phase 03's --load diffs the new chunks against
    that backup, not against the database (INFRA-12).
  * Re-chunking without re-embedding leaves Postgres holding vectors for the old text.
"""

import argparse
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

load_dotenv()

from rag_sec.chunking import chunk_blocks  # noqa: E402
from rag_sec.dataset import load_t2_ragbench  # noqa: E402
from rag_sec.edgar import download_filing, find_10k_accession  # noqa: E402
from rag_sec.parsing import (  # noqa: E402
    TableBlock,
    find_item_boundaries,
    load_parsed_blocks,
    parse_filing,
)

# ─── CONSTANTS ──────────────────────────────────────────────────────────────────
# EXPECTED_FILINGS is an assertion, not a limit: 799 is the measured size of the
# FinQA+ConvFinQA pool (DATA-6). Every published corpus number is against 799, so a
# different count means the upstream dataset moved and the numbers no longer compare.
EXPECTED_FILINGS = 799

DATA_DIR = _ROOT / "data"
FILINGS_DIR = DATA_DIR / "filings"      # raw HTML as downloaded, never rewritten
PARSED_DIR = DATA_DIR / "parsed"        # parse output, the input to --rechunk
CHUNKS_DIR = DATA_DIR / "chunks"        # what phase 03 embeds and phase 04 searches
LOG_PATH = DATA_DIR / "ingest_log.jsonl"  # append-only; one line per filing attempt

# T2-RAGBench subset selector. "all" then dropna on company_cik/report_year is what
# leaves FinQA+ConvFinQA's 799 — TAT-DQA rows carry neither column and drop out here
# (DATA-3). Selecting the two subsets by name instead would give the same set today but
# would silently diverge if a third CIK-bearing subset were added.
DATASET_SUBSET = "all"


def load_targets() -> list[dict]:
    """The full pool, sorted by stem so --limit N is a deterministic prefix."""
    df = load_t2_ragbench(DATASET_SUBSET)
    df = df.dropna(subset=["company_cik", "report_year"])
    unique = df.drop_duplicates(subset=["company_cik", "report_year"])[
        ["company_cik", "report_year", "company_symbol"]
    ]
    targets = [
        {"cik": int(r["company_cik"]), "year": int(r["report_year"]),
         "symbol": r["company_symbol"]}
        for r in unique.to_dict("records")
    ]
    for t in targets:
        t["stem"] = f"{t['symbol']}_{t['year']}_{t['cik']}"
    return sorted(targets, key=lambda t: t["stem"])


def log_event(event: dict) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def process_one(target: dict, rechunk: bool, reparse: bool) -> str:
    """Fetch -> parse -> chunk one filing. Returns the status written to the log."""
    stem, cik, year = target["stem"], target["cik"], target["year"]
    chunks_path = CHUNKS_DIR / f"{stem}.json"
    parsed_path = PARSED_DIR / f"{stem}.json"

    if chunks_path.exists() and not rechunk and not reparse:
        return "skipped"

    # ─── STEP 1: fetch the raw 10-K (skipped if already on disk) ────────────────
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
        raw_path.write_bytes(
            download_filing(cik, filing["accession_number"], filing["primary_document"])
        )
    fetch_s = time.monotonic() - t0

    # ─── STEP 2: parse into blocks (reused unless --reparse) ────────────────────
    t1 = time.monotonic()
    try:
        if parsed_path.exists() and not reparse:
            blocks = load_parsed_blocks(parsed_path)
            n_items = None
        else:
            html = raw_path.read_text(encoding="utf-8", errors="replace")
            blocks = parse_filing(html)
            n_items = len(find_item_boundaries(blocks))
            parsed_path.write_text(
                json.dumps([dataclasses.asdict(b) for b in blocks], ensure_ascii=False)
            )
        parse_s = time.monotonic() - t1
    except Exception as e:
        log_event(
            {
                "stem": stem, "cik": cik, "year": year, "status": "parse_failed",
                "fetch_s": round(fetch_s, 2), "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        return "parse_failed"

    # ─── STEP 3: chunk, write, and log the per-filing record ───────────────────
    t2 = time.monotonic()
    chunks = chunk_blocks(blocks)
    chunk_s = time.monotonic() - t2
    chunks_path.write_text(json.dumps([dataclasses.asdict(c) for c in chunks], ensure_ascii=False))

    sizes = [c.n_tokens for c in chunks]
    log_event(
        {
            "stem": stem, "cik": cik, "year": year, "status": "ok",
            "raw_bytes": raw_path.stat().st_size,
            "fetch_s": round(fetch_s, 2), "parse_s": round(parse_s, 2),
            "chunk_s": round(chunk_s, 2),
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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rechunk", action="store_true", help="re-chunk from data/parsed/")
    ap.add_argument("--reparse", action="store_true",
                    help="re-parse from data/filings/ (implies --rechunk)")
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
