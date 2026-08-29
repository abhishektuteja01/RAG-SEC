"""Day 4: fetch + parse + chunk the next 200 filings toward full 799-filing coverage.

Excludes (cik, year) pairs already in data/chunks/ (the 100 from Day 2), then
samples 200 more from the remaining pool, same stratified-by-subset approach as
day2_ingest.py. Each stage (fetch/parse/chunk) skips its output file if it already
exists, so an interrupted run resumes without redoing work. Run
scripts/index/day3_index_chunks.py afterward to embed the new chunks (it already
skips filing_stems already in Postgres).
"""

import dataclasses
import json
import time
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from rag_sec.chunking import chunk_blocks
from rag_sec.dataset import load_t2_ragbench
from rag_sec.edgar import download_filing, find_10k_accession
from rag_sec.parsing import TableBlock, TextBlock, find_item_boundaries, parse_filing

N_SAMPLES = 250  # above the ~196 remaining toward 799 -- sample_targets() then takes everything
# left instead of a partial draw (DECISIONS.md DATA-6)
SEED = 45  # different from prior runs (42/43/44) so this draws a fresh sample

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
FILINGS_DIR = DATA_DIR / "filings"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
LOG_PATH = DATA_DIR / "day4_ingest_log.jsonl"


def already_done() -> set[tuple[int, int]]:
    done = set()
    for path in CHUNKS_DIR.glob("*.json"):
        _, year, cik = path.stem.rsplit("_", 2)
        done.add((int(cik), int(year)))
    return done


def sample_targets() -> list[dict]:
    df = load_t2_ragbench("all")
    df = df.dropna(subset=["company_cik", "report_year"])
    unique = df.drop_duplicates(subset=["company_cik", "report_year"])[
        ["company_cik", "report_year", "company_symbol", "subset_source"]
    ]
    done = already_done()
    unique = unique[
        ~unique.apply(lambda r: (int(r["company_cik"]), int(r["report_year"])) in done, axis=1)
    ]
    print(f"{len(done)} pairs already done, {len(unique)} remaining in pool")

    per_subset = max(1, N_SAMPLES // unique["subset_source"].nunique())
    parts = [
        g.sample(min(len(g), per_subset), random_state=SEED)
        for _, g in unique.groupby("subset_source")
    ]
    sampled = pd.concat(parts)

    # top off shortfall (e.g. a subset's pool is smaller than its even share)
    # from whichever rows weren't already picked
    shortfall = N_SAMPLES - len(sampled)
    if shortfall > 0:
        leftover = unique.drop(sampled.index)
        sampled = pd.concat([sampled, leftover.sample(min(shortfall, len(leftover)), random_state=SEED)])

    return sampled.to_dict("records")


def log_event(event: dict) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def process_one(target: dict) -> None:
    cik = int(target["company_cik"])
    year = int(target["report_year"])
    symbol = target["company_symbol"]
    stem = f"{symbol}_{year}_{cik}"

    chunks_path = CHUNKS_DIR / f"{stem}.json"
    if chunks_path.exists():
        return

    t0 = time.monotonic()
    raw_candidates = list(FILINGS_DIR.glob(f"{stem}.*"))
    if raw_candidates:
        raw_path = raw_candidates[0]
    else:
        filing = find_10k_accession(cik, year)
        if filing is None:
            log_event({"stem": stem, "cik": cik, "year": year, "status": "no_10k_found"})
            return
        raw_path = FILINGS_DIR / f"{stem}{Path(filing['primary_document']).suffix}"
        content = download_filing(cik, filing["accession_number"], filing["primary_document"])
        raw_path.write_bytes(content)
    fetch_s = time.monotonic() - t0

    t1 = time.monotonic()
    parsed_path = PARSED_DIR / f"{stem}.json"
    try:
        if parsed_path.exists():
            data = json.loads(parsed_path.read_text())
            blocks = [
                TableBlock(rows=d["rows"]) if "rows" in d else TextBlock(text=d["text"], is_title=d["is_title"])
                for d in data
            ]
        else:
            html = raw_path.read_text(encoding="utf-8", errors="replace")
            blocks = parse_filing(html)
            find_item_boundaries(blocks)  # side-effect-free; matches day2_ingest's stats logging
            parsed_path.write_text(json.dumps([dataclasses.asdict(b) for b in blocks], ensure_ascii=False))
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

    t2 = time.monotonic()
    chunks = chunk_blocks(blocks)
    chunk_s = time.monotonic() - t2
    chunks_path.write_text(json.dumps([dataclasses.asdict(c) for c in chunks], ensure_ascii=False))

    sizes = [c.n_tokens for c in chunks]
    log_event(
        {
            "stem": stem,
            "cik": cik,
            "year": year,
            "status": "ok",
            "fetch_s": round(fetch_s, 2),
            "parse_s": round(parse_s, 2),
            "chunk_s": round(chunk_s, 2),
            "n_chunks": len(chunks),
            "median_tokens": sorted(sizes)[len(sizes) // 2] if sizes else 0,
            "max_tokens": max(sizes) if sizes else 0,
        }
    )


def main() -> None:
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

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
