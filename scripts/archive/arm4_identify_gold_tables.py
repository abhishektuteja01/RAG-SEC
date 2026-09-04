"""ARCHIVED -- Day 6's gold-table identifier for Arm 4 (old filename in `git log --follow`).

What it did: worked out which specific raw table in `data/parsed/` answers each dev
question, by IDF-weighted shingle + numeric overlap applied table-vs-table, so the B and C
layouts only had to be built for the ~498 tables that actually matter instead of all 38,959.

Provenance: DECISIONS.md `ARM4-3` (the scoping decision) and `DATA-7`/`DATA-8`/`DATA-9` (the
matching method). Output: `data/day6_gold_tables.json` (208 KB) -- **still read by live
code**: `src/rag_sec/eval.py` and `scripts/corpus/build_table_variants.py`.

**This file holds the last surviving copy of the old gold-labelling method.** Chunk-level
relevance moved to `gold_inds` positional matching under `GOLD-1`; the IDF-weighted
shingle/numeric approach survives only in this script's inlined helpers, which were kept
deliberately (see the comment above `SHINGLE_N`) because table-vs-table matching is a much
lower-noise problem than question-vs-chunk. Deleting this script would erase that approach
from the project.

Safe to run today? Yes -- no database, no GPU, no money; it reads the dataset and
`data/parsed/`. It overwrites `data/day6_gold_tables.json`, which two live consumers read,
so re-run it deliberately rather than casually.

--- original header, kept verbatim ---

Day 6, Arm 4: identifies which of a filing's raw tables (data/parsed/*.json) is the
gold table for each dev question, using the same IDF-weighted shingle+numeric overlap
approach eval.py already uses at chunk granularity (DECISIONS.md DATA-7/DATA-8/DATA-9),
applied table-vs-table instead of question-vs-chunk. Reused rather than a verbatim/substring
match because our own table serialization differs from the dataset's markdown-pipe `table`
field (different cell splits/spacing) -- same problem DATA-8 already solved once at chunk
level.

Scopes Strategy B/C (build_table_variants.py) to only these gold tables instead of all
38,959 tables across the 324 dev-relevant filings -- see DECISIONS.md ARM4-3.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

from rag_sec.chunking import _table_to_text
from rag_sec.dataset import load_t2_ragbench

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PARSED_DIR = DATA_DIR / "parsed"
OUT_PATH = DATA_DIR / "day6_gold_tables.json"

# IDF-weighted shingle+numeric overlap primitives -- inlined rather than imported from
# rag_sec.eval, which moved to gold_inds-based positional labeling for chunk-level
# relevance (DECISIONS.md GOLD-1). Table-vs-table matching here is a different, much
# lower-noise problem (a known table against a handful of whole-table candidates in the
# same filing, not a whole page against ~100-250 chunks), so the old overlap approach
# still holds up fine for it -- no reason to migrate this one too.
SHINGLE_N = 5
RELEVANCE_THRESHOLD = 0.3
NUMBER_RE = re.compile(r"\d+\.\d+")
NUMERIC_RELEVANCE_THRESHOLD = 0.5
MIN_NUMERIC_EVIDENCE = 3


def _shingles(text: str, n: int = SHINGLE_N) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _numbers(text: str) -> set[str]:
    return set(NUMBER_RE.findall(text))


def _idf(token, df: Counter, n_chunks: int) -> float:
    doc_freq = df.get(token, n_chunks)
    return math.log(1 + n_chunks / doc_freq)


def filing_tables(stem: str) -> list[list[list[str]]]:
    blocks = json.loads((PARSED_DIR / f"{stem}.json").read_text())
    return [b["rows"] for b in blocks if "rows" in b]


def gold_table_matches(gold_table_text: str, tables: list[list[list[str]]]) -> list[dict]:
    """Returns every table clearing the relevance threshold (usually one, occasionally
    two if a question's evidence spans adjacent tables) -- mirrors gold_relevant_chunk_evidence.
    """
    gold_sh = _shingles(gold_table_text)
    gold_nums = _numbers(gold_table_text)
    if not gold_sh or not tables:
        return []

    table_texts = [_table_to_text(rows) for rows in tables]
    n = len(tables)

    sh_sets = [_shingles(t) for t in table_texts]
    sh_df = Counter()
    for s in sh_sets:
        sh_df.update(s)
    gold_sh_weight = {t: _idf(t, sh_df, n) for t in gold_sh}
    gold_sh_total = sum(gold_sh_weight.values()) or 1.0

    use_numeric = len(gold_nums) >= MIN_NUMERIC_EVIDENCE
    if use_numeric:
        num_sets = [_numbers(t) for t in table_texts]
        num_df = Counter()
        for s in num_sets:
            num_df.update(s)
        gold_num_weight = {t: _idf(t, num_df, n) for t in gold_nums}
        gold_num_total = sum(gold_num_weight.values()) or 1.0

    matches = []
    for i in range(n):
        sh_score = sum(gold_sh_weight[t] for t in gold_sh & sh_sets[i]) / gold_sh_total
        num_score = 0.0
        if use_numeric:
            num_score = sum(gold_num_weight[t] for t in gold_nums & num_sets[i]) / gold_num_total
        if sh_score >= RELEVANCE_THRESHOLD or num_score >= NUMERIC_RELEVANCE_THRESHOLD:
            matches.append({"table_index": i, "sh_score": sh_score, "num_score": num_score})
    return matches


def main() -> None:
    df = load_t2_ragbench("all")
    df = df[df["subset_source"].isin(["FinQA", "ConvFinQA"])]
    dev = df[df["split"] == "dev"]
    dev = dev[dev["table"].str.strip() != ""].reset_index(drop=True)

    results = []
    unmatched = 0
    table_cache: dict[str, list[list[list[str]]]] = {}
    for _, row in dev.iterrows():
        stem = f"{row['company_symbol']}_{int(row['report_year'])}_{int(row['company_cik'])}"
        if stem not in table_cache:
            table_cache[stem] = filing_tables(stem)
        matches = gold_table_matches(row["table"], table_cache[stem])
        if not matches:
            unmatched += 1
            continue
        for m in matches:
            results.append({"question_id": row["id"], "filing_stem": stem, **m})

    print(f"{len(dev) - unmatched}/{len(dev)} dev questions matched >=1 gold table ({unmatched} unmatched)")
    unique_tables = {(r["filing_stem"], r["table_index"]) for r in results}
    print(f"{len(unique_tables)} unique (filing, table_index) targets across {len(table_cache)} filings")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
