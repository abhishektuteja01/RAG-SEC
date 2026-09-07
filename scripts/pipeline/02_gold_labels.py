"""Pipeline phase 02 — gold labels: the ground truth every metric is scored against.

*** READ THIS FIRST: THE `evidence` LEG CANNOT RUN END TO END TODAY (DECISIONS.md
*** INFRA-15). Its upstream input, `data/day7_gold_inds_matched_full.json`, has NO
*** PRODUCER in this repo — the ticker/year/page join that made it (GOLD-7) was never
*** committed — and the raw datasets it would need (`data/raw`, `data/FinQA`,
*** `data/ConvFinQA`, `data/TAT-DQA`) are ALL ABSENT from this machine. That 4.8 MB file
*** is therefore FROZEN, not rebuildable: it and its output are tracked in git and that
*** is the only backup. Do NOT write a new producer for it and do NOT delete either
*** file — re-deriving the labels would move them, and labels that move are not the
*** labels the published numbers were measured against (RETR-35).
*** `evidence` still exists here because it is the documented, auditable step between the
*** frozen file and what eval.py reads. Run with the raw datasets restored, or not at all.

TWO LEGS, two independent outputs, no shared state:

    evidence   data/day7_gold_evidence_resolved.json   (2.4 MB, chunk-level labels)
               Turns `gold_inds` pointers (`table_6`, `text_1`) into literal evidence —
               raw table row cells, or a literal sentence — by looking them up in the
               ORIGINAL FinQA/ConvFinQA files. Not T2-RAGBench's flattened
               context/pre_text/post_text/table columns, which lose the list structure
               `gold_inds` indexes into (GOLD-1). Read by src/rag_sec/eval.py at scoring
               time, so eval never loads the ~100 MB+ raw files itself.
               Reads: data/day7_gold_inds_matched_full.json + $RAW_DATA_DIR/*.

    tables     data/day6_gold_tables.json               (208 KB, table-level labels)
               Which specific raw table in data/parsed/ answers each dev question, by
               IDF-weighted shingle + numeric overlap applied table-vs-table. Scoped
               Arm 4's B/C layouts to the ~498 tables that matter instead of all 38,959.
               Reads: T2-RAGBench + data/parsed/. No DB, no GPU, no money — it CAN run.
               Still read by live code: src/rag_sec/eval.py and
               scripts/pipeline/06_arm4_tables.py (`variants`), so re-run it deliberately.

DECISIONS.md ROWS THIS BACKS
    INFRA-15  the frozen-input finding above. The reason this phase is half-runnable.
    GOLD-1    gold_inds positional labelling replaced the IDF shingle/numeric labeler for
              CHUNK-level relevance.
    GOLD-7    the join that produced day7_gold_inds_matched_full.json was never saved.
    INFRA-9   the two data-loss guards on the `evidence` leg (see STEP 1 there).
    ARM4-3    scoping B/C to gold tables only — why the `tables` leg exists.
    DATA-7/8/9  the IDF-weighted shingle + numeric overlap matching method itself.

WHEN THIS ACTUALLY RAN
    2026-08-29 00:44   data/day6_gold_tables.json written (`tables` leg).
    2026-08-30 17:31   data/day7_gold_inds_matched_full.json — the frozen input, by the
                       uncommitted GOLD-7 join, NOT by this script.
    2026-08-30 18:53   data/day7_gold_evidence_resolved.json written (`evidence` leg).
    Both committed 2026-08-30. There is no calendar "Day 6" or "Day 7" — the dayN_
    prefixes are plan numbers and the `day6_` file predates the `day7_` ones by a day.

TRAPS
  * The `tables` leg OVERWRITES data/day6_gold_tables.json, which two live consumers read.
  * The IDF/shingle helpers below are the LAST SURVIVING COPY of the old gold-labelling
    method. They are deliberately inlined rather than imported from rag_sec.eval, which
    moved to gold_inds positional matching (GOLD-1). Table-vs-table is a much lower-noise
    problem than question-vs-chunk (a known table against a handful of whole-table
    candidates in one filing, not a whole page against ~100-250 chunks), so the old
    approach still holds up for it. Deleting them erases that approach from the project.
  * `table_N` indexes the `table` field directly for both datasets. ConvFinQA also has
    `table_ori`, but `gold_inds` indexes the normalized `table` — confirmed against a real
    row: JKHY/2009 `table_6` is the "net cash from operating activities" row, at that index
    in `table`, not in `table_ori`, which carries an extra merged header and is off-by-one.
  * `text_N` indexes `pre_text` then `post_text` concatenated, 0-indexed continuously
    (confirmed: ADI/2009/page_49.pdf-1's `text_1` is exactly `pre_text[1]`).
  * ConvFinQA `Double_*` ids need a `qa_0`/`qa_1` suffix (carried in `matched_source_id` as
    `Double_.../page_N.pdf#qa_0`) to pick the right one of two annotation blocks.
    Everything else (FinQA, ConvFinQA `Single_*`) has a single `qa` block.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec.chunking import _table_to_text  # noqa: E402
from rag_sec.dataset import load_t2_ragbench  # noqa: E402

# ─── CONSTANTS ──────────────────────────────────────────────────────────────────
DATA_DIR = _ROOT / "data"
PARSED_DIR = DATA_DIR / "parsed"

# The frozen input. Env-overridable only so a restored copy can be pointed at without
# editing this file; there is no producer for it (INFRA-15).
MATCHED_PATH = Path(os.environ.get("GOLD_MATCHED_PATH",
                                   DATA_DIR / "day7_gold_inds_matched_full.json"))
EVIDENCE_OUT = DATA_DIR / "day7_gold_evidence_resolved.json"  # what eval.py reads
TABLES_OUT = DATA_DIR / "day6_gold_tables.json"  # read by eval.py + build_table_variants

# Raw FinQA/ConvFinQA sources. Not committed (too large, third-party); fetch from the
# original repos. RAW_DATA_DIR is the documented override and defaults to a directory
# that does NOT exist on this machine — that is the INFRA-15 condition, not a typo.
RAW_DATA_DIR = Path(os.environ.get("RAW_DATA_DIR", DATA_DIR / "raw"))
RAW_FILES = [
    RAW_DATA_DIR / "finqa_train.json",
    RAW_DATA_DIR / "finqa_dev.json",
    RAW_DATA_DIR / "finqa_test.json",
    RAW_DATA_DIR / "convfinqa_data" / "data" / "train.json",
    RAW_DATA_DIR / "convfinqa_data" / "data" / "dev.json",
]

# IDF-weighted shingle + numeric overlap thresholds (DATA-7/DATA-8/DATA-9). These are the
# ORIGINAL values the shipped data/day6_gold_tables.json was produced with — changing any
# of them changes the labels, so they are frozen with that artifact, not tuned.
SHINGLE_N = 5                     # 5-word shingles: long enough that a shared financial
                                  # boilerplate phrase does not match on its own
RELEVANCE_THRESHOLD = 0.3         # fraction of the gold table's IDF-weighted shingle mass
                                  # a candidate table must recover
NUMERIC_RELEVANCE_THRESHOLD = 0.5 # higher bar for the numeric channel: decimal figures are
                                  # far more discriminative, so a weak match means nothing
MIN_NUMERIC_EVIDENCE = 3          # below 3 decimals the numeric channel is noise, so it is
                                  # switched off rather than trusted
NUMBER_RE = re.compile(r"\d+\.\d+")  # decimals only — bare integers (years, counts) match
                                     # across unrelated tables constantly


# ─── shared: nothing. The two legs are independent by design (see docstring). ───


def _shingles(text: str, n: int = SHINGLE_N) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _numbers(text: str) -> set[str]:
    return set(NUMBER_RE.findall(text))


def _idf(token, df: Counter, n_chunks: int) -> float:
    doc_freq = df.get(token, n_chunks)
    return math.log(1 + n_chunks / doc_freq)


# ═══ LEG A: `evidence` ══════════════════════════════════════════════════════════


def load_raw_index() -> dict:
    """id -> raw entry, across every fetched FinQA/ConvFinQA source file."""
    index = {}
    for path in RAW_FILES:
        if not path.exists():
            print(f"WARNING: missing {path}, skipping (some ids may fail to resolve)")
            continue
        entries = json.loads(path.read_text())
        for e in entries:
            index[e["id"]] = e
        print(f"loaded {len(entries)} entries from {path.name}")
    return index


def resolve_one(matched_source_id: str, gold_inds: dict, raw_index: dict) -> dict | None:
    if "#" in matched_source_id:
        base_id, qa_key = matched_source_id.split("#", 1)
    else:
        base_id, qa_key = matched_source_id, "qa"

    entry = raw_index.get(base_id)
    if entry is None:
        return None

    table = entry.get("table")
    pre_text = entry.get("pre_text", [])
    post_text = entry.get("post_text", [])
    combined_text = list(pre_text) + list(post_text)

    table_rows, sentences = [], []
    for key in gold_inds:
        if key.startswith("table_"):
            idx = int(key.split("_", 1)[1])
            if table is not None and 0 <= idx < len(table):
                table_rows.append(table[idx])
        elif key.startswith("text_"):
            idx = int(key.split("_", 1)[1])
            if 0 <= idx < len(combined_text):
                sentences.append(combined_text[idx])

    if not table_rows and not sentences:
        return None
    return {"table_rows": table_rows, "sentences": sentences}


def run_evidence(force: bool) -> None:
    # ─── STEP 1: refuse to run into a silent total data loss (INFRA-9) ─────────
    # Two hard-fails, because the failure mode is silent rather than loud: with
    # RAW_DATA_DIR absent every lookup misses, `resolved` comes out {}, and the old code
    # wrote that straight over the 2.4 MB of gold evidence eval.py scores against.
    # Neither guard is a size heuristic — the first says nothing *can* resolve, the
    # second says this run resolved strictly less than the file already on disk.
    present = [p for p in RAW_FILES if p.exists()]
    if not present:
        raise SystemExit(
            "CANNOT RUN: no raw FinQA/ConvFinQA source files under\n"
            f"  {RAW_DATA_DIR}\n"
            "Every id would fail to resolve and\n"
            f"  {EVIDENCE_OUT.name}\n"
            "would be overwritten with nothing. This is the expected state on a fresh\n"
            "clone -- see DECISIONS.md INFRA-15: the raw datasets are not committed and\n"
            "this leg is NOT rebuildable here. Fetch them from the original FinQA and\n"
            "ConvFinQA repos, or set RAW_DATA_DIR, then re-run.\n"
            "The existing output is tracked in git; leave it alone."
        )
    if not MATCHED_PATH.exists():
        raise SystemExit(
            f"CANNOT RUN: missing {MATCHED_PATH}\n"
            "This file has NO PRODUCER in this repo (DECISIONS.md INFRA-15 / GOLD-7) --\n"
            "the ticker/year/page join that made it was never committed. It is FROZEN\n"
            "data, restorable only from git. Do not write a new producer for it: fresh\n"
            "labels are not the labels the published numbers were measured against."
        )

    # ─── STEP 2: resolve every pointer against the raw datasets ────────────────
    matched = json.loads(MATCHED_PATH.read_text())
    print(f"resolving {len(matched)} matched ids")
    raw_index = load_raw_index()

    resolved, unresolved = {}, 0
    for our_id, m in matched.items():
        r = resolve_one(m["matched_source_id"], m["gold_inds"], raw_index)
        if r is None:
            unresolved += 1
            continue
        resolved[our_id] = r

    print(f"resolved: {len(resolved)}/{len(matched)} ({len(resolved) / len(matched):.1%})")
    print("unresolved (base id missing from fetched files, or all indices out of range): "
          f"{unresolved}")

    # ─── STEP 3: second guard, then write ──────────────────────────────────────
    if EVIDENCE_OUT.exists() and not force:
        existing = json.loads(EVIDENCE_OUT.read_text())
        if len(resolved) < len(existing):
            raise SystemExit(
                f"refusing to overwrite: {len(resolved)} resolved now against "
                f"{len(existing)} already in {EVIDENCE_OUT.name}. "
                f"{len(RAW_FILES) - len(present)} of {len(RAW_FILES)} raw files are "
                "missing. Pass --force if the shrink is intended."
            )

    EVIDENCE_OUT.write_text(json.dumps(resolved))
    print(f"wrote {EVIDENCE_OUT}")


# ═══ LEG B: `tables` ════════════════════════════════════════════════════════════


def filing_tables(stem: str) -> list[list[list[str]]]:
    blocks = json.loads((PARSED_DIR / f"{stem}.json").read_text())
    return [b["rows"] for b in blocks if "rows" in b]


def gold_table_matches(gold_table_text: str, tables: list[list[list[str]]]) -> list[dict]:
    """Every table clearing the relevance threshold — usually one, occasionally two if a
    question's evidence spans adjacent tables. Mirrors gold_relevant_chunk_evidence."""
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


def run_tables() -> None:
    # ─── STEP 1: the dev questions that have a table at all ────────────────────
    df = load_t2_ragbench("all")
    df = df[df["subset_source"].isin(["FinQA", "ConvFinQA"])]
    dev = df[df["split"] == "dev"]
    dev = dev[dev["table"].str.strip() != ""].reset_index(drop=True)

    # ─── STEP 2: match each question's gold table against its filing's tables ──
    results, unmatched = [], 0
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

    # ─── STEP 3: report and write ──────────────────────────────────────────────
    print(f"{len(dev) - unmatched}/{len(dev)} dev questions matched >=1 gold table "
          f"({unmatched} unmatched)")
    unique_tables = {(r["filing_stem"], r["table_index"]) for r in results}
    print(f"{len(unique_tables)} unique (filing, table_index) targets across "
          f"{len(table_cache)} filings")

    TABLES_OUT.write_text(json.dumps(results, indent=2))
    print(f"Written to {TABLES_OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="leg", required=True)

    p_ev = sub.add_parser(
        "evidence", help="resolve gold_inds pointers into literal evidence (NOT RUNNABLE "
                         "here -- INFRA-15)",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_ev.add_argument("--force", action="store_true",
                      help="write even if this run resolves fewer ids than the file on disk")

    sub.add_parser(
        "tables", help="identify each dev question's gold raw table (runnable: no DB, no GPU)",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    args = ap.parse_args()
    if args.leg == "evidence":
        run_evidence(force=args.force)
    else:
        run_tables()


if __name__ == "__main__":
    main()
