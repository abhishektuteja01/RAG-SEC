"""Resolve `data/day7_gold_inds_matched_full.json`'s `gold_inds` pointers (e.g.
`table_6`, `text_1`) into literal evidence -- raw table row cells or a literal sentence --
by looking them up in the original FinQA/ConvFinQA source files (not T2-RAGBench's
flattened `context`/`pre_text`/`post_text`/`table` columns, which lose the list structure
`gold_inds` indexes into -- see DECISIONS.md GOLD-1).

One-time, offline. Output (`data/day7_gold_evidence_resolved.json`) is what `eval.py`
reads at scoring time, so it never has to load the ~100MB+ raw dataset files itself.
Raw FinQA/ConvFinQA files aren't committed (too large, third-party) -- fetch them from
the original repos into `RAW_DATA_DIR` (env override) before re-running.

`table_N` indexes the `table` field directly for both datasets (FinQA has one `table`
field; ConvFinQA also has `table_ori`, but `gold_inds` indexes the normalized `table`,
confirmed against a real row: JKHY/2009 `table_6` = the "net cash from operating
activities" row, present at that index in `table`, not `table_ori`, which carries an
extra merged header row and would be off-by-one).

`text_N` indexes the concatenation of `pre_text` then `post_text`, 0-indexed continuously
(confirmed: ADI/2009/page_49.pdf-1's `text_1` is exactly `pre_text[1]`).

ConvFinQA's `Double_*` ids need a `qa_0`/`qa_1` suffix (stored in `matched_source_id` as
`Double_.../page_N.pdf#qa_0`) to pick the right one of two top-level annotation blocks;
everything else (FinQA, ConvFinQA `Single_*`) has one `qa` block.
"""

import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", os.path.join(REPO_ROOT, "data", "raw"))

MATCHED_PATH = os.path.join(REPO_ROOT, "data", "day7_gold_inds_matched_full.json")
OUT_PATH = os.path.join(REPO_ROOT, "data", "day7_gold_evidence_resolved.json")

RAW_FILES = [
    os.path.join(RAW_DATA_DIR, "finqa_train.json"),
    os.path.join(RAW_DATA_DIR, "finqa_dev.json"),
    os.path.join(RAW_DATA_DIR, "finqa_test.json"),
    os.path.join(RAW_DATA_DIR, "convfinqa_data", "data", "train.json"),
    os.path.join(RAW_DATA_DIR, "convfinqa_data", "data", "dev.json"),
]


def load_raw_index() -> dict:
    """id -> raw entry, across every fetched FinQA/ConvFinQA source file."""
    index = {}
    for path in RAW_FILES:
        if not os.path.exists(path):
            print(f"WARNING: missing {path}, skipping (some ids may fail to resolve)")
            continue
        with open(path) as f:
            entries = json.load(f)
        for e in entries:
            index[e["id"]] = e
        print(f"loaded {len(entries)} entries from {os.path.basename(path)}")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="write even if this run resolves fewer ids than the file on disk")
    args = ap.parse_args()

    # Two hard-fails, because the failure mode here is silent and destructive rather than
    # loud: with RAW_DATA_DIR absent, every lookup misses, `resolved` comes out empty, and
    # the old code wrote `{}` straight over the evidence eval.py scores against. Neither
    # guard is a size heuristic -- the first says nothing *can* resolve, the second says
    # this run resolved strictly less than the file already on disk.
    present = [p for p in RAW_FILES if os.path.exists(p)]
    if not present:
        raise SystemExit(
            f"no raw FinQA/ConvFinQA files under {RAW_DATA_DIR} -- every id would fail to\n"
            f"resolve and {os.path.basename(OUT_PATH)} would be overwritten with nothing.\n"
            "Fetch them from the original repos, or set RAW_DATA_DIR."
        )

    with open(MATCHED_PATH) as f:
        matched = json.load(f)
    print(f"resolving {len(matched)} matched ids")

    raw_index = load_raw_index()

    resolved = {}
    unresolved = 0
    for our_id, m in matched.items():
        r = resolve_one(m["matched_source_id"], m["gold_inds"], raw_index)
        if r is None:
            unresolved += 1
            continue
        resolved[our_id] = r

    print(f"resolved: {len(resolved)}/{len(matched)} ({len(resolved)/len(matched):.1%})")
    print(f"unresolved (base id missing from fetched files, or all indices out of range): {unresolved}")

    if os.path.exists(OUT_PATH) and not args.force:
        with open(OUT_PATH) as f:
            existing = json.load(f)
        if len(resolved) < len(existing):
            raise SystemExit(
                f"refusing to overwrite: {len(resolved)} resolved now against "
                f"{len(existing)} already in {os.path.basename(OUT_PATH)}. "
                f"{len(RAW_FILES) - len(present)} of {len(RAW_FILES)} raw files are missing. "
                "Pass --force if the shrink is intended."
            )

    with open(OUT_PATH, "w") as f:
        json.dump(resolved, f)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
