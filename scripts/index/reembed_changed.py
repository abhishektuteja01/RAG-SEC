"""Re-embed only the chunks whose text changed (RETR-7/RETR-8, INFRA-12).

`embed_local.py`/`embed_prepare.py` are INSERT-only and skip any filing_stem already
present, so neither can apply a change to an existing corpus. This can, because RETR-7 was
built to preserve chunk boundaries: (filing_stem, chunk_index, variant='A') still identifies
the same body text, so a changed chunk is an UPDATE of that row, not a delete-and-reinsert.
Scope: 47,312 of 99,654 chunks (47.5%); the other 52.5% keep the embedding they have.

Same three-stage split as embed_prepare/embed_hpc/embed_load -- GPU work happens on HPC and
needs no DB access, so a dropped connection there cannot corrupt anything (ARM3-2). Stage 2
is `embed_hpc.py` **unchanged**: its payload format is already {filing_stem, chunk_index,
text} and its output is keyed the same way.

    python scripts/index/reembed_changed.py --prepare data/retr7_embed_payload.json
    # -> HPC:  python embed_hpc.py retr7_embed_payload.json retr7_embed_results.jsonl
    python scripts/index/reembed_changed.py --load data/retr7_embed_results.jsonl

Which chunks need work is decided by comparing the NEW data/chunks/ against the PRE-FIX
chunks (--baseline-chunks, the backup taken before the re-chunk), never against current DB
state. That is deliberate: it makes the set reproducible, and it re-embeds any row a partial
earlier run already touched, so every vector in the corpus comes from the same GPU pass.

Only variant 'A' is touched. B/C are a dropped experiment (ARM4-*) and stay on their
pre-RETR-7 headings.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag_sec.store import BM25_INDEX_SQL, HNSW_INDEX_SQL, get_conn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_DIR = ROOT / "data" / "chunks"


def changed_chunks(baseline_dir: Path) -> list[dict]:
    """Chunks whose text differs from the pre-fix build. Refuses if a filing's chunk COUNT
    moved -- that would mean boundaries shifted, which this cannot express as an UPDATE and
    which would renumber every gold label."""
    out, mismatched = [], []
    for path in sorted(CHUNKS_DIR.glob("*.json")):
        base_path = baseline_dir / path.name
        if not base_path.exists():
            mismatched.append((path.stem, "missing from baseline", ""))
            continue
        new, old = json.loads(path.read_text()), json.loads(base_path.read_text())
        if len(new) != len(old):
            mismatched.append((path.stem, len(old), len(new)))
            continue
        for idx, (n, o) in enumerate(zip(new, old)):
            if n["text"] != o["text"]:
                out.append(
                    {"filing_stem": path.stem, "chunk_index": idx, "text": n["text"],
                     "heading": n["heading"], "n_tokens": n["n_tokens"]}
                )
    if mismatched:
        print(f"REFUSED {len(mismatched)} filings whose chunk count changed:")
        for row in mismatched[:5]:
            print(f"  {row[0]}: {row[1]} -> {row[2]}")
        sys.exit(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", metavar="PAYLOAD", help="stage 1: write the HPC payload")
    g.add_argument("--load", metavar="RESULTS", help="stage 3: apply embeddings from HPC")
    ap.add_argument("--baseline-chunks", default=str(ROOT / "data" / "chunks_pre_retr7"),
                    help="pre-fix data/chunks/ (from the backup taken before --rechunk)")
    ap.add_argument("--keep-indexes", action="store_true",
                    help="do not drop/rebuild HNSW+BM25 around the load (much slower)")
    args = ap.parse_args()

    baseline = Path(args.baseline_chunks)
    if not baseline.is_dir():
        sys.exit(f"baseline chunks not found: {baseline}\n"
                 "Extract the pre-re-chunk backup there, e.g.\n"
                 "  mkdir -p data/chunks_pre_retr7 && tar -xzf ~/rag-sec-backups/chunks_json_pre_retr7_*.tgz "
                 "-C data/chunks_pre_retr7 --strip-components=2")

    todo = changed_chunks(baseline)

    if args.prepare:
        payload = [{"filing_stem": c["filing_stem"], "chunk_index": c["chunk_index"], "text": c["text"]}
                   for c in todo]
        Path(args.prepare).write_text(json.dumps(payload))
        print(f"{len(payload)} changed chunks -> {args.prepare}")
        print("Next (HPC):  python embed_hpc.py <payload> <results.jsonl>")
        return

    # stage 3: apply
    meta = {(c["filing_stem"], c["chunk_index"]): c for c in todo}
    embs, unknown = {}, 0
    with open(args.load) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["filing_stem"], row["chunk_index"])
            if key not in meta:
                unknown += 1
                continue
            embs[key] = row["embedding"]

    missing = len(meta) - len(embs)
    print(f"changed chunks: {len(meta)} | embeddings present: {len(embs)} | missing: {missing} | not-needed rows in file: {unknown}")
    if missing:
        sys.exit(f"refusing to load: {missing} changed chunks have no embedding. "
                 "Re-run embed_hpc.py against the same payload (it checkpoints and resumes).")

    t0 = time.monotonic()
    with get_conn(check=False) as conn:  # this script rewrites rows, it does not move counts
        if not args.keep_indexes:
            print("dropping HNSW + BM25 indexes for the bulk update...")
            conn.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw")
            conn.execute("DROP INDEX IF EXISTS chunks_bm25_idx")
            conn.commit()
        n = 0
        for (stem, idx), emb in embs.items():
            c = meta[(stem, idx)]
            conn.execute(
                "UPDATE chunks SET text = %s, heading = %s, n_tokens = %s, embedding = %s"
                " WHERE filing_stem = %s AND chunk_index = %s AND variant = 'A'",
                (c["text"], c["heading"], c["n_tokens"], emb, stem, idx),
            )
            n += 1
            if n % 5000 == 0:
                conn.commit()
                print(f"  ...{n}/{len(embs)} rows", flush=True)
        conn.commit()
        if not args.keep_indexes:
            print("rebuilding HNSW + BM25 indexes...")
            conn.execute(HNSW_INDEX_SQL)
            conn.execute(BM25_INDEX_SQL)
            conn.commit()

    print(f"\nrows updated: {n}\nelapsed     : {time.monotonic() - t0:.0f}s")
    print("Now re-run: scripts/checks/candidate_sql.py, scripts/checks/variant_predicates.py")


if __name__ == "__main__":
    main()
