"""Pipeline phase 03 — index: embed data/chunks/ into pgvector, and build BM25 over it.

PRODUCES
    rows in Postgres `chunks` (filing_stem, chunk_index, heading, n_tokens, text,
    embedding), the HNSW vector index, and the pg_search BM25 index.
    On the cluster route it also produces/consumes an on-disk payload and results file.

READS
    data/chunks/ (phase 01), Postgres, and — on the cluster route — the .jsonl of
    embeddings that scripts/pipeline/hpc/embed_hpc.py wrote on the GPU node.

FOUR LEGS. BOTH ROUTES ARE KEPT ON PURPOSE — the local one is how a fresh machine gets a
corpus, the cluster one is how this project's real passes were actually run.

    local            single process, laptop/host: embed every not-yet-loaded filing with
                     BGE-M3, INSERT, then build HNSW. Measured ~1.4 chunks/s on this M3's
                     MPS (INFRA-13) — fine for a smoke test, ~9.3 h for a full pass.
    bm25             build the pg_search BM25 index over the text already loaded. No
                     re-embedding, no GPU.
    new-filings      three-stage cluster round trip for filings NOT YET in Postgres:
                       --prepare  (host) dump their chunk texts to a payload file
                       ->  cluster: python embed_hpc.py <payload> <results.jsonl>
                       --load     (host) join embeddings back with local text, INSERT,
                                  rebuild HNSW
    changed-chunks   three-stage round trip for chunks ALREADY in Postgres whose text
                     changed (RETR-7/RETR-8). Same --prepare / --load split; the middle
                     stage is the identical embed_hpc.py. This is an UPDATE, not an
                     INSERT, and it is the ONLY leg that can apply a change to an existing
                     corpus — `local` and `new-filings` both skip any filing_stem already
                     present, so neither can.

WHY THE SPLIT AT ALL (ARM3-2, INFRA-6): the GPU stage needs no database access, so a
dropped connection on the cluster cannot corrupt anything, and the cluster needs no
`rag_sec` install. Measured payoff: 11 chunks/s on the HPC V100 at BATCH_SIZE=64 vs 1.4
on local MPS, ~8x (INFRA-13).

DECISIONS.md ROWS THIS BACKS
    ARM3-2 / INFRA-6   why GPU work is split out to the cluster and why embed_hpc.py
                       imports nothing from rag_sec.
    INFRA-4 / ARM2-1   real BM25 via pg_search, not tsvector/ts_rank (spec.md's explicit
                       trap: no length normalization or term saturation).
    RETR-7 / RETR-8    the heading fix that made 47.5% of chunks need a new vector.
    INFRA-12           that re-index is INCREMENTAL, not a full pass: chunk count
                       identical at 99,654 across all 799 filings, 52,342 (52.5%) keep
                       their embedding, 47,312 (47.5%) need a new one. Possible only
                       because RETR-7 preserved chunk boundaries, so
                       (filing_stem, chunk_index, variant) still identifies the same body
                       text and a changed chunk is a keyed UPDATE.
    INFRA-13           the throughput figures quoted above.
    ARM4-3             only variant 'A' is live; B/C are a dropped experiment and are
                       never touched here.

WHEN THIS RAN: see the phase 03 row of scripts/README.md. The RETR-7/RETR-8 re-index was
HPC job 9949705 — all 47,312 chunks embedded in ~72 min. "day5_" in
data/day5_embed_payload.json is a plan number, not a date (written 08-28); the
`changed-chunks` artifacts are named retr7_* instead.

TRAPS
  * `local` and `new-filings` are INSERT-only and skip any filing_stem already present, so
    an interrupted run resumes safely — but a re-chunk is invisible to them. Use
    `changed-chunks` after a --rechunk, or delete the filing's rows by hand.
  * `changed-chunks` computes the work set by diffing data/chunks/ against a BACKUP of the
    pre-fix chunks (--baseline-chunks), NEVER against database state. That is deliberate:
    it makes the set reproducible, and it re-embeds any row a partial earlier run already
    touched, so every vector in the corpus comes from one GPU pass. The default baseline
    directory does NOT exist until you extract that backup (see the error message).
  * `changed-chunks` refuses outright if any filing's chunk COUNT moved: that means
    boundaries shifted, which cannot be expressed as an UPDATE and would renumber every
    gold label.
  * The BM25 index needs a full rebuild after any text change; that is Postgres work, no
    GPU. `changed-chunks --load` drops and rebuilds both indexes around the bulk UPDATE.
  * A first attempt at the RETR-7 re-embed ran locally on MPS and was aborted after 382 of
    47,312 rows (INFRA-12). Those rows are in the payload by construction — see above.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

load_dotenv()

from rag_sec.config import EMBED_MODEL_NAME, pick_device  # noqa: E402
from rag_sec.store import (  # noqa: E402
    BM25_INDEX_SQL,
    HNSW_INDEX_SQL,
    get_conn,
    init_schema,
)

# ─── CONSTANTS ──────────────────────────────────────────────────────────────────
DATA_DIR = _ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"

# Local embed batch. 32, not the cluster's 64: this runs on a 16 GiB M3 that must also
# hold Postgres and (during eval) the reranker on the same MPS device (INFRA-6).
EMBED_BATCH_SIZE = 32

# Default payload for the `new-filings` route. Kept at the original name so the artifact
# on disk (data/day5_embed_payload.json, 26,737 chunks, written 2026-08-28) still matches
# the default that produced it. The `changed-chunks` route has NO default payload path on
# purpose — RUNBOOK.md passes retr7_embed_payload.json explicitly, and defaulting it would
# invite overwriting one route's artifact from the other.
NEW_FILINGS_PAYLOAD = DATA_DIR / "day5_embed_payload.json"

# Pre-RETR-7 chunk backup that `changed-chunks` diffs against. Not created by any script:
# it is extracted from the tarball RUNBOOK.md takes before --rechunk.
BASELINE_CHUNKS_DIR = DATA_DIR / "chunks_pre_retr7"

# Commit every N rows during the bulk UPDATE. Large enough that commit overhead is
# negligible over ~47k rows, small enough that an interrupted run leaves most work done.
UPDATE_COMMIT_EVERY = 5000

INSERT_CHUNK_SQL = (
    "INSERT INTO chunks (filing_stem, chunk_index, heading, n_tokens, text, embedding)"
    " VALUES (%s, %s, %s, %s, %s, %s)"
)

# Variant-agnostic by design: "which filings are already loaded" is bookkeeping, not a
# candidate pool. Exempted by name in rag_sec.preflight.ALLOWED (RETR-24).
DONE_STEMS_SQL = "SELECT DISTINCT filing_stem FROM chunks"


# ─── shared helpers, used by more than one leg ──────────────────────────────────


def _done_stems(conn) -> set[str]:
    return {r[0] for r in conn.execute(DONE_STEMS_SQL).fetchall()}


def _insert_rows(conn, rows: list[tuple]) -> None:
    with conn.cursor() as cur:
        cur.executemany(INSERT_CHUNK_SQL, rows)
    conn.commit()


def _chunk_row(stem: str, idx: int, chunk: dict, embedding) -> tuple:
    """The one insert shape both INSERT legs use — identical column order by construction
    rather than by two hand-copied tuples, which is how they could have drifted."""
    return (stem, idx, chunk.get("heading"), chunk.get("n_tokens"), chunk["text"], embedding)


def _load_chunks(stem: str) -> list[dict]:
    return json.loads((CHUNKS_DIR / f"{stem}.json").read_text())


# ═══ LEG: local ═════════════════════════════════════════════════════════════════


def run_local() -> None:
    # ─── STEP 1: schema + model ────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer

    init_schema()
    model = SentenceTransformer(EMBED_MODEL_NAME, device=pick_device())

    paths = sorted(CHUNKS_DIR.glob("*.json"))
    print(f"Embedding chunks from {len(paths)} filings")

    # check=False on every leg here: the corpus assertion compares row counts, and these
    # are the scripts that move those counts.
    with get_conn(check=False) as conn:
        done_stems = _done_stems(conn)
        print(f"{len(done_stems)}/{len(paths)} filings already indexed, skipping those")

        # ─── STEP 2: embed and insert, filing by filing ───────────────────────
        for i, path in enumerate(paths, 1):
            stem = path.stem
            if stem in done_stems:
                continue
            chunks = json.loads(path.read_text())
            texts = [c["text"] for c in chunks]
            if not texts:
                continue
            embeddings = model.encode(
                texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False,
                normalize_embeddings=True,
            )
            rows = [
                _chunk_row(stem, idx, c, emb)
                for idx, (c, emb) in enumerate(zip(chunks, embeddings))
            ]
            _insert_rows(conn, rows)
            print(f"[{i}/{len(paths)}] {stem}: {len(rows)} chunks embedded")

        # ─── STEP 3: build the HNSW index once, at the end ────────────────────
        print("Building HNSW index...")
        conn.execute(HNSW_INDEX_SQL)
        conn.commit()

    print("Done.")


# ═══ LEG: bm25 ══════════════════════════════════════════════════════════════════


def run_bm25() -> None:
    # ─── STEP 1: build the pg_search index over the text already loaded ───────
    init_schema()
    with get_conn(check=False) as conn:  # build-time: this script moves the counts
        print("Building BM25 index (pg_search)...")
        conn.execute(BM25_INDEX_SQL)
        conn.commit()
        n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"Done. BM25 index covers {n} chunks.")


# ═══ LEG: new-filings (three-stage, filings absent from Postgres) ═══════════════


def new_filings_prepare(payload_path: Path) -> None:
    # ─── STEP 1: which filings on disk are not in Postgres ────────────────────
    with get_conn(check=False) as conn:  # build-time: this script moves the counts
        done_stems = _done_stems(conn)

    paths = sorted(CHUNKS_DIR.glob("*.json"))
    todo = [p for p in paths if p.stem not in done_stems]
    print(f"{len(done_stems)}/{len(paths)} filings already indexed, {len(todo)} to embed")

    # ─── STEP 2: dump their chunk texts as the GPU payload ────────────────────
    payload = []
    for path in todo:
        for idx, c in enumerate(json.loads(path.read_text())):
            payload.append({"filing_stem": path.stem, "chunk_index": idx, "text": c["text"]})

    payload_path.write_text(json.dumps(payload))
    print(f"Wrote {len(payload)} chunks (from {len(todo)} filings) to {payload_path}")
    print("Next (cluster):  python embed_hpc.py <payload> <results.jsonl>")


def new_filings_load(results_path: Path) -> None:
    # ─── STEP 1: read the cluster's embeddings, keyed by (stem, chunk_index) ──
    by_stem: dict[str, dict[int, list[float]]] = {}
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                by_stem.setdefault(row["filing_stem"], {})[row["chunk_index"]] = row["embedding"]

    print(f"Loaded embeddings for {len(by_stem)} filings")

    # ─── STEP 2: join with local text/metadata and INSERT ─────────────────────
    init_schema()
    with get_conn(check=False) as conn:  # build-time: this script moves the counts
        done_stems = _done_stems(conn)

        for i, (stem, emb_by_idx) in enumerate(by_stem.items(), 1):
            if stem in done_stems:
                continue
            chunks = _load_chunks(stem)
            rows = []
            for idx, c in enumerate(chunks):
                if idx not in emb_by_idx:
                    print(f"WARNING: missing embedding for {stem} chunk {idx}, skipping filing")
                    rows = []
                    break
                rows.append(_chunk_row(stem, idx, c, emb_by_idx[idx]))
            if not rows:
                continue
            _insert_rows(conn, rows)
            print(f"[{i}/{len(by_stem)}] {stem}: {len(rows)} chunks inserted")

        # ─── STEP 3: rebuild HNSW ─────────────────────────────────────────────
        print("Rebuilding HNSW index...")
        conn.execute(HNSW_INDEX_SQL)
        conn.commit()

    print("Done.")


# ═══ LEG: changed-chunks (three-stage, rows already in Postgres) ════════════════


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


def _require_baseline(baseline: Path) -> None:
    if not baseline.is_dir():
        sys.exit(f"baseline chunks not found: {baseline}\n"
                 "Extract the pre-re-chunk backup there, e.g.\n"
                 "  mkdir -p data/chunks_pre_retr7 && tar -xzf "
                 "~/rag-sec-backups/chunks_json_pre_retr7_*.tgz "
                 "-C data/chunks_pre_retr7 --strip-components=2")


def changed_prepare(payload_path: Path, baseline: Path) -> None:
    # ─── STEP 1: diff new chunks against the pre-fix backup ───────────────────
    _require_baseline(baseline)
    todo = changed_chunks(baseline)

    # ─── STEP 2: write the GPU payload (same format embed_hpc.py already eats) ─
    payload = [
        {"filing_stem": c["filing_stem"], "chunk_index": c["chunk_index"], "text": c["text"]}
        for c in todo
    ]
    payload_path.write_text(json.dumps(payload))
    print(f"{len(payload)} changed chunks -> {payload_path}")
    print("Next (HPC):  python embed_hpc.py <payload> <results.jsonl>")


def changed_load(results_path: Path, baseline: Path, keep_indexes: bool) -> None:
    # ─── STEP 1: recompute the same work set, then match embeddings to it ─────
    # Recomputed rather than read back from the payload so the load stage cannot be fed a
    # results file that belongs to a different diff.
    _require_baseline(baseline)
    todo = changed_chunks(baseline)

    meta = {(c["filing_stem"], c["chunk_index"]): c for c in todo}
    embs, unknown = {}, 0
    with open(results_path) as f:
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
    print(f"changed chunks: {len(meta)} | embeddings present: {len(embs)} | "
          f"missing: {missing} | not-needed rows in file: {unknown}")
    if missing:
        sys.exit(f"refusing to load: {missing} changed chunks have no embedding. "
                 "Re-run embed_hpc.py against the same payload (it checkpoints and resumes).")

    # ─── STEP 2: bulk UPDATE variant 'A' rows in place ────────────────────────
    t0 = time.monotonic()
    with get_conn(check=False) as conn:  # rewrites rows; it does not move the counts
        if not keep_indexes:
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
            if n % UPDATE_COMMIT_EVERY == 0:
                conn.commit()
                print(f"  ...{n}/{len(embs)} rows", flush=True)
        conn.commit()

        # ─── STEP 3: rebuild both indexes ─────────────────────────────────────
        if not keep_indexes:
            print("rebuilding HNSW + BM25 indexes...")
            conn.execute(HNSW_INDEX_SQL)
            conn.execute(BM25_INDEX_SQL)
            conn.commit()

    print(f"\nrows updated: {n}\nelapsed     : {time.monotonic() - t0:.0f}s")
    print("Now re-run: scripts/checks/candidate_sql.py, scripts/checks/variant_predicates.py")


def _add_stage_flags(p: argparse.ArgumentParser, prepare_default: Path | None) -> None:
    """The --prepare/--load split, as the pre-reorg reembed_changed.py defined it (now the
    `new-filings --prepare/--load` legs): mutually exclusive and required, so a stage is always named rather than inferred."""
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", metavar="PAYLOAD", nargs="?" if prepare_default else None,
                   const=str(prepare_default) if prepare_default else None,
                   default=None,
                   help="stage 1 (host): write the GPU payload"
                        + (f" [default: {prepare_default.name}]" if prepare_default else ""))
    g.add_argument("--load", metavar="RESULTS",
                   help="stage 3 (host): apply the embeddings the cluster wrote")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="leg", required=True)

    sub.add_parser("local", description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter,
                   help="embed everything not yet loaded, in this process, then build HNSW")
    sub.add_parser("bm25", description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter,
                   help="build the pg_search BM25 index over text already loaded")

    p_new = sub.add_parser("new-filings", description=__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter,
                           help="cluster round trip for filings not yet in Postgres (INSERT)")
    _add_stage_flags(p_new, NEW_FILINGS_PAYLOAD)

    p_chg = sub.add_parser("changed-chunks", description=__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter,
                           help="cluster round trip for chunks whose text changed (UPDATE)")
    _add_stage_flags(p_chg, None)
    p_chg.add_argument("--baseline-chunks", default=str(BASELINE_CHUNKS_DIR),
                       help="pre-fix data/chunks/ (from the backup taken before --rechunk)")
    p_chg.add_argument("--keep-indexes", action="store_true",
                       help="do not drop/rebuild HNSW+BM25 around the load (much slower)")

    args = ap.parse_args()

    if args.leg == "local":
        run_local()
    elif args.leg == "bm25":
        run_bm25()
    elif args.leg == "new-filings":
        if args.prepare:
            new_filings_prepare(Path(args.prepare))
        else:
            new_filings_load(Path(args.load))
    else:
        baseline = Path(args.baseline_chunks)
        if args.prepare:
            changed_prepare(Path(args.prepare), baseline)
        else:
            changed_load(Path(args.load), baseline, keep_indexes=args.keep_indexes)


if __name__ == "__main__":
    main()
