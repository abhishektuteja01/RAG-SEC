"""Pipeline phase 05 — Arm 3: cross-encoder rerank + company filter + query strip.

THE HEADLINE ARM. Published test recall@10 **0.747** (dev 0.760), `RETR-39`.

PRODUCES
    data/retr7_rr_{dev,test}_scores.jsonl    raw rerank scores, one line per question
                                             (written by the CLUSTER leg, or by `local`)
    data/retr7_arm3_{dev,test}_results.json  the scored 2x2 table (`score --out`)
    data/day8_retr16v2_rerank_payload.json   the cluster's input (`prepare`; see TRAPS,
                                             the default name is stale)

READS
    Postgres `chunks` (variant 'A' only) via rag_sec.candidates, and the matched
    question set via rag_sec.eval.load_matched_questions. `score` reads nothing but a
    scores file on disk -- no GPU, no database, no spend.

THE 2x2 (cells), from one candidate pool per side:
    unfiltered_raw       as-was                             -> the control
    filtered_raw         + company filter                   -> isolates RETR-5
    unfiltered_stripped  + entity framing stripped          -> isolates RETR-6
    filtered_stripped    + both                             -> THE SHIPPED ARM
The fourth cell is what makes the result attributable: without the control, a better
number cannot be assigned to the filter or to the query cleanup.

THREE LEGS, one scores format
    prepare  laptop. Runs first-stage retrieval and dumps a self-contained payload.
    (GPU)    scripts/pipeline/hpc/rerank_hpc.py, ON the cluster. See ARM3-2.
    local    laptop/deploy host. Same model, same candidate pool, single process, no
             cluster -- so the headline arm is reproducible without HPC access. Writes
             the IDENTICAL scores format, so `score` cannot tell the two legs apart.
    score    laptop. Replays a scores file into the published table.

DECISIONS.md ROWS THIS BACKS
    ARM3-1    bge-reranker-v2-m3, one forward pass per pair, not an autoregressive
              reranker -- spec.md measures nDCG gain against latency cost.
    ARM3-2    why reranking is a split job: laptop CPU plateaued at ~110s/question
              (~25h for dev) and a live SSH tunnel to the laptop's DB dies with the
              connection. `local` below is that abandoned laptop route, rebuilt on the
              current library -- still slow, but now correct and resumable.
    RETR-5    company-filtered candidate generation, company resolved from the question
              text alone.
    RETR-6    rerank against the question with company/filing framing removed, while
              candidate generation still sees the full question. Alone +0.015; with the
              filter +0.145 -- once every candidate is the right company, the company
              name only rewards whichever chunk repeats the most boilerplate.
    RETR-16   the 2x2 itself, and why dev could skip a cell it already had.
    RETR-18   test gets the full 2x2, so RETR-22's interaction is confirmed out of sample.
    RETR-24   the missing `variant` predicate that forced the v2 re-run; candidate SQL now
              lives once in rag_sec.candidates and `scripts/checks/candidate_sql.py`
              asserts it.
    RETR-30   test's payload carries all four cells, so no baseline merge is needed.
    RETR-39   the current numbers, post RETR-7/RETR-8 re-index.
    AGENT-16  a scores file as stored is NOT in rank order. `rag_sec.eval.load_ranking`
              sorts on load; reading it raw cost recall@10 0.552 against 0.739.
    AGENT-24  `torch.mps.empty_cache()` per retrieval is why laptop rerank latency is a
              flat ~32.6s instead of climbing to ~48s. `local` inherits it from
              `rag_sec.retrieve.retrieve()`. Do not remove or bypass it.

WHEN THIS ACTUALLY RAN (calendar dates, not "Day N")
    2026-08-28   the Arm 3 cross-encoder HPC pass (its Day-5/6 ancestor,
                 scripts/archive/arm3_rerank_*.py)
    2026-08-30   those results rescored on the laptop
    2026-09-04   the RETR-7/RETR-8 re-index and the current retr7_* dev+test passes;
                 data/retr7_arm3_{dev,test}_results.json are 09-04 09:39 and 11:22
    The `local` leg is NEW in this consolidation and has never produced a published
    number. It exists so someone without a cluster can reproduce one.

TRAPS
  * STALE DEFAULTS, PRESERVED ON PURPOSE. `prepare --out` still defaults to
    `data/day8_retr16v2_rerank_payload.json` and `score --scores` to
    `data/day8_retr16v2_dev_scores.jsonl`, while the LIVE artifacts these legs now
    produce and consume are `data/retr7_rr_{dev,test}_scores.jsonl`. Quote RETR-39 from
    the retr7_* files and pass `--scores` explicitly. The names were left alone so this
    phase is byte-for-byte behaviour-preserving against the scripts it merges;
    `scripts/retrieval/rerank_hpc.sbatch` carries the same drift (`day8_retr18_test_*`).
  * `score`'s CELLS requires all four cells present, and counts what it skipped rather
    than dropping questions silently -- a missing cell would otherwise shrink the
    denominator invisibly and make cells incomparable.
  * The cached dev control `data/day6_arm4_A_rerank_scores.jsonl` was scored against
    PRE-RETR-7 chunk text. Merging it into a post-re-index 2x2 mixes two corpora and
    mis-attributes the ablation. `score` therefore refuses to merge it when the scores
    file already carries `unfiltered_raw` for every question; an explicit `--baseline`
    still wins. This is also why `prepare` now DEFAULTS to all four cells on both splits:
    the three-cell dev payload only ever made sense as "reuse the cell we already have",
    and after the re-index of 2026-09-04 (`RETR-39`) that cell is the wrong corpus. The
    three-cell payload is still reachable with `--reuse-dev-baseline`, which is a
    historical-reproduction flag, not an optimization. `score` is unchanged either way.
  * recall@50 is a property of the candidate POOL, not the reranker. It can only differ
    between filtered and unfiltered cells; identical values there are correct.
  * `local` IS SLOW AND THAT IS NOT A BUG. Rerank alone is ~32.6s/question on an M3
    laptop (AGENT-24) and ~156.9s on the Graviton3 deploy host, PER CELL. Four cells over
    the 1235-question dev split is many hours to days. Use `--n` for a smoke test; use
    the cluster for a full pass.
  * README.md's quickstart still points at `scripts/archive/arm3_*` for the local route.
    Those are the ABANDONED Day-5 scripts. `local` here is the current one.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.candidates import LIVE_VARIANT, bm25, chunk_texts, dense, rrf_fuse  # noqa: E402
from rag_sec.company import resolve as resolve_companies  # noqa: E402
from rag_sec.company import strip_entity_framing  # noqa: E402
from rag_sec.config import EMBED_MODEL_NAME, RERANK_MODEL_NAME  # noqa: E402
from rag_sec.eval import (  # noqa: E402
    ALL_CELLS,
    _filing_stem,
    gold_relevant_chunk_ids,
    load_matched_questions,
    load_ranking,
    mean_and_stderr,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from rag_sec.retrieve import last_call_stats, retrieve  # noqa: E402
from rag_sec.store import get_conn  # noqa: E402

# ─── CONSTANTS ──────────────────────────────────────────────────────────────────
DATA_DIR = _ROOT / "data"

# First-stage pool size handed to the reranker. 50, matching rag_sec.retrieve.CANDIDATE_K,
# because the whole point of the 2x2 is that the rerank cells differ ONLY in the filter and
# the query text -- a different pool size here would make `local` and the cluster leg
# incomparable, and would silently move recall@50, which is a pool property.
CANDIDATE_K = 50

# The four cells, in the order the published table prints them. `unfiltered_raw` must come
# first: `score` reads it as the baseline row that every other row's delta is against.
CELLS = ("unfiltered_raw", "filtered_raw", "unfiltered_stripped", "filtered_stripped")

# cell name -> (company_filter, strip_query) for rag_sec.retrieve.retrieve(). This is the
# whole of the `local` leg's cell semantics: the two flags ARE the 2x2's two axes, so the
# local leg cannot drift from the shipping path's definition of a cell (RETR-5 / RETR-6).
LOCAL_CELL_FLAGS = {
    "unfiltered_raw": (False, False),
    "filtered_raw": (True, False),
    "unfiltered_stripped": (False, True),
    "filtered_stripped": (True, True),
}

# STALE, PRESERVED: the live payload/scores artifacts are retr7_*, not day8_*. Kept as the
# defaults so this phase is behaviour-identical to the scripts it merges. See TRAPS.
PAYLOAD_DEFAULT = DATA_DIR / "day8_retr16v2_rerank_payload.json"
SCORES_DEFAULT = DATA_DIR / "day8_retr16v2_dev_scores.jsonl"

# The published post-re-index artifacts, named so `--help` can point at them.
LIVE_SCORES = {s: DATA_DIR / f"retr7_rr_{s}_scores.jsonl" for s in ("dev", "test")}

# dev's fourth cell was scored on 2026-08-30 and is merged in from here when the scores
# file lacks it. test has no such prior run (RETR-30). PRE-RETR-7 text -- see TRAPS.
BASELINE_DEV = DATA_DIR / "day6_arm4_A_rerank_scores.jsonl"

# `local` writes here by default, NOT over data/retr7_rr_*_scores.jsonl. Those two files
# are a paid cluster booking that cannot be re-made cheaply; a local smoke run must not be
# able to land on top of them by forgetting a flag.
LOCAL_SCORES_TMPL = "local_rr_{split}_scores.jsonl"

# Measured per-question rerank latency, for the runtime warning `local` prints. Per CELL,
# so a four-cell run is ~4x these. Laptop figure is AGENT-24's steady state (M3, mps, with
# torch.mps.empty_cache() per call); the host figure is the Graviton3 deploy box, where
# rerank is 99.5% of /ask latency (DEPLOY-18).
RERANK_S_LAPTOP = 32.6
RERANK_S_DEPLOY_HOST = 156.9

# Transfer/allocation recipe printed by `prepare`. Constants, not buried strings, because
# ARM3-2 is explicit that the xfer host is mandatory: the interactive login node
# throttles or kills a 100MB+ payload mid-copy.
XFER_HOST = "tuteja.a@xfer.discovery.neu.edu"
SRUN_RECIPE = (
    "  srun --partition=gpu --gres=gpu:v100-sxm2:1 --cpus-per-task=4 \\\n"
    "       --mem=48G --time=08:00:00 --pty /bin/bash\n"
    "  module load python/3.13.5 && source ~/rerank-env/bin/activate"
)

def _rel(path: Path) -> str:
    """Repo-relative, for --help text. Absolute paths in a --help make the help unreadable
    and machine-specific; the constants themselves stay absolute so cwd cannot matter."""
    return str(Path(path).relative_to(_ROOT))


# Metric keys, in print order. One tuple so the table header, the accumulator and the
# JSON sidecar cannot fall out of step.
METRIC_KEYS = ("recall_10", "recall_50", "ndcg_10", "mrr")


# ─── STEP 1: prepare the cluster payload (laptop, needs Postgres) ───────────────
def cmd_prepare(args: argparse.Namespace) -> None:
    """Dump one self-contained payload covering every cell the GPU leg must score.

    Chunk text is stored ONCE in a shared `texts` map keyed "stem|chunk_index", not
    inlined per candidate: the filtered and unfiltered pools overlap heavily and chunks
    repeat across questions, so inlining paid ~240MB for one pool in the Day 6 payloads.

    The stripped query is computed HERE and travels as a field, so the GPU leg needs no
    `rag_sec` import and no database (ARM3-2 / INFRA-6).
    """
    from sentence_transformers import SentenceTransformer

    df = load_matched_questions()
    wanted = [x.strip() for x in args.splits.split(",") if x.strip()]
    rows = df[df["split"].isin(wanted)].reset_index(drop=True)
    if args.n:
        rows = rows.head(args.n)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    questions: list[dict] = []
    texts: dict[str, str] = {}
    n_resolved = n_stripped = 0

    with get_conn() as conn:
        for _, row in tqdm(rows.iterrows(), total=len(rows), desc="building RETR-16 payload"):
            q = row["question"]
            emb = model.encode(q, normalize_embeddings=True)
            tickers = resolve_companies(q)
            n_resolved += bool(tickers)

            unfiltered = rrf_fuse([
                dense(conn, emb, CANDIDATE_K, LIVE_VARIANT),
                bm25(conn, q, CANDIDATE_K, LIVE_VARIANT),
            ])[:CANDIDATE_K]
            # No ticker resolved -> the filtered pool IS the unfiltered pool, the same
            # fallback retrieve() uses. Recorded rather than skipped so the filtered arm is
            # scored over all 1235 questions, not just the resolvable ones.
            filtered = (
                rrf_fuse([
                    dense(conn, emb, CANDIDATE_K, LIVE_VARIANT, tickers),
                    bm25(conn, q, CANDIDATE_K, LIVE_VARIANT, tickers),
                ])[:CANDIDATE_K]
                if tickers
                else unfiltered
            )

            need = [p for p in unfiltered + filtered if f"{p[0]}|{p[1]}" not in texts]
            for (stem, idx), text in chunk_texts(conn, need, LIVE_VARIANT).items():
                texts[f"{stem}|{idx}"] = text

            stripped = strip_entity_framing(q)
            n_stripped += stripped != q
            questions.append({
                "id": row["id"],
                "split": row["split"],
                # All four cells by default (RETR-30/RETR-39). --reuse-dev-baseline
                # brings back the pre-re-index shortcut of dropping dev's unfiltered_raw
                # and merging it from BASELINE_DEV, which is now the wrong thing to do.
                "cells": (
                    ["filtered_raw", "filtered_stripped", "unfiltered_stripped"]
                    if row["split"] == "dev" and args.reuse_dev_baseline
                    else list(CELLS)
                ),
                "question": q,
                "question_stripped": stripped,
                "tickers": tickers,
                "cands_unfiltered": [list(p) for p in unfiltered],
                "cands_filtered": [list(p) for p in filtered],
            })

    n = len(questions)
    if not n:
        # Reachable with --splits naming a split the question set does not have. Checked
        # BEFORE the write, not after: --out defaults to the 100MB+ day8_* payload, and a
        # typo'd --splits would otherwise replace it with an empty one -- and the summary
        # below divides by n.
        raise SystemExit(
            f"no questions matched splits={wanted!r} -- nothing written to {args.out}.\n"
            f"  Known splits: {sorted(df['split'].unique())}"
        )

    args.out.write_text(json.dumps({"questions": questions, "texts": texts}))
    pairs = sum(
        len(q["cands_filtered" if c.startswith("filtered") else "cands_unfiltered"])
        for q in questions
        for c in q["cells"]
    )
    print("by split:", dict(Counter(q["split"] for q in questions)))
    print(f"\nquestions {n}   company resolved {n_resolved} ({100 * n_resolved / n:.1f}%)"
          f"   query stripped {n_stripped} ({100 * n_stripped / n:.1f}%)")
    print(f"unique chunks carried: {len(texts)}")
    n_cells = sorted({len(q["cells"]) for q in questions})
    print(f"pairs to score across {'/'.join(map(str, n_cells))} cells per question: {pairs:,}")
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.0f} MB)")
    print("\nCopy via the transfer node, not the login node (DECISIONS.md ARM3-2):")
    print(f"  scp {args.out} {XFER_HOST}:~/{args.out.name}")
    print(f"  scp scripts/pipeline/hpc/rerank_hpc.py {XFER_HOST}:~/")
    print("\nOn the GPU node (inside tmux -- srun --pty dies with the SSH session):")
    print(SRUN_RECIPE)
    # Derived from the payload name, not hardcoded: the dev run's scores file is already on
    # disk, and a second job writing the same name would clobber it on copy-back.
    scores_name = args.out.name.replace("_payload", "_scores").replace(".json", ".jsonl")
    print(f"  python -u rerank_hpc.py {args.out.name} {scores_name}")


# ─── STEP 2 (cluster): scripts/pipeline/hpc/rerank_hpc.py ───────────────────────
# Not a function here on purpose. It runs on a node with no `rag_sec` install and no DB,
# keeps its own basename because the sbatch invokes it by bare name after scp, and takes
# exactly two positionals (payload.json results.jsonl). See ARM3-2 / INFRA-6.


# ─── STEP 2-ALT: rerank locally, single process, no cluster ─────────────────────
def _local_cells(question: str, cells: list[str]) -> tuple[dict[str, list], float]:
    """Score one question's cells through the SHIPPING path, `rag_sec.retrieve.retrieve()`.

    Why retrieve() and not a private copy of the loop: the 2x2's two axes ARE retrieve()'s
    `company_filter` and `strip_query` flags, so going through it makes a local cell
    definitionally the same thing the deployed system does, and inherits the candidate SQL
    (`rag_sec.candidates`, RETR-24), the no-ticker fallback, and AGENT-24's
    `torch.mps.empty_cache()` per call -- the one line that keeps laptop rerank latency
    flat at ~32.6s instead of climbing past 48s. Do not bypass it.

    The full reranked pool comes from `last_call_stats()["reranked"]`, not from
    retrieve()'s return value, which is truncated to top-k: a cell must carry all ~50
    scored candidates or recall@50 and MRR below k become unrecoverable.
    """
    out: dict[str, list] = {}
    rerank_s = 0.0
    for cell in cells:
        company_filter, strip_query = LOCAL_CELL_FLAGS[cell]
        retrieve(question, k=CANDIDATE_K, company_filter=company_filter,
                 strip_query=strip_query)
        stats = last_call_stats()
        # Already 3-wide [stem, int(idx), float(score)] -- the exact entry shape the
        # cluster leg writes. Stored order differs (the cluster zips scores onto the
        # first-stage RRF order, this is score-descending) and that is immaterial:
        # load_ranking sorts on load for both (AGENT-16).
        out[cell] = stats["reranked"]
        rerank_s += stats["timings"]["rerank_s"]
    return out, rerank_s


def cmd_local(args: argparse.Namespace) -> None:
    """Single-process rerank leg: same model, same pool, no HPC. Resumable, slow."""
    out_path = args.out or DATA_DIR / LOCAL_SCORES_TMPL.format(split=args.split)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    unknown = [c for c in cells if c not in LOCAL_CELL_FLAGS]
    if unknown:
        raise SystemExit(f"unknown cell(s) {unknown}; known: {sorted(LOCAL_CELL_FLAGS)}")

    df = load_matched_questions()
    rows = df[df["split"] == args.split].reset_index(drop=True)
    if args.n:
        rows = rows.head(args.n)

    # Resume by question id, the same contract the cluster leg has: a killed run costs
    # nothing, and the file is append-only so a partial file is still loadable.
    done: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [r for _, r in rows.iterrows() if r["id"] not in done]

    est_min = len(todo) * len(cells) * RERANK_S_LAPTOP / 60
    print(f"local rerank leg: {RERANK_MODEL_NAME}")
    print(f"{args.split}: {len(rows)} questions, {len(done)} already done, {len(todo)} to run")
    print(f"cells per question: {len(cells)} {cells}")
    print(f"ESTIMATE ~{est_min:.0f} min on an M3 laptop at {RERANK_S_LAPTOP}s/cell "
          f"(~{len(todo) * len(cells) * RERANK_S_DEPLOY_HOST / 3600:.1f} h on the deploy "
          f"host at {RERANK_S_DEPLOY_HOST}s/cell). Use --n for a smoke test.")
    if not todo:
        print("nothing to do")
        return

    with open(out_path, "a") as fh:
        for i, row in enumerate(todo, 1):
            t0 = time.perf_counter()
            cells_out, rerank_s = _local_cells(row["question"], cells)
            wall = time.perf_counter() - t0
            # Identical key set and semantics to the cluster leg's line, so `score` and
            # rag_sec.eval.load_ranking cannot tell the two legs apart. `latency_s` is the
            # rerank time for this question across its cells -- the cluster's is the same
            # quantity averaged over a batch of 20.
            fh.write(json.dumps({
                "id": row["id"],
                "split": row["split"],
                "cells": cells_out,
                "latency_s": rerank_s,
            }) + "\n")
            fh.flush()
            print(f"  {i}/{len(todo)} {row['id']:<18} rerank={rerank_s:6.1f}s "
                  f"wall={wall:6.1f}s", flush=True)
    print(f"wrote {out_path}")


# ─── STEP 3: score a scores file into the published 2x2 table ──────────────────
def cmd_score(args: argparse.Namespace) -> None:
    """Replay a scores file. No GPU, no database, no spend -- free to re-run."""
    # All cells, ids only, sorted by rerank score -- see rag_sec.eval.load_ranking for why
    # the sort happens on load and why the loader dispatches on the record's own shape.
    ranked: dict[str, dict[str, list]] = load_ranking(args.scores, ALL_CELLS)

    # Resolved AFTER loading, so a scores file that already carries unfiltered_raw is never
    # silently overwritten by the cached pre-RETR-7 one. An explicit --baseline still wins.
    baseline = args.baseline or (BASELINE_DEV if args.split == "dev" else None)
    if (baseline and not args.baseline and ranked
            and all("unfiltered_raw" in v for v in ranked.values())):
        print(f"scores file already has unfiltered_raw for all {len(ranked)} questions "
              f"-- NOT merging the cached baseline {BASELINE_DEV}")
        baseline = None

    if baseline:
        merged = 0
        # cell=None: the baseline is a legacy no-cells file, already in rerank order. Read
        # through the shared loader rather than unpacking `reranked` inline -- that inline
        # copy assumed a 3-wide entry, which only the day6_arm4_* files happen to be.
        for qid, order in load_ranking(baseline, None).items():
            if qid in ranked:
                ranked[qid]["unfiltered_raw"] = order
                merged += 1
        print(f"merged unfiltered_raw for {merged} questions from {baseline}")

    df = load_matched_questions()
    rows = df[df["split"] == args.split].reset_index(drop=True)

    per = {c: {k: [] for k in METRIC_KEYS} for c in CELLS}
    n = 0
    # Counted, not silently skipped: a question dropped for a missing cell would otherwise
    # shrink the denominator invisibly and make cells incomparable.
    skipped_cells = skipped_gold = 0
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="scoring"):
        qid = row["id"]
        if qid not in ranked or any(c not in ranked[qid] for c in CELLS):
            skipped_cells += 1
            continue
        stem = _filing_stem(row)
        rel = [(stem, i) for i in gold_relevant_chunk_ids(row)]
        if not rel:
            skipped_gold += 1
            continue
        n += 1
        for c in CELLS:
            got = ranked[qid][c]
            per[c]["recall_10"].append(recall_at_k(got, rel, 10))
            per[c]["recall_50"].append(recall_at_k(got, rel, 50))
            per[c]["ndcg_10"].append(ndcg_at_k(got, rel, 10))
            per[c]["mrr"].append(mrr(got, rel))

    print(f"\nquestions scored: {n} of {len(rows)} {args.split}"
          f"   (skipped: {skipped_cells} missing a cell, {skipped_gold} with no gold label)\n")
    hdr = f"{'cell':<22}" + "".join(
        f"{k:>18}" for k in ("recall@10", "recall@50", "nDCG@10", "MRR"))
    print(hdr)
    print("-" * len(hdr))
    base, table = {}, {}
    for c in CELLS:
        printed = []
        for k in METRIC_KEYS:
            m, se = mean_and_stderr(per[c][k])
            table.setdefault(c, {})[k] = {"mean": m, "stderr": se}
            if c == "unfiltered_raw":
                base[k] = m
                printed.append(f"{m:.3f} ± {se:.3f}".rjust(18))
            else:
                printed.append(f"{m:.3f} ({m - base[k]:+.3f})".rjust(18))
        print(f"{c:<22}" + "".join(printed))
    print("\n(baseline row shows ± stderr; other rows show the delta against it)")

    # Stderr is stored for every cell, not just the baseline row that prints it: the deltas
    # are what get quoted, and a delta needs both cells' spread to be defensible.
    if args.out:
        args.out.write_text(json.dumps({
            "split": args.split, "scores": str(args.scores), "n_scored": n,
            "skipped_missing_cell": skipped_cells, "skipped_no_gold": skipped_gold,
            "cells": table,
        }, indent=1))
        print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="laptop: build the cluster payload (needs Postgres)",
                       description="Step 1: first-stage retrieval -> self-contained payload "
                                   "for scripts/pipeline/hpc/rerank_hpc.py.",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, help="limit questions (smoke test)")
    p.add_argument("--splits", default="dev,test", help="comma-separated (default: %(default)s)")
    # DEFAULT FLIPPED: the full 2x2 on every split. Before the RETR-7/RETR-8 re-index the
    # dev payload dropped unfiltered_raw and `score` merged it from the cached
    # day6_arm4_A file; that file is PRE-re-index text, so doing it today mixes two
    # corpora and mis-attributes the ablation (see TRAPS). The old behaviour is still
    # reachable, but it now has to be asked for.
    p.add_argument("--reuse-dev-baseline", action="store_true",
                   help="OLD, PRE-RE-INDEX behaviour: omit dev's unfiltered_raw cell and "
                        "let `score` merge it from the cached day6_arm4_A file. That file "
                        "is pre-RETR-7 text, so this mixes two corpora -- do not use it "
                        "for a post-re-index number")
    p.add_argument("--out", type=Path, default=PAYLOAD_DEFAULT,
                   help=f"default is the STALE day8_* name ({_rel(PAYLOAD_DEFAULT)})")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser(
        "local", help="laptop/host: rerank in one process, no HPC. SLOW.",
        description=(
            f"Step 2-ALT: the local rerank leg. Same model ({RERANK_MODEL_NAME}), same "
            f"{CANDIDATE_K}-candidate pool and same cell definitions as the cluster leg, in "
            "one process on CPU/MPS, writing the identical scores format.\n\n"
            f"RUNTIME, HONESTLY: rerank is ~{RERANK_S_LAPTOP}s per question PER CELL on an "
            f"M3 laptop (AGENT-24) and ~{RERANK_S_DEPLOY_HOST}s on the Graviton3 deploy "
            "host. Four cells over the full 1235-question dev split is MANY HOURS on a "
            "laptop and over a day on the host. This leg exists so the headline arm is "
            "reproducible without cluster access, not because it is a good way to spend an "
            "afternoon. Use --n for a smoke test. It is resumable per question id.\n\n"
            "No API calls, no money -- just wall clock."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, help="limit questions (STRONGLY recommended)")
    p.add_argument("--split", default="dev", help="(default: %(default)s)")
    p.add_argument("--cells", default=",".join(CELLS),
                   help="comma-separated cells to score (default: all four, which is what "
                        "`score` requires)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"default data/{LOCAL_SCORES_TMPL} -- deliberately NOT the published "
                        "retr7_rr_*_scores.jsonl, which a cluster booking paid for")
    p.set_defaults(fn=cmd_local)

    p = sub.add_parser(
        "score", help="laptop: replay a scores file into the 2x2 table (free)",
        description=("Step 3: score a scores file from either leg. Reads only -- no GPU, no "
                     "database, no spend.\n\nThe published RETR-39 numbers come from "
                     f"{_rel(LIVE_SCORES['dev'])} (dev, recall@10 0.760) and "
                     f"{_rel(LIVE_SCORES['test'])} "
                     "(test, 0.747); pass --scores/--split explicitly, because the default "
                     "below is the stale day8_* name."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", type=Path, default=SCORES_DEFAULT,
                   help=f"default is the STALE day8_* name ({_rel(SCORES_DEFAULT)})")
    p.add_argument("--split", default="dev", help="(default: %(default)s)")
    p.add_argument("--out", type=Path, default=None,
                   help="write the scored table to JSON as well as printing it")
    p.add_argument("--baseline", type=Path,
                   help="file supplying unfiltered_raw; omit when the scores file already "
                        "has that cell. PRE-RETR-7 text -- see the module docstring's TRAPS")
    p.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
