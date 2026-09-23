"""Pipeline phase 05 — Arm 3: cross-encoder rerank + company filter + query strip.

THE HEADLINE ARM. Two tables live here:
  2x2   the published ablation, test recall@10 **0.747** (dev 0.760), `RETR-39`. Measured
        BEFORE `RETR-40` shipped the year nudge, so its cells pin `year_bias=False`.
  arms  `deployed` vs `n18d` vs `n18d_p10` -- the shipped configuration (filter + strip +
        year_bias, dev 0.791 / test 0.771) against the two first-stage candidates, with
        paired CIs, McNemar counts and the `RETR-45` subgroup split.

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

THE ARMS (cells), one candidate pool each, all reranked against the STRIPPED question:
    deployed   filter + strip + year_bias           -> the bar to beat
    n18d       + dense leg embeds the stripped query
    n18d_p10   + a third RRF list preferring chunks whose TEXT names the question's
               fiscal year, choosing which 50 of a 200-deep set survive
They differ only in how the 50 candidates are CHOSEN, never in what the cross-encoder is
asked, so a delta between them is first-stage and nothing else. Pool and rerank cost are
identical across all three.

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
              reranker. The arm is judged on nDCG gain against latency cost, so the
              cheaper architecture wins over the higher leaderboard score.
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

WHEN THIS RAN: see the phase 05 row of scripts/README.md. The `local` leg is NEW in this
consolidation and has never produced a published number; it exists so someone without a
cluster can reproduce one.

TRAPS
  * STALE DEFAULTS, PRESERVED ON PURPOSE. `prepare --out` still defaults to
    `data/day8_retr16v2_rerank_payload.json` and `score --scores` to
    `data/day8_retr16v2_dev_scores.jsonl`, while the LIVE artifacts these legs now
    produce and consume are `data/retr7_rr_{dev,test}_scores.jsonl`. Quote RETR-39 from
    the retr7_* files and pass `--scores` explicitly. The names were left alone so this
    phase is byte-for-byte behaviour-preserving against the scripts it merges;
    `scripts/pipeline/hpc/rerank_hpc.sbatch` carries the same drift (`day8_retr18_test_*`).
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
  * `local` IS SLOW ON A LAPTOP AND THAT IS NOT A BUG. Rerank alone is ~32.6s/question on
    an M3 (AGENT-24), PER CELL -- four cells over dev is many hours. On the CURRENT deploy
    host it is ~3.27s (g4dn.xlarge T4, fp16, DEPLOY-21), which is a different machine from
    the ~156.9s Graviton3 this file used to quote; DEPLOY-24 terminated that box. A full
    arms pass on both splits is an evening there and about a week on the laptop. Use `--n`
    for a smoke test.
  * CELL FLAGS ARE PINNED PER CELL, including `year_bias`. retrieve()'s defaults move as the
    system ships; a cell's definition must not. `local` inherited the default and so stopped
    reproducing the very 2x2 it prints, silently, the day RETR-40 landed.
  * The 2x2 and the arms tables have DIFFERENT baseline rows (`unfiltered_raw` and
    `deployed`). A delta is only meaningful against the row `--table` chose.
  * README.md's quickstart used to point at the abandoned Day-5 `arm3_*` scripts for the
    local route. Those are deleted; the quickstart now points here. `local` is the only one.
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

from rag_sec.candidates import (  # noqa: E402
    CANDIDATE_K,
    LIVE_VARIANT,
    READ_DEPTH,
    first_stage,
)
from rag_sec.company import resolve as resolve_companies  # noqa: E402
from rag_sec.company import strip_entity_framing  # noqa: E402
from rag_sec.config import EMBED_MODEL_NAME, RERANK_MODEL_NAME, pick_device  # noqa: E402
from rag_sec.fiscal_year import chunk_year, extract_years  # noqa: E402
from rag_sec.eval import (  # noqa: E402
    ALL_CELLS,
    _filing_stem,
    exact_mcnemar,
    gold_relevant_chunk_ids,
    load_matched_questions,
    load_ranking,
    mean_and_stderr,
    mrr,
    ndcg_at_k,
    paired_bootstrap_ci,
    percentile,
    recall_at_k,
)
# CANDIDATE_K imported from rag_sec.candidates above, not re-declared: a pool size of its own
# here would make `local` and the cluster leg incomparable and would silently move recall@50,
# which is a pool property.
from rag_sec.retrieve import last_call_stats, retrieve  # noqa: E402
from rag_sec.store import get_conn  # noqa: E402

# ─── CONSTANTS ──────────────────────────────────────────────────────────────────
DATA_DIR = _ROOT / "data"

# The four cells, in the order the published table prints them. `unfiltered_raw` must come
# first: `score` reads it as the baseline row that every other row's delta is against.
CELLS = ("unfiltered_raw", "filtered_raw", "unfiltered_stripped", "filtered_stripped")

# The ARMS table: the shipped configuration and the two candidates measured against it.
# Separate from the 2x2 above because it asks a different question (which arm ships?) and
# has a different baseline row (`deployed`, not `unfiltered_raw`).
ARM_CELLS = ("deployed", "n18d", "n18d_p10")

# cell name -> kwargs for rag_sec.retrieve.retrieve(). One table, used by BOTH the local leg
# and `prepare`, so a cell means the same thing whichever leg scores it.
#
# year_bias=False ON THE 2x2, DELIBERATELY. RETR-39's published 0.747/0.760 were measured
# before RETR-40 made the year-proximity nudge the shipped default, and retrieve() now
# defaults it ON -- so the local leg had silently stopped reproducing the table it claims to
# reproduce, while the cluster leg (which builds its own pools in `prepare`) had not. Pinning
# it here is what makes the two legs comparable again and keeps the 2x2 a reproduction rather
# than a new measurement. The ARMS cells pinned it to the shipped value instead of inheriting
# it -- and as of DEPLOY-25 no cell here inherits ANY flag from retrieve(), see below.
#
# EVERY cell now pins strip_dense/read_depth/year_text_fusion too, and that is not tidiness.
# `DEPLOY-25` flipped all three ON in retrieve()'s signature, so a cell that LEFT them out
# would inherit the new serving defaults and stop measuring the arm its name promises -- the
# 2x2 would quietly become four P10 cells, and `deployed` would become `n18d_p10`, reporting
# a delta of zero against itself. The pool builder happens to read these through
# `kw.get(..., <literal>)` rather than through retrieve(), so nothing breaks TODAY; pinning
# them makes that an invariant of this table instead of a coincidence two files apart.
# This is the same failure `RETR-53` fixed for the prepare/local legs, one flag-set later.
CELL_FLAGS = {
    "unfiltered_raw": dict(company_filter=False, strip_query=False, year_bias=False,
                           strip_dense=False, read_depth=READ_DEPTH, year_text_fusion=False),
    "filtered_raw": dict(company_filter=True, strip_query=False, year_bias=False,
                         strip_dense=False, read_depth=READ_DEPTH, year_text_fusion=False),
    "unfiltered_stripped": dict(company_filter=False, strip_query=True, year_bias=False,
                                strip_dense=False, read_depth=READ_DEPTH,
                                year_text_fusion=False),
    "filtered_stripped": dict(company_filter=True, strip_query=True, year_bias=False,
                              strip_dense=False, read_depth=READ_DEPTH,
                              year_text_fusion=False),
    # The arm deployed UP TO `DEPLOY-25`: filter + strip + year_bias (RETR-40), and the
    # baseline the two candidates had to beat. It is no longer what `retrieve()` serves --
    # the name is kept because `rerank_hpc.CELL_SPEC` keys off it and `cell_config.py`
    # checks the two agree, so renaming it here alone would fail that guard. Re-scored
    # rather than read off data/retr7_rr_*_scores.jsonl on purpose -- that artifact predates
    # the RETR-50 ef_search fix, so reusing it would fold a ~0.08pt confound into every delta.
    "deployed": dict(company_filter=True, strip_query=True, year_bias=True,
                     strip_dense=False, read_depth=READ_DEPTH, year_text_fusion=False),
    # N18d: the dense leg embeds the stripped query. Query-side only, same pool size, same
    # rerank bill.
    "n18d": dict(company_filter=True, strip_query=True, year_bias=True, strip_dense=True,
                 read_depth=READ_DEPTH, year_text_fusion=False),
    # N18d + P10: a third RRF list preferring chunks whose TEXT names the question's fiscal
    # year, choosing which 50 of a 200-deep candidate set survive. Pool and rerank unchanged;
    # P10 is worth ~0 at depth 50, so the depth is part of the cell, not a separate knob.
    "n18d_p10": dict(company_filter=True, strip_query=True, year_bias=True, strip_dense=True,
                     read_depth=200, year_text_fusion=True),
}

# Which cells SHARE a candidate pool. Two cells differing only in `strip_query` are reranked
# with different query text against the SAME 50 candidates, which is the whole point of the
# 2x2 -- so the pool is built once per group, not once per cell. The GPU leg keys off the
# same names (`cands_<group>` in the payload), so this table and rerank_hpc.CELL_SOURCES
# have to agree; `scripts/checks/cell_config.py` asserts that they do rather than trusting it.
POOL_GROUP = {
    "unfiltered_raw": "unfiltered",
    "unfiltered_stripped": "unfiltered",
    "filtered_raw": "filtered",
    "filtered_stripped": "filtered",
    "deployed": "deployed",
    "n18d": "n18d",
    "n18d_p10": "n18d_p10",
}

# Everything in CELL_FLAGS except `strip_query`, which changes only the rerank query and
# never the pool. DERIVED, not written out again: a second hand-maintained copy of the same
# flags is exactly how `prepare` drifted off `RETR-40` in the first place. The assert is the
# guard -- if two cells in one group ever disagree on a pool-affecting flag, the group is a
# lie and the payload would silently rerank one cell against the other's candidates.
POOL_KWARGS: dict[str, dict] = {}
for _cell, _flags in CELL_FLAGS.items():
    _pool_kw = {k: v for k, v in _flags.items() if k != "strip_query"}
    _g = POOL_GROUP[_cell]
    if _g in POOL_KWARGS and POOL_KWARGS[_g] != _pool_kw:
        raise AssertionError(
            f"cells in pool group {_g!r} disagree on pool-affecting flags: "
            f"{POOL_KWARGS[_g]} vs {_pool_kw} (from {_cell!r})")
    POOL_KWARGS[_g] = _pool_kw


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
# torch.mps.empty_cache() per call).
#
# The host figure is the CURRENT deploy host: g4dn.xlarge (Tesla T4) at fp16, DEPLOY-21's
# 3.27s. It was 156.9s here -- the Graviton3 CPU box of DEPLOY-18, which DEPLOY-24 terminated
# on 2026-09-13. Left stale it estimated a full run at ~5 days on the one machine that can
# actually do it in an evening, which is an argument for the wrong hardware.
RERANK_S_LAPTOP = 32.6
RERANK_S_DEPLOY_HOST = 3.27

# Transfer/allocation recipe printed by `prepare`. Constants, not buried strings, because
# ARM3-2 is explicit that the xfer host is mandatory: the interactive login node
# throttles or kills a 100MB+ payload mid-copy.
XFER_HOST = "<user>@xfer.discovery.neu.edu"
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
    cells_wanted = [c.strip() for c in args.cells.split(",") if c.strip()]
    unknown = [c for c in cells_wanted if c not in CELL_FLAGS]
    if unknown:
        raise SystemExit(f"unknown cell(s) {unknown}; known: {sorted(CELL_FLAGS)}")
    groups = sorted({POOL_GROUP[c] for c in cells_wanted})
    model = SentenceTransformer(EMBED_MODEL_NAME, device=pick_device())

    questions: list[dict] = []
    texts: dict[str, str] = {}
    n_resolved = n_stripped = 0

    with get_conn() as conn:
        for _, row in tqdm(rows.iterrows(), total=len(rows), desc="building rerank payload"):
            q = row["question"]
            tickers = resolve_companies(q)
            n_resolved += bool(tickers)
            stripped = strip_entity_framing(q)
            n_stripped += stripped != q
            years = extract_years(q)
            # Two embeddings, because N18d's whole content is that the dense leg encodes the
            # stripped string. Encoded once per question and reused across every pool group
            # that wants it -- the second encode is ~0.1s against a rerank pass of seconds.
            embs = {False: model.encode(q, normalize_embeddings=True)}
            if any(POOL_KWARGS[g].get("strip_dense") for g in groups):
                embs[True] = model.encode(stripped, normalize_embeddings=True)

            pools: dict[str, list] = {}
            for g in groups:
                kw = POOL_KWARGS[g]
                # candidates.first_stage, the SAME function retrieve() calls. This used to be
                # a hand-rolled copy here and it had drifted: still plain rrf_fuse long after
                # RETR-40 made the year nudge the default, so this leg and the local leg were
                # scoring different pools under one cell name (INFRA-22's shape again).
                pool, ptexts, _ = first_stage(
                    conn, embs[bool(kw.get("strip_dense"))], q,
                    tickers=tickers if kw["company_filter"] else None,
                    variant=LIVE_VARIANT, read_depth=kw.get("read_depth", READ_DEPTH),
                    pool_k=CANDIDATE_K, query_years=years,
                    year_bias=kw["year_bias"], year_text_fusion=kw.get("year_text_fusion", False),
                )
                pools[g] = pool
                for (stem, idx), text in ptexts.items():
                    key = f"{stem}|{idx}"
                    if key not in texts:
                        texts[key] = text

            questions.append({
                "id": row["id"],
                "split": row["split"],
                # All four 2x2 cells by default (RETR-30/RETR-39). --reuse-dev-baseline
                # brings back the pre-re-index shortcut of dropping dev's unfiltered_raw
                # and merging it from BASELINE_DEV, which is now the wrong thing to do.
                "cells": (
                    [c for c in cells_wanted if c != "unfiltered_raw"]
                    if row["split"] == "dev" and args.reuse_dev_baseline
                    else list(cells_wanted)
                ),
                "question": q,
                "question_stripped": stripped,
                "tickers": tickers,
                **{f"cands_{g}": [list(p) for p in pools[g]] for g in groups},
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
    pairs = sum(len(q[f"cands_{POOL_GROUP[c]}"]) for q in questions for c in q["cells"])
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
        # **CELL_FLAGS[cell], not two positional flags: the cells now differ on five knobs
        # (filter, strip, year_bias, strip_dense, read_depth/year_text_fusion), and spelling
        # a subset of them out here is how a cell silently inherits a retrieve() default it
        # was never meant to have -- which is exactly what happened when year_bias shipped.
        retrieve(question, k=CANDIDATE_K, **CELL_FLAGS[cell])
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
    unknown = [c for c in cells if c not in CELL_FLAGS]
    if unknown:
        raise SystemExit(f"unknown cell(s) {unknown}; known: {sorted(CELL_FLAGS)}")

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
          f"host at {RERANK_S_DEPLOY_HOST}s/cell, g4dn.xlarge T4 fp16). "
          f"Use --n for a smoke test.")
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
def _year_mismatch(row, stem: str) -> bool | None:
    """True when the question names a fiscal year and the GOLD filing is not one of them.

    `RETR-45`'s axis, and the one subgroup that has repeatedly moved the opposite way to the
    aggregate: depth-200 alone is +1.0pt overall and -4.0pt here, and raising YEAR_BIAS_ALPHA
    buys +5.5pt on the other side of this split by taking -4.3pt from this one. An arm that
    is positive overall and negative here has not earned a ship. None = the question names no
    year at all, so the split does not apply to it.
    """
    years = extract_years(row["question"])
    if not years:
        return None
    return chunk_year(stem) not in years


def cmd_score(args: argparse.Namespace) -> None:
    """Replay a scores file. No GPU, no database, no spend -- free to re-run."""
    cells = CELLS if args.table == "2x2" else ARM_CELLS
    baseline_cell = cells[0]
    # All cells, ids only, sorted by rerank score -- see rag_sec.eval.load_ranking for why
    # the sort happens on load and why the loader dispatches on the record's own shape.
    ranked: dict[str, dict[str, list]] = load_ranking(args.scores, ALL_CELLS)

    # An `arms` run may reuse an ALREADY-MEASURED deployed cell instead of paying to rerank
    # it again -- data/retr7_rr_dev_scores_year_bias.jsonl is exactly that, and it reproduces
    # the published dev 0.791/0.843 to four places. The cell is named differently there, so
    # the rename is explicit rather than guessed.
    if args.arm_baseline:
        merged = 0
        for qid, byc in load_ranking(args.arm_baseline, ALL_CELLS).items():
            if qid in ranked and args.arm_baseline_cell in byc:
                ranked[qid].setdefault(baseline_cell, byc[args.arm_baseline_cell])
                merged += 1
        print(f"merged {baseline_cell!r} for {merged} questions from {args.arm_baseline} "
              f"(cell {args.arm_baseline_cell!r})")
        print("  NOTE: that artifact predates the RETR-50 ef_search fix. The delta it "
              "produces carries that confound; a same-run baseline does not.")

    # Resolved AFTER loading, so a scores file that already carries unfiltered_raw is never
    # silently overwritten by the cached pre-RETR-7 one. An explicit --baseline still wins.
    baseline = args.baseline or (BASELINE_DEV if args.split == "dev" and args.table == "2x2" else None)
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

    per = {c: {k: [] for k in METRIC_KEYS} for c in cells}
    # Per-question recall@10, kept aligned across cells so the deltas can be PAIRED. A mean
    # of means cannot produce a CI or a McNemar count; only the per-question vectors can.
    paired: dict[str, list[float]] = {c: [] for c in cells}
    hit10: dict[str, list[bool]] = {c: [] for c in cells}
    subgroup: list[bool | None] = []
    n = 0
    # Counted, not silently skipped: a question dropped for a missing cell would otherwise
    # shrink the denominator invisibly and make cells incomparable.
    skipped_cells = skipped_gold = 0
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="scoring"):
        qid = row["id"]
        if qid not in ranked or any(c not in ranked[qid] for c in cells):
            skipped_cells += 1
            continue
        stem = _filing_stem(row)
        rel = [(stem, i) for i in gold_relevant_chunk_ids(row)]
        if not rel:
            skipped_gold += 1
            continue
        n += 1
        subgroup.append(_year_mismatch(row, stem))
        for c in cells:
            got = ranked[qid][c]
            r10 = recall_at_k(got, rel, 10)
            per[c]["recall_10"].append(r10)
            per[c]["recall_50"].append(recall_at_k(got, rel, 50))
            per[c]["ndcg_10"].append(ndcg_at_k(got, rel, 10))
            per[c]["mrr"].append(mrr(got, rel))
            paired[c].append(r10)
            hit10[c].append(r10 > 0)

    print(f"\nquestions scored: {n} of {len(rows)} {args.split}"
          f"   (skipped: {skipped_cells} missing a cell, {skipped_gold} with no gold label)\n")
    hdr = f"{'cell':<22}" + "".join(
        f"{k:>18}" for k in ("recall@10", "recall@50", "nDCG@10", "MRR"))
    print(hdr)
    print("-" * len(hdr))
    base, table = {}, {}
    for c in cells:
        printed = []
        for k in METRIC_KEYS:
            m, se = mean_and_stderr(per[c][k])
            table.setdefault(c, {})[k] = {"mean": m, "stderr": se}
            if c == baseline_cell:
                base[k] = m
                printed.append(f"{m:.3f} ± {se:.3f}".rjust(18))
            else:
                printed.append(f"{m:.3f} ({m - base[k]:+.3f})".rjust(18))
        print(f"{c:<22}" + "".join(printed))
    print(f"\n(baseline row = {baseline_cell}, shown with ± stderr; other rows show the delta)")

    # ── paired statistics, recall@10 against the baseline cell ──────────────────
    # A delta of means is not evidence on its own: the arms are scored on the SAME questions,
    # so the question-to-question spread has to be differenced away before the interval means
    # anything. Reported for every non-baseline cell, on both the continuous coverage metric
    # and the binary any-gold@10, because the two can disagree and the disagreement is
    # informative (`RETR-46`: one gold chunk answers as well as all of them).
    print(f"\nPAIRED vs {baseline_cell}, recall@10 (n={n})")
    ph = f"{'cell':<22}{'delta':>10}{'95% CI':>22}{'win':>7}{'lose':>6}{'McNemar p':>12}"
    print(ph); print("-" * len(ph))
    for c in cells:
        if c == baseline_cell:
            continue
        deltas = [a - b for a, b in zip(paired[c], paired[baseline_cell])]
        mean_d, lo, hi = paired_bootstrap_ci(deltas, n_boot=args.boot)
        n01 = sum(1 for a, b in zip(hit10[c], hit10[baseline_cell]) if a and not b)
        n10 = sum(1 for a, b in zip(hit10[c], hit10[baseline_cell]) if b and not a)
        pval = exact_mcnemar(n10, n01)
        table[c]["paired_recall_10"] = {"delta": mean_d, "ci_lo": lo, "ci_hi": hi,
                                        "win": n01, "lose": n10, "mcnemar_p": pval}
        print(f"{c:<22}{mean_d:>+10.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>22}"
              f"{n01:>7}{n10:>6}{pval:>12.4f}")

    # ── RETR-45 subgroup: does the arm take from one population to pay another? ──
    print(f"\nSUBGROUP recall@10 on the RETR-45 year axis "
          f"(names a year & gold filing is a DIFFERENT year, vs names the gold year)")
    sh = f"{'cell':<22}{'mismatch':>12}{'delta':>9}{'match':>10}{'delta':>9}{'no year':>10}"
    print(sh); print("-" * len(sh))
    idx = {"mismatch": [i for i, v in enumerate(subgroup) if v is True],
           "match": [i for i, v in enumerate(subgroup) if v is False],
           "no_year": [i for i, v in enumerate(subgroup) if v is None]}
    print(f"{'(n)':<22}{len(idx['mismatch']):>12}{'':>9}{len(idx['match']):>10}{'':>9}"
          f"{len(idx['no_year']):>10}")
    for c in cells:
        sub = {}
        for gname, ii in idx.items():
            sub[gname] = (sum(paired[c][i] for i in ii) / len(ii)) if ii else float("nan")
            if c != baseline_cell and ii:
                sub[gname + "_delta"] = sub[gname] - sum(paired[baseline_cell][i] for i in ii) / len(ii)
        table[c]["subgroup_recall_10"] = sub
        if c == baseline_cell:
            print(f"{c:<22}{sub['mismatch']:>12.3f}{'':>9}{sub['match']:>10.3f}{'':>9}"
                  f"{sub['no_year']:>10.3f}")
        else:
            print(f"{c:<22}{sub['mismatch']:>12.3f}{sub.get('mismatch_delta', 0):>+9.3f}"
                  f"{sub['match']:>10.3f}{sub.get('match_delta', 0):>+9.3f}"
                  f"{sub['no_year']:>10.3f}")
    print("\nAn arm positive in aggregate and negative on `mismatch` has NOT earned a ship "
          "(RETR-45; depth-200 alone fails exactly here).")

    # ── rerank wall time, so a GPU pass publishes its own cost ─────────────────
    lat = []
    with open(args.scores) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                if (rec.get("latency_s") is not None and rec.get("cells")
                        and rec.get("split") == args.split):
                    lat.append(rec["latency_s"] / max(1, len(rec["cells"])))
    # The cluster leg writes one latency averaged over a batch of 20, the local leg one per
    # question. Repeated values are the batch signature; p50 == p95 was not, and only ever
    # fired because percentile() was handed 50/95 instead of 0.5/0.95.
    batch_avg = len(set(lat)) < len(lat)
    if lat:
        lat.sort()
        p50, p95 = percentile(lat, 0.5), percentile(lat, 0.95)
        # sum(lat) is the PER-CELL series, so the total must be multiplied back up by the
        # number of cells or it under-reports the GPU bill by exactly that factor.
        print(f"\nrerank wall time per question per cell: p50 {p50:.3f}s  p95 {p95:.3f}s  "
              f"n={len(lat)}  (total {sum(lat) * len(cells) / 3600:.2f} GPU-h across "
              f"{len(cells)} cells)")
        if batch_avg:
            print("  NOTE: per-BATCH averages, not per-question timings -- a spread of batch "
                  "means, not a latency distribution; do not quote it as one.")

    # Stderr is stored for every cell, not just the baseline row that prints it: the deltas
    # are what get quoted, and a delta needs both cells' spread to be defensible.
    if args.out:
        args.out.write_text(json.dumps({
            "split": args.split, "table": args.table, "scores": str(args.scores),
            "baseline_cell": baseline_cell, "n_scored": n,
            "skipped_missing_cell": skipped_cells, "skipped_no_gold": skipped_gold,
            "subgroup_n": {k: len(v) for k, v in idx.items()},
            "rerank_s_per_question_per_cell": (
                {"p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95), "n": len(lat),
                 "batch_averaged": batch_avg}
                if lat else None),
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
    p.add_argument("--cells", default=",".join(CELLS),
                   help="cells to build pools for (default: the published 2x2). For the arms "
                        f"comparison use --cells {','.join(ARM_CELLS)}")
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
            "one process, writing the identical scores format.\n\n"
            f"RUNTIME, HONESTLY: rerank is ~{RERANK_S_LAPTOP}s per question PER CELL on an "
            f"M3 laptop (AGENT-24) and ~{RERANK_S_DEPLOY_HOST}s on the g4dn.xlarge GPU deploy "
            "host (T4, fp16 -- DEPLOY-21/DEPLOY-23). Four cells over the full 1235-question dev split is MANY HOURS on a "
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
    p.add_argument("--table", choices=("2x2", "arms"), default="2x2",
                   help="'2x2' = the published ablation (baseline row unfiltered_raw). "
                        "'arms' = deployed vs n18d vs n18d_p10, with paired CIs, McNemar "
                        "and the RETR-45 subgroup split (baseline row `deployed`)")
    p.add_argument("--arm-baseline", type=Path,
                   help="reuse an already-measured `deployed` cell from this scores file "
                        "instead of reranking it again (e.g. "
                        "data/retr7_rr_dev_scores_year_bias.jsonl, which reproduces the "
                        "published dev 0.791/0.843). Carries the pre-RETR-50 ef_search "
                        "confound -- a same-run baseline does not")
    p.add_argument("--arm-baseline-cell", default="filtered_stripped_year_bias",
                   help="cell name to read from --arm-baseline (default: %(default)s)")
    p.add_argument("--boot", type=int, default=10000,
                   help="bootstrap resamples for the paired CI (default: %(default)s)")
    p.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
