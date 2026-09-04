"""Stage 3 (laptop): sweep token budgets and measure how much gold evidence survives
compression. See DECISIONS.md COST-6/COST-7/COST-11.

Arms over an identical candidate pool:
  full    -- every top-10 chunk, no budget. The control: what agent.py sends today, and the
             reachable ceiling, since selection cannot recover gold the prompt never held.
  chunks  -- whole chunks in rerank order until the budget is spent. The baseline, and the
             one to beat: if it matches, ship it and drop the slice machinery entirely.
  slices  -- slices of the top-10 chunks only, in slice-score order. The cost play: same
             evidence retrieval found, packed finer.
  slices50 -- slices of all 50 fused candidates. The recall play: a slice from the chunk
             ranked 30th can outrank a slice from the chunk ranked 2nd, so this can surface
             evidence chunk-level reranking never returned.

`full` and `chunks` depend only on the chunk ordering, so they can be recomputed against a
new retrieval ordering for free. The slice arms cannot: they need per-slice cross-encoder
scores, which are tied to both the pool and the query text used to produce them (RETR-29).
Hence --arms, so a partial recompute states what it did rather than silently reusing stale
slice scores against a pool they were not measured on.

Metric is gold survival: does the compressed prompt still match this question's gold
evidence, judged by eval.py's own three-layer matcher (`_relevance_evidence`) rather than a
second matcher written here -- a divergent copy would let this sweep and the recall numbers
disagree for reasons unrelated to compression.

Caveat, and note the second half of it was WRONG (corrected by COST-25):
CLUSTER_SHARE_THRESHOLD was calibrated against whole chunks, so absolute survival numbers
carry that assumption. This used to add "the comparison *between* arms is unaffected, since
all three are judged alike." It is not unaffected. The matcher fires on a prose gold_ind
sentence or a partial cluster, and a compressed prompt is exactly the shape that keeps a
table's caption and drops its rows -- so the loose criterion flatters the compressed arm
about twice as much (6.7 pts on the control, 15.6 on slices@1500). Judge compression on
every-gold-figure survival (69.3% @1500), not on matcher survival (84.9%).

Usage:
    python scripts/compression/slice_budget_sweep.py --arms full,chunks
    python scripts/compression/slice_budget_sweep.py \
        --arms full,chunks,slices,slices50 \
        --scores data/day8_slice_scores_t150_filtered_stripped.jsonl \
        --out data/day8_slice_budget_dev_results.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.chunking import count_tokens  # noqa: E402
from rag_sec.compress import chunk_atoms, pack_by_score, slice_atom  # noqa: E402
from rag_sec.eval import (  # noqa: E402
    _gold_evidence_resolved,
    _relevance_evidence,
    load_matched_questions,
)

BUDGETS = (500, 1000, 1500, 2000, 3000)
UNCOMPRESSED = 10**9  # control: every retrieved chunk, what agent.py sends today
CHUNK_SCORES = Path("data/day8_retr16v2_dev_scores.jsonl")  # RETR-29 ordering
CELL = "filtered_stripped"
BUDGETED_ARMS = ("chunks", "slices", "slices50")
SLICE_ARMS = ("slices", "slices50")
TOP_K = 10


def _survives(text: str, resolved, context: str) -> bool:
    return bool(_relevance_evidence(resolved, context, [(0, text)], []))


def _load_chunk_order(path: Path, cell: str) -> dict[str, list[tuple[str, int]]]:
    """Two on-disk shapes. The old Arm 3/4 file holds one already-sorted ranking per
    question under `reranked`; the RETR-16 file holds several cells per question, each an
    unsorted [stem, idx, score] list, so the cell has to be named and sorted here."""
    order: dict[str, list[tuple[str, int]]] = {}
    cells_seen: set[str] = set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "reranked" in rec:
                order[rec["id"]] = [(s, i) for s, i, _v in rec["reranked"]]
            else:
                cells_seen |= set(rec["cells"])
                if cell in rec["cells"]:
                    order[rec["id"]] = [
                        (s, i) for s, i, _sc in sorted(rec["cells"][cell], key=lambda x: -x[2])
                    ]
    if not order:
        raise SystemExit(f"no rankings in {path} for cell {cell!r}; file has {sorted(cells_seen)}")
    return order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", help="day8_slice_scores_*.jsonl; required for the slice arms")
    ap.add_argument("--chunk-scores", type=Path, default=CHUNK_SCORES)
    ap.add_argument("--cell", default=CELL, help="ignored for single-ranking score files")
    ap.add_argument("--arms", default="full," + ",".join(BUDGETED_ARMS))
    ap.add_argument("--n", type=int)
    # Per-question flags, not just the marginals: McNemar's power on COST-13 comes from
    # discordant pairs, which the aggregate survival rates cannot be recovered from.
    ap.add_argument("--dump", type=Path, help="write per-question survival flags as JSON")
    # Aggregate table to disk, alongside the printed one. --dump holds the per-question
    # flags COST-13 needs; this holds the marginals COST-11/COST-18 quote, so the numbers in
    # DECISIONS.md have a file behind them like every earlier arm does.
    ap.add_argument("--out", type=Path, help="write the survival table to JSON as well as printing it")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - {"full", *BUDGETED_ARMS}
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}")
    want_slices = bool(set(arms) & set(SLICE_ARMS))
    if want_slices and not args.scores:
        raise SystemExit(f"--scores is required for {sorted(set(arms) & set(SLICE_ARMS))}")

    slice_scores: dict[str, list] = {}
    target = None
    if want_slices:
        targets = set()
        truncated = 0
        with open(args.scores) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Tolerated so this can run against a partial file copied while the HPC
                    # job is still writing -- only ever the final line, and only ever one.
                    truncated += 1
                    continue
                slice_scores[rec["id"]] = rec["scores"]
                targets.add(rec.get("target"))
        if truncated > 1:
            raise SystemExit(f"{truncated} unparseable lines -- corruption, not a partial copy")
        print(f"loaded {len(slice_scores)} scored questions" + (" (partial file)" if truncated else ""))
        if targets != {t for t in targets if isinstance(t, int)} or len(targets) != 1:
            raise SystemExit(f"scores file must carry exactly one integer slice target, got {targets}")
        target = targets.pop()
        # Slices are stored as positions; resolving them means re-running the same slicing.
        # At any other target the positions point at different text, silently.
        print(f"Slice target from scores file: {target}")

    chunk_order = _load_chunk_order(args.chunk_scores, args.cell)
    print(f"chunk ordering: {args.chunk_scores} cell={args.cell} ({len(chunk_order)} questions)")

    df = load_matched_questions()
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    if args.n:
        dev = dev.head(args.n)
    resolved_all = _gold_evidence_resolved()

    hits: defaultdict = defaultdict(int)
    toks: defaultdict = defaultdict(int)
    flags: dict[str, dict[str, bool]] = {}
    n = 0

    for _, row in tqdm(dev.iterrows(), total=len(dev)):
        qid = row["id"]
        if qid not in chunk_order or (want_slices and qid not in slice_scores):
            continue
        n += 1
        resolved, context = resolved_all.get(qid), row["context"]

        chunk_units = []
        for stem, idx in chunk_order[qid][:TOP_K]:
            try:
                chunk, _ = chunk_atoms(stem, idx)
            except (FileNotFoundError, IndexError):
                continue
            chunk_units.append((chunk.text, chunk.n_tokens))

        slice_units_all, slice_units_top = [], []
        if want_slices:
            top_chunks = set(chunk_order[qid][:TOP_K])
            scored = sorted(slice_scores[qid], key=lambda s: s[4], reverse=True)
            cache: dict[tuple, dict[tuple[int, int], str]] = {}

            def _text(stem, idx, atom_i, piece_i):
                """Slice text by (atom, piece) position, built once per chunk. The HPC stage
                emits positions rather than text, so the payload stays one copy of the corpus
                instead of two -- this rebuilds the same slicing deterministically."""
                key = (stem, idx)
                if key not in cache:
                    try:
                        _, atoms = chunk_atoms(stem, idx)
                    except (FileNotFoundError, IndexError):
                        cache[key] = {}
                    else:
                        cache[key] = {
                            (ai, pi): piece
                            for ai, atom in enumerate(atoms)
                            for pi, piece in enumerate(slice_atom(atom, target))
                        }
                return cache[key].get((atom_i, piece_i))

            for stem, idx, atom_i, piece_i, _score in scored:
                text = _text(stem, idx, atom_i, piece_i)
                if text is None:
                    continue
                unit = (text, count_tokens(text))
                slice_units_all.append(unit)
                if (stem, idx) in top_chunks:
                    slice_units_top.append(unit)

        # Control arm: all TOP_K chunks, no budget. Without it the table only says how the
        # arms rank against each other, not what compression actually costs against today's
        # behaviour.
        row_flags: dict[str, bool] = {}
        if "full" in arms:
            full = "\n\n".join(t for t, _ in chunk_units)
            toks[("full", UNCOMPRESSED)] += count_tokens(full)
            survived = _survives(full, resolved, context)
            hits[("full", UNCOMPRESSED)] += survived
            row_flags["full"] = bool(survived)

        by_arm = {"chunks": chunk_units, "slices": slice_units_top, "slices50": slice_units_all}
        for budget in BUDGETS:
            for arm in BUDGETED_ARMS:
                if arm not in arms:
                    continue
                kept = pack_by_score(by_arm[arm], budget)
                text = "\n\n".join(kept)
                toks[(arm, budget)] += count_tokens(text)
                survived = _survives(text, resolved, context)
                hits[(arm, budget)] += survived
                row_flags[f"{arm}_{budget}"] = bool(survived)
        flags[qid] = row_flags

    if args.dump:
        args.dump.write_text(json.dumps(flags))
        print(f"wrote per-question survival flags for {len(flags)} questions -> {args.dump}")

    cols = [a for a in BUDGETED_ARMS if a in arms]
    tail = f"; slice target {target}" if want_slices else ""
    print(f"\nquestions scored: {n}   ({args.chunk_scores.name}:{args.cell}{tail})")
    if "full" in arms:
        print(
            f"{'FULL':>7}  "
            + f"{100 * hits[('full', UNCOMPRESSED)] / n:5.1f}% @ {toks[('full', UNCOMPRESSED)] / n:6.0f} tok".rjust(22)
            + "   <- uncompressed control"
        )
    if cols:
        print(f"{'budget':>7}  " + "  ".join(f"{a:>22}" for a in cols))
        for budget in BUDGETS:
            cells = [
                f"{100 * hits[(arm, budget)] / n:5.1f}% @ {toks[(arm, budget)] / n:6.0f} tok"
                for arm in cols
            ]
            print(f"{budget:>7}  " + "  ".join(f"{c:>22}" for c in cells))

    if args.out:
        table: dict[str, dict[str, dict[str, float]]] = {}
        for arm, budget in hits:
            key = "uncompressed" if budget == UNCOMPRESSED else str(budget)
            table.setdefault(arm, {})[key] = {
                "survival": hits[(arm, budget)] / n,
                "mean_tokens": toks[(arm, budget)] / n,
            }
        args.out.write_text(json.dumps({
            "chunk_scores": str(args.chunk_scores), "cell": args.cell,
            "slice_scores": str(args.scores) if args.scores else None,
            "slice_target": target, "top_k": TOP_K, "budgets": list(BUDGETS),
            "arms": arms, "n_scored": n, "survival": table,
        }, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
