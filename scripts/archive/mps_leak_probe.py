"""Tests WHY retrieval latency degrades with question count, on an otherwise idle machine.

THE OBSERVATION THIS EXISTS TO EXPLAIN. `latency_measure.py`, running alone, watched
`embed_s` go 0.138s -> 13.2s and `rerank_s` 24.9s -> 49.1s over eleven questions while swap
climbed 2.3 GB -> 9.6 GB, monotonically. Nothing else heavy was running. That kills the
standing explanation (AGENT-17), which blamed contention with the
Docker VM and a second competing process: the contention is absent here and the curve is not.

So the hypothesis under test is INTRA-PROCESS: something inside retrieve() accumulates across
calls. `ps` RSS cannot see it -- it reported 0.1 GB against 7 GB of swap growth -- because MPS
allocations live in unified memory the process does not account for. `torch.mps.
current_allocated_memory()` can, which is why it is sampled here and RSS is only kept as the
contrast that makes the point.

WHAT EACH ARM ISOLATES, same N questions, same order, one process each so no arm inherits
another's heap:
    none          the status quo -- the shape latency_measure.py already produced
    empty_cache   torch.mps.empty_cache() after every question. If this flattens the curve,
                  the growth is MPS's caching allocator holding freed blocks, and the fix is
                  one line in retrieve() rather than a machine to run on.
    gc            gc.collect() only. Separates "Python objects are still referenced" from
                  "MPS is hoarding freed blocks" -- different bugs, different fixes.

Deliberately NOT a fix. It edits nothing in src/; it measures whether a fix would work, so
the decision to change retrieve() is made on a curve rather than on a plausible story.

Usage:
    uv run scripts/archive/mps_leak_probe.py --mode none -n 25
    uv run scripts/archive/mps_leak_probe.py --mode empty_cache -n 25
"""

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import torch  # noqa: E402

from rag_sec.eval import load_matched_questions  # noqa: E402
from rag_sec.retrieve import last_call_stats, retrieve  # noqa: E402

SEED = 42


def swap_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                         text=True).stdout.replace("=", " ").split()
    for i, t in enumerate(out):
        if t == "used":
            return float(out[i + 1].rstrip("M"))
    return float("nan")


def mps_mb() -> float:
    """What ps cannot see. 0.0 rather than an exception when MPS is absent (CI, Linux)."""
    try:
        return torch.mps.current_allocated_memory() / 1e6
    except Exception:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["none", "empty_cache", "gc"], default="none")
    ap.add_argument("-n", type=int, default=25)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = load_matched_questions()
    picks = df[df["split"] == "dev"].sample(frac=1.0, random_state=SEED).head(args.n)
    out = args.out or _ROOT / "data" / f"mps_leak_{args.mode}.jsonl"

    print(f"mode={args.mode} n={args.n}")
    print(f"{'i':>3} {'embed_s':>8} {'rerank_s':>9} {'mps_mb':>9} {'swap_mb':>9}")
    with open(out, "w") as f:
        for i, (_, q) in enumerate(picks.iterrows()):
            t0 = time.perf_counter()
            retrieve(q["question"])
            t = last_call_stats()["timings"]
            if args.mode == "empty_cache":
                torch.mps.empty_cache()
            elif args.mode == "gc":
                gc.collect()
            rec = {"i": i, "id": q["id"], "mode": args.mode, "wall_s": time.perf_counter() - t0,
                   "embed_s": t["embed_s"], "rerank_s": t["rerank_s"],
                   "mps_mb": mps_mb(), "swap_mb": swap_mb()}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{i:>3} {rec['embed_s']:>8.3f} {rec['rerank_s']:>9.2f} "
                  f"{rec['mps_mb']:>9.1f} {rec['swap_mb']:>9.1f}")
    # First-vs-last thirds, not a slope: the growth may be a step rather than a line, and a
    # regression coefficient would report a number either way.
    rows = [json.loads(l) for l in open(out)]
    k = max(1, len(rows) // 3)
    for key in ("embed_s", "rerank_s", "mps_mb", "swap_mb"):
        a = sum(r[key] for r in rows[:k]) / k
        b = sum(r[key] for r in rows[-k:]) / k
        print(f"{key:<10} first-third {a:>9.3f}  last-third {b:>9.3f}  "
              f"{'x%.1f' % (b / a) if a else 'n/a'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
