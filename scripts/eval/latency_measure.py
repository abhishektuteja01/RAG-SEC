"""Clean stage-latency re-measurement for spec.md:121's p50/p95 (and Day 13's gate).

WHY THIS EXISTS SEPARATELY FROM agent_analyze.py. The Day 9 run's `stage_latency` is not
publishable: `embed_s` p50 read 12.89s at n=60 against 0.33-0.57s in a quiet two-question
pilot, with `model_init_s` already excluded. Encoding one short query cannot get 20-40x
slower -- that is paging, not compute (swap grew 1.7 GB -> ~10 GB during the run on a 16 GiB
machine holding the Docker VM plus two transformer models in MPS unified memory). So the
published latency has to come from a pass that is the ONLY heavy thing on the machine.

Retrieval only -- no LLM call, no API spend, no gold scoring. It is the same `retrieve()`
the shipped arm uses, at its shipped defaults, so the stages are the ones spec.md names.

SAME QUESTIONS AS THE RUN, deliberately: identical split, seed and sampling as
`agent_run.py`, so the contaminated numbers and these are the same workload measured under
two machine states rather than two different question mixes.

MODEL CONSTRUCTION IS EXCLUDED the way retrieve() already excludes it -- `model_init_s`
lands on the first question only. That first question is dropped from the percentiles as an
explicit warm-up, and reported separately, because a p50 is a steady-state claim.

SWAP IS RECORDED NEXT TO THE NUMBERS, before and after, because the whole point of the
re-measurement is the machine state, and a latency without it is the thing being replaced
(SESSION.md housekeeping: diagnose with `sysctl vm.swapusage`, not `ps` RSS).

Usage:
    uv run scripts/eval/latency_measure.py -n 100
    uv run scripts/eval/latency_measure.py --report-only
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from tqdm import tqdm  # noqa: E402

from rag_sec.eval import load_matched_questions  # noqa: E402
from rag_sec.retrieve import last_call_stats, retrieve  # noqa: E402

SEED = 42  # agent_run.py's, so the question set matches
OUT = _ROOT / "data" / "day9_latency_clean.jsonl"


def meta_path(out: Path) -> Path:
    """Derived from --out, never a constant: a smoke test pointed at a scratch file would
    otherwise stamp its own n next to the real pass's data, which is this project's
    recurring bug class (RETR-24, AGENT-16) rebuilt in a new file."""
    return out.with_name(out.stem + "_meta.json")
STAGES = ["embed_s", "resolve_s", "search_s", "rerank_s", "total_s"]


def swap() -> dict:
    """(used_mb, free_mb) from vm.swapusage. Returns {} off darwin rather than guessing."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                             text=True, check=True).stdout
    except Exception:
        return {}
    # "total = 10240.00M  used = 9244.81M  free = 995.19M  (encrypted)"
    d = {}
    toks = out.replace("=", " ").split()
    for i, t in enumerate(toks):
        if t in ("total", "used", "free") and i + 1 < len(toks):
            d[t + "_mb"] = float(toks[i + 1].rstrip("M"))
    return d


def pctile(xs: list[float], p: float) -> float:
    """Nearest-rank, not interpolated: at n=100 the p95 is a real observed request, and an
    interpolated one is a number no query actually took."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(p / 100 * len(s) + 0.5)) - 1))
    return s[i]


def report(rows: list[dict], meta: dict) -> None:
    warm = [r for r in rows if not r["warmup"]]
    print(f"\nn={len(warm)} steady-state ({len(rows) - len(warm)} warm-up dropped)")
    if meta.get("swap_before") or meta.get("swap_after"):
        print(f"swap used: {meta.get('swap_before', {}).get('used_mb')} MB before -> "
              f"{meta.get('swap_after', {}).get('used_mb')} MB after")
    init = [r["timings"].get("model_init_s", 0.0) for r in rows if r["warmup"]]
    if init:
        print(f"model construction (excluded from every figure below): {init[0]:.2f}s")
    print(f"\n{'stage':<12} {'p50':>8} {'p95':>8} {'mean':>8} {'max':>8}")
    for s in STAGES:
        xs = [r["timings"][s] for r in warm if s in r["timings"]]
        if not xs:
            continue
        print(f"{s:<12} {pctile(xs, 50):>8.3f} {pctile(xs, 95):>8.3f} "
              f"{sum(xs) / len(xs):>8.3f} {max(xs):>8.3f}")
    # Lock waits are inside embed_s/rerank_s. At concurrency 1 they must be ~0; a non-zero
    # value means something else in this process is holding the GPU, which invalidates the pass.
    for k in ("embed_lock_wait_s", "rerank_lock_wait_s"):
        xs = [r["timings"].get(k, 0.0) for r in warm]
        if xs and max(xs) > 0.05:
            print(f"WARNING {k} max {max(xs):.3f}s -- not a quiet single-threaded pass")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        rows = [json.loads(l) for l in args.out.read_text().splitlines() if l.strip()]
        mp = meta_path(args.out)
        meta = json.loads(mp.read_text()) if mp.exists() else {}
        report(rows, meta)
        return 0

    df = load_matched_questions()
    picks = df[df["split"] == args.split].sample(frac=1.0, random_state=SEED).head(args.n)

    meta = {"swap_before": swap(), "started": datetime.now(timezone.utc).isoformat(),
            "machine": platform.platform(), "n": args.n, "split": args.split}
    rows = []
    with open(args.out, "w") as f:
        for i, (_, q) in enumerate(tqdm(list(picks.iterrows()), desc="retrieving")):
            t0 = time.perf_counter()
            retrieve(q["question"])
            stats = last_call_stats()
            rec = {"id": q["id"], "warmup": i == 0, "wall_s": time.perf_counter() - t0,
                   "timings": stats["timings"], "swap_used_mb": swap().get("used_mb")}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            rows.append(rec)
    meta["swap_after"] = swap()
    meta["finished"] = datetime.now(timezone.utc).isoformat()
    meta_path(args.out).write_text(json.dumps(meta, indent=2) + "\n")
    report(rows, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
