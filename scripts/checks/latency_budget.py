"""Fails when a stored latency file's p95 is over the `/ask` budget (DECISIONS.md DEPLOY-27).

It gates STORED timings, not live traffic. Re-measuring the live host sends every question to
Gemini, so a fresh file costs money and is taken by hand; this check is how that file is then
judged, the same way as the committed one:

    python scripts/checks/latency_budget.py [FILE] [--config NAME]

FILE defaults to the post-flip DEPLOY-25 run: `/ask` end to end on the fixed ten test
questions, inside the deployed container on the g4dn.xlarge (T4). Budgets, in seconds, are the
measured p95 on that host times 1.10, rounded UP to the next 0.5s:

    end_to_end   8.5   /ask latency_s     7.484 p95, n=10   (ask_r51r52_on_10q)
    retrieval    5.0   total_s            4.285 p95, n=110  (flags_3config_110q, serving config)
    rerank       4.5   rerank_s           3.855 p95, n=110  (same)
    generation   4.0   generation_s       3.253 p95, n=10   (ask_r51r52_on_10q)

At n=10 a nearest-rank p95 IS the maximum, so the n=10 legs gate the worst question, not a
tail estimate. The 10% margin is a round default, not a measured one; its only check is that
the pre-flip repeat of the same ten (ask_baseline_year_bias_10q) still passes every leg.

Two file shapes are recognised; anything else is refused, not passed. The same key
`latency_s` means a batch's wall time split across questions in the rerank score files, so a
key match alone proves nothing (INFRA-22):
  ask        {id, latency_s, generation_s, stage_latency: {total_s, rerank_s, ...}}
             -- what api.py returns, plus the harness's generation_s
  retrieval  {id, config, timings: {total_s, rerank_s, ...}} and optional {"_meta": true}
             rows -- retrieve()'s own timings; no generation, no end to end. A file holding
             more than one config needs --config, or three configs would be pooled into one p95

Every run also runs a negative control: the file's own rows doubled must fail every leg it
gates, and a 100-value series with 5 values over budget must fail while one with 4 over must
pass -- which a percentile passed 95 instead of 0.95 gets wrong, as it returns the maximum.

No database, no model, no network.
"""

import argparse
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec.eval import percentile  # noqa: E402

DEFAULT = _ROOT / "research_latency" / "deploy25_ask_r51r52_on_10q.jsonl"
BUDGET_S = {"end_to_end": 8.5, "retrieval": 5.0, "rerank": 4.5, "generation": 4.0}
MIN_N = 10  # the fixed-ten harness (DEPLOY-19); fewer and p95 is one or two questions
Q = 0.95  # a FRACTION -- percentile() indexes int(q * n)


class ShapeError(ValueError):
    pass


def _num(row: dict, key: str, where: str) -> float:
    v = row.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise ShapeError(f"{where}: {key}={v!r} is not a finite non-negative number")
    return float(v)


def load(path: Path, config: str | None) -> tuple[str, dict[str, list[float]]]:
    """Returns (shape, {leg: values}). Raises ShapeError on anything unrecognised."""
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    rows = [r for r in rows if not (isinstance(r, dict) and r.get("_meta") is True)]
    if not rows or not all(isinstance(r, dict) for r in rows):
        raise ShapeError("no rows, or a row that is not a JSON object")

    if all("stage_latency" in r for r in rows):
        shape = "ask"
    elif all("timings" in r and "config" in r for r in rows):
        shape = "retrieval"
        configs = sorted({r["config"] for r in rows})
        if config is None and len(configs) > 1:
            raise ShapeError(f"{len(configs)} configs in one file {configs}; pick one "
                             "with --config")
        if config is not None:
            if config not in configs:
                raise ShapeError(f"--config {config!r} not in {configs}")
            rows = [r for r in rows if r["config"] == config]
    else:
        raise ShapeError("rows are neither the /ask shape (stage_latency) nor the retrieval "
                         "shape (config + timings), or the file mixes the two")
    if config is not None and shape == "ask":
        raise ShapeError("--config only applies to a retrieval-shape file")

    ids = [r.get("id") for r in rows]
    if None in ids or len(set(ids)) != len(ids):
        raise ShapeError("missing or repeated question ids -- two runs concatenated would "
                         "double n without adding a question")
    if len(rows) < MIN_N:
        raise ShapeError(f"n={len(rows)} is under the {MIN_N}-question minimum")

    legs: dict[str, list[float]] = {k: [] for k in BUDGET_S}
    for r in rows:
        where = f"id {r['id']}"
        t = r["stage_latency"] if shape == "ask" else r["timings"]
        if not isinstance(t, dict):
            raise ShapeError(f"{where}: stage timings are not an object")
        total, rerank = _num(t, "total_s", where), _num(t, "rerank_s", where)
        if rerank > total:
            raise ShapeError(f"{where}: rerank_s {rerank} exceeds retrieval total_s {total}")
        legs["retrieval"].append(total)
        legs["rerank"].append(rerank)
        if shape == "ask":
            e2e, gen = _num(r, "latency_s", where), _num(r, "generation_s", where)
            # A part larger than the whole means latency_s is not the round trip -- the
            # rerank score files' split-batch latency_s would land here.
            if total > e2e or gen > e2e:
                raise ShapeError(f"{where}: a stage exceeds latency_s {e2e}")
            legs["end_to_end"].append(e2e)
            legs["generation"].append(gen)
    return shape, {k: v for k, v in legs.items() if v}


def over_budget(legs: dict[str, list[float]]) -> list[str]:
    return [f"{k}: p95 {percentile(v, Q):.3f}s > budget {BUDGET_S[k]:.1f}s (n={len(v)})"
            for k, v in legs.items() if percentile(v, Q) > BUDGET_S[k]]


def negative_control(legs: dict[str, list[float]]) -> list[str]:
    """Problems with the check itself; empty means it can see a regression."""
    problems = []
    doubled = {k: [2 * x for x in v] for k, v in legs.items()}
    caught = {f.split(":")[0] for f in over_budget(doubled)}
    if caught != set(legs):
        problems.append(f"rows doubled were not caught on {sorted(set(legs) - caught)}")
    b = BUDGET_S["end_to_end"]
    for n_over, must_fail in ((5, True), (4, False)):
        series = {"end_to_end": [b * 0.5] * (100 - n_over) + [b * 1.5] * n_over}
        if bool(over_budget(series)) != must_fail:
            problems.append(f"100 values with {n_over} over budget "
                            f"{'passed' if must_fail else 'failed'}; p95 should be the 96th")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", type=Path, default=DEFAULT)
    ap.add_argument("--config", help="retrieval-shape files only: which config to gate")
    args = ap.parse_args()

    try:
        shape, legs = load(args.file, args.config)
    except (ShapeError, json.JSONDecodeError, OSError) as exc:
        print(f"error: {args.file.name}: {exc}", file=sys.stderr)
        return 1

    n = len(next(iter(legs.values())))
    print(f"{args.file.name}: {shape} shape, n={n}"
          + (f", config {args.config}" if args.config else ""))
    if n < 20:
        print(f"note: at n={n} nearest-rank p95 is the maximum")
    print(f"{'leg':<11} {'p50':>7} {'p95':>7} {'budget':>7}")
    for k, v in legs.items():
        p95 = percentile(v, Q)
        flag = "  FAIL" if p95 > BUDGET_S[k] else ""
        print(f"{k:<11} {percentile(v, 0.5):>7.3f} {p95:>7.3f} {BUDGET_S[k]:>7.1f}{flag}")

    problems = negative_control(legs)
    if problems:
        print("negative control did not fire -- the check cannot see a regression:",
              file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("negative control: doubled rows fail every leg; 5/100 over fails, 4/100 passes")

    failures = over_budget(legs)
    if failures:
        print("\nOVER BUDGET (DECISIONS.md DEPLOY-27):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"\nok: every p95 within budget ({len(legs)} legs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
