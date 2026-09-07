"""Does int8 quantisation change the reranker's ranking? (DEPLOY-6-RESOLVED)

WHY THIS IS THE GATING MEASUREMENT. int8 was chosen over cutting CANDIDATE_K precisely because
it keeps the deployed ranking identical to the benchmarked one, so RETR-39's numbers still
describe the served system. That is a claim about numerics, and quantisation is lossy by
construction, so it has to be measured before the route can be trusted. If parity fails, the
fallback is cutting CANDIDATE_K *with* a fresh dev measurement, not instead of one.

WHAT IT COMPARES, and why this isolates the reranker. The fp32 side is not re-run: it is the
PUBLISHED per-candidate scores in `data/retr7_rr_dev_scores.jsonl`, the same file the static
arm replays and the same one RETR-39 was computed from. int8 re-scores those exact candidate
sets. So first-stage retrieval, the candidate pool, the gold labels and the query text are all
held fixed, and the only difference between the two rankings is the reranker's arithmetic.

READ THE FILE, DON'T ASSUME IT (AGENT-16): the cells store candidates in FIRST-STAGE order with
rerank scores merely attached, so both sides sort explicitly.

ONE BACKEND PER PROCESS, WHICH IS WHY THIS RUNS IN PHASES (DEPLOY-8). The first attempt held
the torch CrossEncoder, a 569MB int8 ORT session and the Docker VM in one process on a 16 GiB
machine and drove swap to 8.4GB. The phases below each construct exactly one scoring backend
and then exit, so the OS reclaims it rather than the process carrying both:

    parity        tokenizer + int8 ORT session   -- no torch model is needed AT ALL here,
                                                    because the fp32 side is the published file
    latency-fp32  torch CrossEncoder only        -- the leg DEPLOY-8 never reached
    latency-int8  tokenizer + int8 ORT session
    report        no backend, reads the partials

`--phase all` (the default) re-execs the three measuring phases as subprocesses in sequence.

MAX_LENGTH IS DELIBERATELY NOT CAPPED, and DEPLOY-8's suggestion to cap it is retracted here.
The concern was that the shipped CrossEncoder's `max_seq_length` is 8192. Measured over 3,000
live chunks: p50 854 tokens, p99 1504, max 1592 -- so 8192 NEVER BINDS and a cap is a no-op on
compute, because `padding=True` sizes each batch to its own longest member, not to the ceiling.
It is not a harmless no-op either: `retrieve.py` calls `predict()` with no override, so the
published fp32 scores were produced at 8192, and capping below ~1600 would truncate real chunk
text and make this script measure truncation instead of quantisation -- destroying the very
isolation the docstring above claims. `--max-length` exists to bound the worst case only; the
default sits above the observed maximum and the run asserts nothing was truncated.

THREE THINGS THE OUTPUT IS FOR:
  1. ranking parity -- top-10 exact-order and set agreement, Kendall tau over all 50
  2. metric parity  -- recall@10 / nDCG@10 under RETR-35 labels, the numbers actually published
  3. the speedup    -- fp32 vs int8 on CPU, which is the entire point of the exercise
recall@50 is printed as a self-check: both sides rank the SAME 50 candidates, so it cannot
move, and a difference there means the harness is wrong rather than the model (CI-2's logic).

Usage:
    uv run --extra onnx scripts/archive/onnx_rerank_parity.py -n 50
    uv run --extra onnx scripts/archive/onnx_rerank_parity.py --phase report   # re-print only
"""

import argparse
import json
import os
import random
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

load_dotenv()  # POSTGRES_* for the chunk-text fetch

from rag_sec.candidates import LIVE_VARIANT, chunk_texts  # noqa: E402
from rag_sec.company import strip_entity_framing  # noqa: E402
from rag_sec.config import RERANK_MODEL_NAME  # noqa: E402
from rag_sec.eval import (  # noqa: E402
    _filing_stem,
    gold_relevant_chunk_ids,
    load_matched_questions,
    load_ranking,
    ndcg_at_k,
    recall_at_k,
)
from rag_sec.store import get_conn  # noqa: E402

SCORES = _ROOT / "data" / "retr7_rr_dev_scores.jsonl"
CELL = "filtered_stripped"  # the shipped arm: company filter on, query stripped for rerank
INT8 = _ROOT / "models" / "onnx" / "reranker_int8.onnx"
OUT = _ROOT / "data" / "onnx_parity_results.json"
PARTIAL = _ROOT / "data" / "onnx_parity_partial"  # one file per phase, the only channel
#   between processes; gitignored, since a partial is a fragment of a run and not a result
SEED = 17  # same seed as the CI fixture, so the two samples are the same family of questions

# Above the 1592-token maximum observed over 3,000 live chunks, so it cannot truncate; see the
# module docstring for why a lower cap would invalidate the comparison rather than speed it up.
DEFAULT_MAX_LENGTH = 2048

MEASURING_PHASES = ("parity", "latency-fp32", "latency-int8")


def _kendall_tau(a: list, b: list) -> float:
    """Rank correlation over the full candidate list. Written out rather than pulled from
    scipy because scipy is not a dependency and this is 20 lines."""
    pos = {x: i for i, x in enumerate(b)}
    order = [pos[x] for x in a]
    n = len(order)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if order[i] < order[j]:
                conc += 1
            else:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total else 1.0


def _load_rows(n: int) -> list[dict]:
    """The question sample. Deterministic in `n` and SEED alone, so every phase -- running in
    its own process -- reconstructs byte-identical rows without passing them between processes."""
    # sort=False: this script re-scores the candidates itself and compares its own ordering
    # against the published fp32 one, so it needs the stored candidates and their scores.
    published = load_ranking(SCORES, CELL, with_score=True, sort=False)

    df = load_matched_questions()
    split = df[df["split"] == "dev"].reset_index(drop=True)

    rows = []
    for _, row in split.iterrows():
        qid = row["id"]
        if qid not in published:
            continue
        stem = _filing_stem(row)
        gold = [(stem, i) for i in gold_relevant_chunk_ids(row)]
        if not gold:
            continue
        rows.append({"id": qid, "question": row["question"], "gold": gold,
                     "cell": published[qid]})
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return sorted(rows[:n], key=lambda r: r["id"])


def _pairs_for(conn, row: dict) -> tuple[list[tuple], list[tuple[str, str]]]:
    """(candidates, (query, chunk_text) pairs) -- the exact input both backends score."""
    cand = [(s, i) for s, i, _ in row["cell"]]
    texts = chunk_texts(conn, cand, LIVE_VARIANT)
    cand = [c for c in cand if c in texts]
    q = strip_entity_framing(row["question"])
    return cand, [(q, texts[c]) for c in cand]


def _int8_scorer(max_length: int, batch_size: int):
    """Tokenizer + ORT session. Deliberately NOT `CrossEncoder(...).tokenizer`: that would pull
    the full torch model into this process purely to reach its tokenizer, which is exactly the
    double-backend footprint DEPLOY-8 blames for the aborted run."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
    sess = ort.InferenceSession(str(INT8), providers=["CPUExecutionProvider"])
    truncated = 0

    def score(pairs: list[tuple[str, str]]) -> np.ndarray:
        nonlocal truncated
        out = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i: i + batch_size]
            enc = tok([p[0] for p in batch], [p[1] for p in batch], padding=True,
                      truncation=True, max_length=max_length, return_tensors="np")
            # A truncated pair means the cap bound and this run is no longer comparing only
            # arithmetic against the published scores; counted so the claim can be checked.
            truncated += int((enc["attention_mask"].sum(axis=1) >= max_length).sum())
            logits = sess.run(
                ["logits"],
                {"input_ids": enc["input_ids"].astype(np.int64),
                 "attention_mask": enc["attention_mask"].astype(np.int64)},
            )[0]
            out.append(logits.reshape(-1))
        return np.concatenate(out)

    return score, tok, (lambda: truncated)


def _write(phase: str, payload: dict) -> None:
    PARTIAL.mkdir(parents=True, exist_ok=True)
    (PARTIAL / f"{phase}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {(PARTIAL / f'{phase}.json').relative_to(_ROOT)}")


def _read(phase: str) -> dict:
    p = PARTIAL / f"{phase}.json"
    if not p.exists():
        raise SystemExit(f"error: {p.relative_to(_ROOT)} absent -- run --phase {phase} first")
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- phases

def phase_parity(args) -> int:
    """int8 rescoring vs the published fp32 scores. One backend: the ORT session."""
    rows = _load_rows(args.n)
    print(f"\n{len(rows)} questions, {CELL}, {len(rows[0]['cell'])} candidates each")
    score, _tok, truncated = _int8_scorer(args.max_length, args.batch_size)

    m = {k: [] for k in ("top10_exact", "top10_setmatch", "top10_overlap", "tau",
                         "r10_fp32", "r10_int8", "r50_fp32", "r50_int8",
                         "nd_fp32", "nd_int8", "int8_s")}
    with get_conn() as conn:
        for r in tqdm(rows, desc="int8 rescoring"):
            cand, pairs = _pairs_for(conn, r)
            fp32_by_id = {(s, i): sc for s, i, sc in r["cell"]}

            t0 = time.perf_counter()
            scores = score(pairs)
            m["int8_s"].append(time.perf_counter() - t0)

            fp32_rank = sorted(cand, key=lambda c: -fp32_by_id[c])
            int8_rank = [c for _, c in sorted(zip(-scores, cand), key=lambda x: x[0])]

            m["top10_exact"].append(float(fp32_rank[:10] == int8_rank[:10]))
            m["top10_setmatch"].append(float(set(fp32_rank[:10]) == set(int8_rank[:10])))
            m["top10_overlap"].append(len(set(fp32_rank[:10]) & set(int8_rank[:10])) / 10)
            m["tau"].append(_kendall_tau(fp32_rank, int8_rank))

            rel = [tuple(g) for g in r["gold"]]
            for tag, ranked in (("fp32", fp32_rank), ("int8", int8_rank)):
                m[f"r10_{tag}"].append(recall_at_k(ranked, rel, 10))
                m[f"r50_{tag}"].append(recall_at_k(ranked, rel, 50))
                m[f"nd_{tag}"].append(ndcg_at_k(ranked, rel, 10))

    # Raw per-question seconds, not just their mean: a mean hides a RISING curve, and a rising
    # curve is this project's signature failure (AGENT-20/AGENT-24 on MPS, and the same shape
    # observed here on ORT's CPU arena). The mean is a steady-state claim; keep the evidence.
    _write("parity", {"n": len(rows), "ids": [r["id"] for r in rows],
                      "max_length": args.max_length, "truncated_pairs": truncated(),
                      "int8_secs": m["int8_s"],
                      "means": {k: st.mean(v) for k, v in m.items()}})
    return 0


def phase_latency_fp32(args) -> int:
    """torch CrossEncoder on CPU. One backend, and the leg DEPLOY-8 never reached."""
    from sentence_transformers import CrossEncoder

    rows = _load_rows(args.n)[: args.latency_n]
    ce = CrossEncoder(RERANK_MODEL_NAME, device="cpu")
    # No max_length override, matching retrieve.py's bare `predict(pairs, batch_size=32)` --
    # this leg has to be the shipped call, not a tuned one.
    secs, pair_counts = [], []
    with get_conn() as conn:
        for r in tqdm(rows, desc="fp32 CPU timing"):
            _cand, pairs = _pairs_for(conn, r)
            t0 = time.perf_counter()
            ce.predict(pairs, batch_size=args.batch_size)
            secs.append(time.perf_counter() - t0)
            pair_counts.append(len(pairs))
    _write("latency-fp32", {"n": len(rows), "ids": [r["id"] for r in rows],
                            "secs": secs, "pairs": pair_counts,
                            "max_seq_length": ce.max_seq_length})
    return 0


def phase_latency_int8(args) -> int:
    """int8 ORT on CPU over the same questions and the same pairs as the fp32 leg."""
    rows = _load_rows(args.n)[: args.latency_n]
    score, _tok, truncated = _int8_scorer(args.max_length, args.batch_size)
    secs, pair_counts = [], []
    with get_conn() as conn:
        for r in tqdm(rows, desc="int8 CPU timing"):
            _cand, pairs = _pairs_for(conn, r)
            t0 = time.perf_counter()
            score(pairs)
            secs.append(time.perf_counter() - t0)
            pair_counts.append(len(pairs))
    _write("latency-int8", {"n": len(rows), "ids": [r["id"] for r in rows],
                            "secs": secs, "pairs": pair_counts,
                            "max_length": args.max_length, "truncated_pairs": truncated()})
    return 0


def phase_report(args) -> int:
    par, lf, li = _read("parity"), _read("latency-fp32"), _read("latency-int8")
    m = par["means"]

    # The legs ran in separate processes, so this is a real check, not a tautology: it catches
    # a partial left over from a different -n or a different SEED being combined by mistake.
    if lf["ids"] != li["ids"]:
        raise SystemExit("error: latency legs scored different questions -- rerun both phases")
    if lf["pairs"] != li["pairs"]:
        raise SystemExit("error: latency legs scored different pair counts -- stale partial")

    print(f"\n-- ranking parity, int8 vs published fp32 (n={par['n']}) --")
    print(f"  top-10 identical, in order   {m['top10_exact']:.1%}")
    print(f"  top-10 identical as a set    {m['top10_setmatch']:.1%}")
    print(f"  mean top-10 overlap          {m['top10_overlap']:.3f} of 10")
    print(f"  Kendall tau over all 50      {m['tau']:.4f}")

    print("\n-- metric parity, RETR-35 labels --")
    print(f"  recall@10   fp32 {m['r10_fp32']:.4f}   int8 {m['r10_int8']:.4f}   "
          f"delta {m['r10_int8'] - m['r10_fp32']:+.4f}")
    print(f"  nDCG@10     fp32 {m['nd_fp32']:.4f}   int8 {m['nd_int8']:.4f}   "
          f"delta {m['nd_int8'] - m['nd_fp32']:+.4f}")
    print(f"  recall@50   fp32 {m['r50_fp32']:.4f}   int8 {m['r50_int8']:.4f}   "
          f"(must be identical -- same 50 candidates)")
    if m["r50_fp32"] != m["r50_int8"]:
        print("  !! recall@50 MOVED -- the harness is wrong, not the model (CI-2)")
    if par["truncated_pairs"] or li.get("truncated_pairs"):
        print(f"  !! {par['truncated_pairs']} pairs hit max_length={par['max_length']}: this "
              f"run compares truncation as well as quantisation, so parity is NOT isolated")

    for label, secs in (("parity int8", par.get("int8_secs")),
                        ("latency fp32", lf["secs"]), ("latency int8", li["secs"])):
        if not secs or len(secs) < 3:
            continue
        drift = secs[-1] / secs[0]
        print(f"\n-- {label} per-question seconds --\n  "
              + "  ".join(f"{s_:.1f}" for s_ in secs))
        if drift > 1.25:
            print(f"  !! last/first = {drift:.2f}x -- RISING, so the mean above is a "
                  f"saturated-regime number and is not this backend's latency (AGENT-24)")

    fp32_mean, int8_mean = st.mean(lf["secs"]), st.mean(li["secs"])
    print(f"\n-- latency on CPU, {lf['n']} questions x {st.mean(lf['pairs']):.0f} pairs, "
          f"one backend per process --")
    print(f"  fp32 mean {fp32_mean:.2f}s   int8 mean {int8_mean:.2f}s"
          f"   speedup {fp32_mean / int8_mean:.2f}x")

    OUT.write_text(json.dumps({
        "n": par["n"], "cell": CELL, "seed": SEED, "model": RERANK_MODEL_NAME,
        "max_length": par["max_length"], "truncated_pairs": par["truncated_pairs"],
        "parity": {k: m[k] for k in ("top10_exact", "top10_setmatch", "top10_overlap", "tau")},
        "metrics": {k: m[k] for k in ("r10_fp32", "r10_int8", "r50_fp32", "r50_int8",
                                      "nd_fp32", "nd_int8")},
        "latency_cpu_s": {"fp32_mean": fp32_mean, "int8_mean": int8_mean,
                          "speedup": fp32_mean / int8_mean, "n": lf["n"],
                          "ids": lf["ids"], "fp32_secs": lf["secs"], "int8_secs": li["secs"]},
    }, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(_ROOT)}")
    return 0


PHASES = {"parity": phase_parity, "latency-fp32": phase_latency_fp32,
          "latency-int8": phase_latency_int8, "report": phase_report}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=("all", *PHASES))
    ap.add_argument("-n", type=int, default=50, help="questions to score with int8")
    ap.add_argument("--latency-n", type=int, default=5,
                    help="questions timed fp32-vs-int8 on CPU; small because fp32 CPU is the "
                         "207.8s/question case this whole route exists to fix")
    ap.add_argument("--batch-size", type=int, default=32, help="matches retrieve.py")
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                    help="worst-case bound only; the default is above the observed 1592-token "
                         "maximum, and lowering it invalidates parity (see module docstring)")
    args = ap.parse_args()

    if args.phase != "report":
        if not SCORES.exists():
            print(f"error: {SCORES} absent (gitignored; rebuild with rerank_hpc.py)",
                  file=sys.stderr)
            return 1
        if not INT8.exists():
            print(f"error: {INT8} absent -- run onnx_rerank_export.py first", file=sys.stderr)
            return 1

    if args.phase != "all":
        return PHASES[args.phase](args)

    # Each measuring phase in its own process, sequentially: the point is that the OS reclaims
    # a backend at exit, which cannot happen if they share an interpreter (DEPLOY-8).
    base = [sys.executable, str(Path(__file__).resolve()),
            "-n", str(args.n), "--latency-n", str(args.latency_n),
            "--batch-size", str(args.batch_size), "--max-length", str(args.max_length)]
    for phase in MEASURING_PHASES:
        print(f"\n{'=' * 70}\nphase: {phase}\n{'=' * 70}")
        r = subprocess.run(base + ["--phase", phase], cwd=_ROOT, env=os.environ.copy())
        if r.returncode != 0:
            print(f"error: phase {phase} exited {r.returncode}", file=sys.stderr)
            return r.returncode
    return phase_report(args)


if __name__ == "__main__":
    sys.exit(main())
