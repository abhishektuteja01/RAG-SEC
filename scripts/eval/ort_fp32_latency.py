#!/usr/bin/env python
"""Is DEPLOY-11's 3.1x a PRECISION result or a BACKEND result?

DEPLOY-11 compared `latency-fp32` (torch CrossEncoder, 63.0s) against `latency-int8`
(ONNX Runtime session, 194.7s) and read the gap as int8 being slower than fp32. That
comparison changed two variables at once. torch on this Mac reports
BLAS_INFO=accelerate, so its matmuls reach Apple's AMX coprocessor; ORT's MLAS never
does and gets plain NEON. ORT fp32 was never measured, so the missing cell is the one
that decides which variable moved.

Three legs, one process, identical rows and identical pairs:

    torch-fp32   what DEPLOY-11 called "fp32"   -- torch + Accelerate/AMX
    ort-fp32     THE MISSING CELL               -- ORT + NEON, same precision as above
    ort-int8     what DEPLOY-11 called "int8"   -- ORT + NEON, quantised

If ort-fp32 lands near ort-int8, the gap was the backend and int8 is roughly
break-even within its own runtime -- which also means the Mac measurement says nothing
about Graviton, where ORT's i8mm QGEMM kernels are compiled in (they sit behind
`#if defined(__linux__)` and are absent on macOS despite this M3 reporting FEAT_I8MM=1).
If instead ort-fp32 lands near torch-fp32, DEPLOY-11 stands as written.

Sequential by construction: one session is built, timed, and dropped before the next is
built. Three backends resident at once on 16 GiB is what killed DEPLOY-8.
"""
import argparse
import gc
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "eval"))

from onnx_rerank_parity import (  # noqa: E402
    DEFAULT_MAX_LENGTH,
    INT8,
    RERANK_MODEL_NAME,
    _load_rows,
    _pairs_for,
    get_conn,
)

FP32 = _ROOT / "models" / "onnx" / "reranker_fp32.onnx"
OUT = _ROOT / "data" / "ort_fp32_latency.json"
PARTIAL = _ROOT / "data" / "ort_fp32_partial"  # per-leg, the only channel between processes


def _ort_scorer(model_path: Path, max_length: int, batch_size: int, threads: int):
    """Tokenizer + ORT session. Same shape as the parity script's `_int8_scorer`, but
    parameterised by artifact so fp32 and int8 go through byte-identical code -- if the
    two legs differed in tokenisation or batching the comparison would be worthless."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])

    def score(pairs):
        for i in range(0, len(pairs), batch_size):
            b = pairs[i: i + batch_size]
            enc = tok([p[0] for p in b], [p[1] for p in b], padding=True, truncation=True,
                      max_length=max_length, return_tensors="np")
            sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                  "attention_mask": enc["attention_mask"].astype(np.int64)})

    return score, sess


def _time(label, score_fn, rows, conn):
    secs, pairs_n = [], []
    for r in tqdm(rows, desc=label):
        _cand, pairs = _pairs_for(conn, r)
        t0 = time.perf_counter()
        score_fn(pairs)
        secs.append(time.perf_counter() - t0)
        pairs_n.append(len(pairs))
    print(f"  {label:12s} p50 {st.median(secs):7.2f}s  mean {st.fmean(secs):7.2f}s  "
          f"n={len(secs)} pairs={pairs_n}")
    return {"secs": secs, "pairs": pairs_n}


def report_from_partials() -> int:
    """Combine three --only runs. The id check is real, not a tautology: the legs ran in
    separate processes, so it catches a stale partial from a different -n or --latency-n."""
    got = {}
    for label in ("torch-fp32", "ort-fp32", "ort-int8"):
        f = PARTIAL / f"{label}.json"
        if not f.exists():
            raise SystemExit(f"error: {f.relative_to(_ROOT)} absent -- run --only {label} first")
        got[label] = json.loads(f.read_text())
    if len({tuple(g["ids"]) for g in got.values()}) != 1 or \
       len({tuple(g["pairs"]) for g in got.values()}) != 1:
        raise SystemExit("error: legs scored different questions or pair counts -- stale partial")
    for label, g in got.items():
        print(f"  {label:12s} p50 {st.median(g['secs']):7.2f}s  mean {st.fmean(g['secs']):7.2f}s")
    t, of, oi = (st.median(got[k]["secs"]) for k in ("torch-fp32", "ort-fp32", "ort-int8"))
    _verdict(t, of, oi)
    OUT.write_text(json.dumps(got, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(_ROOT)}")
    return 0


def _verdict(t: float, of: float, oi: float) -> None:
    print(f"\n-- verdict --\n  ort-fp32 / torch-fp32 = {of / t:.2f}x   (backend effect, same precision)")
    print(f"  ort-int8 / ort-fp32   = {oi / of:.2f}x   (precision effect, same backend)")
    print("\n  " + ("BACKEND: DEPLOY-11 conflated the two; int8 is ~break-even in its own runtime"
                    if oi / of < 1.5 else
                    "PRECISION: int8 really is slower than fp32 within one backend; DEPLOY-11 stands"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=50, help="row pool; must match the parity run's -n "
                                                     "or _load_rows returns different questions")
    ap.add_argument("--latency-n", type=int, default=3, help="questions actually timed")
    ap.add_argument("--batch-size", type=int, default=32, help="matches retrieve.py")
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--threads", type=int, default=4, help="ORT intra_op; torch leg is untouched")
    ap.add_argument("--only", choices=("torch-fp32", "ort-fp32", "ort-int8"),
                    help="run ONE leg and write a partial. Three invocations is the safe route "
                         "on 16 GiB -- process exit is the only thing that reliably returns a "
                         "backend's memory, which is DEPLOY-8's lesson. Then --report.")
    ap.add_argument("--report", action="store_true", help="combine partials written by --only")
    args = ap.parse_args()

    if args.report:
        return report_from_partials()

    for p in (FP32, INT8):
        if not p.exists():
            print(f"error: {p} absent -- run onnx_rerank_export.py first", file=sys.stderr)
            return 1

    # Deterministic in (-n, SEED) alone, so separate --only processes reconstruct the SAME
    # questions without passing anything between them -- the parity script's discipline.
    rows = _load_rows(args.n)[: args.latency_n]
    print(f"questions: {[r['id'] for r in rows]}\n")
    legs = (args.only,) if args.only else ("torch-fp32", "ort-fp32", "ort-int8")
    res = {}

    with get_conn() as conn:
        for label in legs:
            if label == "torch-fp32":
                from sentence_transformers import CrossEncoder
                ce = CrossEncoder(RERANK_MODEL_NAME, device="cpu")
                res[label] = _time(label, lambda p: ce.predict(p, batch_size=args.batch_size),
                                   rows, conn)
                del ce
            else:
                score, sess = _ort_scorer(FP32 if label == "ort-fp32" else INT8,
                                          args.max_length, args.batch_size, args.threads)
                res[label] = _time(label, score, rows, conn)
                del score, sess
            gc.collect()
            res[label] |= {"ids": [r["id"] for r in rows], "threads": args.threads,
                           "max_length": args.max_length, "batch_size": args.batch_size}
            PARTIAL.mkdir(parents=True, exist_ok=True)
            (PARTIAL / f"{label}.json").write_text(json.dumps(res[label], indent=2) + "\n")

    if args.only:
        print(f"\nwrote partial {args.only}. Run the other legs, then --report.")
        return 0

    t, of, oi = (st.median(res[k]["secs"]) for k in ("torch-fp32", "ort-fp32", "ort-int8"))
    _verdict(t, of, oi)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
