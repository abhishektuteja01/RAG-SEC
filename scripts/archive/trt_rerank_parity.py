"""TensorRT rerank backend: ruled out (2026-09-14).

WHAT THIS MEASURED. ONNX Runtime's TensorRT execution provider, fp16, against the pinned
cross-encoder, on the same g4dn.xlarge/T4 host DEPLOY-21 measured torch fp16 on. 10 real
test-split questions, same convention as DEPLOY-20/DEPLOY-21: exact top-5 order + score deltas,
now also per-question latency (that pair didn't measure it, having no reason to -- torch fp16
was the thing being adopted there, not compared against a second backend for speed).

RESULT: NO WIN ON EITHER AXIS. 9/10 exact top-5 matches, not 10/10 -- one question
(finqa_test_962) reordered. Max score delta 7.46e-3, against DEPLOY-21's fp16-vs-fp32 deltas
of ~1e-3 -- TensorRT is doing more than a plain precision cast (likely attention/activation
kernel fusion, not just fp16), so it doesn't clear this project's own parity bar for a
precision change alone. And it isn't faster either: mean 3.311s vs torch's 3.217s, a 0.97x
"speedup" -- slower. At CANDIDATE_K=50/batch_size=32 (one or two batches per question), the
torch fp16 path on a T4 already runs near-optimal cuDNN/cuBLAS kernels; TensorRT's extra
kernel-selection work adds tokenize/copy/dispatch overhead without a fusion win big enough to
pay for it. Not measured: larger batch sizes, where TensorRT more often wins -- not worth
chasing since retrieve.py's batch size is fixed by CANDIDATE_K, not free to raise for this.

KEPT FOR PROVENANCE, NOT WIRED IN. No retrieve.py/config.py hook exists for this backend --
reverted after this result. Re-running needs `models/onnx/reranker_trt_fp32.onnx`, exported by
the sibling script this one replaces the export step of; see trt_export_onnx.py in this same
directory (archived alongside this one, same reason).

TWO PHASES, TWO PROCESSES. The first version of this script held both backends in one process
and OOM'd the T4: TensorRT's engine-build tactic search wants GPU memory independent of
whatever else is resident, and the torch fp16 CrossEncoder sitting in the same process was
enough to push a search past 15GB. Same fix as the archived int8 parity check
(onnx_rerank_parity.py, DEPLOY-8), for a GPU-memory reason instead of that script's CPU-RAM one.
Bounding the TensorRT shape profile explicitly (batch<=32, seq<=2048, matching retrieve.py's
batch_size and DEPLOY-6's observed token ceiling) was the other half of that OOM fix -- left
unbounded, the engine builder searches tactics across an unconstrained shape space and asked
for a single 14.3GB allocation.

    torch   torch fp16 CrossEncoder only, writes data/trt_parity_torch.json
    trt     TensorRTCrossEncoder only, writes data/trt_parity_trt.json
    report  reads both partials, prints top-5 agreement + score deltas + latency -- no backend
            loaded

`--phase all` (the default) re-execs `torch` and `trt` as subprocesses in sequence, then
reports. Partials are gitignored.

Usage (GPU host only -- needs cuda, TensorRT's own shared libraries on LD_LIBRARY_PATH, and
the ONNX export already on disk):
    uv run --extra tensorrt scripts/archive/trt_rerank_parity.py -n 10
    uv run --extra tensorrt scripts/archive/trt_rerank_parity.py --phase report
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

load_dotenv()  # POSTGRES_* for the chunk-text fetch

SCORES = _ROOT / "data" / "retr7_rr_test_scores.jsonl"
CELL = "filtered_stripped"  # the shipped arm: company filter on, query stripped for rerank
SEED = 17  # arbitrary, but fixed so re-runs sample the same questions
PARTIAL_TORCH = _ROOT / "data" / "trt_parity_torch.json"
PARTIAL_TRT = _ROOT / "data" / "trt_parity_trt.json"
MEASURING_PHASES = ("torch", "trt")

ONNX_PATH = _ROOT / "models" / "onnx" / "reranker_trt_fp32.onnx"
# 2048: above the 1592-token max observed over 3,000 live chunks (DEPLOY-6), so padding to
# this cap never truncates real chunk text.
DEFAULT_MAX_LENGTH = 2048


class TensorRTCrossEncoder:
    """Matches the one corner of sentence_transformers.CrossEncoder's interface this script
    calls: .predict(pairs, batch_size=...) -> scores. Nothing else. See module docstring for
    why sigmoid is applied here: a raw ORT session returns the logit only, and CrossEncoder
    applies sigmoid as this model's default activation."""

    def __init__(self, max_length: int = DEFAULT_MAX_LENGTH):
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._np = np
        if not ONNX_PATH.exists():
            raise FileNotFoundError(
                f"{ONNX_PATH} missing -- run scripts/archive/trt_export_onnx.py first"
            )

        self._tok = AutoTokenizer.from_pretrained(_rerank_model_name())
        self._max_length = max_length
        cache_dir = ONNX_PATH.parent / "trt_cache"
        cache_dir.mkdir(exist_ok=True)
        shape_bounds = "input_ids:1x1,attention_mask:1x1"
        shape_opt = "input_ids:16x512,attention_mask:16x512"
        shape_max = f"input_ids:32x{max_length},attention_mask:32x{max_length}"
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_dir),
                "trt_timing_cache_enable": True,
                "trt_profile_min_shapes": shape_bounds,
                "trt_profile_opt_shapes": shape_opt,
                "trt_profile_max_shapes": shape_max,
            }),
            "CUDAExecutionProvider",
        ]
        self._sess = ort.InferenceSession(str(ONNX_PATH), providers=providers)

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32):
        np = self._np
        if not pairs:
            return np.array([], dtype=np.float32)
        scores = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            enc = self._tok(
                [p[0] for p in batch], [p[1] for p in batch],
                padding=True, truncation=True, max_length=self._max_length,
                return_tensors="np",
            )
            logits = self._sess.run(
                ["logits"],
                {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
            )[0]
            batch_scores = 1.0 / (1.0 + np.exp(-logits.reshape(-1).astype(np.float64)))
            scores[start:start + len(batch)] = batch_scores.astype(np.float32)
        return scores


def _rerank_model_name() -> str:
    from rag_sec.config import RERANK_MODEL_NAME

    return RERANK_MODEL_NAME


def _rows(n: int) -> list[dict]:
    from rag_sec.eval import _filing_stem, load_matched_questions, load_ranking

    # sort=False: this script re-scores the stored candidates itself rather than trusting
    # the stored score's order (AGENT-16's exact bug), so it needs first-stage order.
    published = load_ranking(SCORES, CELL, with_score=True, sort=False)
    df = load_matched_questions()
    split = df[df["split"] == "test"].reset_index(drop=True)

    rows = []
    for _, row in split.iterrows():
        qid = row["id"]
        if qid not in published:
            continue
        rows.append({"id": qid, "question": row["question"],
                     "stem": _filing_stem(row), "cell": published[qid]})
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return sorted(rows[:n], key=lambda r: r["id"])


def _score_rows(rows: list[dict], scorer) -> dict:
    import time

    from rag_sec.candidates import LIVE_VARIANT, chunk_texts
    from rag_sec.company import strip_entity_framing
    from rag_sec.store import get_conn

    out = {}
    with get_conn() as conn:
        for row in rows:
            cand = [(s, i) for s, i, _ in row["cell"]]
            texts = chunk_texts(conn, cand, LIVE_VARIANT)
            cand = [c for c in cand if c in texts]
            q = strip_entity_framing(row["question"])
            pairs = [(q, texts[c]) for c in cand]
            # Only the predict() call, not the DB fetch above -- that isolates the one thing
            # this check is actually comparing between backends.
            t0 = time.perf_counter()
            scores = [float(s) for s in scorer(pairs)]
            predict_s = time.perf_counter() - t0
            out[row["id"]] = {"cand": cand, "scores": scores, "predict_s": predict_s}
    return out


def _phase_torch(rows: list[dict]) -> None:
    import os

    os.environ["RAG_SEC_DEVICE"] = "cuda"
    from sentence_transformers import CrossEncoder

    print("loading torch fp16 cross-encoder (the shipped backend)")
    ce = CrossEncoder(_rerank_model_name(), device="cuda",
                       model_kwargs={"torch_dtype": "float16"})
    out = _score_rows(rows, lambda pairs: ce.predict(pairs, batch_size=32))
    PARTIAL_TORCH.write_text(json.dumps(out))
    print(f"wrote {PARTIAL_TORCH}")


def _phase_trt(rows: list[dict]) -> None:
    import os

    os.environ["RAG_SEC_DEVICE"] = "cuda"

    print("loading TensorRT cross-encoder (builds/loads the cached engine)")
    ce = TensorRTCrossEncoder()
    out = _score_rows(rows, lambda pairs: ce.predict(pairs, batch_size=32))
    PARTIAL_TRT.write_text(json.dumps(out))
    print(f"wrote {PARTIAL_TRT}")


def _phase_report(rows: list[dict]) -> int:
    import statistics as st

    torch_scores = json.loads(PARTIAL_TORCH.read_text())
    trt_scores = json.loads(PARTIAL_TRT.read_text())

    print(f"\n{len(rows)} questions, seed={SEED}, split=test, cell={CELL}")
    agrees = []
    max_abs_delta = 0.0
    torch_times, trt_times = [], []
    for row in rows:
        qid = row["id"]
        t = torch_scores[qid]
        r = trt_scores[qid]
        assert t["cand"] == r["cand"], f"{qid}: candidate pools differ between phases"
        cand = [tuple(c) for c in t["cand"]]
        ts, rs = t["scores"], r["scores"]
        torch_times.append(t["predict_s"])
        trt_times.append(r["predict_s"])

        torch_top5 = [cand[j] for j in sorted(range(len(cand)), key=lambda j: ts[j],
                                               reverse=True)[:5]]
        trt_top5 = [cand[j] for j in sorted(range(len(cand)), key=lambda j: rs[j],
                                             reverse=True)[:5]]
        agree = torch_top5 == trt_top5
        agrees.append(agree)
        deltas = [abs(a - b) for a, b in zip(ts, rs)]
        max_abs_delta = max(max_abs_delta, max(deltas, default=0.0))
        print(f"  {qid}: top5 exact match={agree}, max score delta={max(deltas, default=0.0):.2e}, "
              f"torch={t['predict_s']:.3f}s, trt={r['predict_s']:.3f}s")

    print(f"\nall {len(rows)} top-5 exact matches: {all(agrees)}")
    print(f"max abs score delta across every candidate, every question: {max_abs_delta:.2e}")
    print(f"\ntorch predict_s: mean={st.mean(torch_times):.3f} median={st.median(torch_times):.3f} "
          f"min={min(torch_times):.3f} max={max(torch_times):.3f}")
    print(f"trt   predict_s: mean={st.mean(trt_times):.3f} median={st.median(trt_times):.3f} "
          f"min={min(trt_times):.3f} max={max(trt_times):.3f}")
    print(f"speedup (mean torch / mean trt): {st.mean(torch_times) / st.mean(trt_times):.2f}x")
    return 0 if all(agrees) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--phase", choices=(*MEASURING_PHASES, "report", "all"), default="all")
    args = ap.parse_args()

    rows = _rows(args.n)

    if args.phase == "all":
        for phase in MEASURING_PHASES:
            subprocess.run([sys.executable, __file__, "--phase", phase, "-n", str(args.n)],
                            check=True)
        return _phase_report(rows)
    if args.phase == "torch":
        _phase_torch(rows)
        return 0
    if args.phase == "trt":
        _phase_trt(rows)
        return 0
    return _phase_report(rows)


if __name__ == "__main__":
    sys.exit(main())
