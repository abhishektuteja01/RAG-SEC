"""Export the pinned cross-encoder to ONNX and quantise it to int8 (DEPLOY-6-RESOLVED).

WHY THIS EXISTS. The CPU serving container spends 207.8s of its 212.4s on the reranker
(DEPLOY-6), which cannot serve interactively. int8 is the one route that keeps CANDIDATE_K at
the benchmarked value, so the deployed ranking stays the ranking the published numbers
describe -- conditional on parity, which `onnx_rerank_parity.py` measures.

WHY NOT `optimum`. The documented route is `export_dynamic_quantized_onnx_model()`, and it is
uninstallable here: every published `optimum-onnx` caps transformers at <4.58 and this project
pins 5.8.1 (INFRA/config.py). Downgrading transformers under the benchmarked models to satisfy
an export tool is not a trade worth making, so this uses `torch.onnx.export` and
`onnxruntime.quantization` directly -- neither of which depends on transformers at all.

WHAT int8 DYNAMIC QUANTIZATION DOES. Weights are stored as 8-bit integers instead of 32-bit
floats (4x smaller, and the memory traffic is what a CPU reranker is actually bound by);
activation ranges are computed per batch at inference, so no calibration dataset is needed.
It is lossy by construction, which is the entire reason parity has to be measured rather than
assumed.

Usage:
    uv run --extra onnx scripts/eval/onnx_rerank_export.py
"""

import sys
import time
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec.config import RERANK_MODEL_NAME  # noqa: E402

OUT = _ROOT / "models" / "onnx"
FP32 = OUT / "reranker_fp32.onnx"
INT8 = OUT / "reranker_int8.onnx"


def _size_mb(p: Path) -> float:
    ext = p.with_suffix(p.suffix + ".data")
    return (p.stat().st_size + (ext.stat().st_size if ext.exists() else 0)) / 1e6


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
    # `eager`, not the default sdpa: transformers 5.x builds sdpa masks with shape logic that
    # torch.onnx tracing cannot follow (IndexError in masking_utils.sdpa_mask). Eager attention
    # is mathematically the same operation, and the parity test verifies that empirically.
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANK_MODEL_NAME, attn_implementation="eager").eval()

    # A real pair, not zeros: export traces the graph on this input, and a degenerate one can
    # trace a degenerate path. Sequence length is a dynamic axis, so the value here is a
    # tracing detail rather than a limit.
    enc = tok([("what was the operating margin", "Operating margin was 12.4% in fiscal 2018.")],
              return_tensors="pt", padding=True, truncation=True, max_length=512)

    print(f"exporting {RERANK_MODEL_NAME} -> {FP32.name}")
    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        ({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},),
        str(FP32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        # Both axes dynamic: batch because retrieve() predicts in batches of 32 and the last
        # batch is short, sequence because chunk texts vary and padding to a fixed 512 would
        # spend compute on padding -- which is the cost we are here to cut.
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17,
        # dynamo=True: the legacy tracer feeds masking_utils a 0-d tensor it indexes into.
        # torch.export handles the same code path, and it is also the supported route in
        # torch 2.13 -- the legacy exporter is deprecated.
        dynamo=True,
    )
    # ".data" too: the model is >2GB so the exporter writes weights to an external file and
    # the .onnx is only the graph. Reporting st_size alone says 0 MB.
    print(f"  {_size_mb(FP32):.0f} MB in {time.perf_counter() - t0:.1f}s")

    # The dynamo exporter writes intermediate `value_info` entries that onnx's own shape
    # inference then re-derives differently (`(1024) vs (1)` on the pooler), and the quantiser
    # runs that inference before it does anything else, so it aborts on a graph that runs fine.
    # Clearing the annotations does not change the computation: graph inputs and outputs keep
    # their types, and every intermediate shape is re-derived at load time anyway.
    print("stripping intermediate value_info")
    m = onnx.load(str(FP32))
    del m.graph.value_info[:]
    onnx.save(m, str(FP32), save_as_external_data=True, all_tensors_to_one_file=True,
              location=FP32.name + ".data", size_threshold=1024)
    del m

    print(f"quantising -> {INT8.name}")
    t0 = time.perf_counter()
    # QUInt8 activations with QInt8 weights is the U8S8 pattern ONNX Runtime documents as the
    # faster path on low-end ARM64 and equivalent on high-end -- and the deployment target is
    # arm64/Graviton (DEPLOY-4).
    # DisableShapeInference: the dynamo exporter emits a graph whose value_info onnx's shape
    # inference re-derives inconsistently (`(1024) vs (1)` on the pooler), which aborts the
    # quantiser before it starts. Dynamic quantisation only needs to find the weight
    # initialisers, not the activation shapes -- those are resolved per batch at inference --
    # so skipping inference costs nothing here. Parity is what proves that claim.
    quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QInt8,
                     extra_options={"DisableShapeInference": True})
    print(f"  {_size_mb(INT8):.0f} MB in {time.perf_counter() - t0:.1f}s "
          f"({_size_mb(FP32) / _size_mb(INT8):.1f}x smaller)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
