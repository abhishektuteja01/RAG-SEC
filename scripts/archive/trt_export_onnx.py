"""Export the pinned cross-encoder to ONNX for the TensorRT rerank backend: ruled out
(research_latency.md item 1, 2026-09-14 -- see trt_rerank_parity.py in this directory for the
result and why nothing was wired into retrieve.py).

SAME EXPORT PATH AS onnx_rerank_export.py, same reason: `optimum` is uninstallable against
this project's transformers==5.8.1 pin (DEPLOY-6), so this uses `torch.onnx.export` directly
rather than the documented `optimum` route. Differs from that script in stopping at fp32, no
int8 quantisation -- ORT's TensorRT execution provider does its own fp16 conversion from this
fp32 graph at engine-build time (see trt_rerank_parity.py), so quantising here would be a
second, redundant precision drop.

Usage:
    uv run --extra onnx scripts/archive/trt_export_onnx.py
"""

import sys
import time
from pathlib import Path

import onnx
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rag_sec.config import RERANK_MODEL_NAME  # noqa: E402

OUT = _ROOT / "models" / "onnx" / "reranker_trt_fp32.onnx"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
    # eager, not the default sdpa: transformers 5.x's sdpa mask shape logic is not traceable
    # by torch.onnx (DEPLOY-6's IndexError in masking_utils.sdpa_mask). Mathematically the
    # same operation; the parity check verifies that empirically rather than assuming it.
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANK_MODEL_NAME, attn_implementation="eager").eval()

    # A real pair, not zeros: export traces the graph on this input. Sequence length is a
    # dynamic axis below, so the value here is a tracing detail, not a limit.
    enc = tok([("what was the operating margin", "Operating margin was 12.4% in fiscal 2018.")],
              return_tensors="pt", padding=True, truncation=True, max_length=512)

    print(f"exporting {RERANK_MODEL_NAME} -> {OUT.name}")
    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        ({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},),
        str(OUT),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        # Both axes dynamic: batch because retrieve() predicts in batches of 32 and the last
        # batch is short, sequence because chunk texts vary.
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17,
        # dynamo=True: the legacy tracer feeds masking_utils a 0-d tensor it indexes into;
        # torch.export handles the same code path and is the supported route in torch 2.13.
        dynamo=True,
    )
    # The dynamo exporter writes intermediate value_info entries that onnx's own shape
    # inference then re-derives differently, which trips up some downstream consumers
    # (DEPLOY-6 hit this in the quantiser) even though the graph runs fine as exported.
    # Clearing them changes nothing about the computation -- every shape is re-derived at
    # load time regardless.
    print("stripping intermediate value_info")
    m = onnx.load(str(OUT))
    del m.graph.value_info[:]
    onnx.save(m, str(OUT), save_as_external_data=True, all_tensors_to_one_file=True,
              location=OUT.name + ".data", size_threshold=1024)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
