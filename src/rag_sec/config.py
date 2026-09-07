"""Pinned model names and device selection, imported everywhere rather than re-hardcoded
per script -- see DECISIONS.md ARM1-1/ARM3-1/INFRA-6 for the choices.
"""

# chunking.py's token budgets are only valid while this tokenizer is the one that embeds.
EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def pick_device() -> str:
    """cuda > mps > cpu -- a machine with both is a Linux box with a discrete GPU.

    `mps` measured 2.1x faster than cpu on the reranker and score-identical to 2e-6
    (INFRA-6). torch is imported lazily: the chunking/eval paths that import this module
    never touch a GPU, and importing torch costs ~1s.

    THE MPS SEGFAULT, in one place -- retrieve._gpu_lock, 07_arm6_loop._refuse_unless_serial
    and the Dockerfile's single-worker CMD all point here. Torch's MPS backend is not
    thread-safe: concurrent model construction / model calls SIGSEGV in
    MetalShaderLibrary::exec_unary_kernel with no traceback and zero rows written, killing
    the process so no except can catch it (AGENT-10/AGENT-17, 2026-09-04; the reproducing
    figure on record is concurrency ">1" and nothing more precise may be quoted).

    `RAG_SEC_DEVICE` overrides the choice, and is the escape hatch from the above:
    `RAG_SEC_DEVICE=cpu` trades INFRA-6's 2.1x for a
    backend that cannot do that. Safe for published numbers precisely because INFRA-6
    measured the two score-identical to 2e-6 -- this is a speed knob, not an accuracy one.
    """
    import os

    forced = os.environ.get("RAG_SEC_DEVICE")
    if forced:
        return forced

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
