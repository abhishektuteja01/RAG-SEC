"""Model names, pinned revisions and device selection. Import these; never re-hardcode them."""

import os

# chunking.py's token budgets are only valid while this tokenizer is the one that embeds.
EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
# Hugging Face commit shas, looked up 2026-09-23. The corpus was embedded with these weights;
# a moved upstream repo would silently change every score.
EMBED_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANK_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

GENERATION_MODEL = "gemini-3.7-flash"


def pick_device() -> str:
    """cuda > mps > cpu, unless RAG_SEC_DEVICE forces one. mps and cpu give the same scores
    (to 2e-6), so the device is a speed choice.

    Torch's MPS backend is not thread-safe: concurrent model loading or model calls crash the
    process with no traceback. retrieve.py puts model calls behind a lock and the API runs one
    worker; `RAG_SEC_DEVICE=cpu` is the escape hatch.
    """
    forced = os.environ.get("RAG_SEC_DEVICE")
    if forced:
        return forced

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
