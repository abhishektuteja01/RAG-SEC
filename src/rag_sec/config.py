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
    """cuda > mps > cpu, unless RAG_SEC_DEVICE forces one.

    Torch's MPS backend is not thread-safe: concurrent model construction or model calls
    segfault the process with no traceback. retrieve.py serialises model calls with a lock
    and the API runs one worker; `RAG_SEC_DEVICE=cpu` is the escape hatch. mps and cpu were
    measured score-identical (to 2e-6), so the device is a speed choice, not an accuracy one.

    torch is imported lazily: the chunking/eval paths never touch a GPU.
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
