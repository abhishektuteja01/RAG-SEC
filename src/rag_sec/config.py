"""Single source of truth for pinned model names shared across chunking, embedding,
and retrieval -- see DECISIONS.md ARM1-1/ARM3-1 for why each was chosen. Chunk token
budgets (chunking.py) are only valid if EMBED_MODEL_NAME's tokenizer here matches the
model actually used to embed, so this is imported rather than re-hardcoded per script.
"""

EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
