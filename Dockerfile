# syntax=docker/dockerfile:1
#
# Serving image for src/rag_sec/api.py -- Arm 3 + filter + strip behind HTTP.
#
# Two things drive every choice below, and both are measured, not stylistic:
#   1. The corpus is embedded with BAAI/bge-m3, so the query embedder cannot be swapped for
#      a hosted one without invalidating every dense-search number. torch ships in the image
#      either way, which is why the cross-encoder reranker stays in-process too.
#   2. CPU only. `RAG_SEC_DEVICE=cpu` is INFRA-6's speed knob, not an accuracy one -- mps and
#      cpu scored identical to 2e-6 -- so a CPU container serves the same rankings the GPU
#      runs published, just slower. Day 13's p95 has to be measured here, not on the laptop.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.1

# ---------------------------------------------------------------------------- deps
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
ARG UV_VERSION
COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /bin/uv

WORKDIR /build
COPY pyproject.toml uv.lock ./

# Exported from the lock rather than `uv sync`, for one reason: `--torch-backend` exists on
# `uv pip install` and NOT on `uv sync` (checked against uv 0.12.1). Without it, linux/amd64
# resolves PyPI's CUDA-bundled torch wheel and adds ~2.5GB to an image that will never see a
# GPU. `--frozen` keeps the exact locked versions -- the pins in pyproject.toml exist so a
# routine upgrade cannot move the numbers in DECISIONS.md, and that has to survive the build.
RUN uv export --frozen --no-dev --no-emit-project --no-hashes --extra serve \
        -o /build/requirements.txt \
    && uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv UV_TORCH_BACKEND=cpu uv pip install -r /build/requirements.txt

# ---------------------------------------------------------------------------- weights
FROM builder AS weights
ARG EMBED_MODEL=BAAI/bge-m3
ARG RERANK_MODEL=BAAI/bge-reranker-v2-m3
ENV HF_HOME=/models
# Baked, not fetched at boot: config.py pins these models by NAME but not by revision, so
# a repo that moves upstream silently changes retrieval. Freezing the weights into the
# image is what makes the pin real, and it drops HuggingFace out of the runtime path.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    /opt/venv/bin/python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${EMBED_MODEL}', ignore_patterns=['*.onnx','*.h5','*.msgpack','onnx/*']); \
snapshot_download('${RERANK_MODEL}', ignore_patterns=['*.onnx','*.h5','*.msgpack','onnx/*'])" \
    && cp -r /models /models-baked

# ---------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    RAG_SEC_DEVICE=cpu \
    PORT=8080

COPY --from=builder /opt/venv /opt/venv
COPY --from=weights /models-baked /models
COPY src /app/src
WORKDIR /app

# Non-root, and it owns nothing writable: the container reads Postgres and HF cache only.
RUN useradd --system --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8080
# Single worker on purpose. Two workers = two copies of both transformer models in RAM, and
# concurrent model construction is what SIGSEGV'd the laptop (AGENT-10/AGENT-17). Scale by
# running more containers, not more workers.
CMD ["sh", "-c", "uvicorn rag_sec.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
