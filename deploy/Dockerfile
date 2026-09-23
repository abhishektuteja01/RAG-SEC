# syntax=docker/dockerfile:1
#
# API image for src/rag_sec/api.py. Build from the repo root:
#   CPU: docker build -f deploy/Dockerfile -t rag-sec-api:cpu .
#   GPU: docker build -f deploy/Dockerfile --build-arg TORCH_BACKEND=cu130 \
#            --build-arg DEVICE=cuda -t rag-sec-api:gpu .
#
# Device is a build-time choice. DEVICE is forced rather than auto-detected on purpose: if a
# GPU image runs where nvidia-container-toolkit is not wired up, torch should fail loudly,
# not fall back to a slow CPU container nobody notices. cpu and GPU give the same rankings
# (fp16 on cuda has exact top-5 parity); only speed differs.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.1
ARG TORCH_BACKEND=cpu
ARG DEVICE=cpu

# `COPY --from=<image>` does not expand variables, so the pinned uv image becomes a named
# stage first. Pinned because it resolves the lock.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uvbin

# ---------------------------------------------------------------------------- deps
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
COPY --from=uvbin /uv /bin/uv
ARG TORCH_BACKEND

WORKDIR /build
COPY pyproject.toml uv.lock ./

# Exported from the lock rather than `uv sync` because `--torch-backend` exists on
# `uv pip install` and not on `uv sync` (uv 0.12.1). `--frozen` keeps the locked versions.
# The CPU build strips the `nvidia-*`/`triton` packages from the export: they are explicit
# requirements there (marked linux), so the backend flag cannot deselect them, and left in
# they add ~3 GB of unused CUDA runtime. The GPU build keeps them.
RUN uv export --frozen --no-dev --no-emit-project --no-hashes \
        -o /build/requirements.full.txt \
    && if [ "$TORCH_BACKEND" = "cpu" ]; then \
           grep -vE '^(nvidia-|triton)' /build/requirements.full.txt > /build/requirements.txt; \
       else \
           cp /build/requirements.full.txt /build/requirements.txt; \
       fi \
    && uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --torch-backend=${TORCH_BACKEND} -r /build/requirements.txt

# ---------------------------------------------------------------------------- weights
FROM builder AS weights
ENV HF_HOME=/models
# The model names and commit shas come from config.py, so the image and the code cannot
# disagree. Baked, not fetched at boot: Hugging Face drops out of the runtime path.
COPY src/rag_sec/config.py /build/rag_sec_config.py
RUN --mount=type=cache,target=/root/.cache/huggingface \
    /opt/venv/bin/python -c "\
import sys; sys.path.insert(0, '/build'); import rag_sec_config as c; \
from huggingface_hub import snapshot_download; \
skip = ['*.onnx', '*.h5', '*.msgpack', 'onnx/*']; \
snapshot_download(c.EMBED_MODEL_NAME, revision=c.EMBED_MODEL_REVISION, ignore_patterns=skip); \
snapshot_download(c.RERANK_MODEL_NAME, revision=c.RERANK_MODEL_REVISION, ignore_patterns=skip)"

# ---------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
ARG DEVICE

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    RAG_SEC_DEVICE=${DEVICE} \
    COMPANY_WORDLIST=/app/data/wordlist_web2.txt \
    PORT=8080

COPY --from=builder /opt/venv /opt/venv
COPY --from=weights /models /models
COPY src /app/src
# The company alias table. Without it, company.load_lexicon() tries to rebuild it from the
# HF question set and data/chunks/, neither of which is in the image.
COPY data/company_lexicon.json /app/data/company_lexicon.json
# The wordlist the company resolver's English-word guard reads. python:slim ships none, and
# a missing list only warns: 'visa applications' would then filter to Visa Inc. It is a
# byte-copy of the macOS/BSD web2 list the numbers were measured with; Debian's wamerican is
# NOT a substitute (it has proper nouns, so 'intel', 'merck', 'nike' stop resolving).
COPY data/wordlist_web2.txt /app/data/wordlist_web2.txt
COPY deploy/container_wordlist.py /app/deploy/container_wordlist.py
WORKDIR /app

# Fail the BUILD if the guard is off or reading a different list. It has no runtime symptom.
RUN python /app/deploy/container_wordlist.py

# Non-root; the container only reads Postgres and the baked weights.
RUN useradd --system --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8080
# One worker: two would load two copies of both models. Scale with containers, not workers.
# The wordlist check runs again at startup, because `docker run -e COMPANY_WORDLIST=...`
# could point it elsewhere after the build.
CMD ["sh", "-c", "python /app/deploy/container_wordlist.py && uvicorn rag_sec.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
