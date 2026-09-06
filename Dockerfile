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

# `COPY --from=<image>` does not expand variables, so the pinned uv image has to become a
# named stage first -- `FROM` is the only instruction that expands a global ARG. Pinning uv
# at all is the point: it resolves the lock, and an unpinned resolver is a silent way for a
# build to install something other than what uv.lock says.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uvbin

# ---------------------------------------------------------------------------- deps
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
COPY --from=uvbin /uv /bin/uv

WORKDIR /build
COPY pyproject.toml uv.lock ./

# Exported from the lock rather than `uv sync`, for one reason: `--torch-backend` exists on
# `uv pip install` and NOT on `uv sync` (checked against uv 0.12.1). Without it, linux/amd64
# resolves PyPI's CUDA-bundled torch wheel and adds ~2.5GB to an image that will never see a
# GPU. `--frozen` keeps the exact locked versions -- the pins in pyproject.toml exist so a
# routine upgrade cannot move the numbers in DECISIONS.md, and that has to survive the build.
# The CUDA packages are STRIPPED FROM THE EXPORT, not just deselected by a flag, and the
# distinction is why the first build of this image failed. uv.lock carries 43 `nvidia-*`
# entries plus `triton` marked `sys_platform == 'linux'`, so `uv export` names them as
# explicit requirements -- at which point `UV_TORCH_BACKEND` cannot help, because that
# governs RESOLUTION and this step installs an already-resolved list. Left in, they pulled
# ~3GB of CUDA runtime into an image whose own header says CPU only, and the build ran the
# disk out during layer export. `--torch-backend=cpu` still goes on the install so torch
# itself resolves to the +cpu variant, which has no nvidia dependencies to reinstate.
RUN uv export --frozen --no-dev --no-emit-project --no-hashes --extra serve \
        -o /build/requirements.full.txt \
    && grep -vE '^(nvidia-|triton)' /build/requirements.full.txt > /build/requirements.txt \
    && uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --torch-backend=cpu -r /build/requirements.txt

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
snapshot_download('${RERANK_MODEL}', ignore_patterns=['*.onnx','*.h5','*.msgpack','onnx/*'])"

# ---------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    RAG_SEC_DEVICE=cpu \
    COMPANY_WORDLIST=/app/data/wordlist_web2.txt \
    PORT=8080

COPY --from=builder /opt/venv /opt/venv
COPY --from=weights /models /models
COPY src /app/src
# The company lexicon, 10KB, baked. Without it `load_lexicon()` falls through to
# `build_lexicon()`, which imports the HF dataset loader and reads `data/chunks/` -- neither
# of which belongs in a serving image, and both of which are excluded from it. The first
# build of this image started, loaded both models, and then died in the lifespan warmup for
# exactly that reason. company.py's own docstring already says lexicon BUILDING is kept out
# of the retrieval path; this is what makes that true at runtime rather than only by
# intention. It is committed, so the image and the eval resolve identical aliases.
COPY data/company_lexicon.json /app/data/company_lexicon.json

# The wordlist RETR-11's English-word guard reads, and the reason COMPANY_WORDLIST is set
# above. `python:slim` ships no /usr/share/dict/words, so without this the guard is OFF in
# the container -- silently, because a missing wordlist degrades to an empty word set and
# only warns: nothing fails, 'visa applications' just filters to Visa Inc. and recall drops.
# It is the FILE, not a package: `apt-get install wamerican` also puts a list at that path,
# but a different one. wamerican carries proper nouns, so lowercased it makes 'intel',
# 'merck', 'nike', 'walmart' and 23 other in-corpus aliases "ordinary English" -> risky ->
# unmatched in lowercase questions. 27 of our 250 aliases/tickers get a different verdict
# from it, which is a resolver the benchmark never measured (DEPLOY-2). data/wordlist_web2.txt
# is a committed byte-copy of the macOS/BSD web2 the numbers were produced against
# (sha256 be41ad97...bab9, 2.5MB), so the image cannot drift with an apt mirror either.
COPY data/wordlist_web2.txt /app/data/wordlist_web2.txt
COPY scripts/checks/container_wordlist.py /app/scripts/checks/container_wordlist.py
WORKDIR /app

# Fail the BUILD if the guard is off or reading a list other than web2. Cheap (stdlib +
# a 10KB lexicon, no torch), and it is the only way this defect is ever noticed -- it has
# no runtime symptom.
RUN python /app/scripts/checks/container_wordlist.py

# Non-root, and it owns nothing writable: the container reads Postgres and HF cache only.
RUN useradd --system --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8080
# Single worker on purpose. Two workers = two copies of both transformer models in RAM, and
# concurrent model construction is what SIGSEGV'd the laptop (AGENT-10/AGENT-17). Scale by
# running more containers, not more workers.
# The same check again at startup, because the build-time one cannot see a COMPANY_WORDLIST
# that `docker run -e` points somewhere else. Refuse to serve rather than serve a resolver
# that is not the benchmarked one.
CMD ["sh", "-c", "python /app/scripts/checks/container_wordlist.py && uvicorn rag_sec.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
