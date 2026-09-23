#!/usr/bin/env bash
# Download the prebuilt chunks database (99,654 chunks, embeddings and both indexes) from
# Hugging Face, verify its checksum, and restore it into the compose Postgres.
#
#   docker compose up -d          # start Postgres first
#   scripts/setup_db.sh
#
# On the GPU host, point it at that compose file:
#   COMPOSE="docker compose -f deploy/docker-compose.gpu.yml --env-file .env" scripts/setup_db.sh
set -euo pipefail

cd "$(dirname "$0")/.."

REPO="abhishektuteja/rag-sec-db"
DUMP="rag_sec_chunks.dump"
DEST="data/db"
COMPOSE="${COMPOSE:-docker compose}"

# ── 1. download ─────────────────────────────────────────────────────────────────────
mkdir -p "$DEST"
uv run hf download "$REPO" "$DUMP" "$DUMP.sha256" --repo-type dataset --local-dir "$DEST"

# ── 2. verify ───────────────────────────────────────────────────────────────────────
# First token of the .sha256 file, so it works whether or not the file names the dump.
expected="$(awk '{print $1; exit}' "$DEST/$DUMP.sha256")"
if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$DEST/$DUMP" | awk '{print $1}')"
else
    actual="$(shasum -a 256 "$DEST/$DUMP" | awk '{print $1}')"
fi
if [ "$expected" != "$actual" ]; then
    echo "checksum mismatch for $DEST/$DUMP" >&2
    echo "  expected $expected" >&2
    echo "  actual   $actual" >&2
    exit 1
fi
echo "checksum ok: $actual"

# ── 3. restore ──────────────────────────────────────────────────────────────────────
# Postgres must be running: $COMPOSE up -d
$COMPOSE exec -T postgres pg_isready >/dev/null

# TODO: exact restore commands go here (the tested sequence is being produced separately).
# The dump is on the host at $DEST/$DUMP; stream it into the container, e.g. with
#   $COMPOSE exec -T postgres sh -c 'pg_restore ...' < "$DEST/$DUMP"
# Then check that both indexes exist -- an exit code alone is not a restore check:
#   chunks_embedding_hnsw (vector) and chunks_bm25_idx (pg_search).
echo "TODO: restore step not filled in yet; see scripts/setup_db.sh" >&2
exit 1
