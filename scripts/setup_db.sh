#!/usr/bin/env bash
# Download the prebuilt chunks database (99,654 chunks, embeddings and both indexes) from
# Hugging Face, verify its checksum, and restore it into the compose Postgres.
# Reads POSTGRES_USER and POSTGRES_DB from .env itself.
#
#   docker compose up -d          # start Postgres first
#   scripts/setup_db.sh
#
# Already have the dump? Skip the download (the checksum is still verified):
#   DUMP_FILE=/path/to/rag_sec_chunks.dump scripts/setup_db.sh
#
# On the GPU host, point it at that compose file:
#   COMPOSE="docker compose -f deploy/docker-compose.gpu.yml --env-file .env" scripts/setup_db.sh
set -euo pipefail

cd "$(dirname "$0")/.."

REPO="abhishektuteja/rag-sec-db"
DUMP="rag_sec_chunks.dump"
SHA256="5782b9d4140fdb56626774ebf7e82a491dc5affbfb56bee00a5576abc3632c6a"
ROWS=99654
COMPOSE="${COMPOSE:-docker compose}"

set -a; . ./.env; set +a

psql() { $COMPOSE exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 "$@"; }

# ── 1. download ─────────────────────────────────────────────────────────────────────
if [ -n "${DUMP_FILE:-}" ]; then
    file="$DUMP_FILE"
else
    mkdir -p data/db
    uv run hf download "$REPO" "$DUMP" --repo-type dataset --local-dir data/db
    file="data/db/$DUMP"
fi

# ── 2. verify ───────────────────────────────────────────────────────────────────────
if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$file" | awk '{print $1}')"
else
    actual="$(shasum -a 256 "$file" | awk '{print $1}')"
fi
if [ "$actual" != "$SHA256" ]; then
    echo "checksum mismatch for $file" >&2
    echo "  expected $SHA256" >&2
    echo "  actual   $actual" >&2
    exit 1
fi
echo "checksum ok"

# ── 3. restore ──────────────────────────────────────────────────────────────────────
$COMPOSE exec -T postgres pg_isready >/dev/null

# The restore uses --clean, which drops the existing table. Refuse if it holds anything
# but variant 'A' (an old research database).
other=0
if [ -n "$(psql -tA -c "SELECT to_regclass('public.chunks')")" ]; then
    other="$(psql -tA -c "SELECT count(*) FROM chunks WHERE variant <> 'A'")"
fi
if [ "$other" != "0" ]; then
    echo "refusing: chunks has $other rows outside variant 'A'; this looks like a research" >&2
    echo "database, and the restore would drop them. Use a fresh volume." >&2
    exit 1
fi

psql -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_search;"
$COMPOSE cp "$file" postgres:/tmp/$DUMP
$COMPOSE exec -T postgres pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    --no-owner --no-privileges --clean --if-exists --exit-on-error -j 4 /tmp/$DUMP
$COMPOSE exec -T postgres rm -f /tmp/$DUMP
psql -c "ANALYZE chunks;"

# ── 4. check ────────────────────────────────────────────────────────────────────────
# An exit code alone is not a restore check.
rows="$(psql -tA -c "SELECT count(*) FROM chunks")"
idx="$(psql -tA -c "SELECT count(*) FROM pg_indexes WHERE tablename = 'chunks'
                     AND indexname IN ('chunks_embedding_hnsw', 'chunks_bm25_idx')")"
if [ "$rows" != "$ROWS" ] || [ "$idx" != "2" ]; then
    echo "restore check failed: $rows rows (want $ROWS), $idx of 2 indexes" >&2
    exit 1
fi
echo "restored: $rows rows, both indexes present"
