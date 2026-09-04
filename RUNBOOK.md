# HPC runbook

Every GPU job in this project uses the same split-job pattern (`DECISIONS.md` `ARM3-2`):
the laptop does the cheap work and writes a payload file, a self-contained checkpointed
script runs on the GPU node with **no database and no network dependency**, and the laptop
loads the results afterwards. A dropped SSH connection can therefore never corrupt a run.

## The cluster

Northeastern **Explorer** (SLURM, V100). Note: `ARM3-2` and the two `.sbatch` files still
say `discovery.neu.edu` — that was the previous cluster name and those references are stale.

| | |
|---|---|
| login | `ssh tuteja.a@login.explorer.northeastern.edu` |
| file transfer | the `xfer` host — **never** scp through the login node, it throttles and kills large transfers. **Exact hostname unconfirmed for Explorer**: `ARM3-2` recorded `xfer.discovery.neu.edu` under the old cluster name. Fill it in below the first time you use it. |
| environment | `module load python/3.13.5` then `source ~/rerank-env/bin/activate` |
| what the venv has | `sentence-transformers`, `torch` 2.5.1+cu121, `tqdm`. No `rag_sec`, no DB driver, by design |
| allocation | `srun --partition=gpu --gres=gpu:v100-sxm2:1 --cpus-per-task=4 --mem=48G --time=08:00:00 --pty /bin/bash` |

**Interactive `srun` is the default**, for live visibility. Always start `tmux` *before*
`srun` — an `srun --pty` session dies with the SSH connection and these jobs run for hours.
The two `.sbatch` files are an unattended fallback, not the normal path.

If two jobs are queued, stagger the second past the first's model load: both read the same
model from one `~/.cache/huggingface`, and two cold downloads to one path risk corrupting it.

## General shape

```bash
NEU=tuteja.a

# 1. laptop: build the payload, ship it through xfer
scp <payload>.json  $NEU@<xfer-host>:~/
scp <gpu-script>.py $NEU@<xfer-host>:~/

# 2. cluster: allocate, then run inside tmux
ssh $NEU@login.explorer.northeastern.edu
tmux new -s <job>
srun --partition=gpu --gres=gpu:v100-sxm2:1 --cpus-per-task=4 --mem=48G \
     --time=08:00:00 --pty /bin/bash
module load python/3.13.5
source ~/rerank-env/bin/activate
cd ~ && python -u <gpu-script>.py <payload>.json <results>.jsonl

# 3. laptop: bring results back and load
scp $NEU@<xfer-host>:~/<results>.jsonl data/
```

Detach `Ctrl-b d`, reattach `tmux attach -t <job>`. Every GPU script here checkpoints per
id and skips what is already in the output file, so an interrupted run is resumed by
re-running the identical command — never by starting over.

**Check the first line of output is `Using device: cuda`.** If it says `cpu` or `mps`, stop:
a BGE-M3 pass on laptop MPS was measured at 1.4 chunks/s, a 9.3-hour ETA for a job that is
about two hours on a V100.

## Job: RETR-7/RETR-8 re-index (INFRA-12)

Re-embeds only the 47,312 of 99,654 chunks (47.5%) whose text changed. Chunk boundaries are
preserved, so this is an `UPDATE` keyed on `(filing_stem, chunk_index, variant='A')`, not a
rebuild, and gold labels keep pointing at the same body text.

**Before anything: back up.** The Postgres volume has no backup and holds hours of GPU work.

```bash
mkdir -p ~/rag-sec-backups
docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc -Z6' \
  > ~/rag-sec-backups/chunks_$(date +%Y%m%d).dump
docker compose exec -T postgres sh -c 'pg_restore --list' \
  < ~/rag-sec-backups/chunks_$(date +%Y%m%d).dump | grep -c "TABLE DATA public chunks"   # expect 1
tar -czf ~/rag-sec-backups/chunks_json_$(date +%Y%m%d).tgz data/chunks
```

```bash
# 1. laptop: re-chunk with both flags on, then build the payload
RAG_SEC_MULTI_HEADING=1 RAG_SEC_STRIP_TITLE_FURNITURE=1 \
  .venv/bin/python scripts/corpus/build_corpus.py --rechunk

mkdir -p data/chunks_pre_retr7
tar -xzf ~/rag-sec-backups/chunks_json_*.tgz -C data/chunks_pre_retr7 --strip-components=2

.venv/bin/python scripts/index/reembed_changed.py --prepare data/retr7_embed_payload.json

# 2. ship up (payload is 164 MB -- xfer host, not login)
scp data/retr7_embed_payload.json $NEU@<xfer-host>:~/
scp scripts/index/embed_hpc.py    $NEU@<xfer-host>:~/

# 3. cluster
ssh $NEU@login.explorer.northeastern.edu
tmux new -s reembed
srun --partition=gpu --gres=gpu:v100-sxm2:1 --cpus-per-task=4 --mem=48G \
     --time=08:00:00 --pty /bin/bash
module load python/3.13.5
source ~/rerank-env/bin/activate
cd ~ && python -u embed_hpc.py retr7_embed_payload.json retr7_embed_results.jsonl

# 4. laptop: apply (~1.03 GB comes back)
scp $NEU@<xfer-host>:~/retr7_embed_results.jsonl data/
.venv/bin/python scripts/index/reembed_changed.py --load data/retr7_embed_results.jsonl

# 5. verify
.venv/bin/python scripts/checks/candidate_sql.py
.venv/bin/python scripts/checks/variant_predicates.py
.venv/bin/python scripts/checks/atom_replay.py       # expect 99,654/99,654
```

`embed_hpc.py` is **unchanged** from the day-5 corpus-growth run: its payload format is
already `{filing_stem, chunk_index, text}` and its output `{filing_stem, chunk_index,
embedding}`. Nothing to port.

The `--load` stage refuses rather than half-applies if any changed chunk has no embedding,
refuses if any filing's chunk count moved, and drops/rebuilds HNSW + BM25 around the bulk
update. Row counts never change, so `store.preflight` stays satisfied.

**Afterwards:** re-score Arm 3 + filter + strip on dev and test, and bundle in Arm 1 and
Arm 2, which have owed a retrieval re-run since `RETR-35`. Then delete
`data/retr7_embed_payload.json` and `data/chunks_pre_retr7/` — the laptop disk runs tight.

**Scale anchors, not measurements.** The day-5 pass was 26,737 chunks and its payload and
results timestamps are 68 minutes apart, so that whole cycle was under an hour and a bit;
this job is 1.77x it. No wall time for a BGE-M3 embed pass has ever been logged — if you
run this, log it and give it a `DECISIONS.md` row.
