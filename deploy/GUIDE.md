# Deploying on a GPU host

Both containers (Postgres and the API) on one NVIDIA GPU machine. Written for an AWS
`g4dn.xlarge` (Tesla T4, x86_64, 16 GiB RAM); any x86_64 Linux box with an NVIDIA GPU works
the same way.

What to expect on a `g4dn.xlarge`: rerank about 3.3 s and generation about 2.6-3.2 s per
question, so `/ask` takes about 6.7 s (median). The CPU-only image gives the same answers but
reranks roughly 40x slower.

## 1. Host

- Ubuntu 22.04 or 24.04, x86_64. Give the root volume at least 60 GB: the GPU API image
  carries CUDA libraries and both models, and the database adds several GB.
- Open port 8080 (the API) in the security group. Keep 5432 closed; the API reaches
  Postgres over the compose network.
- An Elastic IP bills while the instance is stopped. If you stop the host between uses, the
  free auto-assigned public IP (it changes on each start) is the cheaper choice.

## 2. NVIDIA driver, Docker, container toolkit

```bash
# NVIDIA driver (the image uses CUDA 13.0 wheels, so the driver must support CUDA 13.0)
sudo apt-get update
sudo ubuntu-drivers install          # or a specific nvidia-driver-XXX package
sudo reboot
nvidia-smi                           # must list the GPU and a CUDA version >= 13.0

# Docker Engine + compose plugin: https://docs.docker.com/engine/install/ubuntu/
sudo usermod -aG docker $USER && newgrp docker

# NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# (add NVIDIA's apt repository as that page shows, then:)
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# check that containers can see the GPU
docker run --rm --gpus all ubuntu nvidia-smi
```

If the last command fails, fix it before going on. The API image forces `cuda` on purpose,
so without a working toolkit it fails at startup instead of running slowly on the CPU.

## 3. Code and config

```bash
git clone https://github.com/abhishektuteja01/RAG-SEC.git && cd RAG-SEC
cp .env.example .env        # set POSTGRES_PASSWORD and GOOGLE_API_KEY
curl -LsSf https://astral.sh/uv/install.sh | sh   # setup_db.sh uses `uv run hf download`
```

## 4. Start Postgres and load the database

```bash
COMPOSE="docker compose -f deploy/docker-compose.gpu.yml --env-file .env"
$COMPOSE up -d postgres
COMPOSE="$COMPOSE" scripts/setup_db.sh
```

`shm_size: 2gb` in the compose file is required: with Docker's default 64 MB of shared
memory the HNSW index build fails with a misleading "No space left on device", and the
restore still exits looking mostly fine. Check that both indexes exist afterwards:

```bash
$COMPOSE exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\di chunks*"'
# expect chunks_embedding_hnsw and chunks_bm25_idx
```

## 5. Build and start the API

```bash
$COMPOSE up -d --build api   # first build downloads CUDA torch and both models (slow)
$COMPOSE logs -f api         # wait for "Application startup complete" (~20 s model warm-up)
```

The container needs about 6 GB of RAM (the two models are ~4.5 GB). Running the CPU image
under Docker Desktop, raise its memory limit to 8 GB, or the warm-up is killed.

Check it:

```bash
curl localhost:8080/health   # {"status":"ok"}         -- process is up
curl localhost:8080/ready    # {"status":"ready"}      -- DB reachable, 99,654 chunks present
curl -s localhost:8080/ask -H 'Content-Type: application/json' \
     -d '{"question": "What was Visa Inc.'\''s net revenue in 2015?"}'
```

The chat page is at `http://<host>:8080/`.

## Notes

- One worker per container, on purpose: a second worker loads a second copy of both models.
  Scale by running more containers.
- `/ready` fails if the table does not hold exactly 99,654 variant-A chunks, each with an
  embedding. That is deliberate: a half-loaded corpus should fail the health check, not
  quietly return worse answers.
- After a stop/start the containers come back on their own (`restart: unless-stopped`).
  Re-run the three `curl` checks.
