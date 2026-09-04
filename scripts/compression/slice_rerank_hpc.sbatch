#!/bin/bash
# COST-17/RETR-30: rerank slices on the corrected (filtered+stripped) pool.
# 555,337 slices, ~4.3 GPU-h est. See scripts/retrieval/rerank_hpc.sbatch for why sbatch.
#
# Submit this SECOND, once the test job is past model load: both jobs read
# BAAI/bge-reranker-v2-m3 from ~/.cache/huggingface, and two simultaneous cold downloads to
# one cache path is an avoidable corruption risk.
#
#   sbatch scripts/compression/slice_rerank_hpc.sbatch
#
#SBATCH --job-name=day8-slices
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load python/3.13.5
source ~/rerank-env/bin/activate
cd ~
python -u slice_rerank_hpc.py \
    day8_slice_payload_t150_filtered_stripped.json \
    day8_slice_scores_t150_filtered_stripped.jsonl
