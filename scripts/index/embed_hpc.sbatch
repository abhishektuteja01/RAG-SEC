#!/bin/bash
# RETR-7/RETR-8 re-index (INFRA-12): embed the 47,312 chunks whose text changed.
# Node setup mirrors DECISIONS.md ARM3-2's interactive recipe.
#
# Payload comes from `reembed_changed.py --prepare` on the laptop. embed_hpc.py is
# UNCHANGED from the day-5 corpus-growth run -- its payload format is already
# {filing_stem, chunk_index, text} and its output {filing_stem, chunk_index, embedding}.
#
# Resumable: embed_hpc.py skips ids already in the output file, so hitting the wall
# limit is not a loss -- resubmit this same script.
#
# Scale anchor: the day-5 pass was 26,737 chunks; this is 1.77x that. 04:00:00 is
# headroom, not an estimate -- no wall time was ever logged for that run.
#
#   sbatch scripts/index/embed_hpc.sbatch      # from ~ on the cluster
#
#SBATCH --job-name=retr7-reembed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load python/3.13.5
source ~/rerank-env/bin/activate
cd ~
python -u embed_hpc.py \
    retr7_embed_payload.json \
    retr7_embed_results.jsonl
