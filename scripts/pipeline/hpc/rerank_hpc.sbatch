#!/bin/bash
# RETR-18: rerank the untouched test split, full 2x2 (309,160 pairs, ~7.3 GPU-h est).
# Node setup mirrors DECISIONS.md ARM3-2's interactive recipe.
#
# Batch, not `srun --pty`+tmux: this job and the slice job are independent and can hold two
# separate GPUs at once, and sbatch needs no terminal to stay alive for 7+ hours.
#
# Resumable: rerank_hpc.py checkpoints per question id and skips what is already
# in the output file, so hitting the wall limit is not a loss -- resubmit this same script.
#
#   sbatch scripts/pipeline/hpc/rerank_hpc.sbatch      # from ~ on the cluster
#
#SBATCH --job-name=retr18-test
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
python -u rerank_hpc.py \
    day8_retr18_test_payload.json \
    day8_retr18_test_scores.jsonl
