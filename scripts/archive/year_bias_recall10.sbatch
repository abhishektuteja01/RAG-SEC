#!/bin/bash
# Year-bias recall@10 diagnostic: rerank the union candidate pool per question so both
# the base and year-biased conditions can be scored from one pass. See
# year_bias_recall10_prepare.py / year_bias_recall10_hpc.py for the full picture.
#
# Node setup mirrors DECISIONS.md ARM3-2's interactive recipe (scripts/pipeline/hpc/).
# Batch, not `srun --pty`+tmux: sbatch needs no terminal to stay alive.
# Resumable: year_bias_recall10_hpc.py checkpoints per question id.
#
#   sbatch year_bias_recall10.sbatch      # from ~ on the cluster
#
#SBATCH --job-name=year-bias-recall10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
# dev (72,526 pairs) + test (90,758 pairs) est. ~3.85 GPU-h combined at the measured
# ~11.76 pairs/s V100 rate (ARM3-2/RETR-18) -- 5h leaves ~30% margin, same convention
# as the original Arm-3 test job (7.3h estimate against an 8h limit).

set -euo pipefail
module load python/3.13.5
source ~/rerank-env/bin/activate
cd ~
python -u year_bias_recall10_hpc.py \
    year_bias_recall10_dev_payload.json \
    year_bias_recall10_dev_scores.jsonl
python -u year_bias_recall10_hpc.py \
    year_bias_recall10_test_payload.json \
    year_bias_recall10_test_scores.jsonl
