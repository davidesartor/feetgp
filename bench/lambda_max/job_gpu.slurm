#!/bin/bash
#SBATCH --job-name=feetgp_lambda_max
#SBATCH --output=bench/lambda_max/logs/lambda_max_%A_%a.out
#SBATCH --error=bench/lambda_max/logs/lambda_max_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-6

set -euo pipefail
mkdir -p bench/lambda_max/logs

PROFILE=${PROFILE:-rbf}
ALL_SIZES=(32 64 128 256 512 1024 2048)
# one size per array task, unless the caller asked for a whole sweep in one job
TRAIN_SIZE=${SIZES:-${ALL_SIZES[$SLURM_ARRAY_TASK_ID]}}

echo "=========================================="
echo "Node:  $(hostname)"
echo "GPUs:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c)"
echo "Date:  $(date)"
echo "Train: n=${TRAIN_SIZE} profile=${PROFILE}"
echo "=========================================="

# lambda_max must kill every group above it and let one back in below it
uv run python -u -m bench.lambda_max.bench \
    --inclines inc0 --target forces --feet both \
    --train_sizes ${TRAIN_SIZE} --profile "${PROFILE}" \
    --time_budget_s "${BUDGET:-2400}" ${EXTRA_ARGS:-}
