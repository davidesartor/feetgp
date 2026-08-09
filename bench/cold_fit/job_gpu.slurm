#!/bin/bash
#SBATCH --job-name=feetgp_cold_fit
#SBATCH --output=bench/cold_fit/logs/cold_fit_%A_%a.out
#SBATCH --error=bench/cold_fit/logs/cold_fit_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-8

set -euo pipefail
mkdir -p bench/cold_fit/logs

DTYPE=${DTYPE:-float64}
PROFILE=${PROFILE:-rbf}
ALL_SIZES=(16 32 64 128 256 512 1024 2048 4096)
# one size per array task, unless the caller asked for a whole sweep in one job
TRAIN_SIZE=${SIZES:-${ALL_SIZES[$SLURM_ARRAY_TASK_ID]}}

echo "=========================================="
echo "Node:  $(hostname)"
echo "GPUs:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c)"
echo "Date:  $(date)"
echo "Train: n=${TRAIN_SIZE} dtype=${DTYPE} profile=${PROFILE}"
echo "=========================================="

# cold lambda=0 fit, cost is maxiter x x_update, so per-iteration time extrapolates
uv run python -u -m bench.cold_fit.bench \
    --inclines inc0 --target markers --feet both \
    --train_sizes ${TRAIN_SIZE} --dtype "${DTYPE}" --profile "${PROFILE}" \
    --max_repeats 12 --time_budget_s "${BUDGET:-1200}"
