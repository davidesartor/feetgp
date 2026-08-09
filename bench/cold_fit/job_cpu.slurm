#!/bin/bash
#SBATCH --job-name=feetgp_cold_fit_cpu
#SBATCH --output=bench/cold_fit/logs/cold_fit_cpu_%A_%a.out
#SBATCH --error=bench/cold_fit/logs/cold_fit_cpu_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-8

set -euo pipefail
mkdir -p bench/cold_fit/logs

DTYPE=${DTYPE:-float64}
PROFILE=${PROFILE:-rbf}
TRAIN_SIZES=(16 32 64 128 256 512 1024 2048 4096)
TRAIN_SIZE=${TRAIN_SIZES[$SLURM_ARRAY_TASK_ID]}

# no visible gpu, and jax threads are only the cores slurm actually gave us
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "=========================================="
echo "Node:  $(hostname)"
echo "CPU:   $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs) x ${SLURM_CPUS_PER_TASK} cores"
echo "Date:  $(date)"
echo "Train: n=${TRAIN_SIZE} dtype=${DTYPE} profile=${PROFILE}"
echo "=========================================="

# cold lambda=0 fit, cost is maxiter x x_update, so per-iteration time extrapolates
uv run python -u -m bench.cold_fit.bench \
    --inclines inc0 --target markers --feet both \
    --train_sizes "${TRAIN_SIZE}" --dtype "${DTYPE}" --profile "${PROFILE}" \
    --max_repeats 12 --time_budget_s 6000
