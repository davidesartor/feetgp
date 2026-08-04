#!/bin/bash
#SBATCH --job-name=feetgp_fit_a100
#SBATCH --output=logs/bench/fit2_a100_%j.out
#SBATCH --error=logs/bench/fit2_a100_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8

set -euo pipefail
mkdir -p logs logs/bench

echo "=========================================="
echo "Node:       $(hostname)"
echo "GPUs:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c)"
echo "Date:       $(date)"
echo "=========================================="

# does the fit actually reach the residual tolerance now that the nugget saturates
# instead of being clipped? the old clip pinned primal at ~3.5 forever at every lambda
for inner in 20 50; do
    echo "### inner_maxiter=${inner}"
    uv run python -u -m bench.bench_fit \
        --subsample 20 --feet both --target markers --l1_penalty 0.0 \
        --maxiters 25 50 100 200 300 450 600 --chunks 39 \
        --inner_maxiter "${inner}" --inner_tol 1e-4
done
