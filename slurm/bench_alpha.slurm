#!/bin/bash
#SBATCH --job-name=feetgp_alpha
#SBATCH --output=logs/bench/alpha_%j.out
#SBATCH --error=logs/bench/alpha_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8

set -euo pipefail
mkdir -p logs logs/bench
echo "Node: $(hostname)  Date: $(date)"

# the residual stalls around 3-4 rather than falling, which is the signature of a
# constant inexact-subproblem error. two candidate causes, tested independently:
# a loose inner solve (inner 20/1e-4 vs 200/1e-6) and a slow ADMM rate (alpha).
for cfg in "20 1e-4 1.0" "200 1e-6 1.0" "20 1e-4 1.5" "200 1e-6 1.5"; do
    set -- $cfg
    echo "### inner_maxiter=$1 inner_tol=$2 alpha=$3"
    uv run python -u -m bench.bench_fit \
        --subsample 20 --feet both --target markers --l1_penalty 0.0 \
        --maxiters 50 100 150 --chunks 39 \
        --inner_maxiter "$1" --inner_tol "$2" --alpha "$3"
done
