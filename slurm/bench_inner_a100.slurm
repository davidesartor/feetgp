#!/bin/bash
#SBATCH --job-name=feetgp_inner_a100
#SBATCH --output=logs/bench/inner_a100_%j.out
#SBATCH --error=logs/bench/inner_a100_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8

set -euo pipefail
mkdir -p logs logs/bench

echo "=========================================="
echo "Node:       $(hostname)"
echo "GPUs:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c)"
echo "Date:       $(date)"
echo "=========================================="

# is the 1000-step inner cap real work or a termination criterion that never fires?
uv run python -u -m bench.bench_inner \
    --subsample 20 \
    --feet both \
    --target markers \
    --l1_penalty 0.0 \
    --warm_iters 0 3 8 \
    --inner_maxiter 1000 \
    --chunk_size 32
