#!/bin/bash
#SBATCH --job-name=feetgp_straggler_1080ti
#SBATCH --output=logs/bench/straggler_1080ti_%j.out
#SBATCH --error=logs/bench/straggler_1080ti_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1080_ti:1
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8

set -euo pipefail
mkdir -p logs logs/bench

echo "=========================================="
echo "Node:       $(hostname)"
echo "GPUs:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c)"
echo "Revision:   $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"
echo "Date:       $(date)"
echo "=========================================="

# one ADMM iteration, full inner budget on iteration 0: are the vmap lanes ragged?
uv run python -u -m bench.bench_straggler \
    --subsample 20 --feet both --target markers \
    --chunks 78 39 13 --inner_maxiter 5 --inner_maxiter_init 5
