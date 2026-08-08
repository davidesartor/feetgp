#!/bin/bash
#SBATCH --job-name=feetgp_traced
#SBATCH --output=logs/bench/traced_%j.out
#SBATCH --error=logs/bench/traced_%j.err
#SBATCH --time=02:00:00
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
echo "Revision:   $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"
echo "Date:       $(date)"
echo "=========================================="

# does the static bound only win because tiny budgets unroll?
uv run python -u -m bench.bench_traced \
    --subsample 20 --feet both --chunk_size 78 \
    --budgets 1 5 10 30 --repeats 5
