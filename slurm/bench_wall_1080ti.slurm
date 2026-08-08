#!/bin/bash
#SBATCH --job-name=feetgp_wall_1080ti
#SBATCH --output=logs/bench/wall_1080ti_%j.out
#SBATCH --error=logs/bench/wall_1080ti_%j.err
#SBATCH --time=08:00:00
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

# how long does one ADMM fit take under the post-optimistix, post-alpha code
uv run python -u -m bench.bench_wall \
    --subsample 20 --feet both --target markers \
    --chunk_size 78 --maxiter 300 --lambdas 0.0 1.0
