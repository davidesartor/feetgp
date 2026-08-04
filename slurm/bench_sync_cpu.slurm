#!/bin/bash
#SBATCH --job-name=feetgp_bench_cpu
#SBATCH --output=logs/bench/cpu_%j.out
#SBATCH --error=logs/bench/cpu_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=cpu
#SBATCH --constraint="zen4"
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=64

set -euo pipefail
mkdir -p logs logs/bench

echo "=========================================="
echo "Node:       $(hostname)"
echo "CPUs:       $(nproc)"
echo "Model:      $(lscpu | grep 'Model name' | head -1)"
echo "Date:       $(date)"
echo "=========================================="

# the x-update is 78 independent solves, so the interesting axis is how far
# chunk_size scales before the batched n*n kernels stop fitting in cache
uv run python -u -m bench.bench_sync \
    --subsample 20 \
    --feet both \
    --target markers \
    --chunks 1 2 4 8 16 32
