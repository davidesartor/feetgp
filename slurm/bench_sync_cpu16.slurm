#!/bin/bash
#SBATCH --job-name=feetgp_bench_cpu16
#SBATCH --output=logs/bench/cpu16_%j.out
#SBATCH --error=logs/bench/cpu16_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=cpu
#SBATCH --constraint="zen4"
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=16

set -euo pipefail
mkdir -p logs logs/bench

echo "=========================================="
echo "Node:       $(hostname)"
echo "CPUs:       $(nproc)"
echo "Model:      $(lscpu | grep 'Model name' | head -1)"
echo "Date:       $(date)"
echo "=========================================="

uv run python -u -m bench.bench_sync \
    --subsample 20 \
    --feet both \
    --target markers \
    --chunks 1 2 4 8 16 32
