#!/bin/bash
#SBATCH --job-name=feetgp_bench_a100
#SBATCH --output=logs/bench/a100_%j.out
#SBATCH --error=logs/bench/a100_%j.err
#SBATCH --time=08:00:00
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

# same shape as the real a100 jobs: sub=20, both feet grouped
uv run python -u -m bench.bench_sync \
    --subsample 20 \
    --feet both \
    --target markers \
    --chunks 1 2 4 8 16 32
