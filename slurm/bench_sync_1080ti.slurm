#!/bin/bash
#SBATCH --job-name=feetgp_bench_1080ti
#SBATCH --output=logs/bench/1080ti_%j.out
#SBATCH --error=logs/bench/1080ti_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="1080ti"
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

# defaults to the real 1080ti job shape (sub=100); override to compare against
# the sub=20 runs on other devices
uv run python -u -m bench.bench_sync \
    --subsample "${SUBSAMPLE:-100}" \
    --feet both \
    --target markers \
    --chunks 1 2 4 8 16 32
