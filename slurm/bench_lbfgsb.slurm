#!/bin/bash
#SBATCH --job-name=benchlbfgsb
#SBATCH --output=logs/bench_lbfgsb/%x_%j.out
#SBATCH --error=logs/bench_lbfgsb/%x_%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
mkdir -p logs logs/bench_lbfgsb results_bench

TAG="${SLURM_JOB_NAME}"

echo "=========================================="
echo "Node:  $(hostname)"
echo "GPUs:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c)"
echo "Cores: $(nproc)"
echo "Date:  $(date)"
echo "=========================================="

uv run python -m bench.bench_lbfgsb \
    --tag "$TAG" \
    --starts 16 \
    --batch 1024 \
    --repeats 3 \
    --out "results_bench/${TAG}.json"
