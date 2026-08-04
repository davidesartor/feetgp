#!/bin/bash
#SBATCH --job-name=chunk
#SBATCH --output=logs/gp/chunk_%j.out
#SBATCH --error=logs/gp/chunk_%j.err
#SBATCH --time=00:40:00
#SBATCH --qos=short
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run python -u -m bench.bench_chunk \
    --run_dir "results_v6/model=gp/target=markers/feet=both/inclines=all_sub=20" \
    --warm_lambda 26.90332801 --l1 34.97432641 \
    --subsample 20 --feet both --target markers \
    --maxiter 10 --chunk_sizes 26 39 78
