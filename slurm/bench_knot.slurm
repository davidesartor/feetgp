#!/bin/bash
#SBATCH --job-name=knot
#SBATCH --output=logs/gp/knot_%j.out
#SBATCH --error=logs/gp/knot_%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|h100"
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

set -euo pipefail
mkdir -p logs logs/gp
echo "Node: $(hostname)  $(date)"

uv run python -u -m bench.bench_knot \
    --run_dir results_v5/model=gp/target=markers/feet=both/inclines=all_sub=20 \
    --warm_lambda 26.90332801 --knot_lambda 34.97432641 \
    --subsample 20 --feet both --target markers \
    --maxiter 40 --chunk_size 39 \
    --inner_maxiters 200 50 \
    --rhos 8.0
