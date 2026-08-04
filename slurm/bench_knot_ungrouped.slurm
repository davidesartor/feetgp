#!/bin/bash
#SBATCH --job-name=knotu
#SBATCH --output=logs/gp/knotu_%j.out
#SBATCH --error=logs/gp/knotu_%j.err
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
    --run_dir "results_v5/model=gp/target=markers/feet=both_ungrouped/inclines=all_sub=20" \
    --warm_lambda 0.0 --knot_lambda 0.2165895934 \
    --subsample 20 --feet both --ungroup_feet --target markers \
    --maxiter 40 --chunk_size 39 \
    --inner_maxiters \
    --rhos 1.0 0.5 4.0 16.0
