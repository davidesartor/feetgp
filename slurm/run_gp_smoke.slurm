#!/bin/bash
#SBATCH --job-name=feetgp_v4
#SBATCH --output=logs/gp/v4_%j.out
#SBATCH --error=logs/gp/v4_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|h100"
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

set -euo pipefail
mkdir -p logs logs/gp

echo "Node: $(hostname)  Date: $(date)"

# one config, into a fresh output_dir: every pickle under results/ predates the theta
# box being restored, and fit_or_load would happily reuse those as results
uv run python -u -m feetgp.run \
    --output_dir results_v4 \
    --subsample 20 --target markers --inclines all --feet both \
    --chunk_size 39 \
    --maxiter 400 \
    --inner_maxiter 50 \
    --inner_tol 1e-4 \
    --tol 1e-3 \
    --log_every 50 \
    --lambda_step 1.3 \
    --lambda_budget 60 \
    --lambda_refine 100

echo "Done: $(date)"
