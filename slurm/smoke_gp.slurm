#!/bin/bash
#SBATCH --job-name=feetgp_smoke
#SBATCH --output=logs/smoke/%j.out
#SBATCH --error=logs/smoke/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|h100"
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8

set -euo pipefail
mkdir -p logs logs/smoke

# cheap end-to-end shape check: does the path kill groups one at a time, and does
# every lambda actually converge now that the nugget is out of the residual
uv run python -u -m feetgp.run \
    --subsample 200 \
    --target markers \
    --inclines all \
    --feet both \
    --chunk_size 39 \
    --maxiter 300 \
    --inner_maxiter 20 \
    --inner_tol 1e-4 \
    --tol 1e-3 \
    --lambda_step 1.3 \
    --lambda_budget 60 \
    --output_dir results_smoke
