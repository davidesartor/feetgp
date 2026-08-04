#!/bin/bash
#SBATCH --job-name=feetgp_rho
#SBATCH --output=logs/bench/rho_%j.out
#SBATCH --error=logs/bench/rho_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8

set -euo pipefail
mkdir -p logs logs/bench
echo "Node: $(hostname)  Date: $(date)"

# at iteration 150 the alpha bench sat at r=5.35 s=7.26e-01: the primal residual is 7x
# the dual, just under the 10x that would double rho, and rho stopped adapting at 75
# anyway. Does the freeze hold the residual up, or is it slow regardless?
for frac in 0.5 1.0; do
    echo "### adapt_rho_frac=${frac}"
    uv run python -u -m bench.bench_fit \
        --subsample 20 --feet both --target markers --l1_penalty 0.0 \
        --maxiters 50 100 200 300 --chunks 39 \
        --inner_maxiter 20 --inner_tol 1e-4 --alpha 1.0 \
        --adapt_rho_frac "${frac}" --log_every 25
done

echo "Done: $(date)"
