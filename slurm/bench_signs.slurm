#!/bin/bash
#SBATCH --job-name=feetgp_signs2
#SBATCH --output=logs/bench/signs_%j.out
#SBATCH --error=logs/bench/signs_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|v100|2080ti|1080ti"
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8

set -euo pipefail
mkdir -p logs logs/bench
echo "Node: $(hostname)  Date: $(date)"

# lambdas straddling the death region of the sub=200 smoke run, where the residual was
# measured to be the prox and not the box clip (r=1.282e+01, r_box=8.165e+00 at 197)
for l1 in 116.6 197.1 256.2; do
    echo "### lambda=${l1}"
    uv run python -u -m bench.bench_signs \
        --subsample 200 --feet both --target markers --l1_penalty "${l1}" \
        --iterations 150 --chunk_size 39 --inner_maxiter 20 --inner_tol 1e-4 \
        --adapt_rho_iters 75 --log_every 10
done

echo "Done: $(date)"
