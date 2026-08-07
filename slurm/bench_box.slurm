#!/bin/bash
#SBATCH --job-name=feetgp_box
#SBATCH --output=logs/bench/box_%j.out
#SBATCH --error=logs/bench/box_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|v100"
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8

set -euo pipefail
mkdir -p logs logs/bench
echo "Node: $(hostname)  Date: $(date)"

# at lambda=0 the group-lasso prox is the identity, so x = z + u exactly and the primal
# residual can only come from the theta box clip in z_and_u_update. Measured on the
# smoke pickles: r=2.857e-01 of which r_box=2.850e-01. If the box is the whole lambda=0
# stall, widening it drives the residual to tol; if x runs away, max|theta| keeps
# growing with theta_scale and the box is load-bearing after all.
for scale in 1 2 10 100; do
    echo "### theta_scale=${scale}"
    uv run python -u -m bench.bench_fit \
        --subsample 20 --feet both --target markers --l1_penalty 0.0 \
        --maxiters 50 100 200 300 --chunks 39 \
        --inner_maxiter 5 --inner_tol 1e-2 \
        --theta_scale "${scale}" --log_every 25
done

echo "Done: $(date)"
