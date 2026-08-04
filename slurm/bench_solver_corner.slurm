#!/bin/bash
#SBATCH --job-name=solvcnr
#SBATCH --output=logs/gp/solvcnr_%j.out
#SBATCH --error=logs/gp/solvcnr_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

set -euo pipefail
mkdir -p logs/gp
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# results_v6 was deleted on 2026-08-04, so this needs a fresh run dir before it can be
# rerun -- the lambda values below are the v6 grid and will not exist in a new one
RUN="/home/dsartor_umass_edu/feetgp/results_v6/model=gp/target=markers/feet=both/inclines=all_sub=20"

# The two cheap levers have only been measured one at a time: inner_maxiter=3 at the default
# line search (16 iters, 135.9s) and inner_maxiter=12 under a capped one (25 iters, 272.6s).
# At maxiter=3 the straggler still runs 43 evaluations, so the cap should still bite.
echo "### knot lambda=34.97 warmstarted from 26.90, both levers at once"
echo "### optimistix reference:            35 iters, 488.7s (13.96s/iter)"
echo "### L-BFGS-B maxiter=3  ls=30:       16 iters, 135.9s ( 8.49s/iter)"
echo "### L-BFGS-B maxiter=12 ls=5:        25 iters, 272.6s (10.90s/iter)"
uv run python -u -m bench.bench_solver --run_dir "$RUN" \
    --warm_lambda 26.90332801 --knot_lambda 34.97432641 \
    --subsample 20 --feet both --target markers \
    --maxiter 60 --chunk_size 39 --inner_tols 1e-2 \
    --inner_maxiters 3 5 --inner_max_linesearch 5
