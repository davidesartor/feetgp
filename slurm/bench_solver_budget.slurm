#!/bin/bash
#SBATCH --job-name=solvbud
#SBATCH --output=logs/gp/solvbud_%j.out
#SBATCH --error=logs/gp/solvbud_%j.err
#SBATCH --time=04:00:00
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

# bench_budget measured one x-update at 7.60 / 12.49 / 19.21 / 28.24s for maxiter 3 / 5 / 8 / 12,
# i.e. ~2.35s per outer iteration regardless of budget -- the lax.map straggler sets the pace, and
# at_cap stays at 42-74 of 78 throughout. Against optimistix's 13.96s per whole ADMM iteration that
# puts the cost-matched budget at inner_maxiter ~5. This asks whether a fit at that budget still
# converges in the ~27 ADMM iterations the expensive one needed.
echo "### knot lambda=34.97 warmstarted from 26.90, inner_maxiter sweep"
echo "### optimistix reference: 35 iters, 488.7s (13.96s/iter), primal=1.63e-1"
echo "### L-BFGS-B at inner_maxiter=50: 27 iters, 1316.2s (48.75s/iter), primal=1.28e-1"
uv run python -u -m bench.bench_solver --run_dir "$RUN" \
    --warm_lambda 26.90332801 --knot_lambda 34.97432641 \
    --subsample 20 --feet both --target markers \
    --maxiter 60 --chunk_size 39 --inner_tols 1e-2 \
    --inner_maxiters 3 5 8

# a shorter line search cuts the straggler's worst case rather than the mean, which is the term
# that actually sets wall time here
echo "### same again with a capped line search"
uv run python -u -m bench.bench_solver --run_dir "$RUN" \
    --warm_lambda 26.90332801 --knot_lambda 34.97432641 \
    --subsample 20 --feet both --target markers \
    --maxiter 60 --chunk_size 39 --inner_tols 1e-2 \
    --inner_maxiters 8 --inner_max_linesearch 10
