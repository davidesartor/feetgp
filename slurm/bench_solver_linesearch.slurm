#!/bin/bash
#SBATCH --job-name=solvls
#SBATCH --output=logs/gp/solvls_%j.out
#SBATCH --error=logs/gp/solvls_%j.err
#SBATCH --time=03:00:00
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

# bench_budget: maxiter=12 costs 28.24s at max_linesearch=30 and 9.45s at 5, for the same
# objective to 8 digits (-485145.38 against -485145.36) and the same projected gradient
# (6.22e-2). The line-search cap is the straggler's leash, not an accuracy knob -- max evals
# per output fell 78 -> 22. So the accurate-and-cheap corner is a long outer budget under a
# short line search, which is the opposite of what cutting inner_maxiter alone buys.
echo "### knot lambda=34.97 warmstarted from 26.90, short line search"
echo "### optimistix reference: 35 iters, 488.7s (13.96s/iter), primal=1.63e-1"
echo "### L-BFGS-B maxiter=3 ls=30: 16 iters, 135.9s (8.49s/iter), primal=2.18e-1"
uv run python -u -m bench.bench_solver --run_dir "$RUN" \
    --warm_lambda 26.90332801 --knot_lambda 34.97432641 \
    --subsample 20 --feet both --target markers \
    --maxiter 60 --chunk_size 39 --inner_tols 1e-2 \
    --inner_maxiters 12 20 --inner_max_linesearch 5

echo "### cold lambda=0 at the same setting, which the whole grid hangs off"
uv run python -u -m bench.bench_solver --run_dir "$RUN" \
    --knot_lambda 0.0 \
    --subsample 20 --feet both --target markers \
    --maxiter 60 --chunk_size 39 --inner_tols 1e-2 \
    --inner_maxiters 12 --inner_max_linesearch 5
