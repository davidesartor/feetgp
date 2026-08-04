#!/bin/bash
#SBATCH --job-name=solver
#SBATCH --output=logs/gp/solver_%j.out
#SBATCH --error=logs/gp/solver_%j.err
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

echo "### knot: lambda=34.97 warmstarted from 26.90, L-BFGS-B x-update"
echo "### optimistix reference: 35 iters, 488.7s (13.96s/iter), primal=1.63e-1"
uv run python -u -m bench.bench_solver --run_dir "$RUN" \
    --warm_lambda 26.90332801 --knot_lambda 34.97432641 \
    --subsample 20 --feet both --target markers \
    --maxiter 60 --chunk_size 39 --inner_maxiters 50 \
    --inner_tols 1e-2 1e-3 1e-4

echo "### cold lambda=0, which the whole grid hangs off"
uv run python -u -m bench.bench_solver --run_dir "$RUN" \
    --knot_lambda 0.0 \
    --subsample 20 --feet both --target markers \
    --maxiter 60 --chunk_size 39 --inner_maxiters 50 \
    --inner_tols 1e-3
