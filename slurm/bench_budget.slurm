#!/bin/bash
#SBATCH --job-name=budget
#SBATCH --output=logs/gp/budget_%j.out
#SBATCH --error=logs/gp/budget_%j.err
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

echo "### what one unit of L-BFGS-B inner budget costs, at the knot warmstart"
uv run python -u -m bench.bench_budget --run_dir "$RUN" --warm_lambda 26.90332801 \
    --subsample 20 --feet both --target markers --chunk_size 39 \
    --tol 1e-3 --maxiters 3 5 8 12 20 50 --max_linesearches 30

echo "### does a shorter line search buy the same objective for less"
uv run python -u -m bench.bench_budget --run_dir "$RUN" --warm_lambda 26.90332801 \
    --subsample 20 --feet both --target markers --chunk_size 39 \
    --tol 1e-3 --maxiters 12 --max_linesearches 5 10 20
