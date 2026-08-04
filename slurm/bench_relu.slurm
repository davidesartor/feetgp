#!/bin/bash
#SBATCH --job-name=relu
#SBATCH --output=logs/gp/relu_%j.out
#SBATCH --error=logs/gp/relu_%j.err
#SBATCH --time=01:00:00
#SBATCH --qos=short
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|h100"
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

set -euo pipefail
RUN="results_v6/model=gp/target=markers/feet=both/inclines=all_sub=20"

echo "### knot: lambda=34.97 warmstarted from 26.90, rho free (no floor)"
uv run python -u -m bench.bench_knot --run_dir "$RUN" \
    --warm_lambda 26.90332801 --knot_lambda 34.97432641 \
    --subsample 20 --feet both --target markers \
    --maxiter 60 --chunk_size 39 --inner_maxiters 50

echo "### death: can a group reach zero at high lambda?"
uv run python -u -m bench.bench_death --run_dir "$RUN" \
    --warm_lambda 814.8348338 --lambdas 1500 3000 6000 12000 \
    --subsample 20 --feet both --target markers \
    --maxiter 120 --chunk_size 39
