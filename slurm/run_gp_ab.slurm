#!/bin/bash
# Solver A/B: the 67.4s-vs-488.7s lbfgsb-vs-optimistix comparison is one knot, not a
# path. Same config, same revision, --n_starts 1, one run per solver into its own
# output dir; compare wall time per lambda, max_live_kkt, path shape, total ADMM
# iterations via summarize_runs before flipping the --solver default.
#SBATCH --job-name=feetgp_ab
#SBATCH --array=0-1
#SBATCH --output=logs/ab/%A_%a.out
#SBATCH --error=logs/ab/%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|h100"
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8

set -euo pipefail
mkdir -p logs logs/ab

SOLVERS=(optimistix lbfgsb)
SOLVER="${SOLVERS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Solver:     $SOLVER"
echo "Node:       $(hostname)"
echo "GPUs:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c)"
DIRTY=$(git status --porcelain)
echo "Revision:   $(git rev-parse HEAD)${DIRTY:+ DIRTY}"
echo "Date:       $(date)"
echo "=========================================="

# inner_maxiter/inner_tol left unset on purpose: the two solvers stop on different
# criteria and count different things, so each gets its own resolved defaults
uv run python -m feetgp.run \
    --output_dir "results_ab_${SOLVER}" \
    --subsample 20 \
    --target markers \
    --inclines all \
    --feet both \
    --chunk_size 39 \
    --maxiter 300 \
    --tol 1e-3 \
    --lambda_step 1.3 \
    --lambda_budget 100 \
    --n_starts 1 \
    --solver "$SOLVER"
