#!/bin/bash
#SBATCH --job-name=feetgp
#SBATCH --array=0-9
#SBATCH --output=logs/gp/v4_%A_%a.out
#SBATCH --error=logs/gp/v4_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|h100"
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

set -euo pipefail
mkdir -p logs logs/gp

CONFIGS=(
    "markers|both|"
    "markers|both --ungroup_feet|"
    "markers|left_only|"
    "markers|right_only|"
    "markers|both|--relative"
    "markers|both --ungroup_feet|--relative"
    "markers|left_only|--relative"
    "markers|right_only|--relative"
    "forces|both|"
    "forces|both --ungroup_feet|"
)

IFS='|' read -r TARGET FEET_FLAGS REL_FLAGS <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Task ID:    $SLURM_ARRAY_TASK_ID"
echo "Target:     $TARGET"
echo "Feet:       $FEET_FLAGS"
echo "Relative:   ${REL_FLAGS:-none}"
echo "Node:       $(hostname)"
echo "GPUs:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c)"
echo "Date:       $(date)"
echo "=========================================="

uv run python -u -m feetgp.run --output_dir results_v4 \
    --subsample 20 \
    --target $TARGET \
    --inclines all \
    --feet $FEET_FLAGS \
    $REL_FLAGS \
    --chunk_size 39 \
    --maxiter 400 \
    --inner_maxiter 50 \
    --inner_tol 1e-4 \
    --tol 1e-3 \
    --lambda_step 1.3 \
    --lambda_budget 60 --lambda_refine 100 --log_every 50
