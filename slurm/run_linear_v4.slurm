#!/bin/bash
#SBATCH --job-name=feetgp
#SBATCH --array=0-9
#SBATCH --output=logs/linear/%A_%a.out
#SBATCH --error=logs/linear/%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1080ti:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4

set -euo pipefail
mkdir -p logs

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
echo "Date:       $(date)"
echo "=========================================="

uv run python -u -m feetgp.run --output_dir results_v4 --lambda_refine 100 --log_every 0 \
    --subsample 20 \
    --target $TARGET \
    --inclines all \
    --feet $FEET_FLAGS \
    --linear_model \
    $REL_FLAGS \
    --maxiter 3000 \
    --tol 1e-5 \
    --lambda_step 1.3 \
    --lambda_budget 300