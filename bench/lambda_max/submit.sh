#!/bin/bash
# Submit the lambda_max array over the chip x dtype x kernel grid.
set -euo pipefail

GPUS=${GPUS:-"1080ti 2080ti"}
DTYPES=${DTYPES:-"float32 float64"}
PROFILES=${PROFILES:-"rbf matern52"}
QOS=${QOS:-}
SIZES=${SIZES:-}

# short qos allows a single submitted job, so the whole sweep runs serially in it
if [ -n "$SIZES" ]; then
    ARRAY_AND_TIME=(--array=0-0 --time="${TIME:-04:00:00}")
else
    ARRAY_AND_TIME=()
fi

for gpu in $GPUS; do
    for dtype in $DTYPES; do
        for profile in $PROFILES; do
            sbatch ${QOS:+--qos="$QOS"} --constraint="$gpu" "${ARRAY_AND_TIME[@]}" \
                --job-name="lambda_max_${gpu}_${dtype}_${profile}" \
                --export=ALL,DTYPE="$dtype",PROFILE="$profile",SIZES="$SIZES",BUDGET="${BUDGET:-2400}" \
                bench/lambda_max/job_gpu.slurm
        done
    done
done
