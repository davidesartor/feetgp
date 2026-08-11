#!/bin/bash
# Submit the unpenalized-fit array on a given chip, across both gpu partitions.
set -euo pipefail

GPU=${GPU:-2080ti}
PROFILE=${PROFILE:-rbf}
QOS=${QOS:-}
SIZES=${SIZES:-}

# short qos allows a single submitted job, so the whole sweep runs serially in it
if [ -n "$SIZES" ]; then
    ARRAY_AND_TIME=(--array=0-0 --time="${TIME:-04:00:00}")
else
    ARRAY_AND_TIME=()
fi

sbatch ${QOS:+--qos="$QOS"} --constraint="$GPU" "${ARRAY_AND_TIME[@]}" \
    --job-name="unpenalized_fit_${GPU}_${PROFILE}" \
    --export=ALL,PROFILE="$PROFILE",SIZES="$SIZES",BUDGET="${BUDGET:-1200}" \
    bench/unpenalized_fit/job_gpu.slurm
