#!/bin/bash
# Submits the scripts in SCRIPTS (each with its own initial array restriction
# from INIT_ARRAY), then polls sacct and resubmits
# any array task that ends in a non-COMPLETED state (preempted, node failure,
# timeout, crash). feetgp.run resumes from cached lambda=*.pkl files, so
# resubmitting an array index just continues it rather than restarting.
# Also writes each poll's per-task state to a table in $STATUS_FILE.
set -uo pipefail
cd /home/dsartor_umass_edu/feetgp

SCRIPTS=(slurm/run_gp.slurm)
declare -A INIT_ARRAY=(
    [slurm/run_gp.slurm]="0-9"
    [slurm/run_linear.slurm]="0-9"
)
POLL_INTERVAL=${POLL_INTERVAL:-300}
MAX_RETRIES=${MAX_RETRIES:-10}
LOG=logs/watcher.log
STATUS_FILE=logs/watcher_status.log

# array index -> config label, same order as the CONFIGS array in both .slurm scripts
LABELS=(
    "markers/both"
    "markers/both_ungrouped"
    "markers/left"
    "markers/right"
    "markers/both_rel"
    "markers/both_ungrouped_rel"
    "markers/left_rel"
    "markers/right_rel"
    "forces/both"
    "forces/both_ungrouped"
)

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

submit() {
    local script=$1 array=$2
    if [ -n "$array" ]; then
        sbatch --array="$array" "$script" | awk '{print $NF}'
    else
        sbatch "$script" | awk '{print $NF}'
    fi
}

declare -A JOBID RETRIES

for script in "${SCRIPTS[@]}"; do
    JOBID[$script]=$(submit "$script" "${INIT_ARRAY[$script]}")
    RETRIES[$script]=0
    log "submitted $script as job ${JOBID[$script]}"
done

write_status() {
    {
        printf "%-16s %-24s %-6s %-12s %-14s\n" "SCRIPT" "CONFIG" "IDX" "JOB" "STATE"
        printf '%s\n' "--------------------------------------------------------------------"
        for script in "${SCRIPTS[@]}"; do
            jid=${JOBID[$script]}
            if [ -z "$jid" ]; then
                printf "%-16s %-24s %-6s %-12s %-14s\n" "$script" "-" "-" "-" "GAVE_UP"
                continue
            fi
            while IFS='|' read -r taskid state; do
                [ -z "$taskid" ] && continue
                idx=${taskid#${jid}_}
                label=${LABELS[$idx]:-"?"}
                printf "%-16s %-24s %-6s %-12s %-14s\n" "$script" "$label" "$idx" "$jid" "$state"
            done < <(sacct -j "$jid" --format=JobID,State --noheader --parsable2 | grep "^${jid}_[0-9]*|")
        done
        printf '\nlast updated: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    } > "$STATUS_FILE"
}

while true; do
    sleep "$POLL_INTERVAL"
    all_done=true

    for script in "${SCRIPTS[@]}"; do
        jid=${JOBID[$script]}
        [ -z "$jid" ] && continue

        pending_or_running=false
        failed_indices=()
        seen=0
        while IFS='|' read -r taskid state; do
            [ -z "$taskid" ] && continue
            seen=1
            case "$state" in
                COMPLETED) ;;
                PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING) pending_or_running=true ;;
                *) failed_indices+=("${taskid#${jid}_}") ;;
            esac
        done < <(sacct -j "$jid" --format=JobID,State --noheader --parsable2 | grep "^${jid}_[0-9]*|")

        # sacct hasn't registered this job yet -- assume still pending, not done
        [ "$seen" -eq 0 ] && pending_or_running=true

        if [ "$pending_or_running" = true ]; then
            all_done=false
            continue
        fi

        if [ ${#failed_indices[@]} -gt 0 ]; then
            all_done=false
            if [ "${RETRIES[$script]}" -ge "$MAX_RETRIES" ]; then
                log "$script: ${#failed_indices[@]} task(s) still failing after $MAX_RETRIES retries, giving up: ${failed_indices[*]}"
                JOBID[$script]=""
                continue
            fi
            idx_list=$(IFS=,; echo "${failed_indices[*]}")
            newjid=$(submit "$script" "$idx_list")
            RETRIES[$script]=$((RETRIES[$script] + 1))
            JOBID[$script]=$newjid
            log "$script: resubmitted indices $idx_list as job $newjid (retry ${RETRIES[$script]}/$MAX_RETRIES)"
        fi
    done

    write_status

    if [ "$all_done" = true ]; then
        log "all tracked jobs completed (or exhausted retries). watcher exiting."
        break
    fi
done
