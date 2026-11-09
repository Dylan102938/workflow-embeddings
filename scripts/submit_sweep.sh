#!/bin/bash
# Usage (on rnn, from the repo): scripts/submit_sweep.sh sweeps/<file>.txt [run-name-filter-regex]
# Packs the selected runs into jobs of RUNS_PER_JOB runs on GPUS_PER_JOB GPUs (scripts/multi.sbatch); the runs of a job
# share one page-cached copy of the frame cache. Env: RUNS_PER_JOB (default 3), GPUS_PER_JOB (default 1),
# MEM_BASE_GB (default 27; the page cache is reclaimable, so less RAM only means more NAS reads), MEM_PER_RUN_GB (default 15), CPUS_PER_RUN (default 5).
set -euo pipefail
SWEEP=$1; FILTER=${2:-.}
K=${RUNS_PER_JOB:-3}; G=${GPUS_PER_JOB:-1}
GDIR=$(dirname "$SWEEP")/groups; mkdir -p "$GDIR"
mapfile -t LINES < <(grep -v '^\s*#' "$SWEEP" | grep -v '^\s*$' | while read -r l; do
    n=$(echo "$l" | sed -n 's/.*--run \([^ ]*\).*/\1/p'); echo "$n" | grep -Eq "$FILTER" && echo "$l"; done)
stem=$(basename "$SWEEP" .txt)
for ((s = 0; s < ${#LINES[@]}; s += K)); do
    grp=("${LINES[@]:s:K}")
    f="$GDIR/${stem}_g$((s / K)).txt"
    printf '%s\n' "${grp[@]}" > "$f"
    names=$(printf '%s\n' "${grp[@]}" | sed -n 's/.*--run \([^ ]*\).*/\1/p' | paste -sd, -)
    n=${#grp[@]}
    mem=$(( ${MEM_BASE_GB:-27} + n * ${MEM_PER_RUN_GB:-15} ))G
    sbatch --job-name="wf-${stem}_g$((s / K))" --gres=gpu:$G --mem=$mem --cpus-per-task=$(( n * ${CPUS_PER_RUN:-5} )) \
        scripts/multi.sbatch "$f" | sed "s/$/  ($names; ${G} GPU, $mem)/"
done
