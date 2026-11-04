#!/bin/bash
# Submit each line of a sweep file as its own single-GPU job (easier to schedule than one multi-GPU job).
# Usage: scripts/submit_sweep.sh sweeps/r1.txt [extra sbatch args, e.g. --dependency=afterok:JOBID]
set -euo pipefail
SWEEP=$1; shift
cd /nas/ucb/dfeng/code/workflow-embeddings
grep -v '^\s*#' "$SWEEP" | grep -v '^\s*$' | while read -r line; do
    name=$(echo "$line" | sed -n 's/.*--run \([^ ]*\).*/\1/p')
    sbatch --parsable --job-name="wfemb-$name" "$@" scripts/train.sbatch $line | sed "s/^/$name: /"
done
