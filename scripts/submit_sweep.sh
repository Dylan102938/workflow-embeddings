#!/bin/bash
# Usage (on rnn, from the repo): scripts/submit_sweep.sh sweeps/<file>.txt [run-name-filter-regex]
# Submits each non-comment line as its own single-GPU train job (scripts/train.sbatch), so runs start as GPUs free up.
set -euo pipefail
SWEEP=$1; FILTER=${2:-.}
grep -v '^\s*#' "$SWEEP" | grep -v '^\s*$' | while read -r line; do
    name=$(echo "$line" | sed -n 's/.*--run \([^ ]*\).*/\1/p')
    echo "$name" | grep -Eq "$FILTER" || continue
    sbatch --job-name="wf-$name" scripts/train.sbatch $line | sed "s/$/  ($name)/"
done
