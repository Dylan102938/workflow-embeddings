#!/bin/bash
# Submit a job file's runs on rnn (from the repo on rnn): scripts/submit.sh jobs/<name>.txt [id-filter-regex]
# Runs that already have final.json in $WFEMB_DATA/results are skipped (train.py would skip them too). The rest are
# packed RUNS_PER_JOB per Slurm job on GPUS_PER_JOB GPUs (scripts/multi.sbatch); a job's runs share the page-cached
# frame files. Env: RUNS_PER_JOB (default 3), GPUS_PER_JOB (1), MEM_BASE_GB (50), MEM_PER_RUN_GB (15), CPUS_PER_RUN (5).
set -euo pipefail
JOBFILE=$1; FILTER=${2:-.}
source scripts/env.sh >/dev/null
K=${RUNS_PER_JOB:-3}; G=${GPUS_PER_JOB:-1}
GDIR=jobs/.groups; mkdir -p "$GDIR"
mapfile -t LINES < <(sed 's/\s*#.*$//' "$JOBFILE" | grep -v '^\s*$' | while read -r id rest; do
    echo "$id" | grep -Eq "$FILTER" || continue
    dir=$($UV run python -m wfemb.config "id=$id" | sed -n 's/^results: *//p')
    dir=${dir/#*\/results\//$WFEMB_DATA/results/}
    if [ -f "$dir/final.json" ]; then echo "skip (done): $id" >&2; continue; fi
    echo "$id $rest"; done)
stem=$(basename "$JOBFILE" .txt)
for ((s = 0; s < ${#LINES[@]}; s += K)); do
    grp=("${LINES[@]:s:K}")
    f="$GDIR/${stem}_g$((s / K)).txt"
    printf '%s\n' "${grp[@]}" > "$f"
    n=${#grp[@]}
    mem=$(( ${MEM_BASE_GB:-50} + n * ${MEM_PER_RUN_GB:-15} ))G
    sbatch --export=ALL --job-name="wf-${stem}_g$((s / K))" --gres=gpu:$G --mem=$mem --cpus-per-task=$(( n * ${CPUS_PER_RUN:-5} )) \
        scripts/multi.sbatch "$f" | sed "s/$/  ($n runs; ${G} GPU, $mem)/"
done
