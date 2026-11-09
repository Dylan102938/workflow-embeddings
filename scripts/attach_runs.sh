#!/bin/bash
# Start extra training runs inside a RUNNING packed job (scripts/multi.sbatch) whose GPU has spare capacity, reusing the
# job's node-local frame-cache copy (/dev/shm/wfemb_<job> or /tmp/wfemb_<job>). Runs as overlapping job steps.
# CAUTION: the job ends when its own runs finish (Slurm then kills attached steps), so only attach runs that will finish
# before the job's longest remaining run.
# Usage (on rnn, from the repo): scripts/attach_runs.sh JOBID FILE   (FILE: one run's train args per line, like a sweep)
set -euo pipefail
source scripts/env.sh
JOB=$1; FILE=$2
grep -v '^\s*#' "$FILE" | grep -v '^\s*$' | while read -r line; do
    name=$(echo "$line" | sed -n 's/.*--run \([^ ]*\).*/\1/p')
    d=$(echo "$line" | sed -n 's/.*--frames_dir \([^ ]*\).*/\1/p')
    for root in /dev/shm/wfemb_$JOB /tmp/wfemb_$JOB; do
        if srun --jobid="$JOB" --overlap -n1 -c1 --mem=1G test -d "$root/$(basename "$d")" 2>/dev/null; then
            line=${line//"--frames_dir $d"/"--frames_dir $root/$(basename "$d")"}; break
        fi
    done
    nohup setsid srun --jobid="$JOB" --overlap -n1 -c5 --mem=15G \
        bash -c "cd $REPO && $UV run python -m wfemb.train $line --frames_mode memmap" \
        > "$WFEMB_DATA/logs/run-$name.out" 2>&1 < /dev/null &
    echo "attached $name to job $JOB: $(echo "$line" | grep -o -- '--frames_dir [^ ]*')"
    sleep 10
done
