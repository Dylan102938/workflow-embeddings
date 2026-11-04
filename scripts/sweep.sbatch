#!/bin/bash
#SBATCH --job-name=wfemb-sweep
#SBATCH --partition=main
#SBATCH --gres=gpu:A6000:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# Usage: sbatch [--gres=gpu:A6000:N] scripts/sweep.sbatch sweeps/<file>.txt
# Each non-comment line of the file is one run's train args; run i goes to GPU i (one run per GPU).
# All runs share one /dev/shm copy of the frame cache.
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
SWEEP=$1
SHM=/dev/shm/wfemb_frames_$SLURM_JOB_ID
trap 'rm -rf "$SHM" "$SHM.lock"' EXIT
CROP=False; [[ "$(cat $SWEEP)" == *--use_crop* ]] && CROP=True
$UV run python -c "from wfemb.data import stage_to_shm; from wfemb import paths; print(stage_to_shm(paths.FRAMES, '$SLURM_JOB_ID', include_crop=$CROP))"
export WFEMB_FRAMES_DIR=$SHM
echo "git: $(git rev-parse --short HEAD)  host: $(hostname)  gpus: $CUDA_VISIBLE_DEVICES"
mapfile -t RUNS < <(grep -v '^\s*#' "$SWEEP" | grep -v '^\s*$')
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
pids=()
for i in "${!RUNS[@]}"; do
    [ "$i" -ge "${#GPUS[@]}" ] && { echo "more runs than GPUs; skipping: ${RUNS[$i]}"; continue; }
    name=$(echo "${RUNS[$i]}" | sed -n 's/.*--run \([^ ]*\).*/\1/p')
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $UV run python -m wfemb.train ${RUNS[$i]} \
        > "$WFEMB_DATA/logs/run-$name.out" 2>&1 &
    pids+=($!)
    echo "launched $name on GPU ${GPUS[$i]} (pid ${pids[-1]})"
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
exit $fail
