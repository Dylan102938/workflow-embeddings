#!/bin/bash
#SBATCH --job-name=wfemb-train
#SBATCH --partition=main
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# Usage: sbatch scripts/train.sbatch --run NAME [train args...]
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
SHM=/dev/shm/wfemb_frames_$SLURM_JOB_ID
trap 'rm -rf "$SHM" "$SHM.lock"' EXIT
CROP=False; [[ "$*" == *--use_crop* ]] && CROP=True
$UV run python -c "from wfemb.data import stage_to_shm; from wfemb import paths; print(stage_to_shm(paths.FRAMES, '$SLURM_JOB_ID', include_crop=$CROP))"
export WFEMB_FRAMES_DIR=$SHM
echo "git: $(git rev-parse --short HEAD)  host: $(hostname)"
$UV run python -m wfemb.train "$@"
