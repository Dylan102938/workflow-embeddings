#!/bin/bash
#SBATCH --job-name=wfemb-frames
#SBATCH --partition=main
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%A_%a.out
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
$UV run python -m wfemb.cache_frames --shard "$SLURM_ARRAY_TASK_ID" --num_shards 8 --workers 11 "$@"
