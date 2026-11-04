#!/bin/bash
#SBATCH --job-name=wfemb-merge
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%A_%a.out
# Merge each split AgentNet archive into one zip (read directly by cache_frames.py).
set -euo pipefail
RAW=/nas/ucb/dfeng/data/workflow-embeddings/agentnet_raw
GROUPS_=(ubuntu_images win_mac_images)
g=${GROUPS_[$SLURM_ARRAY_TASK_ID]}
cd "$RAW/$g"
if [ ! -f merged.zip ]; then
    rm -f merged_tmp.zip
    zip -s 0 images.zip --out merged_tmp.zip
    mv merged_tmp.zip merged.zip
fi
unzip -l merged.zip | tail -1
unzip -l merged.zip | head -8
echo "merge done: $(date)"
