#!/bin/bash
#SBATCH --job-name=wfemb-download
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# Download AgentNet (xlangai/AgentNet) to NAS, then merge each split zip into one archive.
# The training pipeline reads screenshots straight out of the merged zips (no 450K small files on NFS).
set -euo pipefail
DATA=${DATA:-/nas/ucb/dfeng/data/workflow-embeddings}
RAW=$DATA/agentnet_raw
mkdir -p "$RAW" "$DATA/logs"
export HF_HOME=/nas/ucb/dfeng/huggingface
export TMPDIR=/nas/ucb/dfeng/tmp
export HF_HUB_ENABLE_HF_TRANSFER=0
UV=/nas/ucb/dfeng/bin/uv

# 1) metadata + trajectories first (small), then images
$UV run --no-project --with "huggingface_hub[hf_xet]" hf download xlangai/AgentNet \
    --repo-type dataset --local-dir "$RAW" \
    --include "*.jsonl" "README.md" "LICENSE.txt"
echo "jsonl done: $(date)"
$UV run --no-project --with "huggingface_hub[hf_xet]" hf download xlangai/AgentNet \
    --repo-type dataset --local-dir "$RAW" --max-workers 16 \
    --include "ubuntu_images/*" "win_mac_images/*"
echo "images done: $(date)"

