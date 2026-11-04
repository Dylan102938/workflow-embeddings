#!/bin/bash
#SBATCH --job-name=wfemb-textemb
#SBATCH --partition=main
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
$UV run python -m wfemb.text_embed "$@"
$UV run python -m wfemb.inspect_keys > "$WFEMB_DATA/keys/inspect_keys.txt"
