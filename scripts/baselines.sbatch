#!/bin/bash
#SBATCH --job-name=wfemb-baselines
#SBATCH --partition=main
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
mkdir -p "$WFEMB_DATA/runs"
$UV run python -m wfemb.baselines "$@"
