#!/bin/bash
#SBATCH --job-name=wfemb-train
#SBATCH --partition=main
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# Usage: sbatch scripts/train.sbatch --run NAME [train args...]
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
echo "git: $(git rev-parse --short HEAD)  host: $(hostname)"
$UV run python -m wfemb.train "$@"
