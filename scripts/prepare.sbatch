#!/bin/bash
#SBATCH --job-name=wfemb-prepare
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
$UV run python -m wfemb.prepare "$@"
