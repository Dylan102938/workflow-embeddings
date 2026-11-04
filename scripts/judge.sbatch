#!/bin/bash
#SBATCH --job-name=wfemb-judge
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# Usage: sbatch scripts/judge.sbatch --split val --mode {dry,sync,batch,collect} [--limit N]
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
set -a; source "$REPO/.env"; set +a      # API keys (gitignored)
$UV run python -m wfemb.judge_pool "$@"
