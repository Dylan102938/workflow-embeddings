#!/bin/bash
#SBATCH --job-name=wfemb-oaikeys
#SBATCH --partition=main
#SBATCH --qos=high          # highest-priority QOS available to the account
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# OpenAI text-embedding-3-large keys (training/eval labels); typed-text input features stay Qwen3-0.6B.
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
set -a; source "$REPO/.env"; set +a      # OPENAI_API_KEY (gitignored)
# Usage: sbatch scripts/openai_keys.sbatch [--out DIR] [openai_keys args...]   (DIR defaults to DATA/keys_openai)
OUT=$(echo "$@" | sed -n 's/.*--out \([^ ]*\).*/\1/p'); OUT=${OUT:-$WFEMB_DATA/keys_openai}
$UV run python -m wfemb.openai_keys "$@"
WFEMB_KEYS=$OUT $UV run python -m wfemb.inspect_keys > "$OUT/inspect_keys.txt"
