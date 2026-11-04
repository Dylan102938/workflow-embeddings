#!/bin/bash
#SBATCH --job-name=wfemb-para
#SBATCH --partition=main
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# vLLM pins its own torch, so it runs in a separate, isolated uv environment.
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
export VLLM_CACHE_ROOT=/nas/ucb/dfeng/.cache/vllm
PYTHONPATH=$REPO $UV run --isolated --no-project --python 3.11 --with "vllm==0.8.5" \
    python -m wfemb.paraphrase "$@"
