#!/bin/bash
#SBATCH --job-name=wfemb-para
#SBATCH --partition=main
#SBATCH --qos=high          # highest-priority QOS available to the account (max 8 GPUs per user)
#SBATCH --gres=gpu:1
#SBATCH --exclude=ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu   # A100 nodes only (airl/sac: SXM4-80GB, cirl/rlhf: PCI-80GB)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# vLLM pins its own torch, so it runs in a separate, isolated uv environment.
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
export VLLM_CACHE_ROOT=/nas/ucb/dfeng/.cache/vllm
PYTHONPATH=$REPO $UV run --isolated --no-project --python 3.11 --with "vllm==0.8.5" --with "transformers==4.51.3" \
    python -m wfemb.paraphrase "$@"
