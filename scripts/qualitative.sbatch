#!/bin/bash
#SBATCH --job-name=wfemb-qual
#SBATCH --partition=main
#SBATCH --qos=high          # highest-priority QOS available to the account (max 8 GPUs per user)
#SBATCH --gres=gpu:1
#SBATCH --exclude=ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu   # A100 nodes only (airl/sac: SXM4-80GB, cirl/rlhf: PCI-80GB)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
mkdir -p "$WFEMB_DATA/runs"
$UV run python -m wfemb.qualitative "$@"
