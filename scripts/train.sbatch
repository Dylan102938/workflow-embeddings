#!/bin/bash
#SBATCH --job-name=wfemb-train
#SBATCH --partition=main
#SBATCH --qos=high          # highest-priority QOS available to the account (max 8 GPUs per user)
#SBATCH --gres=gpu:1
#SBATCH --exclude=ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu   # A100 nodes only (airl/sac: SXM4-80GB, cirl/rlhf: PCI-80GB)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# Usage: sbatch scripts/train.sbatch --run NAME [train args...]
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
echo "git: $(git rev-parse --short HEAD)  host: $(hostname)"
$UV run python -m wfemb.train "$@"
