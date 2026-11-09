#!/bin/bash
#SBATCH --job-name=wfemb-train
#SBATCH --partition=main
#SBATCH --qos=high          # highest-priority QOS available to the account (max 8 GPUs per user)
#SBATCH --gres=gpu:1
#SBATCH --exclude=ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu,cirl.ist.berkeley.edu   # A100 nodes only (airl/sac: SXM4-80GB, rlhf: PCI-80GB); cirl excluded: its NFS client livelocks reading keys_openai/para_keys.npy and events.npz (2026-11-07, 2026-11-09)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G          # frames_mode=ram keeps the ~107 GB bf16 frame cache in host RAM
#SBATCH --time=2-00:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# Usage: sbatch scripts/train.sbatch --run NAME [train args...]
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
echo "git: $(git rev-parse --short HEAD)  host: $(hostname)"
$UV run python -m wfemb.train "$@"
