#!/bin/bash
#SBATCH --job-name=wfemb-textemb
#SBATCH --partition=main
#SBATCH --qos=high          # highest-priority QOS available to the account (max 8 GPUs per user)
#SBATCH --gres=gpu:1
#SBATCH --exclude=ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu,cirl.ist.berkeley.edu   # A100 nodes only (airl/sac: SXM4-80GB, rlhf: PCI-80GB); cirl excluded: its NFS client livelocks reading keys_openai/para_keys.npy and events.npz (2026-11-07, 2026-11-09)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/runners/slurm/env.sh
$UV run python -m wfemb.prep.text_embed "$@"
