#!/bin/bash
#SBATCH --job-name=wfemb-eval
#SBATCH --partition=main
#SBATCH --qos=high          # highest-priority QOS available to the account (max 8 GPUs per user)
#SBATCH --gres=gpu:1
#SBATCH --exclude=ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu   # A100 nodes only
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G          # frames_mode=ram keeps the ~107 GB bf16 frame cache in host RAM
#SBATCH --time=06:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# Usage: sbatch scripts/eval_ckpt.sbatch RUN [RUN ...] [--splits val test]
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
$UV run python -m wfemb.eval_ckpt "$@"
