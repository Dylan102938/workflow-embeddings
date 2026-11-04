#!/bin/bash
#SBATCH --job-name=wfemb-frames
#SBATCH --partition=main
#SBATCH --qos=high          # highest-priority QOS available to the account (max 8 GPUs per user)
#SBATCH --gres=gpu:1
#SBATCH --exclude=ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu   # A100 nodes only (airl/sac: SXM4-80GB, cirl/rlhf: PCI-80GB)
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%A_%a.out
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
echo "host $(hostname)  git $(git rev-parse --short HEAD)  args: $*"
$UV run python -c "import sys, torch, transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
$UV run python -m wfemb.cache_frames --shard "$SLURM_ARRAY_TASK_ID" --num_shards 8 --workers 11 "$@"
