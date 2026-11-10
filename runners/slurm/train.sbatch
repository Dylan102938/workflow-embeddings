#!/bin/bash
#SBATCH --job-name=wfemb-train
#SBATCH --partition=main
#SBATCH --qos=high          # highest-priority QOS available to the account (max 8 GPUs per user)
#SBATCH --gres=gpu:1
#SBATCH --exclude=ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu,cirl.ist.berkeley.edu   # A100 nodes only (airl/sac: SXM4-80GB, rlhf: PCI-80GB); cirl excluded: its NFS client livelocks reading keys_openai/para_keys.npy and events.npz (2026-11-07, 2026-11-09)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G           # frames are read with preadv through the (evictable) page cache
#SBATCH --time=2-00:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# One run. Usage: sbatch runners/slurm/train.sbatch id=<run id> [overrides, e.g. train.max_frames=2048]
# (a section of jobs.txt, several runs per GPU job: runners/slurm/submit.sh <section>)
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/runners/slurm/env.sh
echo "git: $(git rev-parse --short HEAD)  host: $(hostname)"
export WFEMB_RESULTS=$WFEMB_DATA/results WFEMB_LAUNCHER="slurm:$SLURM_JOB_ID"
$UV run python -m wfemb.train "$@"
