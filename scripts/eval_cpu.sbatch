#!/bin/bash
#SBATCH --job-name=wfemb-evalcpu
#SBATCH --partition=main
#SBATCH --qos=high
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=cirl.ist.berkeley.edu   # its NFS client livelocks reading the key/event files
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# Checkpoint evaluation on CPU (no GPU needed): sbatch scripts/eval_cpu.sbatch RUN [RUN ...] [--finalize]
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
$UV run python -m wfemb.eval_ckpt "$@" --device cpu
