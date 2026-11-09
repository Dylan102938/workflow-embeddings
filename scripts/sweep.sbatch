#!/bin/bash
#SBATCH --job-name=wfemb-sweep
#SBATCH --partition=main
#SBATCH --qos=high          # highest-priority QOS available to the account (max 8 GPUs per user)
#SBATCH --gres=gpu:4
#SBATCH --exclude=ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu,cirl.ist.berkeley.edu   # A100 nodes only (airl/sac: SXM4-80GB, rlhf: PCI-80GB); cirl excluded: its NFS client livelocks reading keys_openai/para_keys.npy and events.npz (2026-11-07, 2026-11-09)
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# Usage: sbatch [--gres=gpu:N] scripts/sweep.sbatch sweeps/<file>.txt
# Each non-comment line of the file is one run's train args; run i goes to GPU i (one run per GPU).
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
SWEEP=$1
echo "git: $(git rev-parse --short HEAD)  host: $(hostname)  gpus: $CUDA_VISIBLE_DEVICES"
mapfile -t RUNS < <(grep -v '^\s*#' "$SWEEP" | grep -v '^\s*$')
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
pids=()
for i in "${!RUNS[@]}"; do
    [ "$i" -ge "${#GPUS[@]}" ] && { echo "more runs than GPUs; skipping: ${RUNS[$i]}"; continue; }
    name=$(echo "${RUNS[$i]}" | sed -n 's/.*--run \([^ ]*\).*/\1/p')
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $UV run python -m wfemb.train ${RUNS[$i]} \
        > "$WFEMB_DATA/logs/run-$name.out" 2>&1 &
    pids+=($!)
    echo "launched $name on GPU ${GPUS[$i]} (pid ${pids[-1]})"
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
exit $fail
