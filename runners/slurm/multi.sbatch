#!/bin/bash
#SBATCH --job-name=wfemb-multi
#SBATCH --partition=main
#SBATCH --qos=high          # highest-priority QOS available to the account (max 8 GPUs per user)
#SBATCH --gres=gpu:1
#SBATCH --exclude=ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu,cirl.ist.berkeley.edu   # A100 nodes only (airl/sac: SXM4-80GB, rlhf: PCI-80GB); cirl excluded: its NFS client livelocks reading keys_openai/para_keys.npy and events.npz (2026-11-07, 2026-11-09)
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G           # ~15 GB per run (incl. ~6 GB pinned buffers); the page cache of the frame files is evictable
#SBATCH --time=2-00:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# Several runs in one job, packed onto its GPUs (run i -> GPU i mod #GPUs). Frames are read from the NAS cache with
# preadv through the page cache (storage/frames.py explains why not mmap or a /dev/shm copy).
# Usage: sbatch [--gres=gpu:G --mem=.. --cpus-per-task=..] runners/slurm/multi.sbatch GROUP_FILE
#   GROUP_FILE: one run per line: a run id, optionally followed by overrides (written by runners/slurm/submit.sh)
# Results: $WFEMB_DATA/results/<run folder>/ (fetch locally: python -m wfemb.storage.fetch rnn)
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/runners/slurm/env.sh
GROUP=$1
echo "git: $(git rev-parse --short HEAD)  host: $(hostname)  gpus: $CUDA_VISIBLE_DEVICES"
export WFEMB_RESULTS=$WFEMB_DATA/results WFEMB_LAUNCHER="slurm:$SLURM_JOB_ID"
mapfile -t RUNS < <(sed 's/\s*#.*$//' "$GROUP" | grep -v '^\s*$')
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
pids=()
for i in "${!RUNS[@]}"; do
    read -r id overrides <<< "${RUNS[$i]}"
    gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
    CUDA_VISIBLE_DEVICES=$gpu $UV run python -m wfemb.train "id=$id" $overrides \
        > "$WFEMB_DATA/logs/run-$SLURM_JOB_ID-$i.out" 2>&1 &
    pids+=($!)
    echo "launched r$i on GPU $gpu (pid ${pids[-1]}, log run-$SLURM_JOB_ID-$i.out): $id $overrides"
    sleep 20                                  # stagger start-up (data loading)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
exit $fail
