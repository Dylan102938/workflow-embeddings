#!/bin/bash
#SBATCH --job-name=wfemb-multi
#SBATCH --partition=main
#SBATCH --qos=high          # highest-priority QOS available to the account (max 8 GPUs per user)
#SBATCH --gres=gpu:1
#SBATCH --exclude=ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu   # A100 nodes only (airl/sac: SXM4-80GB, cirl/rlhf: PCI-80GB)
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# Several training runs in one job, packed onto its GPUs (run i -> GPU i mod #GPUs), all reading ONE copy of the exact
# bf16 frame cache: every run memory-maps the cache (frames_mode=memmap) and the job pre-warms the page cache, which
# the job's processes share (charged once to the job). Host RAM ~ 110 GB for the cache + ~15 GB per run.
# Usage: sbatch [--gres=gpu:G --mem=.. --cpus-per-task=..] scripts/multi.sbatch GROUP_FILE   (one run's args per line)
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
GROUP=$1
echo "git: $(git rev-parse --short HEAD)  host: $(hostname)  gpus: $CUDA_VISIBLE_DEVICES"
mapfile -t RUNS < <(grep -v '^\s*#' "$GROUP" | grep -v '^\s*$')
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
# pre-warm the page cache with every frame cache the runs use (parallel sequential reads from NAS)
DIRS=$(printf '%s\n' "${RUNS[@]}" | sed -n 's/.*--frames_dir \([^ ]*\).*/\1/p' | sort -u)
t0=$(date +%s)
for d in $DIRS; do for f in "$d"/*.npy; do cat "$f" > /dev/null & done; done; wait
echo "page cache warm in $(( $(date +%s) - t0 ))s: $DIRS"
pids=()
for i in "${!RUNS[@]}"; do
    name=$(echo "${RUNS[$i]}" | sed -n 's/.*--run \([^ ]*\).*/\1/p')
    gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
    CUDA_VISIBLE_DEVICES=$gpu $UV run python -m wfemb.train ${RUNS[$i]} --frames_mode memmap \
        > "$WFEMB_DATA/logs/run-$name.out" 2>&1 &
    pids+=($!)
    echo "launched $name on GPU $gpu (pid ${pids[-1]})"
    sleep 20                                  # stagger start-up (data loading)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
exit $fail
