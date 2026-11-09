#!/bin/bash
#SBATCH --job-name=wfemb-multi
#SBATCH --partition=main
#SBATCH --qos=high          # highest-priority QOS available to the account (max 8 GPUs per user)
#SBATCH --gres=gpu:1
#SBATCH --exclude=ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu,cirl.ist.berkeley.edu   # A100 nodes only (airl/sac: SXM4-80GB, rlhf: PCI-80GB); cirl excluded: its NFS client livelocks reading keys_openai/para_keys.npy and events.npz (2026-11-07, 2026-11-09)
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G          # the ~107 GB frame cache copy in /dev/shm is charged to the job + ~15 GB per run
#SBATCH --time=2-00:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# Several training runs in one job, packed onto its GPUs (run i -> GPU i mod #GPUs), all reading ONE copy of the exact
# bf16 frame cache (frames_mode=memmap). The job first copies the cache from NAS to node-local storage (/dev/shm if it
# has room, else local disk /tmp) and its runs memory-map that copy: memory-mapping the NAS files directly is fast only
# until another process on the node opens them, which invalidates the NFS page cache and drops runs to ~0.15 it/s.
# Host RAM ~ 110 GB for the cache (when in /dev/shm, charged to this job) + ~15 GB per run.
# Usage: sbatch [--gres=gpu:G --mem=.. --cpus-per-task=..] scripts/multi.sbatch GROUP_FILE   (one run's args per line)
set -euo pipefail
source /nas/ucb/dfeng/code/workflow-embeddings/scripts/env.sh
GROUP=$1
echo "git: $(git rev-parse --short HEAD)  host: $(hostname)  gpus: $CUDA_VISIBLE_DEVICES"
mapfile -t RUNS < <(grep -v '^\s*#' "$GROUP" | grep -v '^\s*$')
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
# copy every frame cache the runs use to node-local storage, then point the runs at the copies
DIRS=$(printf '%s\n' "${RUNS[@]}" | sed -n 's/.*--frames_dir \([^ ]*\).*/\1/p' | sort -u)
need=0; for d in $DIRS; do need=$(( need + $(du -sb "$d" | cut -f1) )); done
shm_free=$(df -B1 --output=avail /dev/shm | tail -1); tmp_free=$(df -B1 --output=avail /tmp | tail -1)
LOCAL=""
if (( shm_free > need + 40 * 2**30 )); then LOCAL=/dev/shm/wfemb_$SLURM_JOB_ID
elif (( tmp_free > need + 80 * 2**30 )); then LOCAL=/tmp/wfemb_$SLURM_JOB_ID; fi
cleanup() { [ -n "$LOCAL" ] && rm -rf "$LOCAL"; }
trap cleanup EXIT INT TERM
if [ -n "$LOCAL" ]; then
    t0=$(date +%s)
    for d in $DIRS; do
        dst="$LOCAL/$(basename "$d")"; mkdir -p "$dst"
        for f in "$d"/*; do [ -f "$f" ] && cp "$f" "$dst/" & done
    done; wait
    copied=$(du -sb "$LOCAL" | cut -f1)
    if (( copied < need )); then
        echo "WARNING: copy incomplete ($copied of $need bytes); runs read the NAS copy"; rm -rf "$LOCAL"; LOCAL=""
    fi
fi
if [ -n "$LOCAL" ]; then
    for i in "${!RUNS[@]}"; do for d in $DIRS; do RUNS[$i]=${RUNS[$i]//"--frames_dir $d"/"--frames_dir $LOCAL/$(basename "$d")"}; done; done
    echo "frame caches ($(( need / 2**30 )) GB) copied to $LOCAL in $(( $(date +%s) - t0 ))s"
elif [ -z "$LOCAL" ]; then
    echo "WARNING: no room on /dev/shm or /tmp for $(( need / 2**30 )) GB; runs read the NAS copy (slow if other jobs open it)"
fi
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
