#!/bin/bash
# Build frozen-encoder frame caches on Baseten: AgentNet screenshots (public HF dataset) + our private derived data
# (Dylan123/wfemb-data: prepared sessions/events, key sets, PCA bases fitted on rnn) -> dense caches in the team cache.
# Usage: bash runners/baseten/build.sh "SPEC [SPEC ...]" [agentnet|added]   SPEC = siglip2:<grid> (siglip2:24 -> patch 576)
#   A PCA basis that is missing from the team cache is fitted first (on AgentNet only).
#   added: the caches of the added datasets ($TEAM/bulk/added), projected with AgentNet's bases.
set -euo pipefail
cd "$BT_WORKING_DIR"
source runners/baseten/setup.sh
SPECS=${1:-"siglip2:16 siglip2:24"}
export WFEMB_SET=${2:-agentnet}
export WFEMB_DATA=$BT_SCRATCH_DIR/data WFEMB_BULK=$BT_SCRATCH_DIR/bulk
SUB=""; [ "$WFEMB_SET" = added ] && SUB="added/"
LOCAL_BULK=$WFEMB_BULK/${SUB%/}; TEAM_BULK=$TEAM/bulk/${SUB%/}
mkdir -p "$WFEMB_DATA/$SUB" "$LOCAL_BULK" "$TEAM/data" "$TEAM_BULK"
t0=$(date +%s)                                  # inputs were downloaded to the team cache by runners/baseten/download.sh
cp "$TEAM"/data/bulk/*.npz "$WFEMB_BULK/"
if [ "$WFEMB_SET" = added ]; then
    cp -r "$TEAM/data/added/prepared" "$WFEMB_DATA/added/"
    ln -sfn "$TEAM/data/added/sources" "$WFEMB_DATA/added/sources"
    echo "data ready in $(( $(date +%s) - t0 ))s: $(du -shL "$WFEMB_DATA/added/sources" | cut -f1) screenshots"
else
    cp -r "$TEAM"/data/{prepared,keys,keys_openai} "$WFEMB_DATA/"
    ln -sfn "$TEAM/raw/agentnet_raw" "$WFEMB_DATA/agentnet_raw"   # read in place: the job's local disk quota is ~300 GB
    echo "data ready in $(( $(date +%s) - t0 ))s: $(du -sh "$WFEMB_DATA/agentnet_raw" | cut -f1) raw"
fi
NG=$(python -c "import torch; print(torch.cuda.device_count())")
NCPU=$(nproc)
if [ -f /sys/fs/cgroup/cpu.max ] && [ "$(cut -d' ' -f1 /sys/fs/cgroup/cpu.max)" != max ]; then   # container CPU quota
    NCPU=$(awk '{print int($1 / $2)}' /sys/fs/cgroup/cpu.max); fi
W=$(( NCPU / NG - 2 )); (( W > 24 )) && W=24; (( W < 2 )) && W=2
echo "$NG GPUs, $NCPU CPUs -> $W loader workers per shard; memory limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
NSH=8                                         # shards per cache, built in waves of NG; finished shards leave local disk
for spec in $SPECS; do
    grid=${spec##*:}; pca=512
    name=$(python -c "from wfemb.config import frame_cache_name; print(frame_cache_name($grid))")
    basis=dense_pca_g${grid}_d${pca}.npz
    if [ ! -f "$WFEMB_BULK/$basis" ]; then
        [ "$WFEMB_SET" = agentnet ] || { echo "no $basis: build AgentNet's cache for $spec first"; exit 1; }
        python -m wfemb.prep.cache_frames --mode pca_fit --grid "$grid" --pca_dim "$pca" --workers "$W" \
            > "$BT_SCRATCH_DIR/pca_fit_${grid}_${pca}.log" 2>&1 || { echo "FAILED pca_fit $spec"; tail -20 "$BT_SCRATCH_DIR/pca_fit_${grid}_${pca}.log"; exit 1; }
        cp "$WFEMB_BULK/$basis" "$TEAM/data/bulk/"
        echo "$spec: $(grep -a '^saved' "$BT_SCRATCH_DIR/pca_fit_${grid}_${pca}.log")"
    fi
    part="$TEAM_BULK/$name.partial"; mkdir -p "$part"
    t1=$(date +%s)
    for ((w = 0; w < NSH; w += NG)); do
        pids=()
        for ((g = 0; g < NG && w + g < NSH; g++)); do
            CUDA_VISIBLE_DEVICES=$g python -m wfemb.prep.cache_frames --mode dense --grid "$grid" \
                --pca_dim "$pca" --shard $(( w + g )) --num_shards $NSH --workers "$W" \
                > "$BT_SCRATCH_DIR/cache_${grid}_${pca}_$(( w + g )).log" 2>&1 &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p" || { echo "FAILED $spec"; tail -20 "$BT_SCRATCH_DIR"/cache_${grid}_${pca}_*.log; exit 1; }; done
        for f in "$LOCAL_BULK/$name"/global_*.npy "$LOCAL_BULK/$name"/meta_*.json; do
            [ -f "$f" ] && { cp "$f" "$part/" && rm -f "$f"; } &
        done; wait
        echo "$spec: shards $w-$(( w + NG - 1 )) of $NSH done at $(( $(date +%s) - t1 ))s; local disk $(du -sh "$BT_SCRATCH_DIR" | cut -f1)"
    done
    rm -rf "$TEAM_BULK/$name"; mv "$part" "$TEAM_BULK/$name"
    python runners/baseten/cache_summary.py "$TEAM_BULK/$name"
    echo "$spec built in $(( $(date +%s) - t1 ))s: $(du -sh "$TEAM_BULK/$name" | cut -f1) in the team cache"
done
ls -la "$TEAM_BULK"; echo BUILD_DONE
