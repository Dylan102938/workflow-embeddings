#!/bin/bash
# Build frozen-encoder frame caches on Baseten: AgentNet screenshots (public HF dataset) + our private derived data
# (Dylan123/wfemb-data: prepared sessions/events, key sets, PCA bases fitted on rnn) -> dense caches in the team cache.
# Usage: bash runners/baseten/build.sh "SPEC [SPEC ...]"   SPEC = siglip2:<grid> (e.g. siglip2:24 -> patch 576)
set -euo pipefail
cd "$BT_WORKING_DIR"
source runners/baseten/setup.sh
SPECS=${1:-"siglip2:16 siglip2:24"}
export WFEMB_DATA=$BT_SCRATCH_DIR/data WFEMB_BULK=$BT_SCRATCH_DIR/bulk
mkdir -p "$WFEMB_DATA" "$WFEMB_BULK" "$TEAM/data" "$TEAM/bulk"
t0=$(date +%s)                                  # inputs were downloaded to the team cache by runners/baseten/download.sh
cp -r "$TEAM"/data/{prepared,keys,keys_openai} "$WFEMB_DATA/"
cp "$TEAM"/data/bulk/*.npz "$WFEMB_BULK/"
ln -sfn "$TEAM/raw/agentnet_raw" "$WFEMB_DATA/agentnet_raw"   # read in place: the job's local disk quota is ~300 GB
echo "data ready in $(( $(date +%s) - t0 ))s: $(du -sh "$WFEMB_DATA/agentnet_raw" | cut -f1) raw"
NG=$(python -c "import torch; print(torch.cuda.device_count())")
NCPU=$(nproc)
if [ -f /sys/fs/cgroup/cpu.max ] && [ "$(cut -d' ' -f1 /sys/fs/cgroup/cpu.max)" != max ]; then   # container CPU quota
    NCPU=$(awk '{print int($1 / $2)}' /sys/fs/cgroup/cpu.max); fi
W=$(( NCPU / NG - 2 )); (( W > 24 )) && W=24; (( W < 2 )) && W=2
echo "$NG GPUs, $NCPU CPUs -> $W loader workers per shard; memory limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
NSH=8                                         # shards per cache, built in waves of NG; finished shards leave local disk
for spec in $SPECS; do
    enc=${spec%%:*}; grid=${spec##*:}
    name=frames_dense
    if [ "$grid" != 16 ]; then name+=_g$grid; fi
    part="$TEAM/bulk/$name.partial"; mkdir -p "$part"
    t1=$(date +%s)
    for ((w = 0; w < NSH; w += NG)); do
        pids=()
        for ((g = 0; g < NG && w + g < NSH; g++)); do
            CUDA_VISIBLE_DEVICES=$g python -m wfemb.prep.cache_frames --mode dense --grid "$grid" \
                --pca_dim 512 --shard $(( w + g )) --num_shards $NSH --workers "$W" \
                > "$BT_SCRATCH_DIR/cache_${enc}_${grid}_$(( w + g )).log" 2>&1 &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p" || { echo "FAILED $spec"; tail -20 "$BT_SCRATCH_DIR"/cache_${enc}_${grid}_*.log; exit 1; }; done
        for f in "$WFEMB_BULK/$name"/global_*.npy "$WFEMB_BULK/$name"/meta_*.json; do
            [ -f "$f" ] && { cp "$f" "$part/" && rm -f "$f"; } &
        done; wait
        echo "$spec: shards $w-$(( w + NG - 1 )) of $NSH done at $(( $(date +%s) - t1 ))s; local disk $(du -sh "$BT_SCRATCH_DIR" | cut -f1)"
    done
    rm -rf "$TEAM/bulk/$name"; mv "$part" "$TEAM/bulk/$name"
    echo "$spec built in $(( $(date +%s) - t1 ))s: $(du -sh "$TEAM/bulk/$name" | cut -f1) in the team cache"
done
ls -la "$TEAM/bulk"; echo BUILD_DONE
