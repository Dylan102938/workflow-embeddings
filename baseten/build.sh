#!/bin/bash
# Build frozen-encoder frame caches on Baseten: AgentNet screenshots (public HF dataset) + our private derived data
# (Dylan123/wfemb-data: prepared sessions/events, key sets, PCA bases fitted on rnn) -> dense caches in the team cache.
# Usage: bash baseten/build.sh "SPEC [SPEC ...]"   SPEC = siglip2:16 | siglip2:24 | qwen25vl:16
set -euo pipefail
cd "$BT_WORKING_DIR"
NEED_FLASH_ATTN=1 source baseten/setup.sh
SPECS=${1:-"siglip2:16 siglip2:24 qwen25vl:16"}
export WFEMB_DATA=$BT_SCRATCH_DIR/data WFEMB_BULK=$BT_SCRATCH_DIR/bulk
mkdir -p "$WFEMB_DATA" "$WFEMB_BULK" "$TEAM/data" "$TEAM/bulk"
t0=$(date +%s)
hf download Dylan123/wfemb-data --repo-type dataset --local-dir "$BT_SCRATCH_DIR/wfemb-data" --token "$HF_TOKEN" >/dev/null
cp -r "$BT_SCRATCH_DIR"/wfemb-data/{prepared,keys,keys_openai,keys_openai_p2} "$WFEMB_DATA/"
cp "$BT_SCRATCH_DIR"/wfemb-data/bulk/*.npz "$WFEMB_BULK/"
cp -r "$BT_SCRATCH_DIR"/wfemb-data/{prepared,keys,keys_openai,keys_openai_p2,bulk} "$TEAM/data/"   # for training jobs
hf download xlangai/AgentNet --repo-type dataset --local-dir "$WFEMB_DATA/agentnet_raw" --max-workers 32 \
    --include "*.jsonl" "ubuntu_images/*" "win_mac_images/*" >/dev/null
echo "data ready in $(( $(date +%s) - t0 ))s: $(du -sh "$WFEMB_DATA/agentnet_raw" | cut -f1) raw"
NG=$(python -c "import torch; print(torch.cuda.device_count())")
NCPU=$(nproc)
if [ -f /sys/fs/cgroup/cpu.max ] && [ "$(cut -d' ' -f1 /sys/fs/cgroup/cpu.max)" != max ]; then   # container CPU quota
    NCPU=$(awk '{print int($1 / $2)}' /sys/fs/cgroup/cpu.max); fi
W=$(( NCPU / NG - 2 )); (( W > 24 )) && W=24; (( W < 2 )) && W=2
echo "$NG GPUs, $NCPU CPUs -> $W loader workers per shard; memory limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
for spec in $SPECS; do
    enc=${spec%%:*}; grid=${spec##*:}
    t1=$(date +%s); pids=()
    for ((g = 0; g < NG; g++)); do
        CUDA_VISIBLE_DEVICES=$g python -m wfemb.cache_frames --mode dense --encoder_type "$enc" --grid "$grid" \
            --pca_dim 512 --shard $g --num_shards "$NG" --workers "$W" > "$BT_SCRATCH_DIR/cache_${enc}_${grid}_$g.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || { echo "FAILED $spec"; tail -20 "$BT_SCRATCH_DIR"/cache_${enc}_${grid}_*.log; exit 1; }; done
    name=frames_dense$([ "$enc" = qwen25vl ] && echo _qwen25vl)$([ "$grid" != 16 ] && echo _g$grid)
    echo "$spec built in $(( $(date +%s) - t1 ))s: $(du -sh "$WFEMB_BULK/$name" | cut -f1); copying to the team cache"
    t2=$(date +%s); rm -rf "$TEAM/bulk/$name.partial"; cp -r "$WFEMB_BULK/$name" "$TEAM/bulk/$name.partial"
    rm -rf "$TEAM/bulk/$name"; mv "$TEAM/bulk/$name.partial" "$TEAM/bulk/$name"
    echo "$spec copied in $(( $(date +%s) - t2 ))s"
done
ls -la "$TEAM/bulk"; echo BUILD_DONE
