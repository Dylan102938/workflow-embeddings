#!/bin/bash
# Build caches of vision towers' own tokens on Baseten (wfemb/prep/cache_tower_frames.py), written straight into the
# team cache: a cache is 0.5 to 2.5 TB, more than a job's local disk.
# Usage: bash runners/baseten/build_towers.sh "TOWER:BUDGET [TOWER:BUDGET ...]" [agentnet|added]
#   e.g. "siglip2:576 qwen3.8:576 inkling2:576". A cache that was interrupted continues where it stopped.
set -euo pipefail
cd "$BT_WORKING_DIR"
export WFEMB_TRANSFORMERS="transformers @ https://github.com/huggingface/transformers/archive/2f1bc5e321b9a736200a590d6e7743ccdeb60772.zip"
source runners/baseten/setup.sh
TOWERS=$1
export WFEMB_SET=${2:-agentnet}
export WFEMB_DATA=$BT_SCRATCH_DIR/data WFEMB_BULK=$TEAM/bulk
if [ "$WFEMB_SET" = added ]; then
    mkdir -p "$WFEMB_DATA/added"; cp -r "$TEAM/data/added/prepared" "$WFEMB_DATA/added/"
    ln -sfn "$TEAM/data/added/sources" "$WFEMB_DATA/added/sources"; SUB=added/
else
    mkdir -p "$WFEMB_DATA"; cp -r "$TEAM/data/prepared" "$WFEMB_DATA/"
    ln -sfn "$TEAM/raw/agentnet_raw" "$WFEMB_DATA/agentnet_raw"; SUB=
fi
NG=$(python -c "import torch; print(torch.cuda.device_count())")
NCPU=$(nproc)
if [ -f /sys/fs/cgroup/cpu.max ] && [ "$(cut -d' ' -f1 /sys/fs/cgroup/cpu.max)" != max ]; then   # container CPU quota
    NCPU=$(awk '{print int($1 / $2)}' /sys/fs/cgroup/cpu.max); fi
W=$(( NCPU / NG - 1 ))                        # decoding screenshots is all the loader workers do
echo "$NG GPUs, $NCPU CPUs -> $W decoding workers per GPU"
(while sleep 120; do echo "GPU utilization %: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')"; done) &
NSH=8                                         # shards per cache, built in waves of NG
for spec in $TOWERS; do
    tower=${spec%%:*}; budget=${spec##*:}
    t1=$(date +%s)
    for ((w = 0; w < NSH; w += NG)); do
        pids=(); tails=()
        for ((g = 0; g < NG && w + g < NSH; g++)); do
            log=$BT_SCRATCH_DIR/cache_${tower}_$(( w + g )).log
            CUDA_VISIBLE_DEVICES=$g python -m wfemb.prep.cache_tower_frames --tower "$tower" --budget "$budget" \
                --shard $(( w + g )) --num_shards $NSH --workers "$W" > "$log" 2>&1 &
            pids+=($!)
            (tail -n +1 -F "$log" 2>/dev/null | grep --line-buffered -E 'frames  |^done|shard|Error' \
                | sed -u "s/^/[$tower $(( w + g ))] /") &
            tails+=($!)
        done
        fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
        sleep 3; for t in "${tails[@]}"; do pkill -P "$t" 2>/dev/null; kill "$t" 2>/dev/null; done
        (( fail )) && { echo "FAILED $spec"; tail -20 "$BT_SCRATCH_DIR"/cache_${tower}_*.log; exit 1; }
    done
    name=$(python -c "from wfemb.config import tower_frame_cache_name; print(tower_frame_cache_name('$tower', $budget))")
    echo "$spec built in $(( $(date +%s) - t1 ))s: $(du -sh "$TEAM/bulk/$SUB$name" | cut -f1) in the team cache"
done
echo BUILD_DONE
