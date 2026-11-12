#!/bin/bash
# Download the added datasets (public) on Baseten and convert them there (python -m wfemb.prep.sources): their
# screenshots are too large to upload from a workstation. The image archives, which the frame-cache build reads, stay in
# the team cache; the raw downloads are deleted. Usage: bash runners/baseten/convert.sh "videocua scalecua gui360"
set -euo pipefail
cd "$BT_WORKING_DIR"
source runners/baseten/setup.sh
command -v ffmpeg >/dev/null || { apt-get update -qq >/dev/null && apt-get install -y -qq ffmpeg >/dev/null; }
export WFEMB_DATA=$TEAM/data HF_TOKEN HF_HUB_DISABLE_PROGRESS_BARS=1
NCPU=$(nproc)
if [ -f /sys/fs/cgroup/cpu.max ] && [ "$(cut -d' ' -f1 /sys/fs/cgroup/cpu.max)" != max ]; then   # container CPU quota
    NCPU=$(awk '{print int($1 / $2)}' /sys/fs/cgroup/cpu.max); fi
for dataset in $1; do
    t0=$(date +%s)
    directory=$TEAM/data/added/sources/$dataset
    rm -f "$directory/conversion.json"
    python -m wfemb.prep.sources "$dataset" --download --workers "$NCPU" || true
    [ -f "$directory/conversion.json" ] || { echo "FAILED $dataset"; exit 1; }
    find "$directory" -mindepth 1 -maxdepth 1 ! -name images.zip ! -name sessions.jsonl ! -name image_stats.json \
        ! -name conversion.json -exec rm -rf {} +
    echo "$dataset converted in $(( $(date +%s) - t0 ))s: $(cat "$directory/conversion.json" | tr -d '\n')"
    ls -la "$directory"
done
echo CONVERT_DONE
