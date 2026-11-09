#!/bin/bash
# Download AgentNet (public) + our private derived data into the persistent team cache, with retries (hf download
# resumes; a dropped connection killed the first build). Run as a CPU-only job: bash baseten/download.sh
set -uo pipefail
cd "$BT_WORKING_DIR"
source baseten/setup.sh
retry() { for i in 1 2 3 4 5 6 7 8; do "$@" && return 0; echo "attempt $i failed; retrying in 30s"; sleep 30; done; return 1; }
t0=$(date +%s)
retry hf download Dylan123/wfemb-data --repo-type dataset --local-dir "$TEAM/data" --token "$HF_TOKEN" --exclude "runs/*" >/dev/null || exit 1
retry hf download xlangai/AgentNet --repo-type dataset --local-dir "$TEAM/raw/agentnet_raw" --max-workers 16 \
    --include "*.jsonl" "ubuntu_images/*" "win_mac_images/*" >/dev/null || exit 1
echo "downloaded in $(( $(date +%s) - t0 ))s: $(du -sh "$TEAM/raw/agentnet_raw" | cut -f1) raw, $(du -sh "$TEAM/data" | cut -f1) data"
ls "$TEAM/data" "$TEAM/raw/agentnet_raw"; echo DOWNLOAD_DONE
