#!/bin/bash
# Score saved checkpoints on Baseten: bash baseten/eval.sh RUN [RUN ...]   (runs fetched from Dylan123/wfemb-data runs/)
set -euo pipefail
cd "$BT_WORKING_DIR"
source baseten/setup.sh
export WFEMB_DATA=$BT_SCRATCH_DIR/data WFEMB_BULK=$BT_SCRATCH_DIR/bulk
mkdir -p "$WFEMB_DATA" "$WFEMB_BULK/frames_dense"
cp -r "$TEAM"/data/{prepared,keys,keys_openai,keys_openai_p2} "$WFEMB_DATA/"
for f in "$TEAM/bulk/frames_dense"/*; do cp "$f" "$WFEMB_BULK/frames_dense/" & done; wait
hf download Dylan123/wfemb-data --repo-type dataset --include "runs/*" --local-dir "$BT_SCRATCH_DIR/dl" --token "$HF_TOKEN" >/dev/null
cp -r "$BT_SCRATCH_DIR/dl/runs" "$WFEMB_DATA/"
python -m wfemb.eval_ckpt "$@" --finalize 2>&1 | grep -v -i warn
mkdir -p "${BT_CHECKPOINT_DIR:-$TEAM/ckpt}/runs" "$TEAM/runs"
for r in "$@"; do cp -r "$WFEMB_DATA/runs/$r" "${BT_CHECKPOINT_DIR:-$TEAM/ckpt}/runs/"; cp -r "$WFEMB_DATA/runs/$r" "$TEAM/runs/"; done
echo EVAL_DONE
