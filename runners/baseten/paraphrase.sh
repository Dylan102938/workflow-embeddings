#!/bin/bash
# Paraphrase the instructions of a prepared set on Baseten; vLLM runs in its own environment (it pins its own torch).
# Usage: bash runners/baseten/paraphrase.sh <agentnet|added> [model]
# Reads <set>/prepared from the private data repo; leaves keys/paraphrases.json in the job's checkpoint directory
# (locally: python -m wfemb.storage.fetch baseten --job <job id>) and in the team cache.
set -euo pipefail
cd "$BT_WORKING_DIR"
source runners/baseten/setup.sh
export WFEMB_SET=${1:-added} WFEMB_DATA=$BT_SCRATCH_DIR/data
SUB=""; [ "$WFEMB_SET" = added ] && SUB="added/"
MODEL=${2:-Qwen/Qwen2.5-7B-Instruct}
mkdir -p "$WFEMB_DATA"
retry() { for i in $(seq 40); do "$@" && return 0; echo "attempt $i failed; retrying in 30s"; sleep 30; done; return 1; }
retry hf download Dylan123/wfemb-data --repo-type dataset --local-dir "$WFEMB_DATA" --token "$HF_TOKEN" \
    --include "${SUB}prepared/*" >/dev/null
retry hf download "$MODEL" --token "$HF_TOKEN" >/dev/null       # into HF_HOME, so vLLM starts without the hub
pip install -q --root-user-action=ignore uv
export VLLM_CACHE_ROOT=$TEAM/vllm-cache
log=$BT_SCRATCH_DIR/paraphrase.log
HF_HUB_OFFLINE=1 PYTHONPATH=$PWD uv run --isolated --no-project --python 3.11 --with "vllm==0.8.5" --with "transformers==4.51.3" \
    python -m wfemb.prep.paraphrase --model "$MODEL" > "$log" 2>&1 \
    || { echo "FAILED; errors in the log:"; grep -a -i -B3 -A12 "error\|traceback" "$log" | head -n 200; exit 1; }
tail -n 40 "$log"
for target in "${BT_CHECKPOINT_DIR:-$TEAM/ckpt}/${SUB}keys" "$TEAM/data/${SUB}keys"; do
    mkdir -p "$target"; cp "$WFEMB_DATA/${SUB}keys/paraphrases.json" "$target/"
done
echo PARAPHRASE_DONE
