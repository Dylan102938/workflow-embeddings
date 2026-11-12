#!/bin/bash
# Paraphrase the instructions of a prepared set on Baseten; vLLM runs in its own environment (it pins its own torch).
# Usage: bash runners/baseten/paraphrase.sh <agentnet|added>
# Reads <set>/prepared from the private data repo; leaves keys/paraphrases.json in the job's checkpoint directory
# (locally: python -m wfemb.storage.fetch baseten --job <job id>) and in the team cache.
set -euo pipefail
cd "$BT_WORKING_DIR"
source runners/baseten/setup.sh
export WFEMB_SET=${1:-added} WFEMB_DATA=$BT_SCRATCH_DIR/data
SUB=""; [ "$WFEMB_SET" = added ] && SUB="added/"
mkdir -p "$WFEMB_DATA"
hf download Dylan123/wfemb-data --repo-type dataset --local-dir "$WFEMB_DATA" --token "$HF_TOKEN" \
    --include "${SUB}prepared/*" >/dev/null
pip install -q --root-user-action=ignore uv
export VLLM_CACHE_ROOT=$TEAM/vllm-cache
PYTHONPATH=$PWD uv run --isolated --no-project --python 3.11 --with "vllm==0.8.5" --with "transformers==4.51.3" \
    python -m wfemb.prep.paraphrase 2>&1 | grep -v -i "warn" | tail -n 60
for target in "${BT_CHECKPOINT_DIR:-$TEAM/ckpt}/${SUB}keys" "$TEAM/data/${SUB}keys"; do
    mkdir -p "$target"; cp "$WFEMB_DATA/${SUB}keys/paraphrases.json" "$target/"
done
echo PARAPHRASE_DONE
