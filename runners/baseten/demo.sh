#!/bin/bash
# Demo inputs on Baseten, where the frame caches and the AgentNet screenshots are (team cache, read in place):
#   - test_embeddings.npz for runs that lack it (runs from before the final test pass saved it), via wfemb.eval
#   - the test screenshot cache (once; kept in the team cache), as demo_screens.tar
# Demo bundles are then built locally: python -m wfemb.storage.fetch baseten --job <job id> --emb --screens,
# then python -m wfemb.demo bundle --id <run id>.
# Usage: bash runners/baseten/demo.sh "<run id> ..." ["<old run name> ..."]
#   old run names: pre-refactor runs in $TEAM/runs/<name>/best.pt, converted into the results tree first
set -euo pipefail
cd "$BT_WORKING_DIR"
source runners/baseten/setup.sh
IDS=$1; LEGACY=${2:-}
export WFEMB_DATA=$BT_SCRATCH_DIR/data WFEMB_BULK=$TEAM/bulk WFEMB_RESULTS=$BT_SCRATCH_DIR/results
OUT=${BT_CHECKPOINT_DIR:-$TEAM/ckpt}
mkdir -p "$WFEMB_DATA" "$WFEMB_RESULTS" "$OUT/results"
cp -r "$TEAM"/data/{prepared,keys,keys_openai} "$WFEMB_DATA/"
ln -sfn "$TEAM/raw/agentnet_raw" "$WFEMB_DATA/agentnet_raw"
for name in $LEGACY; do python runners/slurm/import_legacy.py ckpt "$TEAM/runs/$name/best.pt" --root "$WFEMB_RESULTS"; done
for id in $IDS; do
    d=$(python -m wfemb.config "id=$id" | sed -n 's#^results: *.*/results/##p')
    [ -f "$WFEMB_RESULTS/$d/best.pt" ] || { mkdir -p "$WFEMB_RESULTS/$d"; cp "$TEAM/results/$d/best.pt" "$WFEMB_RESULTS/$d/"; }
    t0=$(date +%s)
    python -m wfemb.eval "id=$id" save=True 2>&1 | grep -v -i warn
    mkdir -p "$OUT/results/$d"; cp "$WFEMB_RESULTS/$d/test_embeddings.npz" "$OUT/results/$d/"
    echo "EMBEDDED $id in $(( $(date +%s) - t0 ))s"
done
if [ ! -f "$TEAM/demo_screens.done" ]; then
    python -m wfemb.demo screens --screens "$TEAM/demo_screens" --workers 16 && touch "$TEAM/demo_screens.done"
fi
tar cf "$OUT/demo_screens.tar" -C "$TEAM" --transform 's#^demo_screens#screens#' demo_screens
echo "screenshots: $(du -sh "$OUT/demo_screens.tar" | cut -f1)"; echo DEMO_DONE
