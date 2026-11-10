#!/bin/bash
# Redo the final evaluation of finished runs on Baseten (final.json and test_embeddings.npz), and build the test
# screenshot cache once. For runs whose final evaluation is missing or predates the current metrics.
# Usage: bash runners/baseten/evaluate.sh "<run id> ..." ["<old run name> ..."]
#   old run names: pre-refactor runs in $TEAM/runs/<name>/best.pt, converted into the results tree first
# Locally afterwards: python -m wfemb.storage.fetch baseten --job <job id> --embeddings --screenshots
set -euo pipefail
cd "$BT_WORKING_DIR"
source runners/baseten/setup.sh
RUN_IDS=$1; OLD_RUN_NAMES=${2:-}
export WFEMB_DATA=$BT_SCRATCH_DIR/data WFEMB_BULK=$TEAM/bulk WFEMB_RESULTS=$BT_SCRATCH_DIR/results
UPLOAD_DIR=${BT_CHECKPOINT_DIR:-$TEAM/ckpt}
mkdir -p "$WFEMB_DATA" "$WFEMB_RESULTS" "$UPLOAD_DIR/results"
cp -r "$TEAM"/data/{prepared,keys,keys_openai} "$WFEMB_DATA/"
ln -sfn "$TEAM/raw/agentnet_raw" "$WFEMB_DATA/agentnet_raw"

for name in $OLD_RUN_NAMES; do
    python runners/slurm/import_legacy.py checkpoint "$TEAM/runs/$name/best.pt" --root "$WFEMB_RESULTS"
done
for run_id in $RUN_IDS; do
    run_dir=$(python -m wfemb.config "id=$run_id" | sed -n 's#^results: *.*/results/##p')
    if [ ! -f "$WFEMB_RESULTS/$run_dir/best.pt" ]; then
        mkdir -p "$WFEMB_RESULTS/$run_dir"; cp "$TEAM/results/$run_dir/best.pt" "$WFEMB_RESULTS/$run_dir/"
    fi
    started=$(date +%s)
    python -m wfemb.eval "id=$run_id" finalize=True 2>&1 | grep -v -i warn
    mkdir -p "$UPLOAD_DIR/results/$run_dir"
    cp "$WFEMB_RESULTS/$run_dir"/{final.json,test_embeddings.npz} "$UPLOAD_DIR/results/$run_dir/"
    echo "EVALUATED $run_id in $(( $(date +%s) - started ))s"
done

# The screenshot cache must not fail the job: the evaluations above are already uploaded.
if [ ! -f "$TEAM/demo_screens.done" ]; then
    python -m wfemb.demo screens --screens "$TEAM/demo_screens" --workers 16 && touch "$TEAM/demo_screens.done" \
        || echo "SCREENSHOTS FAILED"
fi
if [ -f "$TEAM/demo_screens.done" ]; then
    tar cf "$UPLOAD_DIR/demo_screens.tar" -C "$TEAM" --transform 's#^demo_screens#screens#' demo_screens
    echo "screenshots: $(du -sh "$UPLOAD_DIR/demo_screens.tar" | cut -f1)"
fi
echo EVALUATE_DONE
