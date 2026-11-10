#!/bin/bash
# Demo bundles (the dashboard's Demo tab) for finished runs, built on Baseten where the frame caches and the AgentNet
# screenshots are (team cache, read in place). The first run's chosen test workflows are added to every other run's
# bundle, so the same workflows can be compared across runs.
# Usage: bash runners/baseten/demo.sh "<run id> ..." ["<old run name> ..."]
#   old run names: pre-refactor runs in $TEAM/runs/<name>/best.pt, converted into the results tree first
# Output: $BT_CHECKPOINT_DIR/bundles/<run id>/ (locally: python -m wfemb.storage.fetch baseten --job <job id>)
set -euo pipefail
cd "$BT_WORKING_DIR"
source runners/baseten/setup.sh
IDS=$1; LEGACY=${2:-}
export WFEMB_DATA=$BT_SCRATCH_DIR/data WFEMB_BULK=$TEAM/bulk WFEMB_RESULTS=$BT_SCRATCH_DIR/results
OUT=$BT_SCRATCH_DIR/bundles
mkdir -p "$WFEMB_DATA" "$WFEMB_RESULTS" "$OUT"
cp -r "$TEAM"/data/{prepared,keys,keys_openai} "$WFEMB_DATA/"
ln -sfn "$TEAM/raw/agentnet_raw" "$WFEMB_DATA/agentnet_raw"            # screenshots, read in place
for name in $LEGACY; do python runners/slurm/import_legacy.py ckpt "$TEAM/runs/$name/best.pt" --root "$WFEMB_RESULTS"; done
for id in $IDS; do                                                       # checkpoints of new-code runs
    d=$(python -m wfemb.config "id=$id" | sed -n 's#^results: *.*/results/##p')
    [ -f "$WFEMB_RESULTS/$d/best.pt" ] || { mkdir -p "$WFEMB_RESULTS/$d"; cp "$TEAM/results/$d/best.pt" "$WFEMB_RESULTS/$d/"; }
done
common=()
for id in $IDS; do
    t0=$(date +%s)
    python -m wfemb.demo embed --id "$id"
    python -m wfemb.demo background --id "$id" --out "$OUT/$id"
    python -m wfemb.demo export --id "$id" --out "$OUT/$id" --auto 8 "${common[@]}"
    python -m wfemb.demo frames --id "$id" --out "$OUT/$id" > /dev/null
    if (( ${#common[@]} == 0 )); then                                    # the first run's picks, for every other run
        mapfile -t picks < <(python -c "import json,sys; [print(e['id']) for e in json.load(open(sys.argv[1]))]" \
                             "$OUT/$id/workflows/index.json")
        for p in "${picks[@]}"; do common+=(--session "$p"); done
    fi
    echo "BUNDLE $id: $(ls "$OUT/$id/workflows" | wc -l) files, $(du -sh "$OUT/$id" | cut -f1), $(( $(date +%s) - t0 ))s"
done
mkdir -p "${BT_CHECKPOINT_DIR:-$TEAM/ckpt}/bundles"
cp -r "$OUT/." "${BT_CHECKPOINT_DIR:-$TEAM/ckpt}/bundles/"
echo DEMO_DONE
