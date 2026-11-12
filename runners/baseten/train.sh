#!/bin/bash
# Run one section of jobs.txt on one Baseten job: all its runs at once, spread round-robin over the job's GPUs.
# Usage: bash runners/baseten/train.sh <section>
# Inputs come from the team cache (runners/baseten/download.sh, runners/baseten/build.sh). Each run writes its results folder under
# $WFEMB_RESULTS; at the end the results tree goes to the job's checkpoint directory (locally:
# python -m wfemb.storage.fetch baseten) and to the team cache.
set -euo pipefail
cd "$BT_WORKING_DIR"
source runners/baseten/setup.sh
SECTION=$1
export WFEMB_DATA=$BT_SCRATCH_DIR/data WFEMB_BULK=$BT_SCRATCH_DIR/bulk WFEMB_RESULTS=$BT_SCRATCH_DIR/results
export WFEMB_LAUNCHER="baseten:${BT_TRAINING_JOB_ID:-$SECTION}"
mkdir -p "$WFEMB_DATA/logs" "$WFEMB_BULK" "$WFEMB_RESULTS"
t0=$(date +%s)
cp -r "$TEAM"/data/{prepared,keys,keys_openai} "$WFEMB_DATA/"
mapfile -t RUNS < <(python -m wfemb.storage.jobs runs "$SECTION")
(( ${#RUNS[@]} )) || { echo "no runs in section [$SECTION]"; exit 1; }
CACHES=$(for l in "${RUNS[@]}"; do python -m wfemb.config "id=${l%% *}" | sed -n 's#^\(added \)\?frames: *.*/bulk/##p'; done | sort -u)
if grep -q "^added/" <<< "$CACHES"; then                       # some run trains on the added datasets
    mkdir -p "$WFEMB_DATA/added"; cp -r "$TEAM"/data/added/{prepared,keys,keys_openai} "$WFEMB_DATA/added/"
fi
for n in $CACHES; do
    mkdir -p "$WFEMB_BULK/$n"                                  # the frame caches the runs need, to local NVMe
    for f in "$TEAM/bulk/$n"/*; do cp "$f" "$WFEMB_BULK/$n/" & done; wait
    for f in "$WFEMB_BULK/$n"/*.npy; do cat "$f" > /dev/null & done; wait          # into the page cache
done
echo "data staged in $(( $(date +%s) - t0 ))s: $(du -sh "$WFEMB_BULK" | cut -f1) frames; mem $(free -g | awk '/Mem/{print $2"G total, "$7"G avail"}')"
NG=$(nvidia-smi -L | wc -l); pids=(); tails=()
for i in "${!RUNS[@]}"; do
    read -r id overrides <<< "${RUNS[$i]}"
    log=$WFEMB_DATA/logs/r$i.out
    CUDA_VISIBLE_DEVICES=$(( i % NG )) python -m wfemb.train "id=$id" $overrides > "$log" 2>&1 &
    pids+=($!)
    (tail -n +1 -F "$log" 2>/dev/null | grep --line-buffered -E '^\[eval\]|^best step|Traceback|Error|Timeout \(|already finished' \
        | sed -u -e 's/^\(.\{400\}\).*$/\1/' -e "s/^/[r$i] /") &        # sed -u: unbuffered
    tails+=($!)
    echo "launched r$i on GPU $(( i % NG )): $id $overrides"; sleep 15
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
sleep 5; for t in "${tails[@]}"; do pkill -P "$t" 2>/dev/null; kill "$t" 2>/dev/null; done   # stop the log streamers
for i in "${!RUNS[@]}"; do                                    # failed runs: the end of their log
    grep -q "^best step" "$WFEMB_DATA/logs/r$i.out" || { echo "[r$i] FAILED, log tail:"; tail -30 "$WFEMB_DATA/logs/r$i.out"; }
done
mkdir -p "${BT_CHECKPOINT_DIR:-$TEAM/ckpt}/results" "$TEAM/results"
cp -r "$WFEMB_RESULTS/." "${BT_CHECKPOINT_DIR:-$TEAM/ckpt}/results/"
cp -r "$WFEMB_RESULTS/." "$TEAM/results/"
for l in "${RUNS[@]}"; do
    python - "${l%% *}" <<'PY'
import json, sys
from pathlib import Path
from wfemb.config import Experiment, default_results_root
experiment = Experiment.from_id(sys.argv[1])
final_path = Path(default_results_root()) / experiment.relative_dir / "final.json"
shown = ("best_step", "val_loss", "test_loss", "test_r@1", "test_r@5", "test_avg_log_dist_100NN")
if final_path.exists():
    result = {name: value for name, value in json.loads(final_path.read_text()).items() if name in shown}
else:
    result = {"error": "no final.json"}
print("RESULT", json.dumps({"id": experiment.id, **result}), flush=True)
PY
done
echo "TRAIN_DONE fail=$fail"; exit $fail
