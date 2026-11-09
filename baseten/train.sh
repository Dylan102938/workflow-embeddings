#!/bin/bash
# Run a packed group of wfemb training runs on one Baseten job (runs spread round-robin over the job's GPUs).
# Usage: bash baseten/train.sh baseten/jobs/<group>.txt     (one run's train args per line; rnn paths are rewritten)
# Data comes from the team cache (built by baseten/build.sh); run dirs go to the synced checkpoint dir + team cache.
set -euo pipefail
cd "$BT_WORKING_DIR"
source baseten/setup.sh
GROUP=$1
export WFEMB_DATA=$BT_SCRATCH_DIR/data WFEMB_BULK=$BT_SCRATCH_DIR/bulk
mkdir -p "$WFEMB_DATA/runs" "$WFEMB_DATA/logs" "$WFEMB_BULK"
t0=$(date +%s)
cp -r "$TEAM"/data/{prepared,keys,keys_openai,keys_openai_p2} "$WFEMB_DATA/"
mapfile -t RUNS < <(grep -v '^\s*#' "$GROUP" | grep -v '^\s*$' \
    | sed "s#/nas/ttl=60d/dfeng/workflow-embeddings#$WFEMB_BULK#g; s#/nas/ucb/dfeng/data/workflow-embeddings#$WFEMB_DATA#g")
for d in $(printf '%s\n' "${RUNS[@]}" | sed -n 's/.*--frames_dir \([^ ]*\).*/\1/p' | sort -u); do
    n=$(basename "$d"); mkdir -p "$WFEMB_BULK/$n"
    for f in "$TEAM/bulk/$n"/*; do cp "$f" "$WFEMB_BULK/$n/" & done; wait
    for f in "$WFEMB_BULK/$n"/*.npy; do cat "$f" > /dev/null & done; wait          # into the page cache
done
echo "data staged in $(( $(date +%s) - t0 ))s: $(du -sh "$WFEMB_BULK" | cut -f1) frames; mem $(free -g | awk '/Mem/{print $2"G total, "$7"G avail"}')"
NG=$(nvidia-smi -L | wc -l); pids=()
for i in "${!RUNS[@]}"; do
    line=${RUNS[$i]}; name=$(echo "$line" | sed -n 's/.*--run \([^ ]*\).*/\1/p'); log=$WFEMB_DATA/logs/run-$name.out
    CUDA_VISIBLE_DEVICES=$(( i % NG )) python -m wfemb.train $line --frames_mode pread > "$log" 2>&1 &
    pids+=($!)
    (tail -n +1 -F "$log" 2>/dev/null | grep --line-buffered -E '"val_loss"|"final"|Traceback|Error|Timeout \(|total time' \
        | cut -c1-400 | sed -u "s/^/[$name] /") &
    echo "launched $name on GPU $(( i % NG ))"; sleep 15
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
mkdir -p "${BT_CHECKPOINT_DIR:-$TEAM/ckpt}/runs" "$TEAM/runs"
for line in "${RUNS[@]}"; do
    name=$(echo "$line" | sed -n 's/.*--run \([^ ]*\).*/\1/p')
    cp "$WFEMB_DATA/logs/run-$name.out" "$WFEMB_DATA/runs/$name/" 2>/dev/null || true
    cp -r "$WFEMB_DATA/runs/$name" "${BT_CHECKPOINT_DIR:-$TEAM/ckpt}/runs/"; cp -r "$WFEMB_DATA/runs/$name" "$TEAM/runs/"
    python - "$WFEMB_DATA/runs/$name" <<'PY'
import json, os, sys
d = sys.argv[1]; f = os.path.join(d, "final.json")
if os.path.exists(f):
    j = json.load(open(f)); v, t = j["val"], j["test"]
    keep = ("loss", "key_R@1", "key_R@5", "key_MRR", "jsession_R@1_cross_app", "jsession_R@5_cross_app",
            "jsession_MRR_cross_app", "jsession_nDCG@10_cross_app")
    print("RESULT", json.dumps({"run": os.path.basename(d), "best_step": j["best_step"],
                                 "val": {k: v.get(k) for k in keep}, "test": {k: t.get(k) for k in keep}}), flush=True)
else:
    print("RESULT", json.dumps({"run": os.path.basename(d), "error": "no final.json"}), flush=True)
PY
done
echo "TRAIN_DONE fail=$fail"; exit $fail
