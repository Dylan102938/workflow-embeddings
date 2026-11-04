#!/bin/bash
#SBATCH --job-name=wfemb-mvframes
#SBATCH --partition=main
#SBATCH --qos=high
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/nas/ucb/dfeng/data/workflow-embeddings/logs/%x-%j.out
# Move the regenerable 8x8 frame cache (71 GB) off the ~2 TiB home quota onto the bulk share; leave a symlink.
set -euo pipefail
SRC=/nas/ucb/dfeng/data/workflow-embeddings/frames
DST="/nas/ttl=60d/dfeng/workflow-embeddings/frames_main"
[ -L "$SRC" ] && { echo "already a symlink: $(readlink $SRC)"; exit 0; }
mkdir -p "$DST"
cp -a "$SRC"/. "$DST"/
if diff <(cd "$SRC" && find . -type f -printf "%P %s\n" | sort) <(cd "$DST" && find . -type f -printf "%P %s\n" | sort); then
    for f in global_03of08.npy crop_05of08.npy meta_00of08.json; do cmp "$SRC/$f" "$DST/$f"; done
    rm -rf "$SRC" && ln -s "$DST" "$SRC" && echo "moved; $SRC -> $DST"
else
    echo "verification failed; original left in place"; exit 1
fi
