"""Filesystem layout. Everything large lives under WFEMB_DATA (on rnn: NAS)."""
import os
from pathlib import Path

DATA = Path(os.environ.get("WFEMB_DATA", "/nas/ucb/dfeng/data/workflow-embeddings"))
RAW = DATA / "agentnet_raw"
PREP = DATA / "prepared"          # sessions, events, splits, vocab
KEYS = DATA / "keys"              # frozen text embeddings (instructions, paraphrases, typed text)
FRAMES = DATA / "frames"          # frozen vision-encoder cache
RUNS = DATA / "runs"              # checkpoints + logs
# Regenerable extra caches live on the 60-day-TTL bulk share (keeps the ~2 TiB home quota free)
BULK = Path(os.environ.get("WFEMB_BULK", "/nas/ttl=60d/dfeng/workflow-embeddings"))
FRAMES_EFFECT = BULK / "frames_effect"   # next-frame crops around the previous action point

ZIPS = {   # last part of each split archive; read in place with wfemb.splitzip (no merged copy: NAS quota)
    "ubuntu": RAW / "ubuntu_images" / "images.zip",
    "win_mac": RAW / "win_mac_images" / "images.zip",
}
JSONLS = {
    "ubuntu": RAW / "agentnet_ubuntu_5k.jsonl",
    "win_mac": RAW / "agentnet_win_mac_18k.jsonl",
}
META = RAW / "meta_data_merged.jsonl"
