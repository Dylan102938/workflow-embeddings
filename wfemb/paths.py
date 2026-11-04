"""Filesystem layout. Everything large lives under WFEMB_DATA (on rnn: NAS)."""
import os
from pathlib import Path

DATA = Path(os.environ.get("WFEMB_DATA", "/nas/ucb/dfeng/data/workflow-embeddings"))
RAW = DATA / "agentnet_raw"
PREP = DATA / "prepared"          # sessions, events, splits, vocab
KEYS = DATA / "keys"              # frozen text embeddings (instructions, paraphrases, typed text)
FRAMES = DATA / "frames"          # frozen vision-encoder cache
RUNS = DATA / "runs"              # checkpoints + logs

ZIPS = {
    "ubuntu": RAW / "ubuntu_images" / "merged.zip",
    "win_mac": RAW / "win_mac_images" / "merged.zip",
}
JSONLS = {
    "ubuntu": RAW / "agentnet_ubuntu_5k.jsonl",
    "win_mac": RAW / "agentnet_win_mac_18k.jsonl",
}
META = RAW / "meta_data_merged.jsonl"
