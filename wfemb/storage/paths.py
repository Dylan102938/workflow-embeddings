"""Where the data lives (layout: docs/DATA_PREP.md). Roots come from WFEMB_DATA and WFEMB_BULK."""

import os
from pathlib import Path

DATA = Path(os.environ.get("WFEMB_DATA", "/nas/ucb/dfeng/data/workflow-embeddings"))
RAW = DATA / "agentnet_raw"
PREP = DATA / "prepared"
SHARED = DATA / "keys"  # embedder-independent files: paraphrases.json, judged_<split>.jsonl
KEYS = Path(os.environ.get("WFEMB_KEYS", SHARED))
BULK = Path(os.environ.get("WFEMB_BULK", "/nas/ttl=60d/dfeng/workflow-embeddings"))

ZIPS = {
    "ubuntu": RAW / "ubuntu_images" / "images.zip",
    "win_mac": RAW / "win_mac_images" / "images.zip",
}
JSONLS = {
    "ubuntu": RAW / "agentnet_ubuntu_5k.jsonl",
    "win_mac": RAW / "agentnet_win_mac_18k.jsonl",
}
META = RAW / "meta_data_merged.jsonl"
