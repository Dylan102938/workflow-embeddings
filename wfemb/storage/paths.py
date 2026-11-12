"""Where the data lives (layout: docs/DATA_PREP.md). Roots come from WFEMB_DATA and WFEMB_BULK.

The datasets added to AgentNet have the same layout below ADDED_DATA and ADDED_BULK. WFEMB_SET=added points every
preparation step (DATA, BULK and the paths built from them) at that copy of the layout.
"""

import os
from pathlib import Path

AGENTNET = "agentnet"
ADDED = "added"
ADDED_DATASETS = ("videocua", "scalecua", "gui360")

AGENTNET_DATA = Path(os.environ.get("WFEMB_DATA", "/nas/ucb/dfeng/data/workflow-embeddings"))
AGENTNET_BULK = Path(os.environ.get("WFEMB_BULK", "/nas/ttl=60d/dfeng/workflow-embeddings"))
ADDED_DATA, ADDED_BULK = AGENTNET_DATA / ADDED, AGENTNET_BULK / ADDED

ON_ADDED_SET = os.environ.get("WFEMB_SET", AGENTNET) == ADDED
DATA = ADDED_DATA if ON_ADDED_SET else AGENTNET_DATA
BULK = ADDED_BULK if ON_ADDED_SET else AGENTNET_BULK

RAW = DATA / "agentnet_raw"
PREP = DATA / "prepared"
SHARED = DATA / "keys"  # embedder-independent files: paraphrases.json, judged_<split>.jsonl
KEYS = Path(os.environ.get("WFEMB_KEYS", SHARED))
SOURCES = ADDED_DATA / "sources"  # one folder per added dataset: the download and its conversion

ZIPS = {
    "ubuntu": RAW / "ubuntu_images" / "images.zip",
    "win_mac": RAW / "win_mac_images" / "images.zip",
    **{dataset: SOURCES / dataset / "images.zip" for dataset in ADDED_DATASETS},
}
JSONLS = {
    "ubuntu": RAW / "agentnet_ubuntu_5k.jsonl",
    "win_mac": RAW / "agentnet_win_mac_18k.jsonl",
}
META = RAW / "meta_data_merged.jsonl"
