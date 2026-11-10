"""Filesystem layout of the data. Everything large lives under WFEMB_DATA / WFEMB_BULK (on rnn: NAS).

  DATA/prepared/   sessions.pkl, events.npz, vocab.json        (prep/prepare.py)
  DATA/keys/       paraphrases.json, judged_<split>.jsonl, Qwen3 text_vecs    (embedder-independent + typed text)
  DATA/keys_openai/  keys.npy, para_keys.npy, para_index.npz, text_vecs.npy  (the label set used for training)
  BULK/frames_dense[_g<side>]/   frozen SigLIP2 dense screen features, one folder per grid size (prep/cache_frames.py)
Training results do not live here: see wfemb/storage/results.py.
"""
import os
from pathlib import Path

DATA = Path(os.environ.get("WFEMB_DATA", "/nas/ucb/dfeng/data/workflow-embeddings"))
RAW = DATA / "agentnet_raw"
PREP = DATA / "prepared"          # sessions, events, splits, vocab
SHARED = DATA / "keys"            # embedder-independent: paraphrases.json, judged_<split>.jsonl
# frozen text embeddings (instructions, paraphrases, typed text) of ONE text embedder; WFEMB_KEYS selects another
KEYS = Path(os.environ.get("WFEMB_KEYS", SHARED))
# Regenerable large caches live on the 60-day-TTL bulk share (keeps the ~2 TiB home quota free)
BULK = Path(os.environ.get("WFEMB_BULK", "/nas/ttl=60d/dfeng/workflow-embeddings"))

ZIPS = {   # last part of each split archive; read in place with prep/splitzip.py (no merged copy: NAS quota)
    "ubuntu": RAW / "ubuntu_images" / "images.zip",
    "win_mac": RAW / "win_mac_images" / "images.zip",
}
JSONLS = {
    "ubuntu": RAW / "agentnet_ubuntu_5k.jsonl",
    "win_mac": RAW / "agentnet_win_mac_18k.jsonl",
}
META = RAW / "meta_data_merged.jsonl"
