"""Training-time data: memory-mapped frame cache + flat event arrays + frozen keys.

Batches are padded B x T (length-bucketed). Everything is gathered on CPU with numpy fancy indexing
and moved to GPU by a background prefetch thread.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import paths


@dataclass
class Batch:
    idx: torch.Tensor          # [B]   session indices (global session order)
    frames: torch.Tensor       # [B, T, n_tok, D_v] bf16
    ev_type: torch.Tensor      # [B, T, E] long
    ev_xy: torch.Tensor        # [B, T, E, 4] float (nan = absent)
    ev_scroll: torch.Tensor    # [B, T, E] float
    ev_keys: torch.Tensor      # [B, T, E, K] long
    ev_text: torch.Tensor      # [B, T, E] long (-1 = none)
    valid: torch.Tensor        # [B, T] bool
    lengths: torch.Tensor      # [B]

    def to(self, device):
        return Batch(**{k: v.to(device, non_blocking=True) for k, v in self.__dict__.items()})


class FrameStore:
    """Concatenation of per-shard memmaps, addressed by global step index."""

    def __init__(self, frames_dir: Path, use_crop: bool):
        metas = sorted(glob.glob(str(frames_dir / "meta_*.json")))
        if not metas:
            raise FileNotFoundError(f"no frame cache in {frames_dir}")
        self.shards = []
        for mp in metas:
            m = json.load(open(mp))
            tag = Path(mp).stem.split("_", 1)[1]
            g = np.load(frames_dir / f"global_{tag}.npy", mmap_mode="r")
            c = np.load(frames_dir / f"crop_{tag}.npy", mmap_mode="r") if use_crop else None
            self.shards.append((m["lo"], m["hi"], g, c))
        self.shards.sort(key=lambda s: s[0])
        self.use_crop = use_crop
        m0 = json.load(open(metas[0]))
        self.dim = m0["dim"]
        self.n_tok = m0["global_tokens"] + (m0["crop_tokens"] if use_crop else 0)
        self.lo = np.array([s[0] for s in self.shards])
        self.hi = np.array([s[1] for s in self.shards])
        self.total = int(self.hi.max())

    def gather(self, gidx: np.ndarray) -> np.ndarray:
        """gidx: 1-D global step indices -> [len, n_tok, D] uint16 (bf16 bits)."""
        out = np.empty((len(gidx), self.n_tok, self.dim), np.uint16)
        sh = np.searchsorted(self.hi, gidx, side="right")
        for k in np.unique(sh):
            lo, hi, g, c = self.shards[k]
            sel = np.nonzero(sh == k)[0]
            rows = gidx[sel] - lo
            order = np.argsort(rows)          # sorted reads are friendlier to the page cache
            rs = rows[order]
            gtok = g[rs]
            if c is not None:
                gtok = np.concatenate([gtok, c[rs]], axis=1)
            out[sel[order]] = gtok
        return out


class SessionData:
    def __init__(self, frames_dir: Path | None = None, use_crop: bool = False):
        with open(paths.PREP / "sessions.pkl", "rb") as f:
            self.sessions = pickle.load(f)
        ev = np.load(paths.PREP / "events.npz")
        self.ev = {k: ev[k] for k in ev.files}
        self.offset = self.ev["sess_offset"]
        self.length = self.ev["sess_len"]
        with open(paths.PREP / "vocab.json") as f:
            vocab = json.load(f)
        self.n_keys = len(vocab["key_vocab"])
        self.keys = np.load(paths.KEYS / "keys.npy").astype(np.float32)
        self.text_vecs = np.load(paths.KEYS / "text_vecs.npy")
        pk = paths.KEYS / "para_keys.npy"
        if pk.exists():
            self.para_keys = np.load(pk).astype(np.float32)
            pi = np.load(paths.KEYS / "para_index.npz")
            self.para_offset, self.para_count = pi["offset"], pi["count"]
        else:
            self.para_keys = np.zeros((0, self.keys.shape[1]), np.float32)
            self.para_offset = np.zeros(len(self.sessions), np.int64)
            self.para_count = np.zeros(len(self.sessions), np.int32)
        self.split = np.array([s["split"] for s in self.sessions])
        groups = {g: i for i, g in enumerate(sorted({s["group"] for s in self.sessions}))}
        self.group = np.array([groups[s["group"]] for s in self.sessions])
        self.frames = FrameStore(frames_dir or paths.FRAMES, use_crop)
        assert self.frames.total == int(self.offset[-1] + self.length[-1]), "frame cache / events mismatch"

    def indices(self, split: str) -> np.ndarray:
        return np.nonzero(self.split == split)[0]

    def make_batch(self, sidx: np.ndarray, max_len: int | None = None) -> Batch:
        lens = self.length[sidx].astype(np.int64)
        if max_len:
            lens = np.minimum(lens, max_len)
        B, T = len(sidx), int(lens.max())
        valid = np.arange(T)[None, :] < lens[:, None]
        gidx = (self.offset[sidx][:, None] + np.minimum(np.arange(T)[None, :], lens[:, None] - 1))
        flat = gidx.reshape(-1)
        frames = self.frames.gather(flat).reshape(B, T, self.frames.n_tok, self.frames.dim)
        e = self.ev
        return Batch(
            idx=torch.from_numpy(sidx.astype(np.int64)),
            frames=torch.from_numpy(frames.view(np.int16)).view(torch.bfloat16),
            ev_type=torch.from_numpy(e["ev_type"][flat].reshape(B, T, -1).astype(np.int64)),
            ev_xy=torch.from_numpy(e["ev_xy"][flat].reshape(B, T, -1, 4).astype(np.float32)),
            ev_scroll=torch.from_numpy(e["ev_scroll"][flat].reshape(B, T, -1).astype(np.float32)),
            ev_keys=torch.from_numpy(e["ev_keys"][flat].reshape(B, T, *e["ev_keys"].shape[1:]).astype(np.int64)),
            ev_text=torch.from_numpy(e["ev_text"][flat].reshape(B, T, -1).astype(np.int64)),
            valid=torch.from_numpy(valid),
            lengths=torch.from_numpy(lens),
        )

    def sample_positive_keys(self, sidx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Each session's positive is a random wording of its instruction (original or a kept paraphrase),
        re-drawn every time the session is seen."""
        out = self.keys[sidx].copy()
        cnt = self.para_count[sidx]
        pick = rng.integers(0, cnt + 1)                   # 0 = original key
        use = pick > 0
        if use.any():
            out[use] = self.para_keys[self.para_offset[sidx][use] + pick[use] - 1]
        return out


def length_bucketed_batches(lengths: np.ndarray, idx: np.ndarray, batch: int, rng: np.random.Generator,
                            bucket_mult: int = 50, drop_last: bool = True) -> list[np.ndarray]:
    """Shuffle, then sort within large buckets so each batch pads among similar lengths."""
    perm = rng.permutation(idx)
    out = []
    span = batch * bucket_mult
    for s in range(0, len(perm), span):
        chunk = perm[s:s + span]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        for b in range(0, len(chunk), batch):
            bb = chunk[b:b + batch]
            if len(bb) == batch or not drop_last:
                out.append(bb)
    order = rng.permutation(len(out))
    return [out[i] for i in order]


class Prefetcher:
    """Builds batches on background threads (in order) and moves them to the GPU (pinned, non_blocking)."""

    def __init__(self, data: SessionData, batches: list[np.ndarray], device, extra=None, depth: int = 6,
                 threads: int = 3, max_len: int | None = None):
        self.data, self.batches, self.device, self.extra, self.max_len = data, batches, device, extra, max_len
        self.depth = depth
        self.pool = ThreadPoolExecutor(threads)

    def _make(self, i):
        b = self.data.make_batch(self.batches[i], self.max_len)
        b = Batch(**{k: v.pin_memory() for k, v in b.__dict__.items()})
        ex = self.extra(self.batches[i]) if self.extra else None
        return b, ex

    def __iter__(self):
        n, nxt, futs = len(self.batches), 0, deque()
        while nxt < min(self.depth, n):
            futs.append(self.pool.submit(self._make, nxt))
            nxt += 1
        for _ in range(n):
            b, ex = futs.popleft().result()
            if nxt < n:
                futs.append(self.pool.submit(self._make, nxt))
                nxt += 1
            yield b.to(self.device), ex
        self.pool.shutdown(wait=False)


def stage_to_shm(src: Path, job_tag: str, include_crop: bool = True) -> Path:
    """Copy the frame cache into /dev/shm (RAM) once per job so several runs on the node share one copy.
    The persistent copy stays on NAS; the shm copy is removed by the sbatch script at exit."""
    dst = Path("/dev/shm") / f"wfemb_frames_{job_tag}"
    done = dst / ".complete"
    if done.exists():
        return dst
    lock = Path(str(dst) + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL)
        os.close(fd)
    except FileExistsError:
        import time
        while not done.exists():
            time.sleep(5)
        return dst
    import shutil
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*")):
        if f.suffix not in (".npy", ".json") or (f.name.startswith("crop_") and not include_crop):
            continue
        shutil.copy(f, dst / f.name)
    done.touch()
    return dst
