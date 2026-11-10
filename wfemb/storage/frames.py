"""Reading the frozen screen-feature cache (exact bf16) and other large arrays from NFS-hosted files.

The cache of one grid size is a folder of shards: global_<tag>.npy [rows, n_tok, 512] (bf16 bits stored as uint16) and
meta_<tag>.json {"lo", "hi", "dim", "global_tokens"}; rows lo..hi-1 are global step indices (all sessions' steps in
order). Hard-won rules (rnn NFS, several training processes per node):
  - read with preadv, never memory-map: page faults on a file mapped by several long-running multithreaded processes
    stall them (1-50 s per batch, and after ~1 h a wedge on the mmap lock);
  - read a session's steps as one contiguous span (one syscall per session, not per frame): per-call GIL hand-offs
    against the training thread made per-frame reads ~50x slower;
  - at most 1 MB per read call: large single reads livelock the NFS client on memory-pressured nodes;
  - no per-batch allocations of large buffers: frames are read into a fixed pool of pinned buffers.
"""
from __future__ import annotations

import glob
import io
import json
import os
import queue
import threading
from pathlib import Path

import numpy as np
import torch

NFS_CHUNK = 1 << 20      # max bytes per read syscall on NFS


def read_chunked(path) -> io.BytesIO:
    """Whole file into memory with reads of at most NFS_CHUNK bytes."""
    size = os.path.getsize(path)
    buf = bytearray(size)
    mv = memoryview(buf)
    with open(path, "rb", buffering=0) as f:
        done = 0
        while done < size:
            n = f.readinto(mv[done:done + NFS_CHUNK])
            if not n:
                raise IOError(f"short read of {path} at {done}")
            done += n
    return io.BytesIO(buf)


def np_load(path):
    """np.load (.npy / .npz) via chunked reads."""
    return np.load(read_chunked(path))


class PreadArray:
    """Row access to a .npy file with os.preadv (no mmap). Thread-safe (explicit offsets)."""

    def __init__(self, path):
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                           else np.lib.format.read_array_header_2_0)
            shape, fortran, dtype = read_header(f)
            assert not fortran, path
            self.offset = f.tell()
        self.fd = os.open(path, os.O_RDONLY)
        self.shape, self.dtype = shape, np.dtype(dtype)
        self.row_bytes = int(np.prod(shape[1:])) * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def read_into(self, buf: np.ndarray, row: int):
        """Fill buf (contiguous, len(buf) rows) with rows row.. of the file."""
        mv, done, offset = memoryview(buf).cast("B"), 0, self.offset + row * self.row_bytes
        while done < len(mv):
            n = os.preadv(self.fd, [mv[done:done + NFS_CHUNK]], offset + done)
            if n <= 0:
                raise IOError(f"short read at {offset + done}")
            done += n


class FrameStore:
    """All steps' screen tokens [total, n_tok, dim] (bf16 bits as uint16), addressed by global step index."""

    def __init__(self, frames_dir: Path):
        frames_dir = Path(frames_dir)
        metas = sorted(glob.glob(str(frames_dir / "meta_*.json")))
        if not metas:
            raise FileNotFoundError(f"no frame cache in {frames_dir}")
        shards = []
        for mp in metas:
            m = json.load(open(mp))
            tag = Path(mp).stem.split("_", 1)[1]
            shards.append((m["lo"], m["hi"], PreadArray(frames_dir / f"global_{tag}.npy")))
        self.shards = sorted(shards, key=lambda s: s[0])
        m0 = json.load(open(metas[0]))
        self.dim, self.n_tok = m0["dim"], m0["global_tokens"]
        self.hi = np.array([s[1] for s in self.shards])
        self.total = int(self.hi.max())
        self._tls = threading.local()

    def gather_into(self, gidx: np.ndarray, out: np.ndarray, gap: int = 8) -> np.ndarray:
        """Rows gidx -> out[:len(gidx)] (uint16 [*, n_tok, dim]) without allocating. Each run of increasing rows (a
        session's kept steps) is one span read, straight into `out` or, if steps were dropped, via a per-thread scratch
        buffer; repeated rows (padding repeats a session's last step) are copied from the previous row."""
        g = np.asarray(gidx, np.int64)
        n = len(g)
        dup = np.r_[False, g[1:] == g[:-1]]
        real = np.nonzero(~dup)[0]
        if len(real):
            gr = g[real]
            sh = np.searchsorted(self.hi, gr, side="right")
            brk = np.nonzero((np.diff(real) != 1) | (np.diff(gr) < 1) | (np.diff(gr) > gap) | (np.diff(sh) != 0))[0] + 1
            for a, b in zip(np.r_[0, brk], np.r_[brk, len(real)]):
                p0, p1 = int(real[a]), int(real[b - 1]) + 1          # contiguous output positions
                lo_shard, _, arr = self.shards[sh[a]]
                lo, hi = int(gr[a]), int(gr[b - 1]) + 1
                if hi - lo == p1 - p0:
                    arr.read_into(out[p0:p1], lo - lo_shard)
                else:
                    tmp = self._scratch(hi - lo)
                    arr.read_into(tmp[:hi - lo], lo - lo_shard)
                    np.take(tmp, gr[a:b] - lo, axis=0, out=out[p0:p1])
        if dup.any():
            src = np.maximum.accumulate(np.where(dup, 0, np.arange(n)))
            out[dup] = out[src[dup]]
        return out

    def gather(self, gidx: np.ndarray) -> np.ndarray:
        return self.gather_into(gidx, np.empty((len(gidx), self.n_tok, self.dim), np.uint16))

    def _scratch(self, rows: int) -> np.ndarray:
        buf = getattr(self._tls, "buf", None)
        if buf is None or len(buf) < rows:
            buf = np.empty((max(rows, 512), self.n_tok, self.dim), np.uint16)
            self._tls.buf = buf
        return buf


def bf16(bits: np.ndarray) -> torch.Tensor:
    """uint16 bf16 bits -> torch bfloat16 (no copy)."""
    return torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)


class PinnedPool:
    """A fixed set of pinned host buffers [rows, n_tok, dim] for frame tokens, allocated once and reused."""

    def __init__(self, n: int, rows: int, n_tok: int, dim: int):
        self.rows = rows
        self.free = queue.Queue()
        for _ in range(n):
            self.free.put(torch.empty((rows, n_tok, dim), dtype=torch.int16).pin_memory())

    def get(self):
        return self.free.get()

    def put(self, buf):
        self.free.put(buf)


_POOLS: dict = {}


def pinned_pool(n: int, rows: int, n_tok: int, dim: int) -> PinnedPool:
    """Process-wide pool (pinning is expensive and pinned memory is never returned)."""
    key = (n, rows, n_tok, dim)
    if key not in _POOLS:
        _POOLS[key] = PinnedPool(n, rows, n_tok, dim)
    return _POOLS[key]
