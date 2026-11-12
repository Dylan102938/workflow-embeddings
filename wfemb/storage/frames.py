"""Reads the frozen screen-feature cache and other large arrays from NFS-hosted files.

The reading strategy (preadv, span reads, small chunks, pinned buffers) is explained in docs/ARCHITECTURE.md.
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

MAX_READ_BYTES = 1 << 20  # larger single reads can livelock the NFS client


def read_file(path) -> io.BytesIO:
    """Whole file into memory, in reads of at most MAX_READ_BYTES."""
    size = os.path.getsize(path)
    buffer = bytearray(size)
    view = memoryview(buffer)
    with open(path, "rb", buffering=0) as file:
        done = 0
        while done < size:
            count = file.readinto(view[done : done + MAX_READ_BYTES])
            if not count:
                raise OSError(f"short read of {path} at byte {done}")
            done += count
    return io.BytesIO(buffer)


def load_numpy(path):
    """np.load for .npy / .npz files on NFS."""
    return np.load(read_file(path))


class NpyRows:
    """Row-range reads from a .npy file with preadv (no memory map). Safe to share between threads."""

    def __init__(self, path):
        with open(path, "rb") as file:
            version = np.lib.format.read_magic(file)
            read_header = (
                np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            )
            shape, fortran_order, dtype = read_header(file)
            assert not fortran_order, path
            self.data_offset = file.tell()
        self.file_descriptor = os.open(path, os.O_RDONLY)
        self.shape, self.dtype = shape, np.dtype(dtype)
        self.row_bytes = int(np.prod(shape[1:])) * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def read_into(self, out: np.ndarray, first_row: int):
        """Fills `out` (contiguous) with the rows starting at first_row."""
        view = memoryview(out).cast("B")
        offset = self.data_offset + first_row * self.row_bytes
        done = 0
        while done < len(view):
            count = os.preadv(self.file_descriptor, [view[done : done + MAX_READ_BYTES]], offset + done)
            if count <= 0:
                raise OSError(f"short read at byte {offset + done}")
            done += count


class FrameStore:
    """Screen tokens of every step, [n_steps, n_patches, dim] as bfloat16 bits, addressed by global step index.

    A cache folder holds shards: global_<tag>.npy and meta_<tag>.json with the shard's step range [lo, hi).
    caches: (folder, global index of the folder's first step) pairs.
    """

    def __init__(self, caches: list[tuple[Path, int]]):
        shards = []
        for cache_dir, first_step in caches:
            cache_dir = Path(cache_dir)
            meta_paths = sorted(glob.glob(str(cache_dir / "meta_*.json")))
            if not meta_paths:
                raise FileNotFoundError(f"no frame cache in {cache_dir}")
            for meta_path in meta_paths:
                meta = json.load(open(meta_path))
                tag = Path(meta_path).stem.split("_", 1)[1]
                rows = NpyRows(cache_dir / f"global_{tag}.npy")
                shards.append((first_step + meta["lo"], first_step + meta["hi"], rows))
        self.shards = sorted(shards, key=lambda shard: shard[0])
        self.shard_ends = np.array([end for _, end, _ in self.shards])
        self.n_steps = int(self.shard_ends.max())
        self.dim, self.n_patches = meta["dim"], meta["global_tokens"]
        self._thread_local = threading.local()

    def read_into(self, step_indices: np.ndarray, out: np.ndarray, max_gap: int = 8) -> np.ndarray:
        """Fills out[:len(step_indices)] without allocating.

        Each run of increasing indices (one session's steps) is a single span read. Repeated indices (padding
        repeats a session's last step) are copied from the previous row.
        """
        indices = np.asarray(step_indices, np.int64)
        is_repeat = np.r_[False, indices[1:] == indices[:-1]]
        positions = np.nonzero(~is_repeat)[0]
        if len(positions):
            steps = indices[positions]
            shard_of = np.searchsorted(self.shard_ends, steps, side="right")
            step_gaps = np.diff(steps)
            run_breaks = (
                np.nonzero(
                    (np.diff(positions) != 1) | (step_gaps < 1) | (step_gaps > max_gap) | (np.diff(shard_of) != 0)
                )[0]
                + 1
            )
            for start, end in zip(np.r_[0, run_breaks], np.r_[run_breaks, len(positions)]):
                shard_start, _, rows = self.shards[shard_of[start]]
                first_step, last_step = int(steps[start]), int(steps[end - 1])
                target = out[int(positions[start]) : int(positions[end - 1]) + 1]
                span = last_step - first_step + 1
                if span == end - start:
                    rows.read_into(target, first_step - shard_start)
                else:
                    scratch = self._scratch(span)
                    rows.read_into(scratch[:span], first_step - shard_start)
                    np.take(scratch, steps[start:end] - first_step, axis=0, out=target)
        if is_repeat.any():
            source = np.maximum.accumulate(np.where(is_repeat, 0, np.arange(len(indices))))
            out[is_repeat] = out[source[is_repeat]]
        return out

    def read(self, step_indices: np.ndarray) -> np.ndarray:
        return self.read_into(step_indices, np.empty((len(step_indices), self.n_patches, self.dim), np.uint16))

    def _scratch(self, n_rows: int) -> np.ndarray:
        scratch = getattr(self._thread_local, "scratch", None)
        if scratch is None or len(scratch) < n_rows:
            scratch = np.empty((max(n_rows, 512), self.n_patches, self.dim), np.uint16)
            self._thread_local.scratch = scratch
        return scratch


def as_bfloat16(bits: np.ndarray) -> torch.Tensor:
    """uint16 bit patterns -> bfloat16 tensor sharing the memory."""
    return torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)


class PinnedBufferPool:
    """A fixed set of pinned host buffers for frame tokens, allocated once and reused for every batch."""

    def __init__(self, n_buffers: int, n_rows: int, n_patches: int, dim: int):
        self.n_rows = n_rows
        self._free = queue.Queue()
        for _ in range(n_buffers):
            self._free.put(torch.empty((n_rows, n_patches, dim), dtype=torch.int16).pin_memory())

    def take(self) -> torch.Tensor:
        return self._free.get()

    def give_back(self, buffer: torch.Tensor):
        self._free.put(buffer)


_pools: dict = {}


def pinned_buffer_pool(n_buffers: int, n_rows: int, n_patches: int, dim: int) -> PinnedBufferPool:
    """One pool per shape for the whole process: pinned memory is never returned to the system."""
    key = (n_buffers, n_rows, n_patches, dim)
    if key not in _pools:
        _pools[key] = PinnedBufferPool(*key)
    return _pools[key]
