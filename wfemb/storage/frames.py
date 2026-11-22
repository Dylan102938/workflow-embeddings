"""Reads the frozen screen-feature cache and other large arrays from NFS-hosted files.

The reading strategy (preadv, span reads, small chunks, pinned buffers) is explained in docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import glob
import io
import json
import os
import queue
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


class TokenFile:
    """Token-range reads from a file of [n_tokens, dim] uint16 with preadv (no memory map). Safe to share between
    threads."""

    def __init__(self, path, dim: int, data_offset: int = 0):
        self.file_descriptor = os.open(path, os.O_RDONLY)
        self.token_bytes, self.data_offset = 2 * dim, data_offset

    def read_into(self, out: np.ndarray, first_token: int):
        """Fills `out` (contiguous) with the tokens starting at first_token."""
        view = memoryview(out).cast("B")
        offset = self.data_offset + first_token * self.token_bytes
        done = 0
        while done < len(view):
            count = os.preadv(self.file_descriptor, [view[done : done + MAX_READ_BYTES]], offset + done)
            if count <= 0:
                raise OSError(f"short read at byte {offset + done}")
            done += count


def npy_data_offset(path) -> int:
    with open(path, "rb") as file:
        version = np.lib.format.read_magic(file)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        read_header(file)
        return file.tell()


class FrameStore:
    """Screen tokens of every step, as bfloat16 bits, addressed by global step index.

    Step s has token_counts[s] tokens of `dim` channels, in row-major order on a grid of grids[s] = (rows, columns).
    A cache folder holds shards; meta_<tag>.json gives a shard's step range [lo, hi). Two formats (docs/DATA_PREP.md):
    a square grid for every step (global_<tag>.npy), or a tower's own grid per step (tokens_<tag>.bin and
    index_<tag>.npy). caches: (folder, global index of the folder's first step) pairs.
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
                n_steps = meta["hi"] - meta["lo"]
                if "tower" in meta:
                    index = load_numpy(cache_dir / f"index_{tag}.npy")
                    file = TokenFile(cache_dir / f"tokens_{tag}.bin", meta["dim"])
                    first_tokens, grids = index[:, 0], index[:, 1:]
                else:
                    tokens_path = cache_dir / f"global_{tag}.npy"
                    file = TokenFile(tokens_path, meta["dim"], npy_data_offset(tokens_path))
                    first_tokens = np.arange(n_steps) * meta["global_tokens"]
                    grids = np.full((n_steps, 2), meta["grid"])
                shards.append((first_step + meta["lo"], file, first_tokens, grids))
        shards.sort(key=lambda shard: shard[0])
        self.files = [file for _, file, _, _ in shards]
        self.first_tokens = np.concatenate([first_tokens for _, _, first_tokens, _ in shards]).astype(np.int64)
        self.grids = np.concatenate([grids for _, _, _, grids in shards]).astype(np.int64)
        self.shard_of_step = np.repeat(np.arange(len(shards)), [len(grids) for _, _, _, grids in shards])
        if [first for first, _, _, _ in shards] != [0, *np.cumsum([len(shard[3]) for shard in shards])[:-1]]:
            raise ValueError(f"the shards of {caches} do not cover every step once")
        self.token_counts = self.grids.prod(1)
        self.n_steps, self.dim, self.max_tokens = len(self.grids), meta["dim"], int(self.token_counts.max())

    def read_into(self, steps: np.ndarray, out: np.ndarray) -> int:
        """Fills out[:n] with the tokens of the given steps, in that order, and returns n.

        Consecutive steps of a shard are one span read.
        """
        steps = np.asarray(steps, np.int64)
        if not len(steps):
            return 0
        ends = self.token_counts[steps].cumsum()
        starts = ends - self.token_counts[steps]
        shards = self.shard_of_step[steps]
        run_breaks = np.nonzero((np.diff(steps) != 1) | (np.diff(shards) != 0))[0] + 1
        for first, end in zip(np.r_[0, run_breaks], np.r_[run_breaks, len(steps)]):
            target = out[starts[first] : ends[end - 1]]
            self.files[shards[first]].read_into(target, int(self.first_tokens[steps[first]]))
        return int(ends[-1])

    def read(self, steps: np.ndarray) -> np.ndarray:
        out = np.empty((int(self.token_counts[np.asarray(steps, np.int64)].sum()), self.dim), np.uint16)
        self.read_into(steps, out)
        return out


class TokenShardWriter:
    """Writes one shard of a tower's own tokens: append() the steps in order, then finish().

    checkpoint() makes what was appended so far durable; a writer opened on an interrupted shard continues after
    its last checkpoint (n_steps says where). The meta file marks the shard complete, so it is written last.
    """

    def __init__(self, cache_dir: Path, tag: str, dim: int):
        self.cache_dir, self.tag, self.dim = Path(cache_dir), tag, dim
        self.partial_index_path = self.cache_dir / f"index_{tag}.partial.npy"
        tokens_path = self.cache_dir / f"tokens_{tag}.bin"
        resumed = self.partial_index_path.exists() and tokens_path.exists()
        self.index = [tuple(row) for row in np.load(self.partial_index_path)] if resumed else []
        self.n_tokens = sum(rows * columns for _, rows, columns in self.index)
        self.file = open(tokens_path, "r+b" if resumed else "wb")
        self.file.truncate(self.n_tokens * 2 * dim)
        self.file.seek(self.n_tokens * 2 * dim)

    @property
    def n_steps(self) -> int:
        return len(self.index)

    def append(self, tokens: np.ndarray, grids: list[tuple[int, int]]):
        """tokens: [n, dim] uint16, the tokens of len(grids) steps one after the other."""
        for rows, columns in grids:
            self.index.append((self.n_tokens, rows, columns))
            self.n_tokens += rows * columns
        self.file.write(memoryview(np.ascontiguousarray(tokens)))

    def checkpoint(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        np.save(self.partial_index_path, np.array(self.index, np.int64).reshape(-1, 3))

    def finish(self, meta: dict):
        self.checkpoint()
        self.file.close()
        self.partial_index_path.rename(self.cache_dir / f"index_{self.tag}.npy")
        meta = {**meta, "dim": self.dim, "tokens": self.n_tokens}
        (self.cache_dir / f"meta_{self.tag}.json").write_text(json.dumps(meta))


def as_bfloat16(bits: np.ndarray) -> torch.Tensor:
    """uint16 bit patterns -> bfloat16 tensor sharing the memory."""
    return torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)


class PinnedBufferPool:
    """A fixed set of pinned host buffers for screen tokens, allocated once and reused for every batch."""

    def __init__(self, n_buffers: int, n_tokens: int, dim: int):
        self._free = queue.Queue()
        for _ in range(n_buffers):
            self._free.put(torch.empty((n_tokens, dim), dtype=torch.int16).pin_memory())

    def take(self) -> torch.Tensor:
        return self._free.get()

    def give_back(self, buffer: torch.Tensor):
        self._free.put(buffer)


_pools: dict = {}


def pinned_buffer_pool(n_buffers: int, n_tokens: int, dim: int) -> PinnedBufferPool:
    """One pool per shape for the whole process: pinned memory is never returned to the system."""
    key = (n_buffers, n_tokens, dim)
    if key not in _pools:
        _pools[key] = PinnedBufferPool(*key)
    return _pools[key]
