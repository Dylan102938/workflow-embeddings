"""Training-time data: frame cache + flat event arrays + frozen keys.

Batches are padded B x T (length-bucketed). Events are gathered on CPU and moved to GPU by a background
prefetch thread. Frame tokens come from
  frames_mode="ram"      the bf16 cache read fully into host RAM (exact; ~107 GB for the dense 16x16x512 cache),
                         gathered per batch by the prefetch threads and copied to the GPU (default)
  frames_mode="memmap"   the same bf16 cache memory-mapped (exact, little RAM; pathologically slow on NFS when several
                         processes map it)
  frames_mode="pread"    the same bf16 cache read row by row with preadv through the page cache (exact, little RAM)
  frames_mode="gpu_fp8"  a copy held in GPU memory as fp8-e4m3 (lossy; ~54 GB for the dense cache).
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
import torch.nn.functional as F

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
    gidx: torch.Tensor         # [B, T] global step index of each (padded) position

    def to(self, device):
        return Batch(**{k: v.to(device, non_blocking=True) for k, v in self.__dict__.items()})


class _PreadArray:
    """Row reads from a .npy file with os.preadv (no mmap). Memory-mapping the NFS-hosted cache from several processes
    makes page faults pathologically slow (1-50 s per batch even with the file in page cache); plain reads go through
    the page cache without faulting pages into each process. Thread-safe (explicit offsets)."""

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

    def _pread_full(self, buf, offset):
        mv, done = memoryview(buf).cast("B"), 0
        while done < len(mv):                              # preadv may return a partial read
            n = os.preadv(self.fd, [mv[done:]], offset + done)
            if n <= 0:
                raise IOError(f"short read at {offset + done}")
            done += n

    def read_rows(self, rows: np.ndarray, gap: int = 8) -> np.ndarray:
        """rows (sorted ascending) -> [len(rows), ...]. Nearby rows are read as one contiguous span (a session's steps
        are contiguous in the cache), so a batch costs ~one syscall per session instead of one per frame: per-call
        GIL hand-offs against the training thread made per-frame reads ~50x slower inside a training process."""
        out = np.empty((len(rows),) + tuple(self.shape[1:]), self.dtype)
        if len(rows) == 0:
            return out
        breaks = np.nonzero(np.diff(rows) > gap)[0] + 1
        for s, e in zip(np.r_[0, breaks], np.r_[breaks, len(rows)]):
            lo, hi = int(rows[s]), int(rows[e - 1]) + 1
            if hi - lo == e - s:                           # fully contiguous: read straight into the output
                self._pread_full(out[s:e], self.offset + lo * self.row_bytes)
            else:
                tmp = np.empty((hi - lo,) + tuple(self.shape[1:]), self.dtype)
                self._pread_full(tmp, self.offset + lo * self.row_bytes)
                out[s:e] = tmp[rows[s:e] - lo]
        return out


def _read_into_ram(mm: np.memmap, chunk_bytes: int = 1 << 28) -> np.ndarray:
    """Copy a read-only memmap into an ordinary in-RAM array with large sequential reads."""
    out = np.empty(mm.shape, mm.dtype)
    flat_in, flat_out = mm.reshape(len(mm), -1), out.reshape(len(out), -1)
    rows = max(1, chunk_bytes // max(1, flat_in.shape[1] * mm.dtype.itemsize))
    for a in range(0, len(mm), rows):
        flat_out[a:a + rows] = flat_in[a:a + rows]
    return out


class FrameStore:
    """Concatenation of per-shard memmaps, addressed by global step index. Each frame's tokens are
    [global grid | action crop (optional) | effect crop of the previous action (optional)]."""
    on_gpu = False

    def __init__(self, frames_dir: Path, use_crop: bool, extras: list[str] | tuple = (), avg_grid: int = 0,
                 in_ram: bool = False, pread: bool = False):
        metas = sorted(glob.glob(str(frames_dir / "meta_*.json")))
        if not metas:
            raise FileNotFoundError(f"no frame cache in {frames_dir}")
        self.shards = []                      # (lo, hi, [token arrays in order])
        n_eff = 0
        for mp in metas:
            m = json.load(open(mp))
            tag = Path(mp).stem.split("_", 1)[1]
            parts = [np.load(frames_dir / f"global_{tag}.npy", mmap_mode="r")]
            if use_crop:
                parts.append(np.load(frames_dir / f"crop_{tag}.npy", mmap_mode="r"))
            n_eff = 0
            for name in extras:                       # extra crop caches on the bulk share (BULK/frames_<name>)
                d = paths.BULK / f"frames_{name}"
                me = json.load(open(d / f"meta_{tag}.json"))
                assert (me["lo"], me["hi"]) == (m["lo"], m["hi"]), f"{name} shard {tag} does not match"
                parts.append(np.load(d / f"{name}_{tag}.npy", mmap_mode="r"))
                n_eff += me["crop_tokens"]
            self.shards.append((m["lo"], m["hi"], parts))
        self.shards.sort(key=lambda s: s[0])
        self.use_crop = use_crop
        m0 = json.load(open(metas[0]))
        self.dim = m0["dim"]
        self.grid = m0.get("grid", int(round(m0["global_tokens"] ** 0.5)))
        self.avg_grid = avg_grid if avg_grid and avg_grid < self.grid else 0
        n_global = self.avg_grid ** 2 if self.avg_grid else m0["global_tokens"]
        self.n_tok = n_global + (m0["crop_tokens"] if use_crop else 0) + n_eff
        self.lo = np.array([s[0] for s in self.shards])
        self.hi = np.array([s[1] for s in self.shards])
        self.total = int(self.hi.max())
        if in_ram:                                # read every token array into host RAM, shards in parallel
            arrs = [a for _, _, parts in self.shards for a in parts]
            with ThreadPoolExecutor(min(16, len(arrs))) as ex:
                loaded = list(ex.map(_read_into_ram, arrs))
            it = iter(loaded)
            self.shards = [(lo, hi, [next(it) for _ in parts]) for lo, hi, parts in self.shards]
        self.pread = pread and not in_ram
        if self.pread:                            # same files, read with preadv instead of page faults
            self.shards = [(lo, hi, [_PreadArray(a.filename) for a in parts]) for lo, hi, parts in self.shards]

    def gather(self, gidx: np.ndarray) -> np.ndarray:
        """gidx: 1-D global step indices -> [len, n_tok, D] uint16 (bf16 bits)."""
        out = np.empty((len(gidx), self.n_tok, self.dim), np.uint16)
        sh = np.searchsorted(self.hi, gidx, side="right")
        for k in np.unique(sh):
            lo, hi, parts = self.shards[k]
            sel = np.nonzero(sh == k)[0]
            rows = gidx[sel] - lo
            order = np.argsort(rows)          # sorted reads are friendlier to the page cache
            rs = rows[order]
            if self.pread:
                urs, inv = np.unique(rs, return_inverse=True)  # padded batches repeat a session's last step
                out[sel[order]] = np.concatenate([p.read_rows(urs)[inv] for p in parts], axis=1)
            else:
                out[sel[order]] = np.concatenate([p[rs] for p in parts], axis=1)
        return out


class GPUFrameStore(FrameStore):
    """All frame tokens resident on the GPU as fp8-e4m3 (stored as uint8), dequantized to bf16 per batch."""
    on_gpu = True
    FP8_MAX = 448.0

    def __init__(self, frames_dir: Path, use_crop: bool, device, extras: list[str] | tuple = (),
                 chunk: int = 8192, avg_grid: int = 0):
        super().__init__(frames_dir, use_crop, extras, avg_grid)
        self.device = device
        self.t = torch.empty((self.total, self.n_tok, self.dim), dtype=torch.uint8, device=device)
        for lo, hi, parts in self.shards:
            for a in range(0, hi - lo, chunk):
                b = min(hi - lo, a + chunk)
                c0 = 0
                for j, arr in enumerate(parts):
                    x = torch.from_numpy(np.ascontiguousarray(arr[a:b]).view(np.int16)).to(device).view(torch.bfloat16)
                    if j == 0 and self.avg_grid:                 # fixed averaging of the dense global grid
                        g = self.grid
                        x = F.adaptive_avg_pool2d(x.float().view(-1, g, g, x.shape[-1]).permute(0, 3, 1, 2),
                                                  self.avg_grid).permute(0, 2, 3, 1).reshape(x.shape[0], -1,
                                                                                             x.shape[-1])
                    q = x.clamp(-self.FP8_MAX, self.FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
                    self.t[lo + a:lo + b, c0:c0 + q.shape[1]] = q
                    c0 += q.shape[1]
        torch.cuda.synchronize(device)

    def gather_torch(self, gidx: torch.Tensor) -> torch.Tensor:
        return self.t[gidx].view(torch.float8_e4m3fn).to(torch.bfloat16)

    def gather(self, gidx: np.ndarray) -> np.ndarray:
        x = self.gather_torch(torch.from_numpy(np.asarray(gidx)).to(self.device))
        return x.view(torch.int16).cpu().numpy().view(np.uint16)


class SessionData:
    def __init__(self, frames_dir: Path | None = None, use_crop: bool = False, frames_mode: str = "ram",
                 device=None, extras: list[str] | tuple = (), avg_grid: int = 0):
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
        if frames_mode == "gpu_fp8":
            self.frames = GPUFrameStore(frames_dir or paths.FRAMES, use_crop, device, extras, avg_grid=avg_grid)
        else:
            assert not avg_grid, "avg_grid needs frames_mode=gpu_fp8"
            self.frames = FrameStore(frames_dir or paths.FRAMES, use_crop, extras, in_ram=frames_mode == "ram",
                                     pread=frames_mode == "pread")
        assert self.frames.total == int(self.offset[-1] + self.length[-1]), "frame cache / events mismatch"

    def indices(self, split: str) -> np.ndarray:
        return np.nonzero(self.split == split)[0]

    def make_batch(self, sidx: np.ndarray, max_len: int | None = None, step_drop: float = 0.0,
                   seed: int | None = None) -> Batch:
        """step_drop > 0 (training only): each intermediate step is dropped with that probability; the final
        step (real final screen) is always kept, and the order of the kept steps is preserved."""
        if step_drop > 0:
            rng = np.random.default_rng(seed)
            rows = []
            for s in sidx:
                L = int(self.length[s])
                keep = rng.random(L) >= step_drop
                keep[-1] = True
                kept = np.nonzero(keep)[0]
                if max_len:
                    kept = kept[:max_len]
                rows.append(self.offset[s] + kept)
            lens = np.array([len(r) for r in rows], np.int64)
            B, T = len(sidx), int(lens.max())
            gidx = np.stack([np.concatenate([r, np.full(T - len(r), r[-1])]) for r in rows])
        else:
            lens = self.length[sidx].astype(np.int64)
            if max_len:
                lens = np.minimum(lens, max_len)
            B, T = len(sidx), int(lens.max())
            gidx = (self.offset[sidx][:, None] + np.minimum(np.arange(T)[None, :], lens[:, None] - 1))
        valid = np.arange(T)[None, :] < lens[:, None]
        flat = gidx.reshape(-1)
        if self.frames.on_gpu:
            frames = torch.empty(0, dtype=torch.bfloat16)          # filled on the GPU by finalize()
        else:
            frames = self.frames.gather(flat).reshape(B, T, self.frames.n_tok, self.frames.dim)
            frames = torch.from_numpy(frames.view(np.int16)).view(torch.bfloat16)
        e = self.ev
        return Batch(
            idx=torch.from_numpy(sidx.astype(np.int64)),
            frames=frames,
            ev_type=torch.from_numpy(e["ev_type"][flat].reshape(B, T, -1).astype(np.int64)),
            ev_xy=torch.from_numpy(e["ev_xy"][flat].reshape(B, T, -1, 4).astype(np.float32)),
            ev_scroll=torch.from_numpy(e["ev_scroll"][flat].reshape(B, T, -1).astype(np.float32)),
            ev_keys=torch.from_numpy(e["ev_keys"][flat].reshape(B, T, *e["ev_keys"].shape[1:]).astype(np.int64)),
            ev_text=torch.from_numpy(e["ev_text"][flat].reshape(B, T, -1).astype(np.int64)),
            valid=torch.from_numpy(valid),
            lengths=torch.from_numpy(lens),
            gidx=torch.from_numpy(gidx.astype(np.int64)),
        )

    def finalize(self, b: Batch) -> Batch:
        """Call after b.to(device): materialize frame tokens for GPU-resident caches."""
        if self.frames.on_gpu:
            b.frames = self.frames.gather_torch(b.gidx)
        return b

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
                            bucket_mult: int = 50, drop_last: bool = True,
                            max_frames: int | None = None) -> list[np.ndarray]:
    """Shuffle, then sort within large buckets so each batch pads among similar lengths.
    max_frames caps B x T: batches of very long sessions are split instead of truncating sessions."""
    perm = rng.permutation(idx)
    out = []
    span = batch * bucket_mult
    for s in range(0, len(perm), span):
        chunk = perm[s:s + span]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        for b in range(0, len(chunk), batch):
            bb = chunk[b:b + batch]
            if len(bb) < batch and drop_last:
                continue
            T = int(lengths[bb].max())
            step = max(1, max_frames // T) if max_frames and len(bb) * T > max_frames else len(bb)
            for c in range(0, len(bb), step):
                out.append(bb[c:c + step])
    order = rng.permutation(len(out))
    return [out[i] for i in order]


class Prefetcher:
    """Builds batches on background threads (in order) and moves them to the GPU (pinned, non_blocking)."""

    def __init__(self, data: SessionData, batches: list[np.ndarray], device, extra=None, depth: int = 6,
                 threads: int = 3, max_len: int | None = None, step_drop: float = 0.0, seed: int = 0):
        self.data, self.batches, self.device, self.extra, self.max_len = data, batches, device, extra, max_len
        self.depth, self.step_drop, self.seed = depth, step_drop, seed
        self.pool = ThreadPoolExecutor(threads)

    def _make(self, i):
        b = self.data.make_batch(self.batches[i], self.max_len, self.step_drop, self.seed * 1_000_003 + i)
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
