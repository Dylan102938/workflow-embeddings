"""Data: what a batch holds, and how train / val / test batches are made.

Sessions. A session is one recorded computer-use workflow from AgentNet: its task instruction and T steps, each the
screenshot the user saw plus the input events they performed on it (the last step is the final screen, without
events). prep/prepare.py fixes the splits once (prepared/sessions.pkl), grouped so nothing leaks between them:
  test   every session of a held-out set of applications, plus any session sharing an instruction group with one
  val    the sessions of a random fraction of the remaining instruction groups
  train  the rest
Sessions with the same normalized instruction form a group; groups never cross splits.

Inputs per step (frozen features computed once, see wfemb/prep):
  frames  [patch, 512]  SigLIP2 so400m dense patch features of the screenshot (channel PCA 1152 -> 512), exact bf16,
                        read per batch from the frame cache (storage/frames.py)
  events  E = 8 slots: event type, up to two points (x, y, x2, y2), scroll amount, up to 4 key tokens, typed text
                        (an index into text_vecs, the frozen Qwen3 embeddings of every typed string)

Labels (frozen): keys.npy holds the OpenAI text-embedding-3-large vector of every session's instruction, para_keys.npy
those of its paraphrases (para_index.npz: offset / count per session). In training the candidates of every step are
the keys of all N training instructions (TrainLabels); each step a random wording is drawn for every column.

Training batches (train_batches): each epoch the training sessions are shuffled, sorted by length within buckets of
50 batches (little padding), cut into batches of `bs` sessions (a batch of long sessions is split so it holds at most
`max_frames` steps), and each intermediate step is dropped with probability `step_drop` (the final screen is kept).
Evaluation batches (eval_batches): all sessions of a split, in a fixed order, nothing dropped.
"""
from __future__ import annotations

import json
import pickle
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .config import DataConfig, Experiment, TrainConfig
from .storage.frames import FrameStore, bf16, np_load, pinned_pool, read_chunked


@dataclass
class Batch:
    """B sessions padded to T steps. Padded positions repeat a session's last step and are masked by `valid`."""
    idx: torch.Tensor          # [B]   session indices (global session order)
    frames: torch.Tensor       # [B, T, patch, 512] bf16
    ev_type: torch.Tensor      # [B, T, E] long (0 = empty slot)
    ev_xy: torch.Tensor        # [B, T, E, 4] float (nan = absent)
    ev_scroll: torch.Tensor    # [B, T, E] float
    ev_keys: torch.Tensor      # [B, T, E, K] long (0 = none)
    ev_text: torch.Tensor      # [B, T, E] long (-1 = none)
    valid: torch.Tensor        # [B, T] bool
    lengths: torch.Tensor      # [B]   number of (kept) steps

    def to(self, device, non_blocking=True) -> "Batch":
        return Batch(**{k: v.to(device, non_blocking=non_blocking) for k, v in self.__dict__.items()})


class SessionData:
    """All sessions: metadata, event arrays, frozen keys (in RAM) and the frame cache (read per batch). A session s
    owns rows offset[s] .. offset[s] + length[s] - 1 of the event arrays and of the frame cache."""

    def __init__(self, cfg: DataConfig, exp: Experiment):
        root = Path(cfg.root)
        self.sessions = pickle.load(read_chunked(root / "prepared" / "sessions.pkl"))
        ev = np_load(root / "prepared" / "events.npz")
        self.ev = {k: ev[k] for k in ev.files}
        self.offset, self.length = self.ev["sess_offset"], self.ev["sess_len"]
        self.n_key_tokens = len(json.load(open(root / "prepared" / "vocab.json"))["key_vocab"])
        kdir = root / cfg.keys
        self.keys = np_load(kdir / "keys.npy").astype(np.float32)          # [S, D] one per session
        self.text_vecs = np_load(kdir / "text_vecs.npy")                    # [n_texts, d_text]
        if (kdir / "para_keys.npy").exists():
            self.para_keys = np_load(kdir / "para_keys.npy").astype(np.float32)
            pi = np_load(kdir / "para_index.npz")
            self.para_offset, self.para_count = pi["offset"], pi["count"]
        else:                                                                 # no paraphrases
            self.para_keys = np.zeros((1, self.keys.shape[1]), np.float32)
            self.para_offset = np.zeros(len(self.sessions), np.int64)
            self.para_count = np.zeros(len(self.sessions), np.int64)
        self.split = np.array([s["split"] for s in self.sessions])
        gid = {g: i for i, g in enumerate(sorted({s["group"] for s in self.sessions}))}
        self.group = np.array([gid[s["group"]] for s in self.sessions])
        self.judged_dir = root / "keys"                                      # judged_<split>.jsonl
        self.frames = FrameStore(cfg.frames_dir(exp))
        assert self.frames.n_tok == exp.patch, f"{cfg.frames_dir(exp)} has {self.frames.n_tok} patches, not {exp.patch}"
        assert self.frames.total == int(self.offset[-1] + self.length[-1]), "frame cache / events mismatch"

    def indices(self, split: str) -> np.ndarray:
        return np.nonzero(self.split == split)[0]

    def make_batch(self, sidx: np.ndarray, step_drop: float = 0.0, seed: int | None = None,
                   frames_out: torch.Tensor | None = None) -> Batch:
        """Sessions sidx -> Batch on the CPU. step_drop > 0 (training): each intermediate step is dropped with that
        probability (order kept, final step always kept). frames_out: a reusable [rows, patch, 512] int16 buffer."""
        if step_drop > 0:
            rng = np.random.default_rng(seed)
            rows = []
            for s in sidx:
                keep = rng.random(int(self.length[s])) >= step_drop
                keep[-1] = True
                rows.append(self.offset[s] + np.nonzero(keep)[0])
            lens = np.array([len(r) for r in rows], np.int64)
            T = int(lens.max())
            gidx = np.stack([np.concatenate([r, np.full(T - len(r), r[-1])]) for r in rows])
        else:
            lens = self.length[sidx].astype(np.int64)
            T = int(lens.max())
            gidx = self.offset[sidx][:, None] + np.minimum(np.arange(T)[None, :], lens[:, None] - 1)
        B, flat, n_tok, dim = len(sidx), gidx.reshape(-1), self.frames.n_tok, self.frames.dim
        if frames_out is not None and len(flat) <= len(frames_out):
            buf = frames_out[:len(flat)]
            self.frames.gather_into(flat, buf.numpy().view(np.uint16))
            frames = buf.view(torch.bfloat16).view(B, T, n_tok, dim)
        else:
            frames = bf16(self.frames.gather(flat)).view(B, T, n_tok, dim)
        e = self.ev
        return Batch(
            idx=torch.from_numpy(np.asarray(sidx, np.int64)),
            frames=frames,
            ev_type=torch.from_numpy(e["ev_type"][flat].reshape(B, T, -1).astype(np.int64)),
            ev_xy=torch.from_numpy(e["ev_xy"][flat].reshape(B, T, -1, 4).astype(np.float32)),
            ev_scroll=torch.from_numpy(e["ev_scroll"][flat].reshape(B, T, -1).astype(np.float32)),
            ev_keys=torch.from_numpy(e["ev_keys"][flat].reshape(B, T, *e["ev_keys"].shape[1:]).astype(np.int64)),
            ev_text=torch.from_numpy(e["ev_text"][flat].reshape(B, T, -1).astype(np.int64)),
            valid=torch.from_numpy(np.arange(T)[None, :] < lens[:, None]),
            lengths=torch.from_numpy(lens),
        )


class TrainLabels:
    """The candidates of every training step: the keys of all N training sessions' instructions, on the GPU."""

    def __init__(self, data: SessionData, device):
        self.idx = data.indices("train")                                     # column j <-> session idx[j]
        self.U = torch.from_numpy(data.keys[self.idx]).to(device)            # [N, D] original wordings
        self.group = torch.from_numpy(data.group[self.idx]).to(device)       # [N]
        self._col = np.full(len(data.sessions), -1, np.int64)
        self._col[self.idx] = np.arange(len(self.idx))
        self._para = torch.from_numpy(data.para_keys).to(device)
        self._para_off = torch.from_numpy(data.para_offset[self.idx].astype(np.int64)).to(device)
        self._para_cnt = torch.from_numpy(data.para_count[self.idx].astype(np.int64)).to(device)
        self.device = device

    def __len__(self):
        return len(self.idx)

    def target(self, sidx: torch.Tensor) -> torch.Tensor:
        """[B] column of each (training) session's own instruction."""
        return torch.from_numpy(self._col[sidx.cpu().numpy()]).to(self.device)

    def sample_wordings(self) -> torch.Tensor:
        """[N, D]: every column a uniformly drawn wording of its instruction (the original or one of its paraphrases),
        redrawn every step, so wording style cannot tell the positive from the negatives."""
        pick = (torch.rand(len(self), device=self.device) * (self._para_cnt + 1).float()).long()   # 0 = original
        alt = self._para[(self._para_off + pick - 1).clamp(min=0)]
        return torch.where((pick > 0)[:, None], alt, self.U)


def length_bucketed_batches(lengths: np.ndarray, idx: np.ndarray, batch: int, rng: np.random.Generator,
                            bucket_mult: int = 50, drop_last: bool = True,
                            max_frames: int | None = None) -> list[np.ndarray]:
    """Shuffle, sort by length within buckets of bucket_mult batches, cut into batches (a batch over max_frames steps
    is split into smaller ones instead of truncating sessions), shuffle the batch order."""
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
    """Builds batches on background threads (in order) and moves them to the GPU. Frame tokens are read straight into
    a fixed pool of pinned buffers and copied to the GPU synchronously, after which the buffer is reused."""

    def __init__(self, data: SessionData, batches: list[np.ndarray], device, step_drop: float = 0.0, seed: int = 0,
                 depth: int = 4, threads: int = 4, max_rows: int = 4096):
        self.data, self.batches, self.device = data, batches, device
        self.step_drop, self.seed, self.depth = step_drop, seed, depth
        self.pool = ThreadPoolExecutor(threads)
        self.pins = pinned_pool(depth + 1, max_rows, data.frames.n_tok, data.frames.dim) if device.type == "cuda" \
            else None

    def _make(self, i):
        buf = self.pins.get() if self.pins is not None else None
        b = self.data.make_batch(self.batches[i], self.step_drop, self.seed * 1_000_003 + i, frames_out=buf)
        in_buf = buf is not None and b.frames.data_ptr() == buf.data_ptr()
        if buf is not None and not in_buf:                    # batch larger than the buffer
            self.pins.put(buf)
            buf = None
        if self.pins is not None:
            b = Batch(**{k: (v if (k == "frames" and in_buf) else v.pin_memory()) for k, v in b.__dict__.items()})
        return b, buf

    def _to_device(self, b: Batch, buf) -> Batch:
        out = {}
        for k, v in b.__dict__.items():
            if k == "frames" and buf is not None:
                out[k] = v.to(self.device, non_blocking=False)   # the buffer is reused right after this copy
                self.pins.put(buf)
            else:
                out[k] = v.to(self.device, non_blocking=True)
        return Batch(**out)

    def __iter__(self) -> Iterator[Batch]:
        n, nxt, futs = len(self.batches), 0, deque()
        try:
            while nxt < min(self.depth, n):
                futs.append(self.pool.submit(self._make, nxt))
                nxt += 1
            for _ in range(n):
                b, buf = futs.popleft().result()
                if nxt < n:
                    futs.append(self.pool.submit(self._make, nxt))
                    nxt += 1
                yield self._to_device(b, buf)
        finally:            # abandoned early (e.g. training ends mid-epoch): return the in-flight batches' buffers
            for f in futs:
                try:
                    _, buf = f.result()
                except Exception:
                    continue
                if buf is not None:
                    self.pins.put(buf)
            self.pool.shutdown(wait=False)


def train_batches(data: SessionData, exp: Experiment, train: TrainConfig, data_cfg: DataConfig, epoch: int,
                  device) -> Iterator[Batch]:
    """One epoch of training batches (see the module docstring), deterministic given (seed, epoch)."""
    rng = np.random.default_rng([exp.seed, epoch])
    batches = length_bucketed_batches(data.length, data.indices("train"), exp.bs, rng, max_frames=train.max_frames)
    return iter(Prefetcher(data, batches, device, step_drop=train.step_drop, seed=exp.seed * 1000 + epoch,
                           depth=data_cfg.prefetch_depth, threads=data_cfg.prefetch_threads,
                           max_rows=train.max_frames))


def eval_batches(data: SessionData, sidx: np.ndarray, batch: int, max_frames: int, device,
                 data_cfg: DataConfig | None = None) -> Iterator[Batch]:
    """Every session of sidx exactly once, nothing dropped, in a fixed order (by length within buckets)."""
    batches = length_bucketed_batches(data.length, np.asarray(sidx), batch, np.random.default_rng(0),
                                      drop_last=False, max_frames=max_frames)
    # its own pool of 2 pinned buffers (validation runs while training's prefetcher holds its buffers)
    d = data_cfg or DataConfig()
    return iter(Prefetcher(data, batches, device, depth=1, threads=d.prefetch_threads, max_rows=max_frames))
