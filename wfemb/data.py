"""Sessions, splits and batches. What the data is and how batches are formed: docs/ARCHITECTURE.md."""

from __future__ import annotations

import json
import pickle
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .config import DataConfig, Experiment, TrainConfig
from .storage.frames import FrameStore, as_bfloat16, load_numpy, pinned_buffer_pool, read_file

# Batch field -> (array in prepared/events.npz, dtype in a batch)
EVENT_ARRAYS = {
    "event_types": ("ev_type", np.int64),
    "event_points": ("ev_xy", np.float32),
    "event_scrolls": ("ev_scroll", np.float32),
    "event_keys": ("ev_keys", np.int64),
    "event_texts": ("ev_text", np.int64),
}


@dataclass
class Batch:
    """B sessions padded to T steps; E event slots per step. Padding repeats a session's last step."""

    session_indices: torch.Tensor  # [B]
    frames: torch.Tensor  # [B, T, n_patches, 512] bfloat16
    event_types: torch.Tensor  # [B, T, E], 0 = empty slot
    event_points: torch.Tensor  # [B, T, E, 4] (x, y, x2, y2), nan = absent
    event_scrolls: torch.Tensor  # [B, T, E]
    event_keys: torch.Tensor  # [B, T, E, K], 0 = none
    event_texts: torch.Tensor  # [B, T, E] index into the typed-text embeddings, -1 = none
    valid: torch.Tensor  # [B, T] bool
    lengths: torch.Tensor  # [B]

    def to(self, device, non_blocking: bool = True) -> Batch:
        return Batch(
            **{field.name: getattr(self, field.name).to(device, non_blocking=non_blocking) for field in fields(self)}
        )


class WorkflowDataset:
    """Every session's metadata, events and frozen label keys in memory; screen features read per batch.

    Session s owns steps step_offsets[s] .. step_offsets[s] + lengths[s] - 1 of the event arrays and frame cache.
    """

    def __init__(self, config: DataConfig, experiment: Experiment):
        root = Path(config.root)
        self.sessions = pickle.load(read_file(root / "prepared" / "sessions.pkl"))
        events = load_numpy(root / "prepared" / "events.npz")
        self.events = {name: events[name] for name in events.files}
        self.step_offsets, self.lengths = self.events["sess_offset"], self.events["sess_len"]
        self.max_events = int(self.events["ev_type"].shape[1])
        self.n_key_tokens = len(json.load(open(root / "prepared" / "vocab.json"))["key_vocab"])

        labels_dir = root / config.label_set
        self.label_keys = load_numpy(labels_dir / "keys.npy").astype(np.float32)
        self.text_embeddings = load_numpy(labels_dir / "text_vecs.npy")
        if (labels_dir / "para_keys.npy").exists():
            self.paraphrase_keys = load_numpy(labels_dir / "para_keys.npy").astype(np.float32)
            index = load_numpy(labels_dir / "para_index.npz")
            self.paraphrase_offsets, self.paraphrase_counts = index["offset"], index["count"]
        else:
            self.paraphrase_keys = np.zeros((1, self.label_keys.shape[1]), np.float32)
            self.paraphrase_offsets = np.zeros(len(self.sessions), np.int64)
            self.paraphrase_counts = np.zeros(len(self.sessions), np.int64)

        self.splits = np.array([session["split"] for session in self.sessions])
        group_ids = {name: index for index, name in enumerate(sorted({s["group"] for s in self.sessions}))}
        self.groups = np.array([group_ids[session["group"]] for session in self.sessions])
        self.judged_dir = root / "keys"

        cache_dir = config.frame_cache_dir(experiment)
        self.frames = FrameStore(cache_dir)
        if self.frames.n_patches != experiment.n_patches:
            raise ValueError(f"{cache_dir} has {self.frames.n_patches} patches per frame, not {experiment.n_patches}")
        if self.frames.n_steps != int(self.step_offsets[-1] + self.lengths[-1]):
            raise ValueError(f"{cache_dir} does not match prepared/events.npz")

    def sessions_in(self, split: str) -> np.ndarray:
        return np.nonzero(self.splits == split)[0]

    def padded_steps(
        self, session_indices: np.ndarray, step_dropout: float = 0.0, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Global step indices [B, T] and lengths [B].

        With step_dropout, each step but a session's last is dropped with that probability.
        """
        if step_dropout > 0:
            rng = np.random.default_rng(seed)
            kept_steps = []
            for session in session_indices:
                keep = rng.random(int(self.lengths[session])) >= step_dropout
                keep[-1] = True
                kept_steps.append(self.step_offsets[session] + np.nonzero(keep)[0])
            lengths = np.array([len(steps) for steps in kept_steps], np.int64)
            n_steps = int(lengths.max())
            steps = np.stack([np.concatenate([kept, np.full(n_steps - len(kept), kept[-1])]) for kept in kept_steps])
        else:
            lengths = self.lengths[session_indices].astype(np.int64)
            position = np.minimum(np.arange(int(lengths.max()))[None, :], lengths[:, None] - 1)
            steps = self.step_offsets[session_indices][:, None] + position
        return steps, lengths

    def make_batch(
        self,
        session_indices: np.ndarray,
        step_dropout: float = 0.0,
        seed: int | None = None,
        frame_buffer: torch.Tensor | None = None,
    ) -> Batch:
        """A batch on the CPU. frame_buffer: a reusable int16 buffer the frames are read into if it is large enough."""
        steps, lengths = self.padded_steps(session_indices, step_dropout, seed)
        batch_shape = steps.shape
        flat_steps = steps.reshape(-1)

        frame_shape = (*batch_shape, self.frames.n_patches, self.frames.dim)
        if frame_buffer is not None and len(flat_steps) <= len(frame_buffer):
            used = frame_buffer[: len(flat_steps)]
            self.frames.read_into(flat_steps, used.numpy().view(np.uint16))
            frames = used.view(torch.bfloat16).view(frame_shape)
        else:
            frames = as_bfloat16(self.frames.read(flat_steps)).view(frame_shape)

        event_tensors = {}
        for field, (array_name, dtype) in EVENT_ARRAYS.items():
            array = self.events[array_name]
            event_tensors[field] = torch.from_numpy(
                array[flat_steps].reshape(*batch_shape, *array.shape[1:]).astype(dtype)
            )

        return Batch(
            session_indices=torch.from_numpy(np.asarray(session_indices, np.int64)),
            frames=frames,
            valid=torch.from_numpy(np.arange(batch_shape[1])[None, :] < lengths[:, None]),
            lengths=torch.from_numpy(lengths),
            **event_tensors,
        )


class TrainingCandidates:
    """The candidates of every training step: the instruction keys of all N training sessions, on the device."""

    def __init__(self, dataset: WorkflowDataset, device):
        self.device = device
        self.session_indices = dataset.sessions_in("train")
        self.keys = torch.from_numpy(dataset.label_keys[self.session_indices]).to(device)
        self.groups = torch.from_numpy(dataset.groups[self.session_indices]).to(device)
        self._column_of_session = np.full(len(dataset.sessions), -1, np.int64)
        self._column_of_session[self.session_indices] = np.arange(len(self.session_indices))
        self._paraphrase_keys = torch.from_numpy(dataset.paraphrase_keys).to(device)
        self._paraphrase_offsets = torch.from_numpy(
            dataset.paraphrase_offsets[self.session_indices].astype(np.int64)
        ).to(device)
        self._paraphrase_counts = torch.from_numpy(dataset.paraphrase_counts[self.session_indices].astype(np.int64)).to(
            device
        )

    def __len__(self) -> int:
        return len(self.session_indices)

    def targets(self, session_indices: torch.Tensor) -> torch.Tensor:
        """[B] the candidate column of each training session's own instruction."""
        return torch.from_numpy(self._column_of_session[session_indices.cpu().numpy()]).to(self.device)

    def sample_wordings(self) -> torch.Tensor:
        """[N, D]: for every candidate, its instruction's key or one of its paraphrases' keys, uniformly."""
        n_wordings = self._paraphrase_counts + 1
        choice = (torch.rand(len(self), device=self.device) * n_wordings.float()).long()
        paraphrase = self._paraphrase_keys[(self._paraphrase_offsets + choice - 1).clamp(min=0)]
        return torch.where((choice > 0)[:, None], paraphrase, self.keys)


def length_bucketed_batches(
    lengths: np.ndarray,
    session_indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    max_frames: int | None = None,
    drop_last: bool = True,
    bucket_batches: int = 50,
) -> list[np.ndarray]:
    """Shuffled batches of sessions with similar lengths.

    A batch whose padded size exceeds max_frames is split into smaller batches (sessions are never truncated).
    """
    shuffled = rng.permutation(session_indices)
    bucket_size = batch_size * bucket_batches
    batches = []
    for bucket_start in range(0, len(shuffled), bucket_size):
        bucket = shuffled[bucket_start : bucket_start + bucket_size]
        bucket = bucket[np.argsort(lengths[bucket], kind="stable")]
        for start in range(0, len(bucket), batch_size):
            batch = bucket[start : start + batch_size]
            if len(batch) < batch_size and drop_last:
                continue
            longest = int(lengths[batch].max())
            if max_frames and len(batch) * longest > max_frames:
                piece_size = max(1, max_frames // longest)
                batches += [batch[i : i + piece_size] for i in range(0, len(batch), piece_size)]
            else:
                batches.append(batch)
    return [batches[i] for i in rng.permutation(len(batches))]


class Prefetcher:
    """Builds batches on background threads, in order, and moves them to the device.

    Frames are read into pinned buffers from a fixed pool and copied to the GPU synchronously, so each buffer
    can be reused right away.
    """

    def __init__(
        self,
        dataset: WorkflowDataset,
        batches: list[np.ndarray],
        device,
        step_dropout: float = 0.0,
        seed: int = 0,
        depth: int = 4,
        threads: int = 4,
        buffer_rows: int = 4096,
    ):
        self.dataset, self.batches, self.device = dataset, batches, device
        self.step_dropout, self.seed, self.depth = step_dropout, seed, depth
        self.executor = ThreadPoolExecutor(threads)
        self.buffers = None
        if device.type == "cuda":
            self.buffers = pinned_buffer_pool(depth + 1, buffer_rows, dataset.frames.n_patches, dataset.frames.dim)

    def _build(self, index: int) -> tuple[Batch, torch.Tensor | None]:
        buffer = self.buffers.take() if self.buffers is not None else None
        batch = self.dataset.make_batch(
            self.batches[index], self.step_dropout, self.seed * 1_000_003 + index, frame_buffer=buffer
        )
        if buffer is not None and batch.frames.data_ptr() != buffer.data_ptr():  # the batch did not fit
            self.buffers.give_back(buffer)
            buffer = None
        if self.buffers is not None:
            for field in fields(batch):
                if not (field.name == "frames" and buffer is not None):
                    setattr(batch, field.name, getattr(batch, field.name).pin_memory())
        return batch, buffer

    def _to_device(self, batch: Batch, buffer: torch.Tensor | None) -> Batch:
        on_device = {}
        for field in fields(batch):
            # A pool buffer is reused as soon as it is given back, so its copy must have finished.
            in_pool_buffer = field.name == "frames" and buffer is not None
            on_device[field.name] = getattr(batch, field.name).to(self.device, non_blocking=not in_pool_buffer)
        if buffer is not None:
            self.buffers.give_back(buffer)
        return Batch(**on_device)

    def __iter__(self) -> Iterator[Batch]:
        pending, next_index = deque(), 0

        def submit_next():
            nonlocal next_index
            if next_index < len(self.batches):
                pending.append(self.executor.submit(self._build, next_index))
                next_index += 1

        try:
            for _ in range(self.depth):
                submit_next()
            while pending:
                batch, buffer = pending.popleft().result()
                submit_next()
                yield self._to_device(batch, buffer)
        finally:
            # Stopped early: batches still in flight hold pool buffers.
            for future in pending:
                try:
                    _, buffer = future.result()
                except Exception:
                    continue
                if buffer is not None:
                    self.buffers.give_back(buffer)
            self.executor.shutdown(wait=False)


def train_batches(
    dataset: WorkflowDataset, experiment: Experiment, train: TrainConfig, data: DataConfig, epoch: int, device
) -> Iterator[Batch]:
    """One epoch of training batches, deterministic given (seed, epoch)."""
    rng = np.random.default_rng([experiment.seed, epoch])
    batches = length_bucketed_batches(
        dataset.lengths, dataset.sessions_in("train"), experiment.batch_size, rng, max_frames=train.max_frames_per_batch
    )
    return iter(
        Prefetcher(
            dataset,
            batches,
            device,
            step_dropout=train.step_dropout,
            seed=experiment.seed * 1000 + epoch,
            depth=data.prefetch_depth,
            threads=data.prefetch_threads,
            buffer_rows=train.max_frames_per_batch,
        )
    )


def eval_batches(
    dataset: WorkflowDataset, session_indices: np.ndarray, batch_size: int, max_frames: int, device, threads: int = 4
) -> Iterator[Batch]:
    """Every given session exactly once, in a fixed order."""
    batches = length_bucketed_batches(
        dataset.lengths,
        np.asarray(session_indices),
        batch_size,
        np.random.default_rng(0),
        max_frames=max_frames,
        drop_last=False,
    )
    # depth 1: its own small pool, since validation runs while the training prefetcher holds its buffers
    return iter(Prefetcher(dataset, batches, device, depth=1, threads=threads, buffer_rows=max_frames))
