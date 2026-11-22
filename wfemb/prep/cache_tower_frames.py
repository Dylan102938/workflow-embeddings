"""Frozen screen-feature cache of a vision tower's own tokens: one token per output cell of every screenshot.

  python -m wfemb.prep.cache_tower_frames --tower qwen3.8 --budget 576 --shard 0 --num_shards 8

Towers, file formats and what keeps the GPU busy: docs/DATA_PREP.md.
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time

import numpy as np
import torch

from ..config import NATIVE_TOWERS, tower_frame_cache_name
from ..storage import paths
from ..storage.frames import TokenShardWriter
from .cache_frames import PLACEHOLDER_IMAGE_SIZE, Frame, all_frames
from .prepare import load_sessions
from .splitzip import SplitZip

CHECKPOINT_BATCHES = 100


def main() -> None:
    args = parse_arguments()
    from .towers import TOWERS

    output_dir = paths.BULK / tower_frame_cache_name(args.tower, args.budget)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = all_frames(load_sessions())
    bounds = np.linspace(0, len(frames), args.num_shards + 1).astype(int)
    first_step, end_step = int(bounds[args.shard]), int(bounds[args.shard + 1])
    tag = f"{args.shard:02d}of{args.num_shards:02d}"
    if (output_dir / f"meta_{tag}.json").exists():
        print(f"shard {tag} is already complete", flush=True)
        return

    tower = TOWERS[args.tower](args.budget)
    writer = TokenShardWriter(output_dir, tag, tower.dim)
    todo = frames[first_step + writer.n_steps : end_step]
    print(f"{args.tower} shard {tag}: frames [{first_step}, {end_step}), {len(todo)} to do", flush=True)
    missing, seconds = encode_frames(tower, todo, writer, args)
    writer.finish(
        {
            "shard": args.shard,
            "num_shards": args.num_shards,
            "lo": first_step,
            "hi": end_step,
            "tower": args.tower,
            "repository": tower.repository,
            "budget": args.budget,
            "stride": tower.stride,
            "missing": missing,
        }
    )
    print(f"done: {len(todo)} frames in {seconds:.0f}s, {writer.n_tokens} tokens, {missing} missing", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", choices=NATIVE_TOWERS, required=True)
    parser.add_argument("--budget", type=int, default=576, help="most tokens per screenshot (the experiment's patch)")
    parser.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--batch", dest="batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


class Screenshots(torch.utils.data.Dataset):
    """Decoded screenshots, uint8 [3, H, W], read from the image archives in place.

    A screenshot that is missing from its archive or unreadable is None.
    """

    def __init__(self, frames: list[Frame]):
        self.frames = frames
        self.archives = None  # opened on first use, so every loader worker gets its own file handles

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> torch.Tensor | None:
        from torchvision.io import ImageReadMode, decode_image

        if self.archives is None:
            sources = {frame.source for frame in self.frames}
            self.archives = {source: SplitZip(paths.ZIPS[source]) for source in sources}
            self.member_paths = {source: archive.basename_index() for source, archive in self.archives.items()}
        frame = self.frames[i]
        try:
            data = self.archives[frame.source].read(self.member_paths[frame.source][frame.image])
            return decode_image(torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=ImageReadMode.RGB)
        except Exception:
            return None


def encode_frames(tower, frames: list[Frame], writer: TokenShardWriter, args: argparse.Namespace) -> tuple[int, float]:
    """Appends every frame's tokens to the writer, in order. Returns (missing screenshots, seconds).

    Loader workers only decode. Resizing, the tower and the copy off the GPU run here; other threads write.
    """
    loader = torch.utils.data.DataLoader(
        Screenshots(frames),
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=list,
        prefetch_factor=4 if args.workers else None,
        pin_memory=True,
    )
    written = BackgroundWriter(writer, buffer_tokens=args.batch_size * args.budget)
    placeholder = torch.zeros(3, *PLACEHOLDER_IMAGE_SIZE[::-1], dtype=torch.uint8)
    started = waiting_since = time.time()
    done, missing, waited = 0, 0, 0.0
    for batch_index, screenshots in enumerate(loader):
        waited += time.time() - waiting_since
        missing += sum(screenshot is None for screenshot in screenshots)
        tokens, grids = encode_batch(tower, [placeholder if s is None else s for s in screenshots])
        written.append(tokens, grids)
        done += len(screenshots)
        if (batch_index + 1) % CHECKPOINT_BATCHES == 0:
            written.checkpoint()
            elapsed = time.time() - started
            print(
                f"{done}/{len(frames)} frames  {done / elapsed:.1f}/s  waiting for screenshots {waited / elapsed:.0%}"
                f" and for the disk {written.blocked_seconds / elapsed:.0%} of the time"
                f"  eta {(len(frames) - done) * elapsed / done / 60:.1f} min",
                flush=True,
            )
        waiting_since = time.time()
    written.close()
    return missing, time.time() - started


def encode_batch(tower, screenshots: list[torch.Tensor]) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """The tokens of the screenshots, one after the other [n, dim] (on the GPU), and each one's (rows, columns).

    Screenshots of one size go through the tower together.
    """
    by_size: dict[tuple, list[int]] = {}
    for index, screenshot in enumerate(screenshots):
        by_size.setdefault(tuple(screenshot.shape), []).append(index)
    tokens, grids = [None] * len(screenshots), [None] * len(screenshots)
    for indices in by_size.values():
        images = torch.stack([screenshots[index].to(tower.device, non_blocking=True) for index in indices])
        group_tokens, grid = tower.encode(images)
        for index, frame_tokens in zip(indices, group_tokens):
            tokens[index], grids[index] = frame_tokens, grid
    return torch.cat(tokens), grids


class BackgroundWriter:
    """Writes token batches to the shard on background threads.

    A batch leaves the GPU through a pinned buffer: a copy to ordinary memory would hold up the tower's next batch.
    Several threads write, because one stream into a network file system is slower than a GPU produces tokens.
    """

    def __init__(self, writer: TokenShardWriter, buffer_tokens: int, threads: int = 4):
        self.writer, self.blocked_seconds, self.error = writer, 0.0, None
        self.queue, self.buffers = queue.Queue(), queue.Queue()
        for _ in range(2 * threads):
            self.buffers.put(torch.empty((buffer_tokens, writer.dim), dtype=torch.int16).pin_memory())
        self.threads = [threading.Thread(target=self._run, daemon=True) for _ in range(threads)]
        for thread in self.threads:
            thread.start()

    def append(self, tokens: torch.Tensor, grids: list[tuple[int, int]]):
        """tokens: bfloat16 [n, dim] on the GPU, the tokens of the next len(grids) steps."""
        started = time.time()
        buffer = self.buffers.get()
        self.blocked_seconds += time.time() - started
        buffer[: len(tokens)].copy_(tokens.view(torch.int16), non_blocking=True)
        copied = torch.cuda.Event()
        copied.record()
        self.queue.put((buffer, len(tokens), copied, self.writer.reserve(grids)))

    def checkpoint(self):
        """Waits until everything appended is written, then makes it durable."""
        started = time.time()
        self.queue.join()
        self.blocked_seconds += time.time() - started
        if self.error is not None:
            raise self.error
        self.writer.checkpoint()

    def close(self):
        self.checkpoint()
        for _ in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def _run(self):
        while (item := self.queue.get()) is not None:
            buffer, n_tokens, copied, offset = item
            try:
                if self.error is None:
                    copied.synchronize()
                    self.writer.write(offset, buffer[:n_tokens].numpy().view(np.uint16))
            except Exception as error:
                self.error = error
            self.buffers.put(buffer)
            self.queue.task_done()


if __name__ == "__main__":
    main()
