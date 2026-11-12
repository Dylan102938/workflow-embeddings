"""Frozen screen-feature cache: SigLIP2 patch features of every screenshot, pooled to a grid and PCA-projected.

  python -m wfemb.prep.cache_frames --mode pca_fit --grid 24                     # once per grid, before the shards
  python -m wfemb.prep.cache_frames --mode dense --grid 24 --shard 0 --num_shards 8

File formats: docs/DATA_PREP.md.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import frame_cache_name
from ..storage import paths
from .prepare import load_sessions
from .splitzip import SplitZip

Image.MAX_IMAGE_PIXELS = None  # large screenshots would otherwise trip PIL's decompression-bomb check
ENCODER = "google/siglip2-so400m-patch16-naflex"
DEFAULT_GRID = 16
PLACEHOLDER_IMAGE_SIZE = (1280, 720)
PCA_REPORT_DIMS = (64, 128, 256, 384, 512, 768)

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class Frame(NamedTuple):
    step: int  # global step index: row of events.npz and of the concatenated cache shards
    source: str
    image: str
    split: str


def main() -> None:
    args = parse_arguments()
    output_dir = paths.BULK / frame_cache_name(args.grid, args.pca_dim)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = all_frames(load_sessions())
    encoder = ScreenEncoder(args)
    if args.mode == "pca_fit":
        if paths.ON_ADDED_SET:
            raise SystemExit("the PCA basis is fitted on AgentNet only")
        fit_pca(encoder, frames, args)
    else:
        write_shard(encoder, frames, args, output_dir)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["pca_fit", "dense"],
        default="dense",
        help="pca_fit: channel PCA of dense tokens over a sample of TRAIN frames; dense: write a shard",
    )
    parser.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--global_patches", type=int, default=1024, help="most SigLIP2 patches per screenshot")
    parser.add_argument(
        "--grid", type=int, default=DEFAULT_GRID, help="tokens per frame = grid^2 (the experiment's patch)"
    )
    parser.add_argument("--pca_dim", type=int, default=512, help="0: keep the encoder's channels (no PCA)")
    parser.add_argument("--pca_frames", type=int, default=16000)
    parser.add_argument("--batch", dest="batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


def pca_path(grid: int, pca_dim: int) -> Path:
    """The basis fitted on AgentNet's training frames; the added datasets are projected with it too."""
    return paths.AGENTNET_BULK / f"dense_pca_g{grid}_d{pca_dim}.npz"


def all_frames(sessions: list[dict]) -> list[Frame]:
    """Every step's screenshot, in global step order."""
    frames: list[Frame] = []
    for session in sessions:
        for image in session["images"]:
            frames.append(Frame(len(frames), session["source"], image, session["split"]))
    return frames


# ------------------------------------------------------------------------------------------------- encoding
class FrameDataset(torch.utils.data.Dataset):
    """Screenshots read from the image archives in place, preprocessed for SigLIP2 NaFlex."""

    def __init__(self, frames: list[Frame], encoder: str, max_patches: int):
        self.frames, self.encoder, self.max_patches = frames, encoder, max_patches
        self.archives = None  # opened on first use, so every loader worker gets its own file handles

    def _open(self) -> None:
        from transformers import AutoImageProcessor

        self.processor = AutoImageProcessor.from_pretrained(self.encoder)
        sources = {frame.source for frame in self.frames}
        self.archives = {source: SplitZip(paths.ZIPS[source]) for source in sources}
        self.member_paths = {source: archive.basename_index() for source, archive in self.archives.items()}

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int):
        if self.archives is None:
            self._open()
        frame = self.frames[i]
        image = self._read_image(frame)
        found = image is not None
        if not found:
            image = Image.new("RGB", PLACEHOLDER_IMAGE_SIZE)
        inputs = self.processor(images=[image], max_num_patches=self.max_patches, return_tensors="pt")
        return (
            frame.step,
            found,
            inputs["pixel_values"][0],
            inputs["pixel_attention_mask"][0],
            inputs["spatial_shapes"][0],
        )

    def _read_image(self, frame: Frame) -> Image.Image | None:
        """None if the screenshot is missing from the archive or unreadable."""
        try:
            member = self.member_paths[frame.source].get(frame.image)
            if member is None:
                return None
            return Image.open(io.BytesIO(self.archives[frame.source].read(member))).convert("RGB")
        except Exception:
            return None


def collate(items: list[tuple]) -> Batch:
    steps, found, pixel_values, attention_masks, spatial_shapes = zip(*items)
    return (
        torch.tensor(steps),
        torch.tensor(found),
        torch.stack(pixel_values),
        torch.stack(attention_masks),
        torch.stack(spatial_shapes),
    )


class ScreenEncoder:
    """SigLIP2 vision tower on the GPU: screenshots -> [batch, grid^2, hidden_size] float32 tokens."""

    def __init__(self, args: argparse.Namespace):
        from transformers import Siglip2VisionModel

        self.args = args
        self.vision = (
            Siglip2VisionModel.from_pretrained(args.encoder, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
            .vision_model.cuda()
            .eval()
        )
        self.hidden_size = self.vision.config.hidden_size

    def loader(self, frames: list[Frame]) -> torch.utils.data.DataLoader:
        dataset = FrameDataset(frames, self.args.encoder, self.args.global_patches)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            collate_fn=collate,
            prefetch_factor=4,
            pin_memory=True,
        )

    def encode(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(global step indices, screenshot found, tokens) of one loader batch."""
        steps, found, pixel_values, attention_mask, spatial_shapes = batch
        hidden = self.vision(
            pixel_values=pixel_values.cuda(non_blocking=True).to(torch.bfloat16),
            attention_mask=attention_mask.cuda(non_blocking=True),
            spatial_shapes=spatial_shapes.cuda(non_blocking=True),
        ).last_hidden_state
        return steps, found, pool_grid(hidden, spatial_shapes, (self.args.grid, self.args.grid))


def pool_grid(hidden: torch.Tensor, spatial_shapes: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """SigLIP2 patch outputs [B, n, D] with spatial shapes [B, 2] -> [B, out_h * out_w, D] (adaptive average)."""
    pooled = []
    for i in range(hidden.shape[0]):
        height, width = (int(side) for side in spatial_shapes[i])
        patch_grid = hidden[i, : height * width].reshape(height, width, -1).permute(2, 0, 1)[None].float()
        pooled.append(F.adaptive_avg_pool2d(patch_grid, output_size)[0].flatten(1).T)
    return torch.stack(pooled)


def to_bf16_bits(values: torch.Tensor) -> np.ndarray:
    """The tensor rounded to bfloat16, as the uint16 bit patterns the cache stores."""
    return values.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)


# -------------------------------------------------------------------------------------------------- PCA fit
def fit_pca(encoder: ScreenEncoder, frames: list[Frame], args: argparse.Namespace) -> None:
    """Fits the channel PCA on the tokens of a random sample of training frames and writes it to pca_path."""
    sample = sample_training_frames(frames, args.pca_frames)
    dim = encoder.hidden_size
    token_sum = torch.zeros(dim, dtype=torch.float64, device="cuda")
    outer_product_sum = torch.zeros(dim, dim, dtype=torch.float64, device="cuda")
    token_count, start_time = 0, time.time()
    with torch.inference_mode():
        for batch_index, batch in enumerate(encoder.loader(sample)):
            _, found, tokens = encoder.encode(batch)
            tokens = tokens[found.cuda()].reshape(-1, dim).double()
            token_sum += tokens.sum(0)
            outer_product_sum += tokens.T @ tokens
            token_count += tokens.shape[0]
            if batch_index % 100 == 0:
                print(
                    f"pca_fit {batch_index * args.batch_size}/{len(sample)} frames  {time.time() - start_time:.0f}s",
                    flush=True,
                )

    mean = token_sum / token_count
    covariance = outer_product_sum / token_count - torch.outer(mean, mean)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues, eigenvectors = eigenvalues.flip(0).clamp(min=0), eigenvectors.flip(1)  # largest first
    cumulative_variance = (eigenvalues.cumsum(0) / eigenvalues.sum()).cpu().numpy()
    output_path = pca_path(args.grid, args.pca_dim)
    np.savez(
        output_path,
        mean=mean.float().cpu().numpy(),
        comps=eigenvectors[:, : args.pca_dim].float().cpu().numpy(),
        cum_var=cumulative_variance,
        n_tokens=token_count,
    )
    print(
        f"saved {output_path}: {token_count} tokens; variance kept at "
        + ", ".join(f"{dims}: {cumulative_variance[dims - 1]:.4f}" for dims in PCA_REPORT_DIMS),
        flush=True,
    )


def sample_training_frames(frames: list[Frame], sample_size: int) -> list[Frame]:
    train_frames = [frame for frame in frames if frame.split == "train"]
    chosen = np.random.default_rng(0).choice(len(train_frames), min(sample_size, len(train_frames)), replace=False)
    return [train_frames[i] for i in sorted(chosen)]


# ---------------------------------------------------------------------------------------------- cache shards
def write_shard(encoder: ScreenEncoder, frames: list[Frame], args: argparse.Namespace, output_dir: Path) -> None:
    """Writes global_<tag>.npy for this shard's steps, then meta_<tag>.json; an interrupted shard resumes."""
    project = args.pca_dim > 0
    pca_file = pca_path(args.grid, args.pca_dim) if project else None
    dim = args.pca_dim if project else encoder.hidden_size
    if project:
        pca = np.load(pca_file)
        mean, components = torch.from_numpy(pca["mean"]).cuda(), torch.from_numpy(pca["comps"]).cuda()
    bounds = np.linspace(0, len(frames), args.num_shards + 1).astype(int)
    first_step, end_step = int(bounds[args.shard]), int(bounds[args.shard + 1])
    tag = f"{args.shard:02d}of{args.num_shards:02d}"
    tokens_per_frame = args.grid**2
    features_path, progress_path = output_dir / f"global_{tag}.npy", output_dir / f"progress_{tag}.json"

    start_step = first_step
    if progress_path.exists() and features_path.exists():
        start_step = json.loads(progress_path.read_text())["next"]
        features = np.load(features_path, mmap_mode="r+")
        print(f"resuming at {start_step}", flush=True)
    else:
        features = np.lib.format.open_memmap(
            features_path, mode="w+", dtype=np.uint16, shape=(end_step - first_step, tokens_per_frame, dim)
        )
    print(f"dense shard {tag}: frames [{first_step}, {end_step}) of {len(frames)}", flush=True)

    start_time, done, missing = time.time(), 0, 0
    remaining = end_step - start_step
    with torch.inference_mode():
        for batch_index, batch in enumerate(encoder.loader(frames[start_step:end_step])):
            steps, found, tokens = encoder.encode(batch)
            rows = (steps - first_step).numpy()
            features[rows] = to_bf16_bits((tokens - mean) @ components if project else tokens)
            done += len(rows)
            missing += int((~found).sum())
            if batch_index % 200 == 0:
                features.flush()
                progress_path.write_text(json.dumps({"next": int(rows.max()) + first_step + 1}))
                rate = done / (time.time() - start_time)
                print(
                    f"{done}/{remaining} frames  {rate:.1f}/s  eta {(remaining - done) / max(rate, 1e-6) / 60:.1f} min",
                    flush=True,
                )
    features.flush()

    # training treats a shard as complete once its meta file exists, so it is written last
    meta = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "lo": first_step,
        "hi": end_step,
        "dim": dim,
        "global_tokens": tokens_per_frame,
        "grid": args.grid,
        "encoder": args.encoder,
        "global_patches": args.global_patches,
        "pca": str(pca_file) if project else None,
        "missing": missing,
    }
    with open(output_dir / f"meta_{tag}.json", "w") as f:
        json.dump(meta, f)
    print("done", meta, flush=True)


if __name__ == "__main__":
    main()
