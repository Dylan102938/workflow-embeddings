"""Frozen screen-feature cache (offline, GPU; one shard per job / array task).

Every step's screenshot (global step order from prepare.py) is encoded with SigLIP2 so400m (NaFlex: aspect ratio
kept, <= --global_patches patches), its patch grid adaptive-average-pooled to --grid x --grid tokens, and every token
projected with a channel PCA (1152 -> --pca_dim) fit on training frames. Stored as exact bf16 (uint16 bits) in
BULK/frames_dense (grid 16) or BULK/frames_dense_g<grid>: global_<shard>.npy [rows, grid^2, pca_dim] +
meta_<shard>.json. Training reads it with storage/frames.py; patch (experiment variable) = grid^2.

  python -m wfemb.prep.cache_frames --mode pca_fit --grid 24                    # once, before the shards
  python -m wfemb.prep.cache_frames --mode dense --grid 24 --shard 0 --num_shards 8
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..storage import paths
from .splitzip import SplitZip

Image.MAX_IMAGE_PIXELS = None
ENCODER = "google/siglip2-so400m-patch16-naflex"


class FrameDataset(torch.utils.data.Dataset):
    """Screenshots read from the AgentNet archives in place, preprocessed for SigLIP2 NaFlex."""

    def __init__(self, items, encoder, global_patches):
        self.items, self.encoder, self.gp = items, encoder, global_patches
        self.zips = None

    def _open(self):
        from transformers import AutoImageProcessor
        self.proc = AutoImageProcessor.from_pretrained(self.encoder)
        self.zips, self.index = {}, {}
        for src, zp in paths.ZIPS.items():
            z = SplitZip(zp)
            self.zips[src] = z
            self.index[src] = z.basename_index()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.zips is None:
            self._open()
        gi, src, name = self.items[i][:3]
        try:
            member = self.index[src].get(name)
            img = None if member is None else Image.open(io.BytesIO(self.zips[src].read(member))).convert("RGB")
        except Exception:
            img = None
        ok = img is not None
        if not ok:
            img = Image.new("RGB", (1280, 720))
        g = self.proc(images=[img], max_num_patches=self.gp, return_tensors="pt")
        return gi, ok, g["pixel_values"][0], g["pixel_attention_mask"][0], g["spatial_shapes"][0]


def collate(batch):
    gi, ok, pv, pm, ps = zip(*batch)
    return torch.tensor(gi), torch.tensor(ok), torch.stack(pv), torch.stack(pm), torch.stack(ps)


def pool_grid(hid, shapes, out_hw):
    """SigLIP2 patch outputs [B, n, D] with spatial shapes [B, 2] -> [B, out_h * out_w, D] (adaptive average)."""
    res = []
    for i in range(hid.shape[0]):
        h, w = (int(v) for v in shapes[i])
        x = hid[i, : h * w].reshape(h, w, -1).permute(2, 0, 1)[None].float()
        res.append(F.adaptive_avg_pool2d(x, out_hw)[0].flatten(1).T)
    return torch.stack(res)


def to_bf16_bits(x: torch.Tensor) -> np.ndarray:
    return x.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)


def out_dir_for(grid: int):
    return paths.BULK / ("frames_dense" + (f"_g{grid}" if grid != 16 else ""))


def pca_path_for(grid: int, pca_dim: int):
    return paths.BULK / f"dense_pca_g{grid}_d{pca_dim}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pca_fit", "dense"], default="dense",
                    help="pca_fit: channel PCA of dense tokens over a sample of TRAIN frames; dense: write a shard")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num_shards", type=int, default=8)
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--global_patches", type=int, default=1024)
    ap.add_argument("--grid", type=int, default=16, help="tokens per frame = grid^2 (the experiment's patch)")
    ap.add_argument("--pca_dim", type=int, default=512)
    ap.add_argument("--pca_frames", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    out_dir, pca_path = out_dir_for(args.grid), pca_path_for(args.grid, args.pca_dim)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.PREP / "sessions.pkl", "rb") as f:
        sessions = pickle.load(f)
    items = [(g, s["source"], img, s["split"]) for g, (s, img) in
             enumerate((s, img) for s in sessions for img in s["images"])]
    grid = (args.grid, args.grid)

    from transformers import Siglip2VisionModel
    vision = Siglip2VisionModel.from_pretrained(args.encoder, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").vision_model.cuda().eval()
    dim = vision.config.hidden_size

    def loader(sel):
        return torch.utils.data.DataLoader(FrameDataset(sel, args.encoder, args.global_patches), batch_size=args.batch,
                                           num_workers=args.workers, collate_fn=collate, prefetch_factor=4,
                                           pin_memory=True)

    def encode_batch(batch):
        gi, ok, pv, pm, ps = batch
        h = vision(pixel_values=pv.cuda(non_blocking=True).to(torch.bfloat16),
                   attention_mask=pm.cuda(non_blocking=True),
                   spatial_shapes=ps.cuda(non_blocking=True)).last_hidden_state
        return gi, ok, pool_grid(h, ps, grid)                            # [B, grid^2, dim] fp32

    if args.mode == "pca_fit":
        train = [it for it in items if it[3] == "train"]
        rng = np.random.default_rng(0)
        sel = [train[i] for i in sorted(rng.choice(len(train), min(args.pca_frames, len(train)), replace=False))]
        S = torch.zeros(dim, dtype=torch.float64, device="cuda")
        C = torch.zeros(dim, dim, dtype=torch.float64, device="cuda")
        n, t0 = 0, time.time()
        with torch.inference_mode():
            for bi, batch in enumerate(loader(sel)):
                gi, ok, x = encode_batch(batch)
                x = x[ok.cuda()].reshape(-1, dim).double()
                S += x.sum(0)
                C += x.T @ x
                n += x.shape[0]
                if bi % 100 == 0:
                    print(f"pca_fit {bi * args.batch}/{len(sel)} frames  {time.time() - t0:.0f}s", flush=True)
        mean = S / n
        evals, evecs = torch.linalg.eigh(C / n - torch.outer(mean, mean))
        evals, evecs = evals.flip(0).clamp(min=0), evecs.flip(1)
        cum = (evals.cumsum(0) / evals.sum()).cpu().numpy()
        np.savez(pca_path, mean=mean.float().cpu().numpy(), comps=evecs[:, :args.pca_dim].float().cpu().numpy(),
                 cum_var=cum, n_tokens=n)
        print(f"saved {pca_path}: {n} tokens; variance kept at "
              + ", ".join(f"{k}: {cum[k - 1]:.4f}" for k in (64, 128, 256, 384, 512, 768)), flush=True)
        return

    pca = np.load(pca_path)
    mean, comps = torch.from_numpy(pca["mean"]).cuda(), torch.from_numpy(pca["comps"]).cuda()   # comps [dim, pca]
    n = len(items)
    bounds = np.linspace(0, n, args.num_shards + 1).astype(int)
    lo, hi = int(bounds[args.shard]), int(bounds[args.shard + 1])
    tag = f"{args.shard:02d}of{args.num_shards:02d}"
    g_path, prog_path = out_dir / f"global_{tag}.npy", out_dir / f"progress_{tag}.json"
    G = args.grid ** 2
    start = lo
    if prog_path.exists() and g_path.exists():                          # resume an interrupted shard
        start = json.load(open(prog_path))["next"]
        g_mm = np.load(g_path, mmap_mode="r+")
        print(f"resuming at {start}", flush=True)
    else:
        g_mm = np.lib.format.open_memmap(g_path, mode="w+", dtype=np.uint16, shape=(hi - lo, G, args.pca_dim))
    print(f"dense shard {tag}: frames [{lo}, {hi}) of {n}", flush=True)
    t0, done, missing = time.time(), 0, 0
    with torch.inference_mode():
        for bi, batch in enumerate(loader(items[start:hi])):
            gi, ok, x = encode_batch(batch)
            rows = (gi - lo).numpy()
            g_mm[rows] = to_bf16_bits((x - mean) @ comps)
            done += len(rows)
            missing += int((~ok).sum())
            if bi % 200 == 0:
                g_mm.flush()
                with open(prog_path, "w") as f:
                    json.dump({"next": int(rows.max()) + lo + 1}, f)
                rate = done / (time.time() - t0)
                print(f"{done}/{hi - start} frames  {rate:.1f}/s  eta {(hi - start - done) / max(rate, 1e-6) / 60:.1f} min",
                      flush=True)
    g_mm.flush()
    meta = {"shard": args.shard, "num_shards": args.num_shards, "lo": lo, "hi": hi, "dim": args.pca_dim,
            "global_tokens": G, "grid": args.grid, "encoder": args.encoder, "global_patches": args.global_patches,
            "pca": str(pca_path), "missing": missing}
    with open(out_dir / f"meta_{tag}.json", "w") as f:
        json.dump(meta, f)
    print("done", meta, flush=True)


if __name__ == "__main__":
    main()
