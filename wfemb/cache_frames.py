"""Step 1e (offline, GPU, Slurm array): frozen vision-encoder cache.

For every step screenshot (global step order from prepare.py):
  global tokens: resize keeping aspect ratio (NaFlex, <= GLOBAL_PATCHES patches), encode, then
                 adaptive-average-pool the patch grid to G x G tokens            -> [G*G, D_v]
  crop tokens:   square crop around the action point (or last pointer position) at high resolution,
                 encode, pool to C x C tokens. Used only when the model enables crops -> [C*C, D_v]
Stored as bf16 bit patterns (uint16) in per-shard .npy files, which are memory-mapped at training time.
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

from . import paths
from .splitzip import SplitZip

Image.MAX_IMAGE_PIXELS = None


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, items, encoder, global_patches, crop_patches, crop_frac, crop_px, do_global=True):
        self.items = items
        self.do_global = do_global
        self.encoder = encoder
        self.gp, self.cp = global_patches, crop_patches
        self.crop_frac, self.crop_px = crop_frac, crop_px
        self.zips = None
        self.proc = None

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

    def _load(self, src, name):
        member = self.index[src].get(name)
        if member is None:
            return None
        return Image.open(io.BytesIO(self.zips[src].read(member))).convert("RGB")

    def __getitem__(self, i):
        if self.zips is None:
            self._open()
        gi, src, name, fx, fy = self.items[i]
        try:
            img = self._load(src, name)
        except Exception:
            img = None
        ok = img is not None
        if not ok:
            img = Image.new("RGB", (1280, 720))
        W, H = img.size
        side = min(max(int(round(self.crop_frac * W)), 224), W, H)
        left = int(round(min(max(fx * W - side / 2, 0), W - side)))
        top = int(round(min(max(fy * H - side / 2, 0), H - side)))
        crop = img.crop((left, top, left + side, top + side)).resize((self.crop_px, self.crop_px), Image.BICUBIC)
        c = self.proc(images=[crop], max_num_patches=self.cp, return_tensors="pt")
        if self.do_global:
            g = self.proc(images=[img], max_num_patches=self.gp, return_tensors="pt")
            gv, gm, gs = g["pixel_values"][0], g["pixel_attention_mask"][0], g["spatial_shapes"][0]
        else:
            gv, gm, gs = torch.zeros(1), torch.zeros(1), torch.zeros(2, dtype=torch.long)
        return (gi, ok, W, H, gv, gm, gs, c["pixel_values"][0], c["pixel_attention_mask"][0], c["spatial_shapes"][0])


def collate(batch):
    cols = list(zip(*batch))
    out = [torch.tensor(cols[0]), torch.tensor(cols[1]), torch.tensor(cols[2]), torch.tensor(cols[3])]
    out += [torch.stack(c) for c in cols[4:]]
    return out


def pool_grid(hid, shapes, out_hw):
    res = []
    for i in range(hid.shape[0]):
        h, w = (int(v) for v in shapes[i])
        x = hid[i, : h * w].reshape(h, w, -1).permute(2, 0, 1)[None].float()
        res.append(F.adaptive_avg_pool2d(x, out_hw)[0].flatten(1).T)
    return torch.stack(res)


def to_bf16_bits(x: torch.Tensor) -> np.ndarray:
    return x.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num_shards", type=int, default=8)
    ap.add_argument("--encoder", default="google/siglip2-so400m-patch16-naflex")
    ap.add_argument("--global_patches", type=int, default=1024)
    ap.add_argument("--grid", type=int, default=8)          # global tokens = grid^2
    ap.add_argument("--crop_patches", type=int, default=576)
    ap.add_argument("--crop_grid", type=int, default=4)     # crop tokens = crop_grid^2
    ap.add_argument("--crop_frac", type=float, default=0.25)
    ap.add_argument("--crop_px", type=int, default=384)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N frames of the shard")
    ap.add_argument("--mode", choices=["full", "crop", "effect"], default="full",
                    help="full: global grid + action crop (main cache). crop / effect: an extra crop-only cache "
                         "around the current / PREVIOUS step's action point, written to BULK/frames_<name>/")
    ap.add_argument("--name", default="", help="extra-cache name (default: the mode)")
    args = ap.parse_args()
    extra = args.mode != "full"
    effect = args.mode == "effect"
    name = args.name or args.mode
    out_dir = paths.BULK / f"frames_{name}" if extra else paths.FRAMES
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(paths.PREP / "sessions.pkl", "rb") as f:
        sessions = pickle.load(f)
    ev = np.load(paths.PREP / "events.npz")
    focus = ev["focus"]
    items = []
    for s in sessions:
        for t, img in enumerate(s["images"]):
            g = len(items)
            if effect:   # where the previous action landed; screen center for the first frame
                fx, fy = (float(focus[g - 1, 0]), float(focus[g - 1, 1])) if t > 0 else (0.5, 0.5)
            else:
                fx, fy = float(focus[g, 0]), float(focus[g, 1])
            items.append((g, s["source"], img, fx, fy))
    n = len(items)
    bounds = np.linspace(0, n, args.num_shards + 1).astype(int)
    lo, hi = int(bounds[args.shard]), int(bounds[args.shard + 1])
    if args.limit:
        hi = min(hi, lo + args.limit)
    tag = f"{args.shard:02d}of{args.num_shards:02d}"
    print(f"shard {tag}: frames [{lo}, {hi}) of {n}", flush=True)

    from transformers import Siglip2VisionModel
    vision = Siglip2VisionModel.from_pretrained(args.encoder, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").vision_model.cuda().eval()
    dim = vision.config.hidden_size

    g_path = out_dir / f"global_{tag}.npy"
    c_path = out_dir / (f"{name}_{tag}.npy" if extra else f"crop_{tag}.npy")
    prog_path = out_dir / f"progress_{tag}.json"
    G, C = args.grid ** 2, args.crop_grid ** 2
    start = lo
    if prog_path.exists() and c_path.exists():
        start = json.load(open(prog_path))["next"]
        g_mm = None if extra else np.load(g_path, mmap_mode="r+")
        c_mm = np.load(c_path, mmap_mode="r+")
        sizes = np.load(out_dir / f"sizes_{tag}.npy", mmap_mode="r+")
        print(f"resuming at {start}", flush=True)
    else:
        g_mm = None if extra else np.lib.format.open_memmap(g_path, mode="w+", dtype=np.uint16,
                                                             shape=(hi - lo, G, dim))
        c_mm = np.lib.format.open_memmap(c_path, mode="w+", dtype=np.uint16, shape=(hi - lo, C, dim))
        sizes = np.lib.format.open_memmap(out_dir / f"sizes_{tag}.npy", mode="w+", dtype=np.int32,
                                          shape=(hi - lo, 3))
    ds = FrameDataset(items[start:hi], args.encoder, args.global_patches, args.crop_patches,
                      args.crop_frac, args.crop_px, do_global=not extra)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
                                     prefetch_factor=4, persistent_workers=False, pin_memory=True)
    t0, done, missing = time.time(), 0, 0
    with torch.inference_mode():
        for bi, (gi, ok, W, H, gpv, gmask, gshape, cpv, cmask, cshape) in enumerate(dl):
            if not extra:
                gh = vision(pixel_values=gpv.cuda(non_blocking=True).to(torch.bfloat16),
                            attention_mask=gmask.cuda(non_blocking=True),
                            spatial_shapes=gshape.cuda(non_blocking=True)).last_hidden_state
            ch = vision(pixel_values=cpv.cuda(non_blocking=True).to(torch.bfloat16),
                        attention_mask=cmask.cuda(non_blocking=True),
                        spatial_shapes=cshape.cuda(non_blocking=True)).last_hidden_state
            cp = pool_grid(ch, cshape, (args.crop_grid, args.crop_grid))
            rows = (gi - lo).numpy()
            if not extra:
                g_mm[rows] = to_bf16_bits(pool_grid(gh, gshape, (args.grid, args.grid)))
            c_mm[rows] = to_bf16_bits(cp)
            sizes[rows] = np.stack([W.numpy(), H.numpy(), ok.numpy().astype(np.int32)], 1)
            done += len(rows)
            missing += int((~ok).sum())
            if bi % 200 == 0 or done == len(ds):
                c_mm.flush(); sizes.flush()
                if g_mm is not None:
                    g_mm.flush()
                with open(prog_path, "w") as f:
                    json.dump({"next": int(rows.max()) + lo + 1}, f)
                rate = done / (time.time() - t0)
                print(f"{done}/{len(ds)} frames  {rate:.1f}/s  eta {(len(ds) - done) / max(rate, 1e-6) / 60:.1f} min"
                      f"  missing={missing}", flush=True)
    c_mm.flush(); sizes.flush()
    if g_mm is not None:
        g_mm.flush()
    meta = {"shard": args.shard, "num_shards": args.num_shards, "lo": lo, "hi": hi, "dim": dim,
            "global_tokens": G, "crop_tokens": C, "encoder": args.encoder, "global_patches": args.global_patches,
            "crop_patches": args.crop_patches, "crop_frac": args.crop_frac, "crop_px": args.crop_px,
            "missing": missing, "mode": args.mode, "name": name,
            "complete": not args.limit}
    with open(out_dir / f"meta_{tag}.json", "w") as f:
        json.dump(meta, f)
    print("done", meta, flush=True)


if __name__ == "__main__":
    main()
