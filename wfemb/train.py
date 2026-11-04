"""Training loop (pseudocode section 5). One run per GPU."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import paths
from .data import Prefetcher, SessionData, length_bucketed_batches
from .evaluate import evaluate
from .loss import false_negative_mask, info_nce
from .model import ModelConfig, SessionStateMachine


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames_dir", default=os.environ.get("WFEMB_FRAMES_DIR", str(paths.FRAMES)))
    ap.add_argument("--frames_mode", default="gpu_fp8", choices=["gpu_fp8", "memmap"],
                    help="gpu_fp8: frame cache resident on GPU as fp8; memmap: exact bf16 read from disk")
    # model
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--h", type=int, default=1024)
    ap.add_argument("--n_queries", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--k_min", type=float, default=1.0)
    ap.add_argument("--k_max", type=float, default=100.0)
    ap.add_argument("--use_crop", action="store_true")
    # optimization
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=30)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max_len", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=4096, help="cap on B x T per batch (long sessions split)")
    # loss
    ap.add_argument("--theta_fn", type=float, default=0.9, help="false-negative threshold used in training")
    ap.add_argument("--eval_theta_fn", type=float, default=0.9, help="fixed across runs so val metrics compare")
    ap.add_argument("--no_para", action="store_true", help="disable paraphrase augmentation")
    ap.add_argument("--center_keys", action="store_true", help="subtract the train-key mean, renormalize")
    # eval / logging
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--select_on", default="loss",
                    help="val metric for best.pt: loss | key_R@5 | jsession_R@5_cross_app | jsession_nDCG@10_cross_app ...")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop early")
    return ap.parse_args(argv)


def cosine_with_warmup(opt, warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def main(argv=None):
    args = get_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    run_dir = paths.RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(run_dir / "log.jsonl", "a")

    def log(rec):
        rec = {"time": time.time(), **rec}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(json.dumps(rec), flush=True)

    t0 = time.time()
    data = SessionData(Path(args.frames_dir), use_crop=args.use_crop, frames_mode=args.frames_mode, device=device)
    if args.center_keys:
        tr = data.indices("train")
        mu = data.keys[tr].mean(0, keepdims=True)
        nz = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        data.keys = nz(data.keys - mu).astype(np.float32)
        if len(data.para_keys):
            data.para_keys = nz(data.para_keys - mu).astype(np.float32)
    if args.no_para:
        data.para_count = np.zeros_like(data.para_count)
    train_idx = data.indices("train")
    print(f"data loaded in {time.time() - t0:.0f}s: {len(train_idx)} train sessions, "
          f"{data.frames.total} frames, n_tok={data.frames.n_tok}", flush=True)

    U = torch.from_numpy(data.keys[train_idx]).to(device)                # N x D, frozen
    gU = torch.from_numpy(data.group[train_idx]).to(device)
    pos_in_U = np.full(len(data.sessions), -1, np.int64)
    pos_in_U[train_idx] = np.arange(len(train_idx))

    cfg = ModelConfig(d_key=U.shape[1], d_vis=data.frames.dim, n_tok=data.frames.n_tok, d=args.d, h=args.h,
                      n_queries=args.n_queries, n_keys=data.n_keys, d_text=data.text_vecs.shape[1],
                      k_min=args.k_min, k_max=args.k_max, dropout=args.dropout)
    model = SessionStateMachine(cfg, torch.from_numpy(data.text_vecs)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))
    steps_per_epoch = len(train_idx) // args.batch
    total = int(args.epochs * steps_per_epoch)
    if args.max_steps:
        total = min(total, args.max_steps)
    sched = cosine_with_warmup(opt, args.warmup, total)
    with open(run_dir / "config.json", "w") as f:
        json.dump({"args": vars(args), "model": dataclasses.asdict(cfg), "n_params": n_params,
                   "steps_per_epoch": steps_per_epoch, "total_steps": total,
                   "slurm_job": os.environ.get("SLURM_JOB_ID")}, f, indent=1)
    print(f"{n_params / 1e6:.2f}M params, {total} steps ({steps_per_epoch}/epoch)", flush=True)

    best = {}
    tracked = {"loss", "key_R@5", "jsession_R@5_cross_app", "jsession_nDCG@10_cross_app", args.select_on}
    step, epoch = 0, 0
    t_last, data_wait = time.time(), 0.0
    run_loss, run_n = 0.0, 0
    done = False
    while not done:
        batches = length_bucketed_batches(data.length, train_idx, args.batch, rng, max_frames=args.max_frames)
        it = iter(Prefetcher(data, batches, device, max_len=args.max_len or None))
        while True:
            tw = time.time()
            try:
                b, _ = next(it)
            except StopIteration:
                break
            data.finalize(b)
            data_wait += time.time() - tw
            bidx = b.idx.cpu().numpy()
            # paraphrase augmentation: a random wording of each instruction, re-drawn every time (main thread rng)
            pos_keys = torch.from_numpy(data.sample_positive_keys(bidx, rng)).to(device, non_blocking=True)
            target = torch.from_numpy(pos_in_U[bidx]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d, k = model(b)
            fn = false_negative_mask(U, gU, U[target], gU[target], target, args.theta_fn)
            loss = info_nce(d, k, b.valid.float(), U, pos_keys, target, fn)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            run_n += 1
            if step % args.log_every == 0:
                dt = time.time() - t_last
                log({"step": step, "epoch": epoch, "train_loss": run_loss / run_n, "lr": sched.get_last_lr()[0],
                     "grad_norm": float(gn), "kappa_mean": float(k[b.valid].mean()),
                     "steps_per_s": args.log_every / dt, "data_wait_frac": data_wait / dt})
                t_last, data_wait, run_loss, run_n = time.time(), 0.0, 0.0, 0
            if step % args.eval_every == 0 or step == total:
                m = evaluate(model, data, "val", device, args.eval_theta_fn, U, train_idx)
                rec = {"step": step, "epoch": epoch, **{f"val_{k_}": v for k_, v in m.items()}}
                log(rec)
                state = {"model": model.state_dict(), "cfg": dataclasses.asdict(cfg), "args": vars(args),
                         "step": step}
                sel = args.select_on if args.select_on in m else "loss"   # metric may be unavailable
                for key in tracked:
                    if key not in m or m[key] != m[key]:                 # missing or nan
                        continue
                    if key not in best or (m[key] < best[key] if key == "loss" else m[key] > best[key]):
                        best[key] = m[key]
                        if key == sel:
                            torch.save(state, run_dir / "best.pt")
                torch.save(state, run_dir / "last.pt")
                t_last = time.time()
            if step >= total:
                done = True
                break
        epoch += 1

    # final: best checkpoint on val + test (held-out apps), with the nuisance probe
    state = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    final = {"best_step": state["step"], "best": best, "select_on": args.select_on}
    for split in ("val", "test"):
        m = evaluate(model, data, split, device, args.eval_theta_fn, U, train_idx, full=True)
        final[split] = m
    with open(run_dir / "final.json", "w") as f:
        json.dump(final, f, indent=1)
    log({"final": True, **{f"{sp}_{k_}": v for sp in ("val", "test") for k_, v in final[sp].items()
                           if not isinstance(v, list)}})
    print(f"total time {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
