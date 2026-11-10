"""Evaluation on val / test (metric definitions: docs/EXPERIMENTS.md, wfemb/metrics.py).

Candidates = the keys of all training instructions + the evaluated split's instructions (val/test instructions never
occur in training, so each session's own key is a new candidate). False negatives are masked at the fixed threshold
EvalConfig.dupmask so every run is scored the same way.

  python -m wfemb.eval id=<run id>                       # the run's best checkpoint on the test set
  python -m wfemb.eval id=<run id> split=val ckpt=last.pt
  python -m wfemb.eval id=<run id> save=True             # also write <split>_embeddings.npz into the run folder

<split>_embeddings.npz holds every step's direction and confidence (the raw data the demo is built from; training
writes test_embeddings.npz at the end): sidx [S] session indices, offset / length [S], d [sum of lengths, D] fp16,
k [sum of lengths] fp32; session i's steps are rows offset[i] .. offset[i] + length[i] - 1.
"""
from __future__ import annotations

import time
from pathlib import Path

import chz
import numpy as np
import torch

from . import metrics as M
from .config import DataConfig, EvalConfig, Experiment, default_results
from .data import SessionData, TrainLabels, eval_batches


@torch.no_grad()
def embed(model, data: SessionData, sidx: np.ndarray, cfg: EvalConfig, device, data_cfg: DataConfig | None = None):
    """Directions d [S, T_max, D] (fp16) and confidences kappa [S, T_max] (fp32) of every step, on the CPU, rows in
    the order of sidx."""
    was_training = model.training
    model.eval()
    T_max = int(data.length[sidx].max())
    d_all = torch.zeros(len(sidx), T_max, data.keys.shape[1], dtype=torch.float16)
    k_all = torch.zeros(len(sidx), T_max)
    row = {int(s): i for i, s in enumerate(sidx)}
    for b in eval_batches(data, sidx, cfg.batch, cfg.max_frames, device, data_cfg):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            d, k = model(b)
        rows = [row[int(s)] for s in b.idx.cpu()]
        d_all[rows, :d.shape[1]] = d.half().cpu()
        k_all[rows, :d.shape[1]] = k.squeeze(-1).float().cpu()
    model.train(was_training)
    return d_all, k_all


def save_embeddings(path: Path, sidx: np.ndarray, d_all: torch.Tensor, k_all: torch.Tensor, lens: torch.Tensor):
    """Every valid step's d (fp16) and kappa, concatenated per session (format: module docstring)."""
    L = lens.numpy()
    tmp = Path(path).with_suffix(".tmp.npz")
    np.savez(tmp, sidx=np.asarray(sidx), offset=np.r_[0, np.cumsum(L)[:-1]], length=L,
             d=torch.cat([d_all[i, :n] for i, n in enumerate(L)]).numpy(),
             k=torch.cat([k_all[i, :n] for i, n in enumerate(L)]).numpy())
    tmp.rename(path)


@torch.no_grad()
def evaluate(model, data: SessionData, split: str, labels: TrainLabels, cfg: EvalConfig, device,
             data_cfg: DataConfig | None = None, embeddings_out: Path | None = None) -> dict:
    """All metrics of one split, keyed f"{split}_<metric>". embeddings_out: also save every step's embedding there."""
    sidx = data.indices(split)
    S = len(sidx)
    d_all, k_all = embed(model, data, sidx, cfg, device, data_cfg)
    lens = torch.from_numpy(data.length[sidx].astype(np.int64))
    if embeddings_out is not None:
        save_embeddings(embeddings_out, sidx, d_all, k_all, lens)
    U_ref = torch.cat([labels.U, torch.from_numpy(data.keys[sidx]).to(device)])          # [N + S, D]
    g_ref = torch.cat([labels.group, torch.from_numpy(data.group[sidx]).to(device)])
    tgt_all = torch.arange(S, device=device) + len(labels)
    loss_sum, ranks = 0.0, []
    for s0 in range(0, S, 64):                                           # [64, T, N + S] logits per chunk
        sl = slice(s0, min(S, s0 + 64))
        L = lens[sl].to(device)
        T = int(L.max())
        valid = torch.arange(T, device=device)[None, :] < L[:, None]
        target = tgt_all[sl]
        fn = M.false_negative_mask(U_ref, g_ref, U_ref[target], g_ref[target], target, cfg.dupmask)
        lg = M.logits(d_all[sl, :T].to(device), k_all[sl, :T, None].to(device), U_ref, fn)
        loss, _ = M.info_nce(lg, target, valid)
        loss_sum += loss.item() * len(target)
        ranks.append(M.target_rank(M.terminal(lg, L), target).cpu())
    ranks = torch.cat(ranks)
    E = M.terminal(d_all, lens).float()                                  # workflow embeddings [S, D]
    out = {"loss": loss_sum / S, "r@1": M.recall_at_k(ranks, 1), "r@5": M.recall_at_k(ranks, 5),
           f"avg_log_dist_{cfg.knn}NN": float(M.log_dist_knn(E.to(device), U_ref, g_ref, g_ref[tgt_all], cfg.knn).mean())}
    rel = M.load_judged(data.judged_dir / f"judged_{split}.jsonl", sidx, data.group[sidx])
    if rel is not None:
        out.update(M.judged_retrieval(E, rel))
    return {f"{split}_{k}": v for k, v in out.items()}


ORDER = ("loss", "r@1", "r@5", "avg_log_dist_100NN", "judged_r@1", "judged_r@5", "judged_ndcg@10")


def format_metrics(m: dict, title: str = "") -> str:
    """A table: one row per metric, one column per split present in m."""
    splits = [s for s in ("val", "test") if any(k.startswith(s + "_") for k in m)]
    names = [n for n in ORDER if any(f"{s}_{n}" in m for s in splits)]
    lines = [title] if title else []
    lines.append(f"{'metric':<22}" + "".join(f"{s:>10}" for s in splits))
    for n in names:
        vals = [m.get(f"{s}_{n}") for s in splits]
        lines.append(f"{n:<22}" + "".join(f"{v:>10.4f}" if isinstance(v, (int, float)) else f"{'-':>10}" for v in vals))
    return "\n".join(lines)


def main(id: str, split: str = "test", ckpt: str = "best.pt", results: str = "", device: str = "", save: bool = False,
         data: DataConfig = DataConfig(), eval: EvalConfig = EvalConfig()):
    from .model import load_checkpoint
    exp = Experiment.from_id(id)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cuda.matmul.allow_tf32 = True
    run_dir = Path(results or default_results()) / exp.path
    sd = SessionData(data, exp)
    model, state = load_checkpoint(run_dir / ckpt, torch.from_numpy(sd.text_vecs), dev)
    t0 = time.time()
    m = evaluate(model, sd, split, TrainLabels(sd, dev), eval, dev, data,
                 embeddings_out=run_dir / f"{split}_embeddings.npz" if save else None)
    print(format_metrics(m, f"{exp.id}\n{ckpt} (step {state['step']}), {split}, {time.time() - t0:.0f}s"))
    return m


if __name__ == "__main__":
    chz.entrypoint(main)
