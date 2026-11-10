"""Training (docs/EXPERIMENTS.md). One run per process; everything it records goes to its results folder.

  python -m wfemb.train id=<run id>                              # e.g. a line of jobs.txt
  python -m wfemb.train exp.beta=2 exp.seed=1                    # the best model with changes
  python -m wfemb.train id=<run id> train.max_steps=200 results=/tmp/results     # quick test

training_step: one optimizer update on one batch (inputs and update spelled out in its docstring).
training_loop: epochs of training batches; validation every eval_steps; the best checkpoint by val_loss; at the end
the test evaluation of that checkpoint, written to final.json and printed.
"""
from __future__ import annotations

import copy
import faulthandler
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass

import chz
import numpy as np
import torch

from . import metrics as M
from .config import Experiment, RunConfig, TrainConfig
from .data import Batch, SessionData, TrainLabels, train_batches
from .eval import evaluate, format_metrics
from .model import build_model, model_config
from .storage.results import RunWriter, dumps, vars_of


@dataclass
class TrainState:
    model: torch.nn.Module
    ema: torch.nn.Module              # EMA of the weights: evaluated and checkpointed
    opt: torch.optim.Optimizer
    sched: torch.optim.lr_scheduler.LambdaLR
    total: int                        # optimizer steps of the whole run
    step: int = 0
    epoch: int = 0


def training_step(st: TrainState, batch: Batch, labels: TrainLabels, exp: Experiment, train: TrainConfig) -> dict:
    model = st.model
    model.train()
    target = labels.target(batch.idx)
    U, g = labels.U, labels.group
    words = labels.sample_wordings()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=batch.frames.is_cuda):
        d, kappa = model(batch)

    fn = M.false_negative_mask(U, g, U[target], g[target], target, train.dupmask)
    lg = M.logits(d, kappa, words, fn)
    ramp = min(1.0, st.step / max(1.0, train.hard_neg_ramp * st.total)) if train.hard_neg_ramp > 0 else 1.0
    beta = exp.beta * ramp
    soft = M.smooth_targets(U, U[target], target, fn, exp.smooth, train.smooth_tau) if exp.smooth > 0 else None
    lg_train = lg + M.hard_negative_bias(U, U[target], target, fn, beta)[:, None, :] if beta > 0 else lg
    loss, _ = M.info_nce(lg_train, target, batch.valid, soft_targets=soft, fn_mask=fn)

    st.opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train.grad_clip)
    st.opt.step()
    with torch.no_grad():
        for pe, pm in zip(st.ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1.0 - train.ema)

    lr = st.sched.get_last_lr()[0]
    st.sched.step()
    st.step += 1

    with torch.no_grad():
        rigid, _ = M.info_nce(lg.detach(), target, batch.valid)
        dist = M.avg_dist(M.terminal(d.detach(), batch.lengths), U[target], g[target])
        vals = torch.stack([loss.detach(), rigid, dist, grad_norm.float()]).tolist()     # one GPU sync

    return {"loss": vals[0], "rigid_loss": vals[1], "avg_dist": vals[2], "lr": lr, "grad_norm": vals[3],
            "beta": beta}


def cosine_with_warmup(opt, warmup: int, total: int):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def make_state(exp: Experiment, train: TrainConfig, data: SessionData, device) -> tuple[TrainState, object]:
    mcfg = model_config(exp, train, data)
    model = build_model(mcfg, torch.from_numpy(data.text_vecs)).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "emb" not in n]   # matrices, not embeddings
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": train.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=exp.lr, betas=train.adam_betas)
    total = exp.ep * (len(data.indices("train")) // exp.bs)       # epochs of N/bs steps (split batches count as steps)
    if train.max_steps:
        total = min(total, train.max_steps)
    return TrainState(model, ema, opt, cosine_with_warmup(opt, train.warmup, total), total), mcfg


def setup_process(seed: int):
    # Prefetch threads hand the GIL back and forth with this Python-heavy loop; at the default 5 ms switch interval
    # batch building ran ~4.5x slower.
    sys.setswitchinterval(5e-4)
    torch.set_num_threads(4)                                   # several runs share a machine's cores
    faulthandler.dump_traceback_later(900, exit=False)         # a run stuck for 15 min dumps every thread's stack
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _git() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "wfemb"], capture_output=True, text=True,
                                    timeout=10).stdout.strip())
        return {"git_commit": sha or None, "git_dirty": dirty}
    except Exception:
        return {"git_commit": os.environ.get("WFEMB_GIT_COMMIT"), "git_dirty": None}


def training_loop(cfg: RunConfig) -> dict | None:
    exp, train = cfg.experiment, cfg.train
    run_dir = cfg.run_dir
    if (run_dir / "final.json").exists() and not cfg.overwrite:
        print(f"{exp.id}: already finished ({run_dir}); overwrite=True to rerun", flush=True)
        return None
    setup_process(exp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    data = SessionData(cfg.data, exp)
    labels = TrainLabels(data, device)
    st, mcfg = make_state(exp, train, data, device)
    n_params = sum(p.numel() for p in st.model.parameters())
    out = RunWriter(run_dir, fresh=True)
    out.config({"id": exp.id, "experiment": vars_of(exp), "train": chz.asdict(train), "eval": chz.asdict(cfg.eval),
                "data": chz.asdict(cfg.data), "model": mcfg.to_dict(),
                "meta": {**_git(), "host": socket.gethostname(), "launcher": cfg.launcher,
                         "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_params": n_params,
                         "n_train_sessions": len(labels), "steps_per_epoch": len(labels) // exp.bs,
                         "total_steps": st.total, "torch": torch.__version__}})
    print(f"{exp.id}\n{n_params / 1e6:.2f}M params, {st.total} steps, data loaded in {time.time() - t0:.0f}s",
          flush=True)

    def checkpoint(name):
        torch.save({"model": st.ema.state_dict(), "model_cfg": mcfg.to_dict(), "id": exp.id,
                    "experiment": vars_of(exp), "step": st.step}, run_dir / name)

    best = float("inf")
    t_run = time.time()
    while st.step < st.total:
        batches = train_batches(data, exp, train, cfg.data, st.epoch, device)
        while st.step < st.total:
            t_wait = time.time()
            batch = next(batches, None)
            if batch is None:
                break
            t_step = time.time()
            m = training_step(st, batch, labels, exp, train)
            out.step({"step": st.step, "epoch": st.epoch, "time": round(time.time() - t_run, 3), **m,
                      "data_s": round(t_step - t_wait, 4), "step_s": round(time.time() - t_step, 4)})
            if st.step % 50 == 0:
                faulthandler.cancel_dump_traceback_later()
                faulthandler.dump_traceback_later(900, exit=False)
            if st.step % cfg.eval.eval_steps == 0 or st.step == st.total:
                v = evaluate(st.ema, data, "val", labels, cfg.eval, device, cfg.data)
                row = {"step": st.step, "epoch": st.epoch, "time": round(time.time() - t_run, 3), **v,
                       "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None}
                out.eval(row)
                print("[eval]", dumps(row), flush=True)
                if v["val_loss"] < best:
                    best = v["val_loss"]
                    checkpoint("best.pt")
                checkpoint("last.pt")
        batches.close()                       # returns the prefetcher's in-flight buffers
        st.epoch += 1

    # final: the checkpoint with the lowest val_loss, on val and test
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    st.ema.load_state_dict(state["model"])
    final = {"best_step": state["step"], "selected_on": "val_loss"}
    for split in ("val", "test"):
        final.update(evaluate(st.ema, data, split, labels, cfg.eval, device, cfg.data))
    final["train_minutes"] = round((time.time() - t0) / 60, 2)
    out.final(final)
    out.close()
    faulthandler.cancel_dump_traceback_later()
    print(format_metrics(final, f"{exp.id}\nbest step {state['step']} of {st.total}, "
                                f"{final['train_minutes']:.1f} min"), flush=True)
    return final


def main(cfg: RunConfig):
    return training_loop(cfg)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
