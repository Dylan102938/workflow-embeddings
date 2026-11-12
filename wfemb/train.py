"""Training: one run per process, recorded in its results folder (docs/EXPERIMENTS.md).

python -m wfemb.train id=<run id>
python -m wfemb.train experiment.hard_negative_beta=2 experiment.seed=1
python -m wfemb.train id=<run id> train.max_steps=200 results_root=/tmp/results
"""

from __future__ import annotations

import copy
import faulthandler
import math
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chz
import numpy as np
import torch

from . import metrics
from .config import Experiment, RunConfig, TrainConfig, parse_run_config
from .data import Batch, TrainingCandidates, WorkflowDataset, evaluation_candidates, train_batches
from .eval import evaluate, format_metrics
from .model import ModelConfig, build_model, model_config
from .storage.results import RunWriter, to_json

STALL_SECONDS = 900  # a process without progress for this long dumps every thread's stack to its log


@dataclass
class TrainState:
    model: torch.nn.Module
    ema_model: torch.nn.Module  # the model that is evaluated and checkpointed
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    total_steps: int
    step: int = 0
    epoch: int = 0


def training_step(
    state: TrainState, batch: Batch, candidates: TrainingCandidates, experiment: Experiment, train: TrainConfig
) -> dict:
    """One optimizer update on one batch. Returns the step's metrics."""
    keys, groups = candidates.keys, candidates.groups
    targets = candidates.targets(batch.session_indices)
    mask = metrics.false_negative_mask(keys, groups, targets, train.duplicate_threshold)
    wordings = candidates.sample_wordings()

    state.model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=batch.frames.is_cuda):
        directions, confidences = state.model(batch)
    logits = metrics.candidate_logits(directions, confidences, wordings, mask)

    beta = experiment.hard_negative_beta * hard_negative_ramp(state, train)
    training_logits = logits
    if beta > 0:
        training_logits = logits + metrics.hard_negative_bias(keys, targets, mask, beta)[:, None, :]
    soft_targets = None
    if experiment.label_smoothing > 0:
        soft_targets = metrics.smoothed_targets(
            keys, targets, mask, experiment.label_smoothing, train.smoothing_temperature
        )
    loss, _ = metrics.info_nce(training_logits, targets, batch.valid, soft_targets, mask)

    state.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), train.grad_clip)
    state.optimizer.step()
    update_ema(state.ema_model, state.model, train.ema_decay)
    learning_rate = state.scheduler.get_last_lr()[0]
    state.scheduler.step()
    state.step += 1

    with torch.no_grad():
        rigid_loss, _ = metrics.info_nce(logits.detach(), targets, batch.valid)
        embeddings = metrics.at_last_step(directions.detach(), batch.lengths)
        avg_dist = metrics.avg_dist(embeddings, keys[targets], groups[targets])
        # one transfer from the GPU for all four values
        loss, rigid_loss, avg_dist, grad_norm = torch.stack(
            [loss.detach(), rigid_loss, avg_dist, grad_norm.float()]
        ).tolist()
    return {
        "loss": loss,
        "rigid_loss": rigid_loss,
        "avg_dist": avg_dist,
        "lr": learning_rate,
        "grad_norm": grad_norm,
        "beta": beta,
    }


def hard_negative_ramp(state: TrainState, train: TrainConfig) -> float:
    """Fraction of the hard-negative beta in effect at this step: rises linearly from 0 to 1."""
    if train.hard_negative_ramp <= 0:
        return 1.0
    return min(1.0, state.step / max(1.0, train.hard_negative_ramp * state.total_steps))


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter, 1.0 - decay)


def make_optimizer(model: torch.nn.Module, experiment: Experiment, train: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only (not on biases, norms or embedding tables)."""
    decayed, not_decayed = [], []
    for name, parameter in model.named_parameters():
        is_decayed = parameter.ndim >= 2 and "embedding" not in name
        (decayed if is_decayed else not_decayed).append(parameter)
    groups = [{"params": decayed, "weight_decay": train.weight_decay}, {"params": not_decayed, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=experiment.learning_rate, betas=train.adam_betas)


def cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def make_state(
    config: RunConfig, model_cfg: ModelConfig, dataset: WorkflowDataset, n_train_sessions: int, device
) -> TrainState:
    experiment, train = config.experiment, config.train
    model = build_model(model_cfg, torch.from_numpy(dataset.text_embeddings)).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    total_steps = experiment.n_epochs * (n_train_sessions // experiment.batch_size)
    if train.max_steps:
        total_steps = min(total_steps, train.max_steps)
    optimizer = make_optimizer(model, experiment, train)
    scheduler = cosine_schedule_with_warmup(optimizer, train.warmup_steps, total_steps)
    return TrainState(model, ema_model, optimizer, scheduler, total_steps)


def setup_process(seed: int):
    # With the default 5 ms switch interval the prefetch threads starve behind this loop (batches ~4x slower).
    sys.setswitchinterval(5e-4)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    restart_stall_watchdog()


def restart_stall_watchdog():
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(STALL_SECONDS, exit=False)


def git_state() -> dict:
    def git(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], capture_output=True, text=True, timeout=10).stdout.strip()

    try:
        return {
            "git_commit": git("rev-parse", "HEAD") or None,
            "git_dirty": bool(git("status", "--porcelain", "wfemb")),
        }
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def run_description(config: RunConfig, model_cfg: ModelConfig, state: TrainState, n_train_sessions: int) -> dict:
    """The contents of config.json."""
    experiment = config.experiment
    return {
        "id": experiment.id,
        "experiment": chz.asdict(experiment),
        "train": chz.asdict(config.train),
        "eval": chz.asdict(config.eval),
        "data": chz.asdict(config.data),
        "model": model_cfg.to_dict(),
        "meta": {
            **git_state(),
            "host": socket.gethostname(),
            "launcher": config.launcher,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "n_params": sum(parameter.numel() for parameter in state.model.parameters()),
            "n_train_sessions": n_train_sessions,
            "steps_per_epoch": n_train_sessions // experiment.batch_size,
            "total_steps": state.total_steps,
            "torch": torch.__version__,
        },
    }


def save_checkpoint(path: Path, state: TrainState, model_cfg: ModelConfig, experiment: Experiment):
    torch.save(
        {
            "model": state.ema_model.state_dict(),
            "model_config": model_cfg.to_dict(),
            "id": experiment.id,
            "experiment": chz.asdict(experiment),
            "step": state.step,
        },
        path,
    )


def final_evaluation(
    config: RunConfig, state: TrainState, dataset: WorkflowDataset, evaluation: TrainingCandidates, device
) -> dict:
    """Validation and test metrics of the checkpoint with the lowest validation loss."""
    run_dir = config.run_dir
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    state.ema_model.load_state_dict(best["model"])
    final = {"best_step": best["step"], "selected_on": "val_loss"}
    final.update(evaluate(state.ema_model, dataset, "val", evaluation, config.eval, device))
    embeddings_path = run_dir / "test_embeddings.npz" if config.eval.save_test_embeddings else None
    final.update(evaluate(state.ema_model, dataset, "test", evaluation, config.eval, device, embeddings_path))
    return final


def training_loop(config: RunConfig) -> dict | None:
    experiment, train, run_dir = config.experiment, config.train, config.run_dir
    if (run_dir / "final.json").exists() and not config.overwrite:
        print(f"{experiment.id}: already finished ({run_dir}); overwrite=True to rerun", flush=True)
        return None

    setup_process(experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    dataset = WorkflowDataset(config.data, experiment)
    candidates = TrainingCandidates(dataset, device)
    evaluation = evaluation_candidates(dataset, candidates)
    model_cfg = model_config(experiment, train, dataset)
    state = make_state(config, model_cfg, dataset, len(candidates), device)

    writer = RunWriter(run_dir)
    description = run_description(config, model_cfg, state, len(candidates))
    writer.write_config(description)
    print(
        f"{experiment.id}\n{description['meta']['n_params'] / 1e6:.2f}M parameters, {state.total_steps} steps",
        flush=True,
    )

    def seconds_elapsed() -> float:
        return round(time.time() - started, 3)

    best_val_loss = float("inf")
    while state.step < state.total_steps:
        batches = train_batches(dataset, experiment, train, config.data, state.epoch, device)
        while state.step < state.total_steps:
            waiting_since = time.time()
            batch = next(batches, None)
            if batch is None:
                break
            step_started = time.time()
            step_metrics = training_step(state, batch, candidates, experiment, train)
            writer.write_step(
                {
                    "step": state.step,
                    "epoch": state.epoch,
                    "time": seconds_elapsed(),
                    **step_metrics,
                    "data_s": round(step_started - waiting_since, 4),
                    "step_s": round(time.time() - step_started, 4),
                }
            )
            if state.step % 50 == 0:
                restart_stall_watchdog()

            if state.step % config.eval.eval_steps == 0 or state.step == state.total_steps:
                val_metrics = evaluate(state.ema_model, dataset, "val", evaluation, config.eval, device)
                max_memory = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
                row = {
                    "step": state.step,
                    "epoch": state.epoch,
                    "time": seconds_elapsed(),
                    **val_metrics,
                    "max_mem_gb": max_memory,
                }
                writer.write_eval(row)
                print("[eval]", to_json(row), flush=True)
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    save_checkpoint(run_dir / "best.pt", state, model_cfg, experiment)
                save_checkpoint(run_dir / "last.pt", state, model_cfg, experiment)
        batches.close()  # gives the prefetcher's in-flight buffers back to the pool
        state.epoch += 1

    final = final_evaluation(config, state, dataset, evaluation, device)
    final["train_minutes"] = round((time.time() - started) / 60, 2)
    writer.write_final(final)
    writer.close()
    faulthandler.cancel_dump_traceback_later()
    title = f"{experiment.id}\nbest step {final['best_step']} of {state.total_steps}, {final['train_minutes']:.1f} min"
    print(format_metrics(final, title), flush=True)
    return final


if __name__ == "__main__":
    training_loop(parse_run_config())
