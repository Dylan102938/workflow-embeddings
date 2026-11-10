"""Evaluation on the validation or test split. Metric definitions: docs/EXPERIMENTS.md.

python -m wfemb.eval id=<run id> [split=val] [checkpoint=last.pt] [save_embeddings=True]
python -m wfemb.eval id=<run id> finalize=True     # redo a run's final evaluation and write final.json
"""

from __future__ import annotations

import time
from pathlib import Path

import chz
import numpy as np
import torch

from . import metrics
from .config import DataConfig, EvalConfig, Experiment, default_results_root
from .data import TrainingCandidates, WorkflowDataset, eval_batches
from .storage.results import to_json

SCORING_CHUNK = 64  # sessions scored at once: logits are [chunk, T, N]
METRIC_ORDER = ("loss", "r@1", "r@5", "avg_log_dist")


@torch.no_grad()
def embed_sessions(
    model, dataset: WorkflowDataset, session_indices: np.ndarray, config: EvalConfig, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every step's direction [S, T, D] (float16) and confidence [S, T] on the CPU, in the given session order."""
    was_training = model.training
    model.eval()
    max_steps = int(dataset.lengths[session_indices].max())
    directions = torch.zeros(len(session_indices), max_steps, dataset.label_keys.shape[1], dtype=torch.float16)
    confidences = torch.zeros(len(session_indices), max_steps)
    row_of_session = {int(session): row for row, session in enumerate(session_indices)}
    for batch in eval_batches(dataset, session_indices, config.batch_size, config.max_frames_per_batch, device):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            batch_directions, batch_confidences = model(batch)
        rows = [row_of_session[int(session)] for session in batch.session_indices.cpu()]
        n_steps = batch_directions.shape[1]
        directions[rows, :n_steps] = batch_directions.half().cpu()
        confidences[rows, :n_steps] = batch_confidences.squeeze(-1).float().cpu()
    model.train(was_training)
    return directions, confidences


def save_embeddings(
    path: Path, session_indices: np.ndarray, directions: torch.Tensor, confidences: torch.Tensor, lengths: np.ndarray
):
    """The valid steps of every session, concatenated: session i owns rows step_offsets[i] + range(lengths[i])."""
    temporary = Path(path).with_suffix(".tmp.npz")
    np.savez(
        temporary,
        session_indices=np.asarray(session_indices),
        step_offsets=np.r_[0, np.cumsum(lengths)[:-1]],
        lengths=lengths,
        directions=torch.cat([directions[row, :length] for row, length in enumerate(lengths)]).numpy(),
        confidences=torch.cat([confidences[row, :length] for row, length in enumerate(lengths)]).numpy(),
    )
    temporary.rename(path)


@torch.no_grad()
def evaluate(
    model,
    dataset: WorkflowDataset,
    split: str,
    candidates: TrainingCandidates,
    config: EvalConfig,
    device,
    embeddings_path: Path | None = None,
) -> dict:
    """All metrics of one split, named <split>_<metric>. The candidates are the training keys plus the split's."""
    session_indices = dataset.sessions_in(split)
    n_sessions = len(session_indices)
    lengths = dataset.lengths[session_indices].astype(np.int64)
    directions, confidences = embed_sessions(model, dataset, session_indices, config, device)
    if embeddings_path is not None:
        save_embeddings(embeddings_path, session_indices, directions, confidences, lengths)

    keys = torch.cat([candidates.keys, torch.from_numpy(dataset.label_keys[session_indices]).to(device)])
    groups = torch.cat([candidates.groups, torch.from_numpy(dataset.groups[session_indices]).to(device)])
    targets = torch.arange(n_sessions, device=device) + len(candidates)
    lengths = torch.from_numpy(lengths)

    loss_sum, ranks = 0.0, []
    for start in range(0, n_sessions, SCORING_CHUNK):
        chunk = slice(start, start + SCORING_CHUNK)
        chunk_lengths = lengths[chunk].to(device)
        n_steps = int(chunk_lengths.max())
        valid = torch.arange(n_steps, device=device)[None, :] < chunk_lengths[:, None]
        mask = metrics.false_negative_mask(keys, groups, targets[chunk], config.duplicate_threshold)
        logits = metrics.candidate_logits(
            directions[chunk, :n_steps].to(device), confidences[chunk, :n_steps, None].to(device), keys, mask
        )
        loss, _ = metrics.info_nce(logits, targets[chunk], valid)
        loss_sum += loss.item() * len(chunk_lengths)
        ranks.append(metrics.target_ranks(metrics.at_last_step(logits, chunk_lengths), targets[chunk]).cpu())
    ranks = torch.cat(ranks)

    embeddings = metrics.at_last_step(directions, lengths).float()
    log_distances = metrics.avg_log_dist_knn(embeddings.to(device), keys, groups, groups[targets], config.n_neighbors)
    results = {
        "loss": loss_sum / n_sessions,
        "r@1": metrics.recall_at_k(ranks, 1),
        "r@5": metrics.recall_at_k(ranks, 5),
        f"avg_log_dist_{config.n_neighbors}NN": float(log_distances.mean()),
    }
    return {f"{split}_{name}": value for name, value in results.items()}


def format_metrics(results: dict, title: str = "") -> str:
    """A table with one row per metric and one column per split in `results`."""
    splits = [split for split in ("val", "test") if any(name.startswith(f"{split}_") for name in results)]
    metric_names = {name.split("_", 1)[1] for name in results if name.startswith(tuple(f"{s}_" for s in splits))}
    ordered = sorted(
        metric_names,
        key=lambda name: next(
            (rank for rank, prefix in enumerate(METRIC_ORDER) if name.startswith(prefix)), len(METRIC_ORDER)
        ),
    )
    lines = [title] if title else []
    lines.append(f"{'metric':<22}" + "".join(f"{split:>10}" for split in splits))
    for name in ordered:
        values = [results.get(f"{split}_{name}") for split in splits]
        cells = [f"{value:>10.4f}" if isinstance(value, (int, float)) else f"{'-':>10}" for value in values]
        lines.append(f"{name:<22}" + "".join(cells))
    return "\n".join(lines)


def main(
    id: str,
    split: str = "test",
    checkpoint: str = "best.pt",
    save_embeddings: bool = False,
    finalize: bool = False,
    results_root: str = "",
    data: DataConfig = DataConfig(),
    eval: EvalConfig = EvalConfig(),
):
    """finalize: redo a run's final evaluation (val and test, test embeddings saved) and write its final.json."""
    from .model import load_checkpoint

    experiment = Experiment.from_id(id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    run_dir = Path(results_root or default_results_root()) / experiment.relative_dir
    dataset = WorkflowDataset(data, experiment)
    candidates = TrainingCandidates(dataset, device)
    model, saved = load_checkpoint(run_dir / checkpoint, torch.from_numpy(dataset.text_embeddings), device)

    started = time.time()
    results = {}
    for evaluated_split in ("val", "test") if finalize else (split,):
        save = save_embeddings or (finalize and evaluated_split == "test")
        embeddings_path = run_dir / f"{evaluated_split}_embeddings.npz" if save else None
        results.update(evaluate(model, dataset, evaluated_split, candidates, eval, device, embeddings_path))
    if finalize:
        final = {"best_step": saved["step"], "selected_on": "val_loss", **results}
        (run_dir / "final.json").write_text(to_json(final, indent=1))
    title = f"{experiment.id}\n{checkpoint} (step {saved['step']}), {time.time() - started:.0f}s"
    print(format_metrics(results, title))
    return results


if __name__ == "__main__":
    chz.entrypoint(main)
