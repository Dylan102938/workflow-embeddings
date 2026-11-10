"""The training objective and every tracked metric. Definitions: docs/EXPERIMENTS.md.

Shapes: B sessions, T steps, N candidate keys, D label dimension. Distances are cosine distances.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------- objective
def false_negative_mask(
    keys: torch.Tensor, groups: torch.Tensor, targets: torch.Tensor, duplicate_threshold: float
) -> torch.Tensor:
    """[B, N], True where a candidate must not count as a negative for the session.

    That is: it has the target's instruction group, or its key is closer to the target's than the threshold.
    The target itself is never masked.
    """
    same_group = groups[targets][:, None] == groups[None, :]
    near_duplicate = keys[targets] @ keys.T > duplicate_threshold
    mask = same_group | near_duplicate
    mask[torch.arange(len(targets), device=keys.device), targets] = False
    return mask


def candidate_logits(
    directions: torch.Tensor, confidences: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """[B, T, N] float32: confidence * cos(direction, key), with masked candidates at -inf."""
    similarities = torch.einsum("btd,nd->btn", directions.float(), keys.float())
    return (confidences.float() * similarities).masked_fill(mask[:, None, :], float("-inf"))


def smoothed_targets(
    keys: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, smoothing: float, temperature: float
) -> torch.Tensor:
    """[B, N]: 1 - smoothing on the target, the rest spread over the other unmasked candidates by key similarity."""
    rows = torch.arange(len(targets), device=keys.device)
    similarities = (keys[targets] @ keys.T / temperature).masked_fill(mask, float("-inf"))
    similarities[rows, targets] = float("-inf")
    distribution = smoothing * similarities.softmax(-1)
    distribution[rows, targets] += 1.0 - smoothing
    return distribution


def hard_negative_bias(keys: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    """[B, N] logit bias that weights negative j by exp(beta * cos(target key, key j)), normalized to mean 1."""
    rows = torch.arange(len(targets), device=keys.device)
    is_negative = ~mask
    is_negative[rows, targets] = False
    scores = beta * (keys[targets] @ keys.T)
    log_mean_weight = (
        torch.logsumexp(scores.masked_fill(~is_negative, float("-inf")), -1, keepdim=True)
        - is_negative.sum(-1, keepdim=True).log()
    )
    return (scores - log_mean_weight).masked_fill(~is_negative, 0.0)


def info_nce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    soft_targets: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
):
    """Cross-entropy of every step, averaged over each session's valid steps, then over sessions.

    soft_targets [B, N] replace the one-hot targets (and need the false-negative mask). Returns the loss and the
    per-step cross-entropy [B, T].
    """
    batch_size, n_steps, n_candidates = logits.shape
    if soft_targets is None:
        per_step = F.cross_entropy(
            logits.reshape(-1, n_candidates), targets.repeat_interleave(n_steps), reduction="none"
        ).view(batch_size, n_steps)
    else:
        log_probabilities = logits.log_softmax(-1).masked_fill(mask[:, None, :], 0.0)
        per_step = -(soft_targets[:, None, :] * log_probabilities).sum(-1)
    valid = valid.float()
    per_step = per_step * valid
    return (per_step.sum(1) / valid.sum(1)).mean(), per_step


# ---------------------------------------------------------------------------------------------- embedding metrics
def at_last_step(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """[B, T, ...] -> [B, ...]: each session's value at its last valid step."""
    return values[torch.arange(len(lengths), device=values.device), lengths.to(values.device) - 1]


def avg_dist(embeddings: torch.Tensor, label_keys: torch.Tensor, label_groups: torch.Tensor) -> torch.Tensor:
    """Mean distance from each session's embedding to the labels of the other sessions in the batch.

    Sessions with the same instruction group share a label and are excluded.
    """
    distances = 1.0 - F.normalize(embeddings.float(), dim=-1) @ label_keys.float().T
    is_other = label_groups[:, None] != label_groups[None, :]
    return (distances * is_other).sum() / is_other.sum().clamp(min=1)


def target_ranks(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """[B] rank of each target among the unmasked candidates (1 = best), from logits [B, N]."""
    target_logits = logits.gather(1, targets[:, None])
    return (logits > target_logits).sum(1) + 1


def recall_at_k(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= k).float().mean())


def avg_log_dist_knn(
    embeddings: torch.Tensor,
    keys: torch.Tensor,
    groups: torch.Tensor,
    target_groups: torch.Tensor,
    n_neighbors: int = 100,
    chunk_size: int = 256,
) -> torch.Tensor:
    """[S] mean log distance from each embedding to its nearest candidate keys, its own label excluded."""
    results = []
    for start in range(0, len(embeddings), chunk_size):
        chunk = F.normalize(embeddings[start : start + chunk_size].float(), dim=-1)
        similarities, neighbors = (chunk @ keys.float().T).topk(min(n_neighbors, len(keys)), dim=1)
        is_other = groups[neighbors] != target_groups[start : start + chunk_size, None]
        log_distances = torch.log((1.0 - similarities).clamp(min=1e-6))
        results.append((log_distances * is_other).sum(1) / is_other.sum(1).clamp(min=1))
    return torch.cat(results)


# ----------------------------------------------------------------------------------------------- judged retrieval
SAME_WORKFLOW = 2  # relevance grades: 2 = same workflow, 1 = related, 0 = different


def load_judged(path: Path, session_indices: np.ndarray, groups: np.ndarray) -> dict | None:
    """Graded relevance {query position: {other position: grade}} among the given sessions.

    Combines the LLM-judged pairs in `path` with duplicate-instruction groups (grade 2). None without the file.
    """
    if not Path(path).exists():
        return None
    position = {int(session): index for index, session in enumerate(session_indices)}
    relevance = collections.defaultdict(dict)
    for line in open(path):
        pair = json.loads(line)
        first, second = position.get(pair["a"]), position.get(pair["b"])
        if first is None or second is None:
            continue
        grade = int(pair["label"])
        relevance[first][second] = max(relevance[first].get(second, 0), grade)
        relevance[second][first] = max(relevance[second].get(first, 0), grade)
    members = collections.defaultdict(list)
    for index, group in enumerate(np.asarray(groups).tolist()):
        members[group].append(index)
    for group_members in members.values():
        for first in group_members:
            for second in group_members:
                if first != second:
                    relevance[first][second] = SAME_WORKFLOW
    return relevance


def judged_retrieval(embeddings: torch.Tensor, relevance: dict, k: int = 10) -> dict:
    """Session-to-session retrieval scored with pooled judged relevance (unjudged pairs count as 0)."""
    embeddings = F.normalize(embeddings.float(), dim=-1)
    similarities = embeddings @ embeddings.T
    similarities.fill_diagonal_(-float("inf"))
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    hits_at_1, hits_at_5, ndcgs = [], [], []
    for query, grades in relevance.items():
        if not any(grade > 0 for grade in grades.values()):
            continue
        ranking = similarities[query].argsort(descending=True).tolist()
        gains = np.array([grades.get(candidate, 0) for candidate in ranking[:k]], float)
        ideal = np.sort(np.array(list(grades.values()), float))[::-1][:k]
        dcg = ((2**gains - 1) * discounts[: len(gains)]).sum()
        ndcgs.append(dcg / ((2**ideal - 1) * discounts[: len(ideal)]).sum())
        if SAME_WORKFLOW in grades.values():
            rank = (
                next(index for index, candidate in enumerate(ranking) if grades.get(candidate, 0) == SAME_WORKFLOW) + 1
            )
            hits_at_1.append(rank <= 1)
            hits_at_5.append(rank <= 5)

    def mean(values):
        return float(np.mean(values)) if values else float("nan")

    return {"judged_r@1": mean(hits_at_1), "judged_r@5": mean(hits_at_5), f"judged_ndcg@{k}": mean(ndcgs)}
