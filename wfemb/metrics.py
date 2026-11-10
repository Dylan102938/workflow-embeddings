"""The training objective and every tracked metric (definitions: docs/EXPERIMENTS.md).

Shapes: B sessions, T steps, N candidate keys, D label dimension. d [B, T, D] unit directions, kappa [B, T, 1]
confidences, valid [B, T] step mask, target [B] each session's own column among the candidates. A session's workflow
embedding is its direction at its last step (terminal()). Distances are cosine distances 1 - cos.

Objective:  false_negative_mask, logits, smooth_targets, hard_negative_bias, info_nce
Training:   loss (info_nce with smoothing + hard negatives), rigid_loss (info_nce one-hot), avg_dist
Evaluation: info_nce (loss), target_rank -> recall_at_k (r@k), log_dist_knn (avg_log_dist_100NN),
            load_judged + judged_retrieval (judged_r@k, judged_ndcg@10)
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------------------- objective
def false_negative_mask(U, cand_group, target_key, target_group, target, theta: float) -> torch.Tensor:
    """[B, N] True where a candidate must not count as a negative: the same instruction group as the target, or a
    near-duplicate intent, cos(target key, candidate) > theta. The target column itself is never masked."""
    m = (target_group[:, None] == cand_group[None, :]) | (target_key @ U.T > theta)
    m[torch.arange(len(target), device=U.device), target] = False
    return m


def logits(d, kappa, U, fn_mask) -> torch.Tensor:
    """[B, T, N] fp32: kappa_t * cos(d_t, U_j); masked candidates -inf."""
    out = kappa.float() * torch.einsum("btd,nd->btn", d.float(), U.float())
    return out.masked_fill(fn_mask[:, None, :], float("-inf"))


def smooth_targets(U, target_key, target, fn_mask, eps: float, tau: float) -> torch.Tensor:
    """[B, N] (1 - eps) * one-hot(target) + eps * softmax(cos(target key, U) / tau) over the unmasked candidates other
    than the target itself (its self-similarity would otherwise absorb the smoothing mass)."""
    rows = torch.arange(len(target), device=U.device)
    sims = (target_key @ U.T / tau).masked_fill(fn_mask, float("-inf"))
    sims[rows, target] = float("-inf")
    q = eps * sims.softmax(-1)
    q[rows, target] += 1.0 - eps
    return q


def hard_negative_bias(U, target_key, target, fn_mask, beta: float) -> torch.Tensor:
    """[B, N] additive logit bias weighting each negative j by w_j = exp(beta * cos(target key, U_j)), normalized so a
    row's negative weights average 1 (Robinson et al. 2021, hardness measured in the label space). Target column: 0."""
    rows = torch.arange(len(target), device=U.device)
    s = beta * (target_key @ U.T)
    neg = ~fn_mask
    neg[rows, target] = False
    lme = torch.logsumexp(s.masked_fill(~neg, float("-inf")), -1, keepdim=True) - neg.sum(-1, keepdim=True).log()
    return (s - lme).masked_fill(~neg, 0.0)


def info_nce(lg, target, valid, soft_targets=None, fn_mask=None):
    """Per-step InfoNCE. lg [B, T, N] logits, valid [B, T]. soft_targets [B, N] (training label smoothing; needs
    fn_mask) replace the one-hot target. Returns (mean over sessions of the mean over their valid steps, ce [B, T])."""
    B, T, N = lg.shape
    if soft_targets is None:
        ce = F.cross_entropy(lg.reshape(B * T, N), target.repeat_interleave(T), reduction="none").view(B, T)
    else:
        logp = lg.log_softmax(-1).masked_fill(fn_mask[:, None, :], 0.0)     # masked columns have q = 0
        ce = -(soft_targets[:, None, :] * logp).sum(-1)
    valid = valid.float()
    ce = ce * valid
    return (ce.sum(1) / valid.sum(1)).mean(), ce


# ---------------------------------------------------------------------------------------------- embedding metrics
def terminal(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """[B, T, ...] -> [B, ...] at each session's last (valid) step."""
    return x[torch.arange(len(lengths), device=x.device), lengths.to(x.device) - 1]


def avg_dist(e, label_keys, label_group) -> torch.Tensor:
    """Mean distance between each session's workflow embedding e [B, D] and the label keys [B, D] of the other sessions
    in the batch, excluding its own label (its key and those of sessions with the same instruction group)."""
    dist = 1.0 - F.normalize(e.float(), dim=-1) @ label_keys.float().T          # [B, B]
    other = label_group[:, None] != label_group[None, :]
    return (dist * other).sum() / other.sum().clamp(min=1)


def target_rank(lg_last, target) -> torch.Tensor:
    """[B] rank of the target among the unmasked candidates (1 = first), lg_last [B, N] logits at the last step."""
    true = lg_last.gather(1, target[:, None])
    return (lg_last > true).sum(1) + 1


def recall_at_k(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= k).float().mean())


def log_dist_knn(e, U_ref, ref_group, target_group, k: int = 100, chunk: int = 256) -> torch.Tensor:
    """[S] for each workflow embedding e [S, D]: the k candidates of U_ref [N, D] nearest to it, minus its own label
    (candidates of the target's instruction group) if among them; the mean log distance to the rest."""
    out = []
    for s in range(0, len(e), chunk):
        es = F.normalize(e[s:s + chunk].float(), dim=-1)
        sims, idx = (es @ U_ref.float().T).topk(min(k, len(U_ref)), dim=1)        # [c, k]
        keep = ref_group[idx] != target_group[s:s + chunk, None]
        logd = torch.log((1.0 - sims).clamp(min=1e-6))
        out.append((logd * keep).sum(1) / keep.sum(1).clamp(min=1))
    return torch.cat(out)


# ------------------------------------------------------------------------------------------- judged retrieval
def load_judged(path: Path, sidx: np.ndarray, groups: np.ndarray):
    """Graded relevance between the sessions of a split, keyed by position in sidx: LLM-judged pairs (0/1/2, from
    prep/judge_pool.py) plus duplicate-instruction groups (gain 2). None if the split has no judgments."""
    if not Path(path).exists():
        return None
    pos = {int(s): i for i, s in enumerate(sidx)}
    rel = collections.defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        a, b = pos.get(r["a"]), pos.get(r["b"])
        if a is None or b is None:
            continue
        g = int(r["label"])
        rel[a][b] = max(rel[a].get(b, 0), g)
        rel[b][a] = max(rel[b].get(a, 0), g)
    members = collections.defaultdict(list)
    for i, g in enumerate(np.asarray(groups).tolist()):
        members[g].append(i)
    for ms in members.values():
        for a in ms:
            for b in ms:
                if a != b:
                    rel[a][b] = 2
    return rel


def judged_retrieval(E: torch.Tensor, rel, k: int = 10) -> dict:
    """Session-to-session retrieval across apps (the query's own app is not excluded) with pooled judged relevance
    (unjudged = 0). r@1 / r@5 over queries with a 'same workflow' (gain 2) partner; nDCG@k (gains 2^g - 1) over
    queries with any positive gain."""
    E = F.normalize(E.float(), dim=-1)
    S = E @ E.T
    S.fill_diagonal_(-float("inf"))
    r1, r5, ndcg = [], [], []
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for q, cands in rel.items():
        if not any(g > 0 for g in cands.values()):
            continue
        order = S[q].argsort(descending=True).tolist()
        gains = np.array([cands.get(j, 0) for j in order[:k]], float)
        ideal = np.sort(np.array(list(cands.values()), float))[::-1][:k]
        ndcg.append(((2 ** gains - 1) * disc[:len(gains)]).sum() / ((2 ** ideal - 1) * disc[:len(ideal)]).sum())
        if any(g == 2 for g in cands.values()):
            rank = next(i for i, j in enumerate(order) if cands.get(j, 0) == 2) + 1
            r1.append(rank <= 1)
            r5.append(rank <= 5)
    return {"judged_r@1": float(np.mean(r1)) if r1 else float("nan"),
            "judged_r@5": float(np.mean(r5)) if r5 else float("nan"),
            f"judged_ndcg@{k}": float(np.mean(ndcg)) if ndcg else float("nan")}
